"""
Predict with retrieval labels only:
  - Classification: majority voting over retrieved labels.
  - Regression: mean over retrieved labels.
No model training or checkpoint loading is used.
"""
import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import BCELoss, L1Loss
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from dataset import (
    EntityTimeSeriesBuilder,
    create_ar_dataloaders,
    create_random_ar_dataloaders,
)


parser = argparse.ArgumentParser(description="Predict using retrieved labels (vote/mean)")
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--task", type=str, default="driver-top3")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--cache_dir", type=str, default=os.path.expanduser("/data/starlab/relbench_examples"))
parser.add_argument("--backbone", type=str, default="relgnn", choices=["rdl", "relgnn", "relgt"])
parser.add_argument("--index_path", type=str, default="/data/relts/snapshots")
parser.add_argument("--window_size", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--mode", type=str, default="recent", choices=["recent", "random"])
parser.add_argument("--top_k", type=int, default=5, help="Number of retrieved labels to use at prediction time")
parser.add_argument(
    "--ref_baseline",
    type=str,
    default="retrieval",
    choices=["retrieval", "random"],
    help="retrieval: use retrieved refs; random: random baseline",
)
parser.add_argument("--report", action="store_true", help="Report results to JSON")
parser.add_argument("--report_path", type=str, default="./results")
parser.add_argument("--tag", type=str, default="default", help="Tag for the experiment")
parser.add_argument(
    "--ignore_train_config",
    action="store_true",
    help="If set, ignore train_config from relgnn.utils and use CLI arguments as-is.",
)

args = parser.parse_args()


def cli_flag_provided(flag_name: str) -> bool:
    option = f"--{flag_name}"
    return any(arg == option or arg.startswith(f"{option}=") for arg in sys.argv[1:])

# Use train_config from relgnn.utils.get_configs when available.
train_config = None
if not args.ignore_train_config:
    try:
        from relgnn.utils import get_configs

        res = get_configs(args.dataset, args.task, args.backbone)
        if res is not None and len(res) == 3:
            train_config = res[2]
    except Exception:
        pass

if train_config:
    explicit_cli = {
        "window_size": cli_flag_provided("window_size"),
        "seed": cli_flag_provided("seed"),
    }
    if "window_size" in train_config and not explicit_cli["window_size"]:
        args.window_size = int(train_config["window_size"])
    if "seed" in train_config and not explicit_cli["seed"]:
        args.seed = int(train_config["seed"])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

get_dataset(args.dataset, download=True)
task: EntityTask = get_task(args.dataset, args.task, download=True)

clamp_min, clamp_max = None, None
if task.task_type == TaskType.BINARY_CLASSIFICATION:
    loss_fn = BCELoss()
    tune_metric = "roc_auc"
elif task.task_type == TaskType.REGRESSION:
    loss_fn = L1Loss()
    tune_metric = "mae"
    train_table = task.get_table("train")
    clamp_min, clamp_max = np.percentile(train_table.df[task.target_col].to_numpy(), [2, 98])
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    raise ValueError("main_predict_vote.py currently supports binary classification or regression only.")
else:
    raise ValueError(f"Task type {task.task_type} is unsupported")

builder = EntityTimeSeriesBuilder(
    index_path=args.index_path,
    dataset_name=args.dataset,
    task_name=args.task,
    task=task,
    backbone=args.backbone,
    use_random_embedding=False,
)

if args.mode == "recent":
    ar_loader_dict = create_ar_dataloaders(
        entity_sequences=builder.entity_sequences,
        split_indices=builder.split_indices,
        window_size=args.window_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_input_length=0,
    )
else:
    ar_loader_dict = create_random_ar_dataloaders(
        entity_sequences=builder.entity_sequences,
        split_indices=builder.split_indices,
        window_size=args.window_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_input_length=0,
        samples_per_epoch=None,
        seed=args.seed,
    )

retrieval_root = os.path.join(args.index_path, args.backbone, args.dataset, args.task)
cls_meta = torch.load(os.path.join(retrieval_root, "cls_meta.pt"), map_location="cpu")
cls_entity_ids = cls_meta["entity_ids"].long()
cls_timestamps = cls_meta["timestamps"].long()
cls_lookup = np.load(os.path.join(retrieval_root, "cls_lookup.npz"))
cls_key_sorted = cls_lookup["key_sorted"].astype(np.uint64, copy=False)
cls_row_ids_sorted = cls_lookup["row_ids_sorted"].astype(np.int64, copy=False)
n_total = int(cls_entity_ids.size(0))

# Build (entity_id, timestamp) -> label from the same source as indexing/dataset.
id_time_to_label = {}
train_labels_list = []
for split in ("train", "val", "test"):
    for entity_id, seq in builder.entity_sequences[split].items():
        for ts, _emb, label, _ent_emb in seq:
            id_time_to_label[(int(entity_id), int(ts))] = float(label)
            if split == "train":
                train_labels_list.append(float(label))

if not train_labels_list:
    raise ValueError("No train labels found for fallback prediction")

classification_fallback = float(np.mean(train_labels_list))
regression_fallback = float(np.mean(train_labels_list))

cls_labels = torch.zeros(n_total, dtype=torch.float32)
for i in range(n_total):
    key = (int(cls_entity_ids[i].item()), int(cls_timestamps[i].item()))
    cls_labels[i] = float(id_time_to_label.get(key, regression_fallback))


top5_npy_path = os.path.join(retrieval_root, "top5_indices.npy")
if not os.path.exists(top5_npy_path):
    raise FileNotFoundError(f"Missing retrieval top-k file: {top5_npy_path}")


top5_indices_mm = np.load(top5_npy_path, mmap_mode="r")
if args.top_k <= 0:
    raise ValueError(f"--top_k must be >= 1, got {args.top_k}")
if top5_indices_mm.ndim != 2:
    raise ValueError(f"Expected top-k index array with 2 dims, got shape {top5_indices_mm.shape}")
available_top_k = int(top5_indices_mm.shape[1])
effective_top_k = min(args.top_k, available_top_k)
if effective_top_k < args.top_k:
    print(
        f"Requested top_k={args.top_k}, but only {available_top_k} refs are available in index. "
        f"Using top_k={effective_top_k}."
    )
cls_labels_np = cls_labels.numpy().astype(np.float32, copy=False)

# RNG for random ref baseline (reproducible with same seed).
rng = np.random.default_rng(args.seed)


def lookup_row_indices(entity_ids_tensor, timestamps_tensor, key_sorted, row_ids_sorted):
    """Vectorized lookup for (entity_id, timestamp) -> row index; returns -1 for missing."""
    eids = entity_ids_tensor.detach().cpu().numpy().astype(np.uint64)
    ts = timestamps_tensor.detach().cpu().long().numpy().astype(np.uint64)
    query = (eids << np.uint64(32)) | (ts & np.uint64(0xFFFFFFFF))
    pos = np.searchsorted(key_sorted, query)
    idx = np.full(query.shape[0], -1, dtype=np.int64)
    in_bounds = pos < key_sorted.shape[0]
    found = np.zeros(query.shape[0], dtype=bool)
    found[in_bounds] = key_sorted[pos[in_bounds]] == query[in_bounds]
    idx[found] = row_ids_sorted[pos[found]]
    return idx


def augment_batch_with_retrieval_labels(batch):
    entity_ids = batch["entity_id"].cpu()
    timestamps = batch["target_timestamp"].cpu()
    batch_idx = lookup_row_indices(entity_ids, timestamps, cls_key_sorted, cls_row_ids_sorted)

    batch_idx_safe = np.maximum(batch_idx, 0)
    top5_for_batch = top5_indices_mm[batch_idx_safe, :effective_top_k]
    ref_mask_np = (batch_idx[:, None] >= 0) & (top5_for_batch >= 0)
    flat_idx = np.maximum(top5_for_batch, 0)

    retrieved_labels_np = cls_labels_np[flat_idx]
    batch["retrieved_labels"] = torch.from_numpy(np.asarray(retrieved_labels_np)).to(
        device=device, dtype=torch.float32
    )
    batch["retrieved_ref_mask"] = torch.from_numpy(ref_mask_np).to(device=device, dtype=torch.bool)
    return batch


def augment_batch_with_random_labels(batch):
    bsz = batch["entity_id"].size(0)
    rand_indices = rng.integers(0, n_total, size=(bsz, effective_top_k), dtype=np.int64)
    retrieved_labels_np = cls_labels_np[rand_indices]

    batch["retrieved_labels"] = torch.from_numpy(np.asarray(retrieved_labels_np)).to(
        device=device, dtype=torch.float32
    )
    batch["retrieved_ref_mask"] = torch.ones(bsz, effective_top_k, dtype=torch.bool, device=device)
    return batch


def augment_batch(batch):
    if args.ref_baseline == "random":
        return augment_batch_with_random_labels(batch)
    return augment_batch_with_retrieval_labels(batch)


def predict_from_retrieved_labels(retrieved_labels, retrieved_ref_mask):
    mask_f = retrieved_ref_mask.float()
    valid_count = retrieved_ref_mask.sum(dim=1)

    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        hard = (retrieved_labels >= 0.5).float()
        vote_sum = (hard * mask_f).sum(dim=1)
        denom = valid_count.clamp(min=1).float()
        vote_ratio = vote_sum / denom
        fallback = torch.full_like(vote_ratio, classification_fallback)
        vote_ratio = torch.where(valid_count > 0, vote_ratio, fallback)
        return vote_ratio

    if task.task_type == TaskType.REGRESSION:
        denom = valid_count.clamp(min=1).float()
        pred = (retrieved_labels * mask_f).sum(dim=1) / denom
        fallback = torch.full_like(pred, regression_fallback)
        pred = torch.where(valid_count > 0, pred, fallback)
        return pred.clamp(clamp_min, clamp_max)

    raise ValueError(f"Task type {task.task_type} is unsupported")


@torch.no_grad()
def evaluate(loader, split_name):
    total_loss, total_samples = 0.0, 0
    all_preds, all_targets = [], []

    for batch in tqdm(loader, desc=f"Eval-{split_name}"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch = augment_batch(batch)

        target = batch["target_label"].float()
        preds = predict_from_retrieved_labels(batch["retrieved_labels"], batch["retrieved_ref_mask"])

        loss = loss_fn(preds, target)
        total_loss += loss.item() * target.size(0)
        total_samples += target.size(0)

        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

    avg_loss = total_loss / total_samples if total_samples else 0.0
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = {}
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        try:
            metrics["roc_auc"] = float(roc_auc_score(all_targets, all_preds))
        except ValueError:
            metrics["roc_auc"] = float("nan")
        pred_cls = (all_preds > 0.5).astype(int)
        metrics["accuracy"] = float(accuracy_score(all_targets, pred_cls))
        metrics["f1"] = float(f1_score(all_targets, pred_cls))
    elif task.task_type == TaskType.REGRESSION:
        metrics["mae"] = float(mean_absolute_error(all_targets, all_preds))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(all_targets, all_preds)))
        metrics["r2"] = float(r2_score(all_targets, all_preds))

    return avg_loss, metrics


print("\nPredicting from retrieval labels only (no training)")
print(f"Ref baseline: {args.ref_baseline}")
print(f"Top-k labels used at predict time: {effective_top_k} (requested={args.top_k})")
print("=" * 80)

val_loss, val_metrics = evaluate(ar_loader_dict["val"], "val")
print(f"Val Loss: {val_loss:.4f}")
print(f"Val Metrics: {val_metrics}")
if tune_metric in val_metrics:
    print(f"Val {tune_metric}: {val_metrics[tune_metric]:.4f}")

print("\nEvaluating on test set...")
test_loss, test_metrics = evaluate(ar_loader_dict["test"], "test")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Metrics: {test_metrics}")
if tune_metric in test_metrics:
    print(f"Test {tune_metric}: {test_metrics[tune_metric]:.4f}")
print("=" * 80)

if args.report:
    results_path = Path(args.report_path) / f"{args.dataset}_{args.task}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if results_path.exists():
        try:
            with open(results_path, "r") as f:
                content = f.read().strip()
                all_results = json.loads(content) if content else {}
        except (json.JSONDecodeError, ValueError):
            all_results = {}
    else:
        all_results = {}

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hyperparams = json.dumps(
        {
            "stage": "predict_vote",
            "backbone": args.backbone,
            "mode": args.mode,
            "top_k": args.top_k,
            "effective_top_k": effective_top_k,
            "ref_baseline": args.ref_baseline,
            "tag": args.tag,
            "window_size": args.window_size,
        }
    )

    result_entry = {
        "source": "predict_vote",
        "seed": int(args.seed),
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }
    all_results.setdefault(hyperparams, {})[timestamp] = result_entry

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")


gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
