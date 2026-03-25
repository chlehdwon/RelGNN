"""
Train predictor with in-memory retrieval cache augmentation.

Key differences from main_predict.py:
  - No dependency on precomputed top-k file from main_retrieve.py.
  - Build retrieval top-k cache once at startup (in memory only, no file write).
  - Training/eval batches use fast lookup + gather instead of per-sample dynamic retrieval.
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
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task

from dataset import (
    EntityTimeSeriesBuilder,
    create_ar_dataloaders,
    create_random_ar_dataloaders,
)
from model import RelTS_Model


parser = argparse.ArgumentParser(
    description="Train predictor with in-memory retrieval-cache augmentation"
)
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--task", type=str, default="driver-top3")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("/data/starlab/relbench_examples"),
)
parser.add_argument(
    "--backbone",
    type=str,
    default="relgnn",
    choices=["rdl", "relgnn", "relgt"],
)
parser.add_argument("--index_path", type=str, default="/data/relts/snapshots")
parser.add_argument("--window_size", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--weight_decay", type=float, default=5e-5)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--ff_dim", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--mode", type=str, default="recent", choices=["recent", "random"])
parser.add_argument("--top_k", type=int, default=5, help="Number of retrieved refs")
parser.add_argument(
    "--retrieval_chunk_size",
    type=int,
    default=512,
    help="Chunk size for one-time retrieval cache precompute",
)
parser.add_argument("--use_entity_embedding", action=argparse.BooleanOptionalAction)
parser.add_argument(
    "--ref_baseline",
    type=str,
    default="retrieval",
    choices=["retrieval", "random"],
    help="retrieval: context-key retrieval, random: random augmentation baseline",
)
parser.add_argument("--report", action="store_true")
parser.add_argument("--report_path", type=str, default="./results")
parser.add_argument("--tag", type=str, default="default")
parser.add_argument(
    "--ignore_train_config",
    action="store_true",
    help="If set, ignore train_config from relgnn.utils and use CLI arguments as-is.",
)
args = parser.parse_args()


def cli_flag_provided(flag_name: str) -> bool:
    option = f"--{flag_name}"
    return any(arg == option or arg.startswith(f"{option}=") for arg in sys.argv[1:])

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
        "lr": cli_flag_provided("lr"),
        "weight_decay": cli_flag_provided("weight_decay"),
        "window_size": cli_flag_provided("window_size"),
        "seed": cli_flag_provided("seed"),
    }
    if "lr" in train_config and not explicit_cli["lr"]:
        args.lr = float(train_config["lr"])
    if "weight_decay" in train_config and not explicit_cli["weight_decay"]:
        args.weight_decay = float(train_config["weight_decay"])
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

if task.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION):
    out_channels = 1
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    out_channels = task.num_labels
else:
    raise ValueError(f"Task type {task.task_type} is unsupported")

clamp_min, clamp_max = None, None
if task.task_type == TaskType.BINARY_CLASSIFICATION:
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "roc_auc"
    higher_is_better = True
elif task.task_type == TaskType.REGRESSION:
    loss_fn = L1Loss()
    tune_metric = "mae"
    higher_is_better = False
    train_table = task.get_table("train")
    clamp_min, clamp_max = np.percentile(
        train_table.df[task.target_col].to_numpy(), [2, 98]
    )
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "multilabel_auprc_macro"
    higher_is_better = True
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

channels = None
for split in ("train", "val", "test"):
    sequences = builder.entity_sequences[split]
    if sequences:
        first_seq = next(iter(sequences.values()))
        if first_seq:
            channels = first_seq[0][1].shape[0]
            break
if channels is None:
    raise ValueError("Could not determine embedding dimension from snapshots")

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

entity_embed_dim = builder.entity_embeddings.shape[1] if args.use_entity_embedding else None
model = RelTS_Model(
    channels=channels,
    task_type=task.task_type,
    entity_embed_dim=entity_embed_dim,
    use_entity_embedding=args.use_entity_embedding,
    num_heads=args.num_heads,
    num_layers=args.num_layers,
    ff_dim=args.ff_dim,
    dropout=args.dropout,
    num_classes=out_channels,
).to(device)


# Retrieval DB: backbone snapshot embeddings + metadata.
retrieval_root = os.path.join(args.index_path, args.backbone, args.dataset, args.task)
snapshot_emb_path = os.path.join(retrieval_root, "snapshot_embeddings.npy")
snapshot_meta_path = os.path.join(retrieval_root, "snapshot_meta.npz")
if not os.path.exists(snapshot_emb_path):
    raise FileNotFoundError(f"Missing snapshot embeddings: {snapshot_emb_path}")
if not os.path.exists(snapshot_meta_path):
    raise FileNotFoundError(f"Missing snapshot metadata: {snapshot_meta_path}")

snapshot_embeddings_mm = np.load(snapshot_emb_path, mmap_mode="r")
snapshot_meta = np.load(snapshot_meta_path)
snapshot_entity_ids = snapshot_meta["entity_ids"].astype(np.int64, copy=False)
snapshot_timestamps = snapshot_meta["timestamps"].astype(np.int64, copy=False)
n_total = int(snapshot_embeddings_mm.shape[0])

if args.top_k <= 0:
    raise ValueError(f"--top_k must be >= 1, got {args.top_k}")
effective_top_k = int(args.top_k)

# Build label lookup from the same source as dataloaders.
id_time_to_label = {}
train_labels_list = []
for split in ("train", "val", "test"):
    for entity_id, seq in builder.entity_sequences[split].items():
        for ts, _emb, label, _ent_emb in seq:
            id_time_to_label[(int(entity_id), int(ts))] = float(label)
            if split == "train":
                train_labels_list.append(float(label))

if not train_labels_list:
    raise ValueError("No train labels found for fallback label values")

fallback_label = float(np.mean(train_labels_list))
snapshot_labels = np.full((n_total,), fallback_label, dtype=np.float32)
for i in range(n_total):
    key = (int(snapshot_entity_ids[i]), int(snapshot_timestamps[i]))
    if key in id_time_to_label:
        snapshot_labels[i] = float(id_time_to_label[key])

# Build packed-key lookup: (entity_id, timestamp) -> snapshot row index.
packed_keys = (
    snapshot_entity_ids.astype(np.uint64) << np.uint64(32)
) | (snapshot_timestamps.astype(np.uint64) & np.uint64(0xFFFFFFFF))
packed_order = np.argsort(packed_keys, kind="mergesort")
key_sorted = packed_keys[packed_order]
row_ids_sorted = np.arange(n_total, dtype=np.int64)[packed_order]


def lookup_row_indices(entity_ids_tensor, timestamps_tensor):
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


def build_retrieval_topk_cache():
    """
    Precompute retrieval top-k for all snapshot rows (different entity + strict past).
    Returns int64 array of shape [n_total, K] with -1 padding.
    """
    print("\nPrecomputing retrieval top-k cache in memory...")
    print(
        f"Rule: candidate timestamp < target timestamp AND candidate entity_id != target entity_id"
    )
    print(f"n_total={n_total}, top_k={effective_top_k}, chunk_size={args.retrieval_chunk_size}")

    if n_total == 0:
        return np.full((0, effective_top_k), -1, dtype=np.int64)

    compute_device = device if torch.cuda.is_available() else torch.device("cpu")
    emb_np = np.asarray(snapshot_embeddings_mm, dtype=np.float32)
    emb_all = torch.from_numpy(emb_np).to(compute_device)
    emb_all = F.normalize(emb_all, p=2, dim=1)
    ent_all = torch.from_numpy(snapshot_entity_ids).to(compute_device)
    ts_all = torch.from_numpy(snapshot_timestamps).to(compute_device)

    topk_cache = np.full((n_total, effective_top_k), -1, dtype=np.int64)
    chunk = max(int(args.retrieval_chunk_size), 1)
    k_query = min(effective_top_k, n_total)
    with torch.no_grad():
        for start in tqdm(range(0, n_total, chunk), desc="BuildTopK"):
            end = min(start + chunk, n_total)
            q_emb = emb_all[start:end]  # [B, C]
            q_ent = ent_all[start:end]  # [B]
            q_ts = ts_all[start:end]    # [B]

            # cosine sim because both query/db are normalized
            scores = torch.matmul(q_emb, emb_all.t())  # [B, N]

            valid = (q_ts.unsqueeze(1) > ts_all.unsqueeze(0)) & (
                q_ent.unsqueeze(1) != ent_all.unsqueeze(0)
            )
            scores = scores.masked_fill(~valid, float("-inf"))

            topk_scores, topk_idx = torch.topk(scores, k=k_query, dim=1)
            valid_topk = torch.isfinite(topk_scores)

            idx_np = topk_idx.detach().cpu().numpy().astype(np.int64, copy=False)
            valid_np = valid_topk.detach().cpu().numpy()
            idx_np[~valid_np] = -1
            topk_cache[start:end, :k_query] = idx_np

    # Release large temporary tensors
    del emb_all, ent_all, ts_all
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Top-k cache precompute complete.")
    return topk_cache


topk_cache = None
if args.ref_baseline == "retrieval":
    topk_cache = build_retrieval_topk_cache()

rng = np.random.default_rng(args.seed)

def augment_batch_with_retrieval(batch: dict):
    """Fast retrieval via precomputed top-k cache + batch lookup/gather."""
    bsz = int(batch["entity_id"].size(0))
    batch_idx = lookup_row_indices(batch["entity_id"], batch["target_timestamp"])
    batch_idx_safe = np.maximum(batch_idx, 0)
    topk_for_batch = topk_cache[batch_idx_safe, :effective_top_k]
    ref_mask_np = (batch_idx[:, None] >= 0) & (topk_for_batch >= 0)
    flat_idx = np.maximum(topk_for_batch, 0)

    retrieved_emb = np.asarray(snapshot_embeddings_mm[flat_idx], dtype=np.float32)
    retrieved_labels = snapshot_labels[flat_idx]
    retrieved_ts = snapshot_timestamps[flat_idx].astype(np.float32, copy=False)
    retrieved_emb = retrieved_emb * ref_mask_np[..., None].astype(np.float32, copy=False)

    dev = batch["input_embeddings"].device
    batch["retrieved_cls_emb"] = torch.from_numpy(retrieved_emb).to(dev)
    batch["retrieved_labels"] = torch.from_numpy(retrieved_labels).to(dev)
    batch["retrieved_timestamps"] = torch.from_numpy(retrieved_ts).to(dev)
    batch["retrieved_ref_mask"] = torch.from_numpy(ref_mask_np).to(dev)
    return batch


def augment_batch_with_random_refs(batch: dict):
    bsz = int(batch["entity_id"].size(0))
    rand_indices = rng.integers(0, n_total, size=(bsz, effective_top_k), dtype=np.int64)
    dev = batch["input_embeddings"].device

    batch["retrieved_cls_emb"] = torch.from_numpy(
        np.asarray(snapshot_embeddings_mm[rand_indices], dtype=np.float32)
    ).to(dev)
    batch["retrieved_labels"] = torch.from_numpy(snapshot_labels[rand_indices]).to(dev)
    batch["retrieved_timestamps"] = torch.from_numpy(
        snapshot_timestamps[rand_indices].astype(np.float32, copy=False)
    ).to(dev)
    batch["retrieved_ref_mask"] = torch.ones(
        bsz, effective_top_k, dtype=torch.bool, device=dev
    )
    return batch


def augment_batch(batch: dict):
    if args.ref_baseline == "random":
        return augment_batch_with_random_refs(batch)
    return augment_batch_with_retrieval(batch)


optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train_epoch(loader):
    model.train()
    total_loss, total_samples, steps = 0.0, 0, 0
    for batch in tqdm(loader, desc="Train"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch = augment_batch(batch)
        logits = model(batch)
        target = batch["target_label"]
        if task.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION):
            loss = loss_fn(logits.squeeze(-1), target)
        else:
            loss = loss_fn(logits, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * target.size(0)
        total_samples += target.size(0)
        steps += 1
        if steps >= args.max_steps_per_epoch:
            break
    return total_loss / total_samples if total_samples else 0.0


@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_preds, all_targets = [], []

    for batch in tqdm(loader, desc="Eval"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch = augment_batch(batch)
        logits = model(batch)
        target = batch["target_label"]

        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            loss = loss_fn(logits.squeeze(-1), target)
            preds = torch.sigmoid(logits.squeeze(-1))
        elif task.task_type == TaskType.REGRESSION:
            loss = loss_fn(logits.squeeze(-1), target)
            preds = logits.squeeze(-1).clamp(clamp_min, clamp_max)
        else:
            loss = loss_fn(logits, target)
            preds = torch.sigmoid(logits)

        total_loss += loss.item() * target.size(0)
        total_samples += target.size(0)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

    avg_loss = total_loss / total_samples if total_samples else 0.0
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = {}
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        metrics["roc_auc"] = float(roc_auc_score(all_targets, all_preds))
        pred_cls = (all_preds > 0.5).astype(int)
        metrics["accuracy"] = float(accuracy_score(all_targets, pred_cls))
        metrics["f1"] = float(f1_score(all_targets, pred_cls))
    elif task.task_type == TaskType.REGRESSION:
        metrics["mae"] = float(mean_absolute_error(all_targets, all_preds))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(all_targets, all_preds)))
        metrics["r2"] = float(r2_score(all_targets, all_preds))
    else:
        aucs = []
        for i in range(all_targets.shape[1]):
            aucs.append(roc_auc_score(all_targets[:, i], all_preds[:, i]))
        metrics["multilabel_auprc_macro"] = float(np.mean(aucs))

    return avg_loss, metrics


print("\nTraining with in-memory retrieval cache")
print(f"Ref baseline: {args.ref_baseline}")
print(f"Top-k refs used at predict time: {effective_top_k}")
print("=" * 80)
print("Early stopping patience: 5 epochs")

best_val_metric = -float("inf") if higher_is_better else float("inf")
best_epoch = 0
best_state_dict = None
patience, patience_counter = 5, 0

for epoch in range(1, args.epochs + 1):
    print(f"\nEpoch {epoch}/{args.epochs}")
    print("-" * 80)
    train_loss = train_epoch(ar_loader_dict["train"])
    print(f"Train Loss: {train_loss:.4f}")

    val_loss, val_metrics = evaluate(ar_loader_dict["val"])
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Metrics: {val_metrics}")

    val_metric = val_metrics.get(tune_metric, val_loss)
    print(f"Val {tune_metric}: {val_metric:.4f}")

    is_best = (higher_is_better and val_metric > best_val_metric) or (
        (not higher_is_better) and val_metric < best_val_metric
    )
    if is_best:
        best_val_metric = val_metric
        best_epoch = epoch
        best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
        print(f"✓ New best model! (epoch {epoch}, {tune_metric}={val_metric:.4f})")
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epoch(s)")

    if patience_counter >= patience:
        print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
        print(f"Best model was at epoch {best_epoch} with {tune_metric}={best_val_metric:.4f}")
        break

print("\n" + "=" * 80)
print("Training completed!")
print(f"Best epoch: {best_epoch}")
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
if best_val_metric not in (-float("inf"), float("inf")):
    print(f"Best val {tune_metric}: {best_val_metric:.4f}")
print("=" * 80)

print("\nEvaluating on test set...")
test_loss, test_metrics = evaluate(ar_loader_dict["test"])
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Metrics: {test_metrics}")
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
            "stage": "only_predict",
            "backbone": args.backbone,
            "mode": args.mode,
            "top_k": args.top_k,
            "effective_top_k": effective_top_k,
            "ref_baseline": args.ref_baseline,
            "tag": args.tag,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "window_size": args.window_size,
        }
    )

    metrics_for_json = {k: float(v) for k, v in test_metrics.items()}
    result_entry = {
        "source": "only_predict",
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "test_metrics": metrics_for_json,
    }
    if best_val_metric not in (-float("inf"), float("inf")):
        result_entry["best_val_metric"] = float(best_val_metric)

    all_results.setdefault(hyperparams, {})[timestamp] = result_entry
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")


# Explicitly release memory at the end of the run to avoid buildup across repeated experiments.
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
