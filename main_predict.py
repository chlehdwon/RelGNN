"""
Fine-tune pretrained transformer with retrieved CLS refs:
  [CLS] [ref_1] ... [ref_5] [SEP] [ctx1] ... [ctx_k] -> predict target.
Ref token = retrieved cls_embedding + label_embedding (5 retrieved past same-entity snapshots).
"""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

from model import RelTS_Model
from dataset import EntityTimeSeriesBuilder, create_ar_dataloaders, create_random_ar_dataloaders

parser = argparse.ArgumentParser(description="Fine-tune with retrieved CLS ref tokens")
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--task", type=str, default="driver-top3")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--cache_dir", type=str, default=os.path.expanduser("/data/starlab/relbench_examples"))
parser.add_argument("--backbone", type=str, default="relgnn", choices=["rdl", "relgnn", "relgt"])
parser.add_argument("--results_path", type=str, default="/data/relts/ckpts")
parser.add_argument("--index_path", type=str, default="/data/relts/snapshots")
parser.add_argument("--window_size", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--weight_decay", type=float, default=1e-6)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--ff_dim", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--mode", type=str, default="recent", choices=["recent", "random"])
parser.add_argument("--use_entity_embedding", action=argparse.BooleanOptionalAction)
parser.add_argument("--ref_baseline", type=str, default="retrieval", choices=["retrieval", "random"],
                    help="retrieval: use top-5 retrieved refs; random: random augmentation baseline (sample 5 random refs from index)")
parser.add_argument("--report", action="store_true", help="Report results to JSON (same style as main_pretrain)")
parser.add_argument("--report_path", type=str, default="./results")
parser.add_argument("--tag", type=str, default="default", help="Tag for the experiment")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

get_dataset(args.dataset, download=True)
task: EntityTask = get_task(args.dataset, args.task, download=True)

if task.task_type == TaskType.BINARY_CLASSIFICATION or task.task_type == TaskType.REGRESSION:
    out_channels = 1
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    out_channels = task.num_labels
else:
    raise ValueError(f"Task type {task.task_type} is unsupported")

clamp_min, clamp_max = None, None
if task.task_type == TaskType.REGRESSION:
    train_table = task.get_table("train")
    clamp_min, clamp_max = np.percentile(
        train_table.df[task.target_col].to_numpy(), [2, 98]
    )
higher_is_better = task.task_type != TaskType.REGRESSION
tune_metric = "roc_auc" if task.task_type == TaskType.BINARY_CLASSIFICATION else "mae" if task.task_type == TaskType.REGRESSION else "multilabel_auprc_macro"

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
    entity_embed_dim=entity_embed_dim,
    use_entity_embedding=args.use_entity_embedding,
    num_heads=args.num_heads,
    num_layers=args.num_layers,
    ff_dim=args.ff_dim,
    dropout=args.dropout,
    num_classes=out_channels,
).to(device)

pretrained_ckpt = os.path.join(args.results_path, "transformers", f"{args.dataset}_{args.task}_{args.backbone}.pth")
if not os.path.exists(pretrained_ckpt):
    raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_ckpt}")
model.load_state_dict(torch.load(pretrained_ckpt, map_location=device))

retrieval_root = os.path.join(args.index_path, args.backbone, args.dataset, args.task)
cls_embeddings = torch.load(os.path.join(retrieval_root, "cls_embeddings.pt"), map_location=device)
cls_entity_ids = torch.load(os.path.join(retrieval_root, "cls_entity_ids.pt"), map_location=device)
cls_timestamps = torch.load(os.path.join(retrieval_root, "cls_timestamps.pt"), map_location=device)
top5_indices = torch.load(os.path.join(retrieval_root, "top5_indices.pt"), map_location=device)  # (N, 5)

with open(os.path.join(retrieval_root, "cls_mapping.json"), "r") as f:
    id_time_to_index = json.load(f)
n_total = len(id_time_to_index)

# Build (entity_id, timestamp) -> label from builder.entity_sequences (same source as indexing/dataset)
id_time_to_label = {}
train_labels_list = []
for split in ("train", "val", "test"):
    for entity_id, seq in builder.entity_sequences[split].items():
        for ts, _emb, label, _ent_emb in seq:
            key = f"({entity_id}, {float(ts)})"
            id_time_to_label[key] = float(label)
            if split == "train":
                train_labels_list.append(float(label))

# cls_labels[i] = label for (cls_entity_ids[i], cls_timestamps[i]); key format matches id_time_to_label
def _id_ts_key(eid, ts):
    return f"({eid}, {float(ts)})"

if task.task_type == TaskType.REGRESSION:
    train_median = np.median(train_labels_list)
    cls_labels = torch.zeros(n_total, dtype=torch.long)
    for i in range(n_total):
        key = _id_ts_key(cls_entity_ids[i].item(), cls_timestamps[i].item())
        cls_labels[i] = 1 if id_time_to_label.get(key, 0.0) > train_median else 0
else:
    cls_labels = torch.zeros(n_total, dtype=torch.long)
    for i in range(n_total):
        key = _id_ts_key(cls_entity_ids[i].item(), cls_timestamps[i].item())
        lbl = id_time_to_label.get(key, 0.0)
        cls_labels[i] = int(round(lbl)) if task.task_type == TaskType.BINARY_CLASSIFICATION else (1 if lbl != 0 else 0)
    cls_labels = cls_labels.clamp(0, 1)

cls_embeddings = cls_embeddings.to(device)
cls_timestamps = cls_timestamps.to(device)
cls_labels = cls_labels.to(device)
top5_indices = top5_indices.to(device)

# RNG for random ref baseline (reproducible across runs with same seed)
rng = np.random.default_rng(args.seed)


def augment_batch_with_retrieval(batch, id_time_to_index, top5_indices, cls_embeddings, cls_labels, cls_timestamps):
    """Add retrieved_cls_emb (B, 5, C), retrieved_labels (B, 5), retrieved_timestamps (B, 5), retrieved_ref_mask (B, 5)."""
    entity_ids = batch["entity_id"].cpu()
    timestamps = batch["target_timestamp"].cpu()
    B = entity_ids.size(0)
    C = cls_embeddings.size(1)
    device = batch["input_embeddings"].device
    retrieved_cls_emb = torch.zeros(B, 5, C, device=device, dtype=cls_embeddings.dtype)
    retrieved_labels = torch.zeros(B, 5, device=device, dtype=torch.long)
    retrieved_timestamps = torch.zeros(B, 5, device=device, dtype=torch.float32)
    retrieved_ref_mask = torch.zeros(B, 5, dtype=torch.bool, device=device)
    for b in range(B):
        eid, ts = entity_ids[b].item(), timestamps[b].item()
        key = _id_ts_key(eid, ts)
        idx = id_time_to_index.get(key)
        if idx is None and float(ts) == int(ts):
            idx = id_time_to_index.get(f"({eid}, {int(ts)})")
        if idx is None:
            continue
        top5 = top5_indices[idx]
        for j in range(5):
            if top5[j] >= 0:
                retrieved_cls_emb[b, j] = cls_embeddings[top5[j]]
                retrieved_labels[b, j] = cls_labels[top5[j]]
                retrieved_timestamps[b, j] = cls_timestamps[top5[j]]
                retrieved_ref_mask[b, j] = True
    batch["retrieved_cls_emb"] = retrieved_cls_emb
    batch["retrieved_labels"] = retrieved_labels
    batch["retrieved_timestamps"] = retrieved_timestamps
    batch["retrieved_ref_mask"] = retrieved_ref_mask
    return batch


def augment_batch_with_random_refs(batch, n_total, cls_embeddings, cls_labels, cls_timestamps, rng):
    """Random augmentation baseline: sample 5 random indices from [0, n_total) per sample.
    Adds same keys as augment_batch_with_retrieval."""
    B = batch["entity_id"].size(0)
    C = cls_embeddings.size(1)
    device = batch["input_embeddings"].device
    # (B, 5) random indices in [0, n_total)
    rand_indices = rng.integers(0, n_total, size=(B, 5))
    rand_indices = torch.from_numpy(rand_indices).to(device)
    retrieved_cls_emb = cls_embeddings[rand_indices]
    retrieved_labels = cls_labels[rand_indices]
    retrieved_timestamps = cls_timestamps[rand_indices]
    retrieved_ref_mask = torch.ones(B, 5, dtype=torch.bool, device=device)
    batch["retrieved_cls_emb"] = retrieved_cls_emb
    batch["retrieved_labels"] = retrieved_labels
    batch["retrieved_timestamps"] = retrieved_timestamps
    batch["retrieved_ref_mask"] = retrieved_ref_mask
    return batch


def augment_batch(batch):
    """Dispatch to retrieval or random ref augmentation based on args.ref_baseline."""
    if args.ref_baseline == "random":
        return augment_batch_with_random_refs(
            batch, n_total, cls_embeddings, cls_labels, cls_timestamps, rng
        )
    return augment_batch_with_retrieval(
        batch, id_time_to_index, top5_indices, cls_embeddings, cls_labels, cls_timestamps
    )


if task.task_type == TaskType.BINARY_CLASSIFICATION:
    loss_fn = BCEWithLogitsLoss()
elif task.task_type == TaskType.REGRESSION:
    loss_fn = L1Loss()
else:
    loss_fn = BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train_epoch(loader):
    model.train()
    total_loss, total_samples, steps = 0.0, 0, 0
    for batch in tqdm(loader, desc="Train"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch = augment_batch(batch)
        logits = model(batch)
        target = batch["target_label"]
        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            loss = loss_fn(logits.squeeze(-1), target)
        elif task.task_type == TaskType.REGRESSION:
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
        all_preds.append(preds.cpu().numpy())
        all_targets.append(target.cpu().numpy())
    avg_loss = total_loss / total_samples if total_samples else 0.0
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = {}
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        metrics["roc_auc"] = roc_auc_score(all_targets, all_preds)
        metrics["accuracy"] = accuracy_score(all_targets, (all_preds > 0.5).astype(int))
        metrics["f1"] = f1_score(all_targets, (all_preds > 0.5).astype(int))
    elif task.task_type == TaskType.REGRESSION:
        metrics["mae"] = mean_absolute_error(all_targets, all_preds)
        metrics["rmse"] = np.sqrt(mean_squared_error(all_targets, all_preds))
        metrics["r2"] = r2_score(all_targets, all_preds)
    else:
        metrics["multilabel_auprc_macro"] = float(np.mean([roc_auc_score(all_targets[:, i], all_preds[:, i]) for i in range(all_targets.shape[1])]))
    return avg_loss, metrics


print("\nFine-tuning with [CLS] [ref_1..ref_5] [SEP] [ctx...] ...")
print(f"Ref baseline: {args.ref_baseline}")
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
        not higher_is_better and val_metric < best_val_metric
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
if best_val_metric is not None and best_val_metric != -float("inf") and best_val_metric != float("inf"):
    print(f"Best val {tune_metric}: {best_val_metric:.4f}")
print("=" * 80)

# Evaluate on test set
print("\nEvaluating on test set...")
test_loss, test_metrics = evaluate(ar_loader_dict["test"])
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Metrics: {test_metrics}")
print(f"Test {tune_metric}: {test_metrics[tune_metric]:.4f}")
print("=" * 80)

if args.report:
    results_path = Path(args.report_path) / f"{args.dataset}_{args.task}_predict.json"
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
    hyperparams = json.dumps({
        "stage": "predict",
        "backbone": args.backbone,
        "mode": args.mode,
        "ref_baseline": args.ref_baseline,
        "tag": args.tag,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "window_size": args.window_size,
    })
    result_entry = {
        "seed": args.seed,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
    }
    if best_val_metric is not None and best_val_metric != -float("inf") and best_val_metric != float("inf"):
        result_entry["best_val_metric"] = best_val_metric
    all_results.setdefault(hyperparams, {})[timestamp] = result_entry
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")
