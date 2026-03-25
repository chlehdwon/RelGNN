"""
Fine-tune with cross-attention retrieval merge:
  Base path: [CLS] [ctx1] ... [ctx_k] [target]
  Retrieved references are merged into CLS via cross-attention + gating.
  Retrieval memory can be either one CLS vector per reference or recent context tokens per reference.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import math

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, L1Loss
import torch.nn.functional as F
from torch_geometric.seed import seed_everything
from tqdm import tqdm
import gc

from relbench.base import EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

from model_cross import RelTS_Cross_Model
from dataset import EntityTimeSeriesBuilder, create_ar_dataloaders, create_random_ar_dataloaders

parser = argparse.ArgumentParser(description="Fine-tune with cross-attention retrieval merge")
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
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--weight_decay", type=float, default=5e-5)
parser.add_argument(
    "--scheduler",
    type=str,
    default="none",
    choices=["none", "cosine"],
    help="Learning-rate scheduler type",
)
parser.add_argument(
    "--warmup_ratio",
    type=float,
    default=0.0,
    help="Warmup ratio for scheduler, applied over total optimization steps",
)
parser.add_argument(
    "--min_lr_ratio",
    type=float,
    default=0.1,
    help="Final LR ratio for cosine decay",
)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--ff_dim", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--cross_heads", type=int, default=4, help="Number of heads for cross-attention merge")
parser.add_argument("--cross_dropout", type=float, default=0.1, help="Dropout for cross-attention merge block")
parser.add_argument("--mode", type=str, default="recent", choices=["recent", "random"])
parser.add_argument("--top_k", type=int, default=5, help="Number of retrieved references to use at prediction time")
parser.add_argument(
    "--retrieved_token_source",
    type=str,
    default="cls",
    choices=["cls", "recent"],
    help="cls: one retrieved CLS vector per ref; recent: recent history tokens per retrieved ref",
)
parser.add_argument(
    "--ref_token_window",
    type=int,
    default=5,
    help="Number of recent history tokens per retrieved reference when --retrieved_token_source=recent",
)
parser.add_argument(
    "--max_ref_context_tokens",
    type=int,
    default=25,
    help="Maximum total retrieved context tokens passed to cross-attention when using recent token source",
)
parser.add_argument("--use_entity_embedding", action=argparse.BooleanOptionalAction)
parser.add_argument(
    "--freeze_backbone",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Freeze the loaded base model during cross-gating training",
)
parser.add_argument(
    "--train_base_classifier",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="When backbone is frozen, optionally train the base classifier head too",
)
parser.add_argument(
    "--backbone_lr",
    type=float,
    default=None,
    help="LR for unfrozen backbone params; defaults to 0.1 * --lr",
)
parser.add_argument(
    "--classifier_lr",
    type=float,
    default=None,
    help="LR for base classifier params when trainable; defaults to --lr",
)
parser.add_argument(
    "--loss_reweighting",
    type=str,
    default="none",
    choices=["none", "balanced"],
    help="Binary classification loss reweighting mode",
)
parser.add_argument("--ref_baseline", type=str, default="retrieval", choices=["retrieval", "random"],
                    help="retrieval: use retrieved refs; random: random augmentation baseline")
parser.add_argument("--report", action="store_true", help="Report results to JSON (same style as main_pretrain)")
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

# Use train_config from relgnn.utils.get_configs when available; override only if user left CLI at default.
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

if task.task_type == TaskType.BINARY_CLASSIFICATION or task.task_type == TaskType.REGRESSION:
    out_channels = 1
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    out_channels = task.num_labels
else:
    raise ValueError(f"Task type {task.task_type} is unsupported")

clamp_min, clamp_max = None, None
if task.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "roc_auc"
    higher_is_better = True
elif task.task_type == TaskType.REGRESSION:
    out_channels = 1
    loss_fn = L1Loss()
    tune_metric = "mae"
    higher_is_better = False
    # Get the clamp value at inference time
    train_table = task.get_table("train")
    clamp_min, clamp_max = np.percentile(
        train_table.df[task.target_col].to_numpy(), [2, 98]
    )
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    out_channels = task.num_labels
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
model = RelTS_Cross_Model(
    channels=channels,
    task_type=task.task_type,
    entity_embed_dim=entity_embed_dim,
    use_entity_embedding=args.use_entity_embedding,
    num_heads=args.num_heads,
    num_layers=args.num_layers,
    ff_dim=args.ff_dim,
    dropout=args.dropout,
    num_classes=out_channels,
    cross_heads=args.cross_heads,
    cross_dropout=args.cross_dropout,
    use_ref_time_label=True,
    freeze_backbone=args.freeze_backbone,
    train_base_classifier=args.train_base_classifier,
).to(device)

pretrained_ckpt = os.path.join(args.results_path, "transformers", f"{args.dataset}_{args.task}_{args.backbone}.pth")
if not os.path.exists(pretrained_ckpt):
    raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_ckpt}")
model.load_base_state_dict(torch.load(pretrained_ckpt, map_location=device))

retrieval_root = os.path.join(args.index_path, args.backbone, args.dataset, args.task)
cls_meta = torch.load(os.path.join(retrieval_root, "cls_meta.pt"), map_location="cpu")
cls_entity_ids = cls_meta["entity_ids"].long()
cls_timestamps = cls_meta["timestamps"].long()
cls_lookup = np.load(os.path.join(retrieval_root, "cls_lookup.npz"))
cls_key_sorted = cls_lookup["key_sorted"].astype(np.uint64, copy=False)
cls_row_ids_sorted = cls_lookup["row_ids_sorted"].astype(np.int64, copy=False)
n_total = int(cls_entity_ids.size(0))

# Build (entity_id, timestamp) -> label from builder.entity_sequences (same source as indexing/dataset)
id_time_to_label = {}
train_labels_list = []
for split in ("train", "val", "test"):
    for entity_id, seq in builder.entity_sequences[split].items():
        for ts, _emb, label, _ent_emb in seq:
            key = (int(entity_id), int(ts))
            id_time_to_label[key] = float(label)
            if split == "train":
                train_labels_list.append(float(label))

cls_labels = torch.zeros(n_total, dtype=torch.float32)
for i in range(n_total):
    key = (int(cls_entity_ids[i].item()), int(cls_timestamps[i].item()))
    # Use the original target label value for this (entity_id, timestamp).
    # For regression this is a real value; for classification it is the original class/indicator as stored.
    cls_labels[i] = float(id_time_to_label.get(key, 0.0))

top5_npy_path = os.path.join(retrieval_root, "top5_indices.npy")
if not os.path.exists(top5_npy_path):
    raise FileNotFoundError(f"Missing retrieval top-k file: {top5_npy_path}")
top5_indices_mm = np.load(top5_npy_path, mmap_mode="r")
if args.top_k <= 0:
    raise ValueError(f"--top_k must be >= 1, got {args.top_k}")
if args.ref_token_window <= 0:
    raise ValueError(f"--ref_token_window must be >= 1, got {args.ref_token_window}")
if args.max_ref_context_tokens <= 0:
    raise ValueError(
        f"--max_ref_context_tokens must be >= 1, got {args.max_ref_context_tokens}"
    )
if top5_indices_mm.ndim != 2:
    raise ValueError(f"Expected top-k index array with 2 dims, got shape {top5_indices_mm.shape}")
available_top_k = int(top5_indices_mm.shape[1])
effective_top_k = min(args.top_k, available_top_k)
if effective_top_k < args.top_k:
    print(
        f"Requested top_k={args.top_k}, but only {available_top_k} refs are available in index. "
        f"Using top_k={effective_top_k}."
    )
effective_ref_top_k = effective_top_k
if (
    args.retrieved_token_source == "recent"
    and effective_ref_top_k > args.max_ref_context_tokens
):
    effective_ref_top_k = args.max_ref_context_tokens
    print(
        "Recent-token retrieval requires at least one token per ref. "
        f"Capping retrieved refs to {effective_ref_top_k} to respect "
        f"max_ref_context_tokens={args.max_ref_context_tokens}."
    )

if args.retrieved_token_source == "cls":
    effective_ref_token_window = 1
    cls_emb_npy_path = os.path.join(retrieval_root, "cls_embeddings.npy")
    if not os.path.exists(cls_emb_npy_path):
        raise FileNotFoundError(f"Missing retrieval embedding file: {cls_emb_npy_path}")
    cls_embeddings_mm = np.load(cls_emb_npy_path, mmap_mode="r")
else:
    effective_ref_token_window = min(
        args.ref_token_window,
        max(1, args.max_ref_context_tokens // max(effective_ref_top_k, 1)),
    )
    cls_embeddings_mm = None

effective_ref_context_tokens = effective_ref_top_k * effective_ref_token_window
cls_labels_np = cls_labels.numpy().astype(np.float32, copy=False)
cls_entity_ids_np = cls_entity_ids.numpy().astype(np.int64, copy=False)
cls_timestamps_np = cls_timestamps.numpy().astype(np.int64, copy=False)

full_entity_sequences = None
cls_sequence_positions_np = None
if args.retrieved_token_source == "recent":
    full_entity_sequences = {}
    for split_name in ("train", "val", "test"):
        for entity_id, seq in builder.entity_sequences[split_name].items():
            merged = full_entity_sequences.setdefault(entity_id, [])
            merged.extend(seq)
    for entity_id, seq in full_entity_sequences.items():
        seq.sort(key=lambda item: int(item[0]))
    if not full_entity_sequences:
        raise ValueError("Could not build full entity sequences for recent-token retrieval.")
    id_time_to_position = {}
    for entity_id, seq in full_entity_sequences.items():
        for pos, (ts, _emb, _label, _ent_emb) in enumerate(seq):
            id_time_to_position[(int(entity_id), int(ts))] = pos
    cls_sequence_positions_np = np.full(n_total, -1, dtype=np.int64)
    for i in range(n_total):
        cls_sequence_positions_np[i] = id_time_to_position.get(
            (int(cls_entity_ids_np[i]), int(cls_timestamps_np[i])),
            -1,
        )

# RNG for random ref baseline (reproducible across runs with same seed)
rng = np.random.default_rng(args.seed)


def lookup_row_indices(entity_ids_tensor, timestamps_tensor, key_sorted, row_ids_sorted):
    """Vectorized lookup for (entity_id, timestamp) -> row index; returns int64 array with -1 for missing."""
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


def get_recent_ref_context_arrays(row_indices, ref_mask_np):
    """Expand retrieved row ids into recent history tokens per retrieved reference."""
    batch_size, num_refs = row_indices.shape
    ref_tokens_np = np.zeros(
        (batch_size, num_refs, effective_ref_token_window, channels),
        dtype=np.float32,
    )
    ref_labels_np = np.zeros(
        (batch_size, num_refs, effective_ref_token_window),
        dtype=np.float32,
    )
    ref_timestamps_np = np.zeros(
        (batch_size, num_refs, effective_ref_token_window),
        dtype=np.float32,
    )
    ref_token_mask_np = np.zeros(
        (batch_size, num_refs, effective_ref_token_window),
        dtype=bool,
    )

    for batch_pos in range(batch_size):
        for ref_pos in range(num_refs):
            if not ref_mask_np[batch_pos, ref_pos]:
                continue

            row_id = int(row_indices[batch_pos, ref_pos])
            seq_pos = int(cls_sequence_positions_np[row_id])
            if seq_pos <= 0:
                continue

            entity_id = int(cls_entity_ids_np[row_id])
            sequence = full_entity_sequences.get(entity_id)
            if not sequence:
                continue

            history = sequence[max(0, seq_pos - effective_ref_token_window):seq_pos]
            if not history:
                continue

            for token_pos, (ts, emb, label, _ent_emb) in enumerate(history):
                ref_tokens_np[batch_pos, ref_pos, token_pos] = np.asarray(
                    emb,
                    dtype=np.float32,
                )
                ref_labels_np[batch_pos, ref_pos, token_pos] = float(label)
                ref_timestamps_np[batch_pos, ref_pos, token_pos] = float(ts)
                ref_token_mask_np[batch_pos, ref_pos, token_pos] = True

    return (
        ref_tokens_np.reshape(batch_size, num_refs * effective_ref_token_window, channels),
        ref_labels_np.reshape(batch_size, num_refs * effective_ref_token_window),
        ref_timestamps_np.reshape(batch_size, num_refs * effective_ref_token_window),
        ref_token_mask_np.reshape(batch_size, num_refs * effective_ref_token_window),
    )


def attach_retrieved_memory(batch, row_indices, ref_mask_np):
    """Attach retrieved token memory to batch for cross-attention."""
    device = batch["input_embeddings"].device
    row_indices_safe = np.maximum(row_indices, 0)

    if args.retrieved_token_source == "recent":
        (
            retrieved_ref_tokens_np,
            retrieved_labels_np,
            retrieved_timestamps_np,
            retrieved_ref_mask_np,
        ) = get_recent_ref_context_arrays(row_indices_safe, ref_mask_np)
    else:
        retrieved_ref_tokens_np = np.asarray(
            cls_embeddings_mm[row_indices_safe],
            dtype=np.float32,
        )
        retrieved_labels_np = np.asarray(cls_labels_np[row_indices_safe], dtype=np.float32)
        retrieved_timestamps_np = np.asarray(
            cls_timestamps_np[row_indices_safe],
            dtype=np.float32,
        )
        retrieved_ref_mask_np = ref_mask_np

    retrieved_ref_tokens = torch.from_numpy(retrieved_ref_tokens_np).to(
        device=device,
        dtype=torch.float32,
    )
    retrieved_labels = torch.from_numpy(retrieved_labels_np).to(
        device=device,
        dtype=torch.float32,
    )
    retrieved_timestamps = torch.from_numpy(retrieved_timestamps_np).to(
        device=device,
        dtype=torch.float32,
    )
    retrieved_ref_mask = torch.from_numpy(retrieved_ref_mask_np).to(
        device=device,
        dtype=torch.bool,
    )
    retrieved_ref_tokens = retrieved_ref_tokens * retrieved_ref_mask.unsqueeze(-1).to(
        retrieved_ref_tokens.dtype
    )

    batch["retrieved_ref_tokens"] = retrieved_ref_tokens
    batch["retrieved_labels"] = retrieved_labels
    batch["retrieved_timestamps"] = retrieved_timestamps
    batch["retrieved_ref_mask"] = retrieved_ref_mask
    return batch


def augment_batch_with_retrieval(batch):
    """Low-memory retrieval path using memmap arrays and per-batch GPU transfer."""
    entity_ids = batch["entity_id"].cpu()
    timestamps = batch["target_timestamp"].cpu()
    batch_idx = lookup_row_indices(entity_ids, timestamps, cls_key_sorted, cls_row_ids_sorted)  # np.int64 [B]

    batch_idx_safe = np.maximum(batch_idx, 0)
    top5_for_batch = top5_indices_mm[batch_idx_safe, :effective_ref_top_k]  # np.int64 [B, K]
    ref_mask_np = (batch_idx[:, None] >= 0) & (top5_for_batch >= 0)
    return attach_retrieved_memory(batch, top5_for_batch, ref_mask_np)


def augment_batch_with_random_refs(batch):
    """Low-memory random baseline using memmap arrays and per-batch GPU transfer."""
    B = batch["entity_id"].size(0)
    rand_indices = rng.integers(0, n_total, size=(B, effective_ref_top_k), dtype=np.int64)
    ref_mask_np = np.ones((B, effective_ref_top_k), dtype=bool)
    return attach_retrieved_memory(batch, rand_indices, ref_mask_np)


def augment_batch(batch):
    """Dispatch to retrieval or random ref augmentation based on args.ref_baseline."""
    if args.ref_baseline == "random":
        return augment_batch_with_random_refs(batch)
    return augment_batch_with_retrieval(batch)


if task.task_type == TaskType.BINARY_CLASSIFICATION:
    loss_fn = BCEWithLogitsLoss()
elif task.task_type == TaskType.REGRESSION:
    loss_fn = L1Loss()
else:
    loss_fn = BCEWithLogitsLoss()

binary_class_weights = None
if task.task_type == TaskType.BINARY_CLASSIFICATION and args.loss_reweighting == "balanced":
    train_table = task.get_table("train")
    train_targets_np = train_table.df[task.target_col].to_numpy(dtype=np.float32)
    pos_count = float(train_targets_np.sum())
    neg_count = float(train_targets_np.shape[0] - pos_count)
    pos_weight = neg_count / max(pos_count, 1.0)
    neg_weight = pos_count / max(neg_count, 1.0)
    binary_class_weights = (float(neg_weight), float(pos_weight))


def compute_loss(logits, target):
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        logits_flat = logits.squeeze(-1)
        if binary_class_weights is None:
            return loss_fn(logits_flat, target)
        raw_loss = F.binary_cross_entropy_with_logits(logits_flat, target.float(), reduction="none")
        neg_weight, pos_weight = binary_class_weights
        sample_weight = torch.where(
            target > 0.5,
            torch.full_like(target, pos_weight),
            torch.full_like(target, neg_weight),
        )
        return (raw_loss * sample_weight).mean()
    if task.task_type == TaskType.REGRESSION:
        return loss_fn(logits.squeeze(-1), target)
    return loss_fn(logits, target)


cross_params = list(model.cross_attn.parameters())
cross_params += list(model.cross_norm.parameters())
cross_params += list(model.cross_ff.parameters())
cross_params += list(model.cross_ff_norm.parameters())
cross_params += list(model.gate_proj.parameters())
classifier_params = [
    p for p in model.base_model.classifier.parameters() if p.requires_grad
]
cross_param_ids = {id(p) for p in cross_params}
classifier_param_ids = {id(p) for p in classifier_params}
backbone_params = [
    p
    for p in model.base_model.parameters()
    if p.requires_grad and id(p) not in classifier_param_ids
]

param_groups = []
if cross_params:
    param_groups.append({"params": cross_params, "lr": args.lr})
if classifier_params:
    param_groups.append(
        {
            "params": classifier_params,
            "lr": args.classifier_lr if args.classifier_lr is not None else args.lr,
        }
    )
if backbone_params:
    param_groups.append(
        {
            "params": backbone_params,
            "lr": args.backbone_lr if args.backbone_lr is not None else args.lr * 0.1,
        }
    )

trainable_params = [p for group in param_groups for p in group["params"]]
if not trainable_params:
    raise ValueError("No trainable parameters in cross-merge module.")
optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

steps_per_epoch = min(len(ar_loader_dict["train"]), args.max_steps_per_epoch)
total_optimization_steps = max(1, steps_per_epoch * args.epochs)
scheduler = None
if args.scheduler == "cosine":
    warmup_steps = int(total_optimization_steps * args.warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_optimization_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_epoch(loader):
    model.train()
    total_loss, total_samples, steps = 0.0, 0, 0
    for batch in tqdm(loader, desc="Train"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch = augment_batch(batch)
        logits = model(batch)
        target = batch["target_label"]
        loss = compute_loss(logits, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
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
            loss = compute_loss(logits, target)
            preds = torch.sigmoid(logits.squeeze(-1))
        elif task.task_type == TaskType.REGRESSION:
            loss = compute_loss(logits, target)
            preds = logits.squeeze(-1).clamp(clamp_min, clamp_max)
        else:
            loss = compute_loss(logits, target)
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


print("\nFine-tuning with cross-attention retrieval merge")
print(f"Ref baseline: {args.ref_baseline}")
print(f"Top-k refs used at predict time: {effective_top_k} (requested={args.top_k})")
print(
    f"Retrieved token source: {args.retrieved_token_source} "
    f"(effective_ref_top_k={effective_ref_top_k})"
)
if args.retrieved_token_source == "recent":
    print(
        f"Recent tokens per ref: {effective_ref_token_window} "
        f"(requested={args.ref_token_window}, max_total={args.max_ref_context_tokens})"
    )
print(f"Retrieved context tokens per sample: {effective_ref_context_tokens}")
print(
    f"freeze_backbone={args.freeze_backbone}, "
    f"train_base_classifier={args.train_base_classifier}"
)
print(
    f"Scheduler: {args.scheduler}, warmup_ratio={args.warmup_ratio}, "
    f"min_lr_ratio={args.min_lr_ratio}"
)
if binary_class_weights is not None:
    neg_weight, pos_weight = binary_class_weights
    print(
        "Using balanced BCE reweighting:",
        f"neg_weight={neg_weight:.6f}, pos_weight={pos_weight:.6f}",
    )
print("=" * 80)
print("Early stopping patience: 5 epochs")

best_val_metric = -float("inf") if higher_is_better else float("inf")
best_epoch = 0
best_state_dict = None
patience, patience_counter = 5, 0

for epoch in range(1, args.epochs + 1):
    print(f"\nEpoch {epoch}/{args.epochs}")
    print("-" * 80)
    print("Current LRs:", [f"{group['lr']:.8f}" for group in optimizer.param_groups])
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
if best_state_dict is not None:
    print(f"Loading best checkpoint from epoch {best_epoch} before test evaluation...")
    model.load_state_dict(best_state_dict)
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
    hyperparams = json.dumps({
        "stage": "predict_cross",
        "backbone": args.backbone,
        "mode": args.mode,
        "top_k": args.top_k,
        "effective_top_k": effective_top_k,
        "retrieved_token_source": args.retrieved_token_source,
        "effective_ref_top_k": effective_ref_top_k,
        "ref_token_window": args.ref_token_window,
        "effective_ref_token_window": effective_ref_token_window,
        "max_ref_context_tokens": args.max_ref_context_tokens,
        "effective_ref_context_tokens": effective_ref_context_tokens,
        "ref_baseline": args.ref_baseline,
        "cross_heads": args.cross_heads,
        "cross_dropout": args.cross_dropout,
        "use_ref_time_label": True,
        "freeze_backbone": args.freeze_backbone,
        "train_base_classifier": args.train_base_classifier,
        "tag": args.tag,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "warmup_ratio": args.warmup_ratio,
        "min_lr_ratio": args.min_lr_ratio,
        "loss_reweighting": args.loss_reweighting,
        "backbone_lr": args.backbone_lr,
        "classifier_lr": args.classifier_lr,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "window_size": args.window_size,
    })
    # Ensure JSON-serializable types
    metrics_for_json = {k: float(v) for k, v in test_metrics.items()}
    result_entry = {
        "source": "predict_cross",
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "test_metrics": metrics_for_json,
    }
    if (
        best_val_metric is not None
        and best_val_metric != -float("inf")
        and best_val_metric != float("inf")
    ):
        result_entry["best_val_metric"] = float(best_val_metric)
    all_results.setdefault(hyperparams, {})[timestamp] = result_entry
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")

# Explicitly release memory at the end of the run to avoid buildup across repeated experiments.
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
