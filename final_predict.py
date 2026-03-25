"""
Train final retrieval-augmented predictor:
  1. read precomputed final retrieval cache from final_retrieve.py
  2. attach retrieved recent context blocks
  3. train backbone + cross-attention fusion together

This script intentionally does NOT import relgnn.utils.get_configs or any
transformer train_config. All transformer / training hyperparameters come from
CLI args only.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from dataset import EntityTimeSeriesBuilder, create_ar_dataloaders, create_random_ar_dataloaders
from model_cross import RelTS_Cross_Model
from relbench.base import EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task


def build_snapshot_lookup(snapshot_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    meta = np.load(snapshot_dir / "snapshot_meta.npz")
    entity_ids = meta["entity_ids"].astype(np.int64, copy=False)
    timestamps = meta["timestamps"].astype(np.int64, copy=False)
    keys = (entity_ids.astype(np.uint64) << np.uint64(32)) | (
        timestamps.astype(np.uint64) & np.uint64(0xFFFFFFFF)
    )
    order = np.argsort(keys, kind="mergesort")
    return entity_ids, timestamps, keys[order], order.astype(np.int64, copy=False)


def lookup_row_indices(entity_ids_tensor, timestamps_tensor, key_sorted, row_ids_sorted):
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


def build_full_entity_sequences(builder: EntityTimeSeriesBuilder):
    full_sequences = {}
    for split_name in ("train", "val", "test"):
        for entity_id, seq in builder.entity_sequences[split_name].items():
            merged = full_sequences.setdefault(entity_id, [])
            merged.extend(seq)
    for entity_id, seq in full_sequences.items():
        seq.sort(key=lambda item: int(item[0]))
    id_time_to_position = {}
    for entity_id, seq in full_sequences.items():
        for pos, (ts, _emb, _label, _ent_emb) in enumerate(seq):
            id_time_to_position[(int(entity_id), int(ts))] = pos
    return full_sequences, id_time_to_position


def get_recent_ref_context_arrays(
    row_indices: np.ndarray,
    ref_mask_np: np.ndarray,
    snapshot_entity_ids_np: np.ndarray,
    snapshot_timestamps_np: np.ndarray,
    full_entity_sequences,
    id_time_to_position,
    ref_token_window: int,
    channels: int,
):
    batch_size, num_refs = row_indices.shape
    ref_tokens_np = np.zeros((batch_size, num_refs, ref_token_window, channels), dtype=np.float32)
    ref_labels_np = np.zeros((batch_size, num_refs, ref_token_window), dtype=np.float32)
    ref_timestamps_np = np.zeros((batch_size, num_refs, ref_token_window), dtype=np.float32)
    ref_token_mask_np = np.zeros((batch_size, num_refs, ref_token_window), dtype=bool)

    for batch_pos in range(batch_size):
        for ref_pos in range(num_refs):
            if not ref_mask_np[batch_pos, ref_pos]:
                continue
            row_id = int(row_indices[batch_pos, ref_pos])
            entity_id = int(snapshot_entity_ids_np[row_id])
            timestamp = int(snapshot_timestamps_np[row_id])
            seq_pos = int(id_time_to_position.get((entity_id, timestamp), -1))
            if seq_pos <= 0:
                continue
            sequence = full_entity_sequences.get(entity_id)
            if not sequence:
                continue

            history = sequence[max(0, seq_pos - ref_token_window) : seq_pos]
            if not history:
                continue
            for token_pos, (ts, emb, label, _ent_emb) in enumerate(history):
                ref_tokens_np[batch_pos, ref_pos, token_pos] = np.asarray(emb, dtype=np.float32)
                ref_labels_np[batch_pos, ref_pos, token_pos] = float(label)
                ref_timestamps_np[batch_pos, ref_pos, token_pos] = float(ts)
                ref_token_mask_np[batch_pos, ref_pos, token_pos] = True

    return (
        ref_tokens_np.reshape(batch_size, num_refs * ref_token_window, channels),
        ref_labels_np.reshape(batch_size, num_refs * ref_token_window),
        ref_timestamps_np.reshape(batch_size, num_refs * ref_token_window),
        ref_token_mask_np.reshape(batch_size, num_refs * ref_token_window),
    )


def attach_retrieved_memory(
    batch,
    row_indices: np.ndarray,
    ref_mask_np: np.ndarray,
    snapshot_entity_ids_np: np.ndarray,
    snapshot_timestamps_np: np.ndarray,
    full_entity_sequences,
    id_time_to_position,
    ref_token_window: int,
    channels: int,
):
    device = batch["input_embeddings"].device
    row_indices_safe = np.maximum(row_indices, 0)
    (
        retrieved_ref_tokens_np,
        retrieved_labels_np,
        retrieved_timestamps_np,
        retrieved_ref_mask_np,
    ) = get_recent_ref_context_arrays(
        row_indices_safe,
        ref_mask_np,
        snapshot_entity_ids_np=snapshot_entity_ids_np,
        snapshot_timestamps_np=snapshot_timestamps_np,
        full_entity_sequences=full_entity_sequences,
        id_time_to_position=id_time_to_position,
        ref_token_window=ref_token_window,
        channels=channels,
    )

    batch["retrieved_ref_tokens"] = torch.from_numpy(retrieved_ref_tokens_np).to(device=device, dtype=torch.float32)
    batch["retrieved_labels"] = torch.from_numpy(retrieved_labels_np).to(device=device, dtype=torch.float32)
    batch["retrieved_timestamps"] = torch.from_numpy(retrieved_timestamps_np).to(device=device, dtype=torch.float32)
    batch["retrieved_ref_mask"] = torch.from_numpy(retrieved_ref_mask_np).to(device=device, dtype=torch.bool)
    batch["retrieved_ref_tokens"] = batch["retrieved_ref_tokens"] * batch["retrieved_ref_mask"].unsqueeze(-1).to(
        batch["retrieved_ref_tokens"].dtype
    )
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final retrieval-augmented predictor")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="relgnn", choices=["rdl", "relgnn", "relgt"])
    parser.add_argument("--results_path", type=str, default="/data/relts/ckpts")
    parser.add_argument("--index_path", type=str, default="/data/relts/snapshots")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cross_heads", type=int, default=4)
    parser.add_argument("--cross_dropout", type=float, default=0.1)
    parser.add_argument("--mode", type=str, default="recent", choices=["recent", "random"])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--ref_token_window", type=int, default=5)
    parser.add_argument("--max_ref_context_tokens", type=int, default=25)
    parser.add_argument("--use_entity_embedding", action=argparse.BooleanOptionalAction)
    parser.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_base_classifier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss_reweighting", type=str, default="none", choices=["none", "balanced"])
    parser.add_argument("--ref_baseline", type=str, default="retrieval", choices=["retrieval", "random"])
    parser.add_argument("--retrieval_indices_name", type=str, default="final_top5_indices.npy")
    parser.add_argument("--save_best_checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report_name", type=str, default="summary.json")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            args.results_path,
            "final_predict",
            f"{args.dataset}_{args.task}_{args.backbone}",
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        clamp_min, clamp_max = np.percentile(train_table.df[task.target_col].to_numpy(), [2, 98])
    else:
        loss_fn = BCEWithLogitsLoss()
        tune_metric = "multilabel_auprc_macro"
        higher_is_better = True

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
        raise RuntimeError("Could not determine embedding dimension from snapshots.")

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

    retrieval_root = Path(args.index_path) / args.backbone / args.dataset / args.task
    retrieval_path = retrieval_root / args.retrieval_indices_name
    if not retrieval_path.exists():
        raise FileNotFoundError(
            f"Missing retrieval cache: {retrieval_path}. Run final_retrieve.py first."
        )
    topk_indices_mm = np.load(retrieval_path, mmap_mode="r")
    if topk_indices_mm.ndim != 2:
        raise ValueError(f"Expected retrieval cache with 2 dims, got shape {topk_indices_mm.shape}")
    available_top_k = int(topk_indices_mm.shape[1])
    effective_top_k = min(args.top_k, available_top_k)
    if effective_top_k < args.top_k:
        print(f"Requested top_k={args.top_k}, but cache has only {available_top_k}. Using {effective_top_k}.")
    if effective_top_k <= 0:
        raise ValueError("No retrieved references available.")

    effective_ref_token_window = min(
        args.ref_token_window,
        max(1, args.max_ref_context_tokens // max(effective_top_k, 1)),
    )
    effective_ref_context_tokens = effective_top_k * effective_ref_token_window

    snapshot_entity_ids_np, snapshot_timestamps_np, snapshot_key_sorted, snapshot_row_ids_sorted = build_snapshot_lookup(
        retrieval_root
    )
    full_entity_sequences, id_time_to_position = build_full_entity_sequences(builder)

    rng = np.random.default_rng(args.seed)

    def augment_batch_with_retrieval(batch):
        entity_ids = batch["entity_id"].cpu()
        timestamps = batch["target_timestamp"].cpu()
        batch_idx = lookup_row_indices(entity_ids, timestamps, snapshot_key_sorted, snapshot_row_ids_sorted)
        batch_idx_safe = np.maximum(batch_idx, 0)
        ref_rows = topk_indices_mm[batch_idx_safe, :effective_top_k]
        ref_mask_np = (batch_idx[:, None] >= 0) & (ref_rows >= 0)
        return attach_retrieved_memory(
            batch,
            ref_rows,
            ref_mask_np,
            snapshot_entity_ids_np=snapshot_entity_ids_np,
            snapshot_timestamps_np=snapshot_timestamps_np,
            full_entity_sequences=full_entity_sequences,
            id_time_to_position=id_time_to_position,
            ref_token_window=effective_ref_token_window,
            channels=channels,
        )

    def augment_batch_with_random(batch):
        B = batch["entity_id"].size(0)
        rand_indices = rng.integers(0, len(snapshot_entity_ids_np), size=(B, effective_top_k), dtype=np.int64)
        ref_mask_np = np.ones((B, effective_top_k), dtype=bool)
        return attach_retrieved_memory(
            batch,
            rand_indices,
            ref_mask_np,
            snapshot_entity_ids_np=snapshot_entity_ids_np,
            snapshot_timestamps_np=snapshot_timestamps_np,
            full_entity_sequences=full_entity_sequences,
            id_time_to_position=id_time_to_position,
            ref_token_window=effective_ref_token_window,
            channels=channels,
        )

    def augment_batch(batch):
        if args.ref_baseline == "random":
            return augment_batch_with_random(batch)
        return augment_batch_with_retrieval(batch)

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

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters. Check freeze_backbone/train_base_classifier flags.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

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
        return total_loss / max(total_samples, 1)

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
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())
        avg_loss = total_loss / max(total_samples, 1)
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
            metrics["multilabel_auprc_macro"] = float(
                np.mean([roc_auc_score(all_targets[:, i], all_preds[:, i]) for i in range(all_targets.shape[1])])
            )
        return avg_loss, metrics

    print("\nFinal retrieval + predict pipeline")
    print(f"Retrieval cache: {retrieval_path}")
    print(f"Ref baseline: {args.ref_baseline}")
    print(f"Top-k refs: {effective_top_k}")
    print(f"Ref token window: {effective_ref_token_window}")
    print(f"Retrieved context tokens per sample: {effective_ref_context_tokens}")
    print(f"freeze_backbone={args.freeze_backbone}, train_base_classifier={args.train_base_classifier}")
    print(f"No relgnn train_config was loaded. Using CLI hyperparameters as-is.")
    print("=" * 80)

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
        is_best = (higher_is_better and val_metric > best_val_metric) or (
            not higher_is_better and val_metric < best_val_metric
        )
        if is_best:
            best_val_metric = val_metric
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"✓ New best model! epoch={epoch} {tune_metric}={val_metric:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} stagnant epochs.")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print("\nEvaluating best checkpoint on test set...")
    test_loss, test_metrics = evaluate(ar_loader_dict["test"])
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Metrics: {test_metrics}")

    summary = {
        "dataset": args.dataset,
        "task": args.task,
        "backbone": args.backbone,
        "output_dir": str(output_dir),
        "window_size": int(args.window_size),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "max_steps_per_epoch": int(args.max_steps_per_epoch),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "scheduler": args.scheduler,
        "retrieval_cache": str(retrieval_path),
        "ref_baseline": args.ref_baseline,
        "top_k": int(effective_top_k),
        "ref_token_window": int(effective_ref_token_window),
        "freeze_backbone": bool(args.freeze_backbone),
        "train_base_classifier": bool(args.train_base_classifier),
        "best_epoch": int(best_epoch),
        "best_val_metric": float(best_val_metric),
        "tune_metric": tune_metric,
        "test_loss": float(test_loss),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }
    with open(output_dir / args.report_name, "w") as f:
        json.dump(summary, f, indent=2)
    if args.save_best_checkpoint and best_state_dict is not None:
        torch.save(best_state_dict, output_dir / "best_model.pth")
    print(f"Saved summary to {output_dir / args.report_name}")
    if args.save_best_checkpoint and best_state_dict is not None:
        print(f"Saved best checkpoint to {output_dir / 'best_model.pth'}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
