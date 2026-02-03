"""
Run retrieval: optionally train the retriever objective (CLtime + InfoNCE) first,
then load the model and compute top-k CLS indices per (entity_id, timestamp).

Usage:
  # Retrieval only (load pretrained or existing retriever checkpoint):
  python main_retrieve.py --dataset rel-f1 --task driver-top3

  # Train retriever then run retrieval in one script:
  python main_retrieve.py --dataset rel-f1 --task driver-top3 --train_retriever --epochs 5 --alpha_aug 0.1
"""
import argparse
import json
import os
import random
from collections import defaultdict

import torch
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task

from model import (
    RelTS_Model,
    cltime_retriever_loss,
    info_nce_retriever_loss,
    mask_correlated_samples_retriever,
)
from dataset import (
    EntityTimeSeriesBuilder,
    create_ar_dataloaders,
    create_random_ar_dataloaders,
)

parser = argparse.ArgumentParser(
    description="Retrieval: optionally train retriever (CLtime+InfoNCE), then compute and save top-k CLS indices"
)
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--task", type=str, default="driver-top3")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--backbone", type=str, default="relgnn", choices=["rdl", "relgnn", "relgt"])
parser.add_argument("--results_path", type=str, default="/data/relts/ckpts")
parser.add_argument("--index_path", type=str, default="/data/relts/snapshots", help="Root path for snapshots / CLS index")
parser.add_argument("--window_size", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--ff_dim", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--mode", type=str, default="recent", choices=["recent", "random"])
parser.add_argument("--use_entity_embedding", action=argparse.BooleanOptionalAction)

# Retriever training (used when --train_retriever)
parser.add_argument("--train_retriever", action=argparse.BooleanOptionalAction, default=False, help="If set, train retriever objective (CLtime+InfoNCE) before running retrieval.")
parser.add_argument("--epochs", type=int, default=5, help="Epochs for retriever training")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for retriever training")
parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for contrastive loss")
parser.add_argument("--lambda_decay", type=float, default=0.1, help="Time decay rate for CLtime")
parser.add_argument("--alpha_aug", type=float, default=0.1, help="Weight for InfoNCE augmentation loss (0 to disable)")
parser.add_argument("--aug_mask_prob", type=float, default=0.15, help="Per-event masking probability for InfoNCE augmented view (0 = dropout only)")
parser.add_argument("--train_batch_size", type=int, default=64, help="Batch size for retriever training")
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

# Use smaller batch for training when train_retriever; same batch for retrieval
train_batch_size = args.train_batch_size if args.train_retriever else args.batch_size
if args.mode == "recent":
    ar_loader_dict = create_ar_dataloaders(
        entity_sequences=builder.entity_sequences,
        split_indices=builder.split_indices,
        window_size=args.window_size,
        batch_size=train_batch_size,
        num_workers=args.num_workers,
        min_input_length=0,
    )
else:
    ar_loader_dict = create_random_ar_dataloaders(
        entity_sequences=builder.entity_sequences,
        split_indices=builder.split_indices,
        window_size=args.window_size,
        batch_size=train_batch_size,
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

retrieval_root = os.path.join(args.index_path, args.backbone, args.dataset, args.task)
cls_embeddings = torch.load(os.path.join(retrieval_root, "cls_embeddings.pt"), map_location=device)
cls_entity_ids = torch.load(os.path.join(retrieval_root, "cls_entity_ids.pt"), map_location=device)
cls_timestamps = torch.load(os.path.join(retrieval_root, "cls_timestamps.pt"), map_location=device)
cls_embeddings = cls_embeddings / (cls_embeddings.norm(dim=1, keepdim=True) + 1e-8)

with open(os.path.join(retrieval_root, "cls_mapping.json"), "r") as f:
    id_time_to_index = json.load(f)
n_total = len(id_time_to_index)

# Load checkpoint: always use pretrained transformer (no separate retriever checkpoint)
ckpt_path = os.path.join(args.results_path, "transformers", f"{args.dataset}_{args.task}_{args.backbone}.pth")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
model.load_state_dict(torch.load(ckpt_path, map_location=device))
print(f"Loaded checkpoint: {ckpt_path}")


def mask_sequence_events(batch, mask_prob, device):
    """
    Build augmented batch by masking each valid event with probability mask_prob.
    Masked positions get input_mask set to False so the transformer does not attend to them.
    """
    if mask_prob <= 0:
        return batch
    batch_aug = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_aug[k] = v.clone()
        else:
            batch_aug[k] = v
    valid = batch_aug["input_mask"]
    B, W = valid.shape
    rand = torch.rand(B, W, device=device, dtype=torch.float32)
    drop = (rand < mask_prob) & valid
    new_mask = valid & ~drop
    # Ensure at least one valid position per sample (avoid empty context)
    all_dropped = new_mask.sum(dim=1) == 0
    if all_dropped.any():
        new_mask[all_dropped] = valid[all_dropped]
    batch_aug["input_mask"] = new_mask
    return batch_aug


def get_positive_negative_indices(batch, entity_to_sorted_list, n_total, device):
    """For each sample, sample one positive (same entity, earlier time) and one hard negative. Returns (positive_idx, negative_idx, valid_mask)."""
    entity_ids = batch["entity_id"].cpu().numpy()
    target_ts = batch["target_timestamp"].cpu().numpy()
    B = len(entity_ids)
    positive_idx = torch.zeros(B, dtype=torch.long, device=device)
    negative_idx = torch.zeros(B, dtype=torch.long, device=device)
    valid = torch.zeros(B, dtype=torch.bool, device=device)
    for b in range(B):
        eid = int(entity_ids[b])
        ts = float(target_ts[b])
        candidates = entity_to_sorted_list.get(eid, [])
        earlier = [(t, idx) for t, idx in candidates if t < ts]
        if not earlier:
            valid[b] = False
            positive_idx[b] = 0
            negative_idx[b] = random.randint(0, n_total - 1) if n_total > 0 else 0
            continue
        valid[b] = True
        pos_ts, pos_i = random.choice(earlier)
        positive_idx[b] = pos_i
        neg_i = random.randint(0, n_total - 1) if n_total > 1 else 0
        while n_total > 1 and neg_i == pos_i:
            neg_i = random.randint(0, n_total - 1)
        negative_idx[b] = neg_i
    return positive_idx, negative_idx, valid


def train_retriever_epoch(loader, model, optimizer, entity_to_sorted_list, n_total, cls_embeddings, cls_timestamps, args, device):
    model.train()
    total_loss, total_cl, total_aug = 0.0, 0.0, 0.0
    num_batches = 0
    for batch in tqdm(loader, desc="Retriever train"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        pos_idx, neg_idx, valid = get_positive_negative_indices(batch, entity_to_sorted_list, n_total, device)
        if valid.sum() == 0:
            continue
        mask = valid
        B_full = batch["entity_id"].size(0)
        batch_reduced = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) == B_full:
                batch_reduced[k] = v[mask]
            else:
                batch_reduced[k] = v
        pos_idx_r, neg_idx_r = pos_idx[mask], neg_idx[mask]
        anchors = model.encode_cls(batch_reduced)
        anchors = anchors / (anchors.norm(dim=1, keepdim=True) + 1e-8)
        positives = cls_embeddings[pos_idx_r].to(device)
        hard_negatives = cls_embeddings[neg_idx_r].to(device)
        anchor_time = batch_reduced["target_timestamp"]
        pos_time = cls_timestamps[pos_idx_r].float().to(device)
        neg_time = cls_timestamps[neg_idx_r].float().to(device)
        cl_loss = cltime_retriever_loss(
            anchors, positives, hard_negatives,
            anchor_time, pos_time, neg_time,
            temperature=args.temperature, decay_rate=args.lambda_decay,
        )
        loss = cl_loss
        aug_loss_val = 0.0
        if args.alpha_aug > 0 and mask.sum() >= 2:
            z_i = model.encode_cls(batch_reduced)
            batch_aug = mask_sequence_events(batch_reduced, args.aug_mask_prob, device)
            z_j = model.encode_cls(batch_aug)
            z_i = z_i / (z_i.norm(dim=1, keepdim=True) + 1e-8)
            z_j = z_j / (z_j.norm(dim=1, keepdim=True) + 1e-8)
            b_aug = z_i.size(0)
            mask_nce_local = mask_correlated_samples_retriever(b_aug, device)
            aug_loss = info_nce_retriever_loss(z_i, z_j, temperature=args.temperature, mask=mask_nce_local)
            loss = loss + args.alpha_aug * aug_loss
            aug_loss_val = aug_loss.item()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        total_cl += cl_loss.item()
        total_aug += aug_loss_val
        num_batches += 1
    n = max(num_batches, 1)
    return total_loss / n, total_cl / n, total_aug / n


if args.train_retriever:
    entity_to_sorted_list = defaultdict(list)
    for key, idx in id_time_to_index.items():
        parts = key.strip("()").split(",")
        entity_id = int(parts[0].strip())
        ts = int(float(parts[1].strip()))
        entity_to_sorted_list[entity_id].append((ts, idx))
    for eid in entity_to_sorted_list:
        entity_to_sorted_list[eid].sort(key=lambda x: x[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        train_loss, train_cl, train_aug = train_retriever_epoch(
            ar_loader_dict["train"], model, optimizer,
            entity_to_sorted_list, n_total, cls_embeddings, cls_timestamps, args, device,
        )
        print(f"Epoch {epoch + 1}/{args.epochs} | loss={train_loss:.4f} | cl_loss={train_cl:.4f} | aug_loss={train_aug:.4f}")

# Recreate dataloaders with retrieval batch_size if we used train_batch_size for training
if args.train_retriever and args.batch_size != args.train_batch_size:
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


def get_topk_cls(batch, model, cls_embeddings, cls_entity_ids, cls_timestamps, k=5, chunk_size=256):
    cls_query = model.encode_cls(batch)
    cls_query = cls_query / (cls_query.norm(dim=1, keepdim=True) + 1e-8)
    batch_entity_ids = batch["entity_id"]
    batch_timestamps = batch["target_timestamp"]
    entity_match = batch_entity_ids.unsqueeze(1) == cls_entity_ids.unsqueeze(0)
    time_match = batch_timestamps.unsqueeze(1) > cls_timestamps.unsqueeze(0)
    valid_mask = entity_match & time_match
    valid_indices = [torch.nonzero(valid_mask[b], as_tuple=False).squeeze(-1) for b in range(valid_mask.size(0))]
    topk_indices = [None] * cls_query.size(0)
    for start in range(0, cls_query.size(0), chunk_size):
        end = min(start + chunk_size, cls_query.size(0))
        scores = torch.matmul(cls_query[start:end], cls_embeddings.t())
        for local_i, i in enumerate(range(start, end)):
            valid_idx = valid_indices[i]
            k_eff = min(k, valid_idx.numel())
            if k_eff == 0:
                topk_indices[i] = torch.full((k,), -1, dtype=torch.long, device=scores.device)
                continue
            _, local_idx = torch.topk(scores[local_i, valid_idx], k=k_eff, dim=0)
            indices_i = valid_idx[local_idx]
            padded = torch.full((k,), -1, dtype=torch.long, device=scores.device)
            padded[:k_eff] = indices_i
            topk_indices[i] = padded
    return torch.stack(topk_indices, dim=0)


def retrieve_topk_for_split(split_name, loader):
    all_topk, all_entity_ids, all_timestamps = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Retrieving top5 ({split_name})"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            topk_idx = get_topk_cls(batch, model, cls_embeddings, cls_entity_ids, cls_timestamps, k=5, chunk_size=256)
            all_topk.append(topk_idx.cpu())
            all_entity_ids.append(batch["entity_id"].detach().cpu())
            all_timestamps.append(batch["target_timestamp"].detach().cpu())
    return (
        torch.cat(all_topk, dim=0),
        torch.cat(all_entity_ids, dim=0),
        torch.cat(all_timestamps, dim=0),
    )


os.makedirs(retrieval_root, exist_ok=True)
top5_train, entity_train, ts_train = retrieve_topk_for_split("train", ar_loader_dict["train"])
top5_val, entity_val, ts_val = retrieve_topk_for_split("val", ar_loader_dict["val"])
top5_test, entity_test, ts_test = retrieve_topk_for_split("test", ar_loader_dict["test"])

top5_result = torch.full((n_total, 5), -1, dtype=torch.long)
for top5, entity_ids, timestamps in [
    (top5_train, entity_train, ts_train),
    (top5_val, entity_val, ts_val),
    (top5_test, entity_test, ts_test),
]:
    for i in range(len(entity_ids)):
        key = f"({entity_ids[i].item()}, {float(timestamps[i].item())})"
        idx = id_time_to_index[key]
        top5_result[idx] = top5[i]

torch.save(top5_result, os.path.join(retrieval_root, "top5_indices.pt"))
print(f"Saved top5 indices to {retrieval_root} (order matches cls_embeddings / cls_mapping)")
