"""
Run retrieval and compute top-k CLS indices per (entity_id, timestamp).

Usage:
  # Retrieval:
  python main_retrieve.py --dataset rel-f1 --task driver-top3
"""
import argparse
import os
import gc

import numpy as np
import torch
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task

from model import RelTS_Model
from dataset import (
    EntityTimeSeriesBuilder,
    create_ar_dataloaders,
    create_random_ar_dataloaders,
)

parser = argparse.ArgumentParser(
    description="Retrieval: compute and save top-k CLS indices"
)
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--task", type=str, default="driver-top3")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--backbone", type=str, default="relgnn", choices=["rdl", "relgnn", "relgt"])
parser.add_argument("--results_path", type=str, default="/data/relts/ckpts")
parser.add_argument("--index_path", type=str, default="/data/relts/snapshots", help="Root path for snapshots / CLS index")
parser.add_argument("--window_size", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--top_k", type=int, default=5, help="Number of retrieved references to save per sample")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--ff_dim", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--mode", type=str, default="recent", choices=["recent", "random"])
parser.add_argument("--use_entity_embedding", action=argparse.BooleanOptionalAction)
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

retrieval_root = os.path.join(args.index_path, args.backbone, args.dataset, args.task)
cls_emb_npy_path = os.path.join(retrieval_root, "cls_embeddings.npy")
if not os.path.exists(cls_emb_npy_path):
    raise FileNotFoundError(f"Missing retrieval embedding file: {cls_emb_npy_path}")
cls_embeddings_mm = np.load(cls_emb_npy_path, mmap_mode="r")
cls_embeddings = torch.from_numpy(np.asarray(cls_embeddings_mm)).to(device=device, dtype=torch.float32)
cls_meta = torch.load(os.path.join(retrieval_root, "cls_meta.pt"), map_location="cpu")
cls_entity_ids = cls_meta["entity_ids"].long().to(device)
cls_timestamps = cls_meta["timestamps"].long().to(device)
cls_embeddings = cls_embeddings / (cls_embeddings.norm(dim=1, keepdim=True) + 1e-8)
cls_lookup = np.load(os.path.join(retrieval_root, "cls_lookup.npz"))
cls_key_sorted = cls_lookup["key_sorted"].astype(np.uint64, copy=False)
cls_row_ids_sorted = cls_lookup["row_ids_sorted"].astype(np.int64, copy=False)
n_total = int(cls_entity_ids.size(0))

# Load checkpoint: always use pretrained transformer (no separate retriever checkpoint)
ckpt_path = os.path.join(args.results_path, "transformers", f"{args.dataset}_{args.task}_{args.backbone}.pth")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
model.load_state_dict(torch.load(ckpt_path, map_location=device))
print(f"Loaded checkpoint: {ckpt_path}")


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


def retrieve_topk_for_split(split_name, loader, k_value):
    all_topk, all_entity_ids, all_timestamps = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Retrieving top{k_value} ({split_name})"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            topk_idx = get_topk_cls(
                batch, model, cls_embeddings, cls_entity_ids, cls_timestamps, k=k_value, chunk_size=256
            )
            all_topk.append(topk_idx.cpu())
            all_entity_ids.append(batch["entity_id"].detach().cpu())
            all_timestamps.append(batch["target_timestamp"].detach().cpu())
    return (
        torch.cat(all_topk, dim=0),
        torch.cat(all_entity_ids, dim=0),
        torch.cat(all_timestamps, dim=0),
    )


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

os.makedirs(retrieval_root, exist_ok=True)
print(f"Retrieval top_k: {args.top_k}")
k_value = args.top_k
top5_train, entity_train, ts_train = retrieve_topk_for_split("train", ar_loader_dict["train"], k_value)
top5_val, entity_val, ts_val = retrieve_topk_for_split("val", ar_loader_dict["val"], k_value)
top5_test, entity_test, ts_test = retrieve_topk_for_split("test", ar_loader_dict["test"], k_value)

top5_result = torch.full((n_total, k_value), -1, dtype=torch.long)
for top5, entity_ids, timestamps in [
    (top5_train, entity_train, ts_train),
    (top5_val, entity_val, ts_val),
    (top5_test, entity_test, ts_test),
]:
    row_idx = lookup_row_indices(entity_ids, timestamps, cls_key_sorted, cls_row_ids_sorted)
    valid = row_idx >= 0
    if np.any(valid):
        top5_result[torch.from_numpy(row_idx[valid]).long()] = top5[torch.from_numpy(valid).bool()]

np.save(
    os.path.join(retrieval_root, "top5_indices.npy"),
    top5_result.numpy().astype(np.int64, copy=False),
)
print(
    f"Saved top-k indices (k={k_value}) to {retrieval_root}/top5_indices.npy "
    f"(order matches cls_embeddings / cls_lookup)"
)

# Explicitly release memory at the end of the run to avoid buildup across repeated experiments.
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
