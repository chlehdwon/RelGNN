"""
Build final retrieval cache:
  1. coarse retrieval by raw context similarity
  2. rerank with context residual + entity semantic + topology + temporal recency
  3. save final top-k indices aligned to snapshot row order

This script intentionally does NOT import relgnn.utils.get_configs or any
transformer train_config. All retrieval hyperparameters come from CLI args.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch_geometric.seed import seed_everything

from relbench.datasets import get_dataset
from relbench.tasks import get_task


TASKS = [
    ("rel-f1", "driver-dnf"),
    ("rel-f1", "driver-top3"),
    ("rel-avito", "user-clicks"),
    ("rel-avito", "user-visits"),
]


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def to_unix_seconds(series) -> np.ndarray:
    return (series.astype("int64") // 10**9).to_numpy(dtype=np.int64, copy=False)


def stratified_sample(indices: np.ndarray, labels: np.ndarray, max_items: int, seed: int) -> np.ndarray:
    if max_items <= 0 or len(indices) <= max_items:
        return np.sort(indices.astype(np.int64, copy=False))
    valid = indices[~np.isnan(labels[indices])]
    if len(valid) <= max_items:
        return np.sort(valid.astype(np.int64, copy=False))
    rng = np.random.default_rng(seed)
    y = labels[valid].astype(np.int64, copy=False)
    idx0 = valid[y == 0]
    idx1 = valid[y == 1]
    n_total = len(valid)
    take1 = min(len(idx1), max(1, round(max_items * len(idx1) / max(1, n_total))))
    take0 = min(len(idx0), max_items - take1)
    if take0 == 0 and len(idx0) > 0:
        take0 = 1
        take1 = min(len(idx1), max_items - take0)
    part0 = rng.choice(idx0, size=take0, replace=False) if take0 > 0 else np.array([], dtype=np.int64)
    part1 = rng.choice(idx1, size=take1, replace=False) if take1 > 0 else np.array([], dtype=np.int64)
    sampled = np.concatenate([part0, part1])
    rng.shuffle(sampled)
    return np.sort(sampled.astype(np.int64, copy=False))


def build_label_map(task) -> dict[tuple[int, int], float]:
    label_map: dict[tuple[int, int], float] = {}
    for split in ("train", "val", "test"):
        table = task.get_table(split, mask_input_cols=False)
        df = table.df
        entity_ids = df[task.entity_col].to_numpy(np.int64, copy=False)
        timestamps = to_unix_seconds(df[task.time_col])
        labels = df[task.target_col].astype(float).to_numpy(np.float32, copy=False)
        for entity_id, timestamp, label in zip(entity_ids, timestamps, labels):
            label_map[(int(entity_id), int(timestamp))] = float(label)
    return label_map


def load_context_rows(index_dir: Path, dataset: str, task_name: str):
    task = get_task(dataset, task_name, download=False)
    meta = np.load(index_dir / "snapshot_meta.npz")
    embeddings = np.asarray(np.load(index_dir / "snapshot_embeddings.npy", mmap_mode="r"), dtype=np.float32)
    entity_ids = meta["entity_ids"].astype(np.int64, copy=False)
    timestamps = meta["timestamps"].astype(np.int64, copy=False)
    split_offsets = meta["split_offsets"].astype(np.int64, copy=False)
    splits = np.empty(int(split_offsets[-1]), dtype=object)
    for i, split_name in enumerate(("train", "val", "test")):
        splits[split_offsets[i] : split_offsets[i + 1]] = split_name

    label_map = build_label_map(task)
    labels = np.full(len(entity_ids), np.nan, dtype=np.float32)
    for i, (entity_id, timestamp) in enumerate(zip(entity_ids, timestamps)):
        label = label_map.get((int(entity_id), int(timestamp)))
        if label is not None:
            labels[i] = label
    return embeddings, entity_ids, timestamps, splits, labels


def load_entity_mapping(index_dir: Path, prefix: str, dataset: str, task_name: str) -> dict[int, int]:
    mapping_path = index_dir / f"{prefix}_mapping.json"
    entity_ids_path = index_dir / f"{prefix}_entity_ids.npy"
    if mapping_path.exists():
        with open(mapping_path) as f:
            raw = json.load(f)
        return {int(k): int(v) for k, v in raw.items()}
    if entity_ids_path.exists():
        entity_ids = np.load(entity_ids_path)
        return {int(entity_id): int(idx) for idx, entity_id in enumerate(entity_ids.tolist())}

    dataset_obj = get_dataset(dataset, download=True)
    task = get_task(dataset, task_name, download=False)
    table = dataset_obj.get_db().table_dict[task.entity_table]
    pkey = table.pkey_col
    if pkey is None:
        raise RuntimeError(f"Could not recover entity mapping for {dataset}/{task_name}/{prefix}.")
    entity_ids = table.df[pkey].to_numpy()
    return {int(entity_id): int(idx) for idx, entity_id in enumerate(entity_ids.tolist())}


def load_entity_view(index_dir: Path, prefix: str, dataset: str, task_name: str, row_entity_ids: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(np.load(index_dir / f"{prefix}_embeddings.npy", mmap_mode="r"), dtype=np.float32)
    mapping = load_entity_mapping(index_dir, prefix, dataset, task_name)
    row_idx = np.array([mapping[int(entity_id)] for entity_id in row_entity_ids.tolist()], dtype=np.int64)
    out = embeddings[row_idx]
    return normalize_rows(out)


def build_context_topm(
    query_idx: np.ndarray,
    bank_idx: np.ndarray,
    context_emb: np.ndarray,
    entity_ids: np.ndarray,
    timestamps: np.ndarray,
    top_m: int,
    query_chunk_size: int,
    cand_chunk_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    topm = np.full((int(query_idx.shape[0]), top_m), -1, dtype=np.int64)
    topm_scores = np.full((int(query_idx.shape[0]), top_m), -np.inf, dtype=np.float32)

    bank_context = torch.from_numpy(context_emb[bank_idx]).to(device)
    bank_entity = torch.from_numpy(entity_ids[bank_idx].astype(np.int64, copy=False)).to(device)
    bank_ts = torch.from_numpy(timestamps[bank_idx].astype(np.int64, copy=False)).to(device)
    query_entity_all = torch.from_numpy(entity_ids[query_idx].astype(np.int64, copy=False)).to(device)
    query_ts_all = torch.from_numpy(timestamps[query_idx].astype(np.int64, copy=False)).to(device)

    for q_start in range(0, len(query_idx), query_chunk_size):
        q_end = min(q_start + query_chunk_size, len(query_idx))
        q_rows = query_idx[q_start:q_end]
        q_context = torch.from_numpy(context_emb[q_rows]).to(device)
        q_entity = query_entity_all[q_start:q_end]
        q_ts = query_ts_all[q_start:q_end]

        best_scores = torch.full((len(q_rows), top_m), float("-inf"), device=device)
        best_idx = torch.full((len(q_rows), top_m), -1, dtype=torch.long, device=device)

        for c_start in range(0, len(bank_idx), cand_chunk_size):
            c_end = min(c_start + cand_chunk_size, len(bank_idx))
            cand_context = bank_context[c_start:c_end]
            cand_entity = bank_entity[c_start:c_end]
            cand_ts = bank_ts[c_start:c_end]

            scores = q_context @ cand_context.T
            valid = (q_ts.unsqueeze(1) > cand_ts.unsqueeze(0)) & (q_entity.unsqueeze(1) != cand_entity.unsqueeze(0))
            scores = scores.masked_fill(~valid, float("-inf"))

            cand_idx_global = torch.from_numpy(bank_idx[c_start:c_end]).to(device)
            cand_idx_global = cand_idx_global.unsqueeze(0).expand(len(q_rows), -1)
            merged_scores = torch.cat([best_scores, scores], dim=1)
            merged_idx = torch.cat([best_idx, cand_idx_global], dim=1)
            new_scores, new_pos = torch.topk(merged_scores, k=top_m, dim=1)
            new_idx = torch.gather(merged_idx, 1, new_pos)
            best_scores, best_idx = new_scores, new_idx

        valid_best = torch.isfinite(best_scores)
        best_np = best_idx.cpu().numpy().astype(np.int64, copy=False)
        best_scores_np = best_scores.cpu().numpy().astype(np.float32, copy=False)
        mask_np = valid_best.cpu().numpy()
        best_np[~mask_np] = -1
        best_scores_np[~mask_np] = -np.inf
        topm[q_start:q_end] = best_np
        topm_scores[q_start:q_end] = best_scores_np

    return topm, topm_scores


def build_feature_tensor(
    query_idx: np.ndarray,
    coarse_topm: np.ndarray,
    coarse_scores: np.ndarray,
    sem_emb: np.ndarray,
    topo_emb: np.ndarray,
    timestamps: np.ndarray,
    tau_days: float,
) -> tuple[np.ndarray, np.ndarray]:
    safe = np.maximum(coarse_topm, 0)
    valid = coarse_topm >= 0
    q_sem = sem_emb[query_idx]
    q_topo = topo_emb[query_idx]
    c_sem = sem_emb[safe]
    c_topo = topo_emb[safe]
    sim_sem = np.einsum("qd,qkd->qk", q_sem, c_sem, optimize=True)
    sim_topo = np.einsum("qd,qkd->qk", q_topo, c_topo, optimize=True)
    delta_days = (timestamps[query_idx][:, None] - timestamps[safe]) / 86400.0
    delta_days = np.maximum(delta_days, 0.0).astype(np.float32, copy=False)
    recency = np.exp(-delta_days / float(tau_days)).astype(np.float32, copy=False)
    feats = np.stack([coarse_scores, sim_sem, sim_topo, recency], axis=-1).astype(np.float32, copy=False)
    feats[~valid] = 0.0
    return feats, valid


def build_pairwise_examples(
    query_idx: np.ndarray,
    coarse_topm: np.ndarray,
    feats: np.ndarray,
    valid: np.ndarray,
    labels: np.ndarray,
    max_pairs_per_query: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_rows = []
    y_rows = []
    for i, q in enumerate(query_idx.tolist()):
        y = labels[q]
        if np.isnan(y):
            continue
        cand_idx = coarse_topm[i][valid[i]]
        cand_labels = labels[cand_idx]
        good = ~np.isnan(cand_labels)
        cand_labels = cand_labels[good].astype(np.int64, copy=False)
        cand_feats = feats[i][valid[i]][good]
        if len(cand_labels) == 0:
            continue
        pos = np.flatnonzero(cand_labels == int(y))
        neg = np.flatnonzero(cand_labels != int(y))
        if len(pos) == 0 or len(neg) == 0:
            continue
        n_pairs = min(max_pairs_per_query, len(pos), len(neg))
        pos_sel = rng.choice(pos, size=n_pairs, replace=len(pos) < n_pairs)
        neg_sel = rng.choice(neg, size=n_pairs, replace=len(neg) < n_pairs)
        x_rows.append(cand_feats[pos_sel] - cand_feats[neg_sel])
        y_rows.append(np.ones(n_pairs, dtype=np.float32))
        x_rows.append(cand_feats[neg_sel] - cand_feats[pos_sel])
        y_rows.append(np.zeros(n_pairs, dtype=np.float32))
    if not x_rows:
        return np.empty((0, feats.shape[-1]), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.concatenate(x_rows, axis=0), np.concatenate(y_rows, axis=0)


class LinearReranker(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def vote_auc(query_labels: np.ndarray, bank_labels: np.ndarray, topk: np.ndarray, ks: list[int]) -> dict[int, float]:
    row = {}
    for k in ks:
        ref_k = topk[:, :k]
        valid = ref_k >= 0
        safe = np.maximum(ref_k, 0)
        ref_labels = bank_labels[safe]
        valid = valid & ~np.isnan(ref_labels)
        count = valid.sum(axis=1)
        coverage = count > 0
        scores = np.full(len(query_labels), np.nan, dtype=np.float32)
        if np.any(coverage):
            scores[coverage] = ((ref_labels * valid).sum(axis=1)[coverage] / count[coverage])
        mask = ~np.isnan(query_labels) & ~np.isnan(scores)
        y_true = query_labels[mask].astype(np.int64, copy=False)
        y_score = scores[mask].astype(np.float32, copy=False)
        if len(np.unique(y_true)) < 2:
            row[k] = float("nan")
        else:
            row[k] = float(roc_auc_score(y_true, y_score))
    return row


def topk_from_features(
    model: LinearReranker,
    feats: np.ndarray,
    coarse_topm: np.ndarray,
    valid: np.ndarray,
    rerank_limit: int | None,
    top_k: int,
    device: torch.device,
) -> np.ndarray:
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(feats).to(device)).cpu().numpy().astype(np.float32, copy=False)
    logits = logits.copy()
    logits[~valid] = -np.inf

    if rerank_limit is None:
        order = np.argsort(-logits, axis=1)
        reranked = np.take_along_axis(coarse_topm, order, axis=1)
        valid_sorted = np.take_along_axis(valid, order, axis=1)
    else:
        limit = min(rerank_limit, coarse_topm.shape[1])
        reranked = coarse_topm.copy()
        valid_sorted = valid.copy()
        prefix_order = np.argsort(-logits[:, :limit], axis=1)
        reranked[:, :limit] = np.take_along_axis(coarse_topm[:, :limit], prefix_order, axis=1)
        valid_sorted[:, :limit] = np.take_along_axis(valid[:, :limit], prefix_order, axis=1)

    reranked[~valid_sorted] = -1
    return reranked[:, :top_k]


def train_reranker(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_feats: np.ndarray,
    val_coarse_topm: np.ndarray,
    val_valid: np.ndarray,
    val_query_labels: np.ndarray,
    bank_labels: np.ndarray,
    rerank_limit: int | None,
    top_k_eval: int,
    device: torch.device,
    epochs: int,
) -> tuple[LinearReranker, dict]:
    model = LinearReranker(in_dim=train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-2, weight_decay=1e-4)
    x_t = torch.from_numpy(train_x).to(device)
    y_t = torch.from_numpy(train_y).to(device)
    best_state = None
    best_val_mean = float("-inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(x_t)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 != 0 and epoch != epochs:
            continue
        model.eval()
        val_topk = topk_from_features(
            model,
            val_feats,
            val_coarse_topm,
            val_valid,
            rerank_limit=rerank_limit,
            top_k=top_k_eval,
            device=device,
        )
        val_auc = vote_auc(val_query_labels, bank_labels, val_topk, [5, 10, min(20, top_k_eval)])
        val_mean = float(np.nanmean(list(val_auc.values())))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "val_auc_mean": val_mean,
                "val_auc_at_5": float(val_auc[5]),
                "val_auc_at_10": float(val_auc[10]),
                "val_auc_at_20": float(val_auc[min(20, top_k_eval)]),
            }
        )
        if val_mean > best_val_mean:
            best_val_mean = val_mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), {"best_val_auc_mean": float(best_val_mean), "history": history}


def save_json(path: Path, payload) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final top-k retrieval cache with ctx residual reranking")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="relgnn", choices=["rdl", "relgnn", "relgt"])
    parser.add_argument("--index_path", type=str, default="/data/relts/snapshots")
    parser.add_argument("--top_m", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--rerank_limit", type=int, default=0, help="0 means rerank the full top-m list")
    parser.add_argument("--query_chunk_size", type=int, default=256)
    parser.add_argument("--cand_chunk_size", type=int, default=8192)
    parser.add_argument("--max_train_queries", type=int, default=4000)
    parser.add_argument("--max_val_queries", type=int, default=2500)
    parser.add_argument("--max_pairs_per_query", type=int, default=4)
    parser.add_argument("--reranker_epochs", type=int, default=30)
    parser.add_argument("--tau_days", type=float, nargs="*", default=[3.0, 7.0, 14.0, 30.0, 90.0, 180.0, 365.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_name", type=str, default="final_top5_indices.npy")
    parser.add_argument("--output_scores_name", type=str, default="final_top5_scores.npy")
    parser.add_argument("--output_meta_name", type=str, default="final_retrieval_meta.json")
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top_k must be >= 1")
    if args.top_m < args.top_k:
        raise ValueError("--top_m must be >= --top_k")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    get_dataset(args.dataset, download=True)
    get_task(args.dataset, args.task, download=True)

    retrieval_root = Path(args.index_path) / args.backbone / args.dataset / args.task
    text_root = Path(args.index_path) / "text" / args.dataset / args.task
    topo_root = Path(args.index_path) / "topology" / args.dataset / args.task

    context_emb, entity_ids, timestamps, splits, labels = load_context_rows(retrieval_root, args.dataset, args.task)
    context_emb = normalize_rows(context_emb)
    sem_emb = load_entity_view(text_root, "entity_text", args.dataset, args.task, entity_ids)
    topo_emb = load_entity_view(topo_root, "entity_topology", args.dataset, args.task, entity_ids)

    train_idx = np.flatnonzero((splits == "train") & ~np.isnan(labels))
    val_idx = np.flatnonzero((splits == "val") & ~np.isnan(labels))
    test_idx = np.flatnonzero((splits == "test") & ~np.isnan(labels))

    train_queries = stratified_sample(train_idx, labels, args.max_train_queries, args.seed)
    val_queries = stratified_sample(val_idx, labels, args.max_val_queries, args.seed + 1)

    coarse_train, coarse_train_scores = build_context_topm(
        train_queries,
        train_idx,
        context_emb,
        entity_ids,
        timestamps,
        args.top_m,
        args.query_chunk_size,
        args.cand_chunk_size,
        device,
    )
    coarse_val, coarse_val_scores = build_context_topm(
        val_queries,
        train_idx,
        context_emb,
        entity_ids,
        timestamps,
        args.top_m,
        args.query_chunk_size,
        args.cand_chunk_size,
        device,
    )

    rerank_limit = None if int(args.rerank_limit) <= 0 else int(args.rerank_limit)
    tau_grid = [float(x) for x in args.tau_days]
    best_model = None
    best_tau = None
    best_val_mean = float("-inf")
    tau_rows = []
    history_by_tau = {}

    for tau_days in tau_grid:
        train_feats, train_valid = build_feature_tensor(
            train_queries, coarse_train, coarse_train_scores, sem_emb, topo_emb, timestamps, tau_days
        )
        val_feats, val_valid = build_feature_tensor(
            val_queries, coarse_val, coarse_val_scores, sem_emb, topo_emb, timestamps, tau_days
        )
        train_x, train_y = build_pairwise_examples(
            train_queries,
            coarse_train,
            train_feats,
            train_valid,
            labels,
            max_pairs_per_query=args.max_pairs_per_query,
            seed=args.seed + int(tau_days),
        )
        if len(train_x) == 0:
            continue

        model, meta = train_reranker(
            train_x=train_x,
            train_y=train_y,
            val_feats=val_feats,
            val_coarse_topm=coarse_val,
            val_valid=val_valid,
            val_query_labels=labels[val_queries],
            bank_labels=labels,
            rerank_limit=rerank_limit,
            top_k_eval=max(args.top_k, 20),
            device=device,
            epochs=args.reranker_epochs,
        )
        val_topk = topk_from_features(
            model,
            val_feats,
            coarse_val,
            val_valid,
            rerank_limit=rerank_limit,
            top_k=max(args.top_k, 20),
            device=device,
        )
        val_auc = vote_auc(labels[val_queries], labels, val_topk, [5, 10, 20])
        val_mean = float(np.nanmean(list(val_auc.values())))
        tau_rows.append(
            {
                "tau_days": float(tau_days),
                "val_auc_at_5": float(val_auc[5]),
                "val_auc_at_10": float(val_auc[10]),
                "val_auc_at_20": float(val_auc[20]),
                "val_auc_mean_5_10_20": val_mean,
                "train_pairs": int(len(train_x)),
            }
        )
        history_by_tau[str(tau_days)] = meta["history"]
        if val_mean > best_val_mean:
            best_val_mean = val_mean
            best_tau = float(tau_days)
            best_model = model

    if best_model is None or best_tau is None:
        raise RuntimeError("Could not train reranker or choose tau.")

    final_topk = np.full((len(entity_ids), args.top_k), -1, dtype=np.int64)
    final_scores = np.full((len(entity_ids), args.top_k), -np.inf, dtype=np.float32)

    split_groups = {
        "train": np.flatnonzero(splits == "train"),
        "val": np.flatnonzero(splits == "val"),
        "test": np.flatnonzero(splits == "test"),
    }
    for split_name, query_rows in split_groups.items():
        if len(query_rows) == 0:
            continue
        coarse_split, coarse_split_scores = build_context_topm(
            query_rows,
            train_idx,
            context_emb,
            entity_ids,
            timestamps,
            args.top_m,
            args.query_chunk_size,
            args.cand_chunk_size,
            device,
        )
        feats_split, valid_split = build_feature_tensor(
            query_rows, coarse_split, coarse_split_scores, sem_emb, topo_emb, timestamps, best_tau
        )
        reranked = topk_from_features(
            best_model,
            feats_split,
            coarse_split,
            valid_split,
            rerank_limit=rerank_limit,
            top_k=args.top_k,
            device=device,
        )
        final_topk[query_rows] = reranked

        # Save the reranker scores for the chosen top-k.
        model = best_model.to(device)
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(feats_split).to(device)).cpu().numpy().astype(np.float32, copy=False)
        logits = logits.copy()
        logits[~valid_split] = -np.inf
        if rerank_limit is None:
            order = np.argsort(-logits, axis=1)
            score_sorted = np.take_along_axis(logits, order, axis=1)
        else:
            limit = min(rerank_limit, logits.shape[1])
            order = np.tile(np.arange(logits.shape[1]), (logits.shape[0], 1))
            prefix_order = np.argsort(-logits[:, :limit], axis=1)
            order[:, :limit] = np.take_along_axis(order[:, :limit], prefix_order, axis=1)
            score_sorted = np.take_along_axis(logits, order, axis=1)
        final_scores[query_rows] = score_sorted[:, : args.top_k]

    np.save(retrieval_root / args.output_name, final_topk)
    np.save(retrieval_root / args.output_scores_name, final_scores)
    save_json(
        retrieval_root / args.output_meta_name,
        {
            "dataset": args.dataset,
            "task": args.task,
            "backbone": args.backbone,
            "output_name": args.output_name,
            "output_scores_name": args.output_scores_name,
            "top_m": int(args.top_m),
            "top_k": int(args.top_k),
            "rerank_limit": None if rerank_limit is None else int(rerank_limit),
            "best_tau_days": float(best_tau),
            "best_val_auc_mean": float(best_val_mean),
            "tau_grid": tau_grid,
            "tau_tuning": tau_rows,
            "reranker_state_dict": {k: v.detach().cpu().numpy().tolist() for k, v in best_model.state_dict().items()},
            "history_by_tau": history_by_tau,
        },
    )
    print(f"Saved retrieval cache to {retrieval_root / args.output_name}")
    print(f"Saved score cache to {retrieval_root / args.output_scores_name}")
    print(f"Saved metadata to {retrieval_root / args.output_meta_name}")


if __name__ == "__main__":
    main()
