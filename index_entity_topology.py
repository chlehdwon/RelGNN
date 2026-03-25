import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import normalize
from tqdm import tqdm

from relbench.base import Table
from relbench.datasets import get_dataset
from relbench.tasks import get_task


parser = argparse.ArgumentParser(
    description="Create retrieval-specific entity topology embeddings using relation-aware random walks."
)
parser.add_argument("--dataset", type=str, default="rel-amazon")
parser.add_argument("--task", type=str, default="user-churn")
parser.add_argument(
    "--index_path",
    type=str,
    default="/data/relts/snapshots",
    help="Root path for saving retrieval indices",
)
parser.add_argument(
    "--output_prefix",
    type=str,
    default="entity_topology",
    help="Prefix for saved embedding files",
)
parser.add_argument(
    "--walk_length",
    type=int,
    default=8,
    help="Number of transitions per random walk.",
)
parser.add_argument(
    "--walks_per_entity",
    type=int,
    default=32,
    help="Number of random walks sampled per entity.",
)
parser.add_argument(
    "--hash_dim",
    type=int,
    default=384,
    help="Output dimensionality for hashed topology features.",
)
parser.add_argument(
    "--ngram_max",
    type=int,
    default=3,
    help="Maximum n-gram size over walk tokens.",
)
parser.add_argument(
    "--chunk_rows",
    type=int,
    default=20000,
    help="Number of entities to process per chunk.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for walk sampling.",
)
parser.add_argument(
    "--max_mapping_json_entities",
    type=int,
    default=500000,
    help="Skip writing mapping json when entity count exceeds this threshold. Use entity_ids.npy instead.",
)
parser.add_argument(
    "--save_walks",
    action="store_true",
    help="If set, save sampled random walks as jsonl for inspection/debugging.",
)
args = parser.parse_args()


@dataclass
class ExactMatchIndex:
    values: np.ndarray
    row_indices: np.ndarray

    def find_one(self, query_value) -> Optional[int]:
        if pd.isna(query_value) or self.values.size == 0:
            return None
        left = np.searchsorted(self.values, query_value, side="left")
        if left >= self.values.size or self.values[left] != query_value:
            return None
        return int(self.row_indices[left])

    def sample_one(self, query_value, rng: np.random.Generator) -> Optional[int]:
        if pd.isna(query_value) or self.values.size == 0:
            return None
        left = np.searchsorted(self.values, query_value, side="left")
        if left >= self.values.size or self.values[left] != query_value:
            return None
        right = np.searchsorted(self.values, query_value, side="right")
        if right <= left:
            return None
        if right == left + 1:
            return int(self.row_indices[left])
        sample_pos = int(rng.integers(left, right))
        return int(self.row_indices[sample_pos])


@dataclass
class TableState:
    name: str
    pkey_col: Optional[str]
    pkey_values: Optional[np.ndarray]
    fkey_cols: Dict[str, str]
    columns: Dict[str, np.ndarray]
    num_rows: int


def build_exact_index(values: np.ndarray) -> ExactMatchIndex:
    valid_mask = ~pd.isna(values)
    valid_values = values[valid_mask]
    valid_rows = np.nonzero(valid_mask)[0].astype(np.int64, copy=False)
    if valid_values.size == 0:
        return ExactMatchIndex(np.array([], dtype=values.dtype), np.array([], dtype=np.int64))
    order = np.argsort(valid_values, kind="mergesort")
    return ExactMatchIndex(valid_values[order], valid_rows[order])


def collect_table_states(dataset_name: str, task_name: str) -> Tuple[Dict[str, TableState], Dict[str, ExactMatchIndex], Dict[str, List[Tuple[str, str, ExactMatchIndex]]], str]:
    dataset = get_dataset(dataset_name, download=True)
    task = get_task(dataset_name, task_name, download=True)
    db = dataset.get_db()

    table_states: Dict[str, TableState] = {}
    pkey_indices: Dict[str, ExactMatchIndex] = {}
    incoming_relations: Dict[str, List[Tuple[str, str, ExactMatchIndex]]] = {
        table_name: [] for table_name in db.table_dict.keys()
    }

    for table_name, table in db.table_dict.items():
        required_cols = set(table.fkey_col_to_pkey_table.keys())
        if table.pkey_col is not None:
            required_cols.add(table.pkey_col)
        columns = {
            col: table.df[col].to_numpy(copy=False)
            for col in required_cols
        }
        pkey_values = columns.get(table.pkey_col) if table.pkey_col is not None else None
        table_states[table_name] = TableState(
            name=table_name,
            pkey_col=table.pkey_col,
            pkey_values=pkey_values,
            fkey_cols=dict(table.fkey_col_to_pkey_table),
            columns=columns,
            num_rows=len(table.df),
        )
        if pkey_values is not None:
            pkey_indices[table_name] = build_exact_index(pkey_values)

    for child_table, state in table_states.items():
        for fk_col, parent_table in state.fkey_cols.items():
            incoming_relations[parent_table].append(
                (child_table, fk_col, build_exact_index(state.columns[fk_col]))
            )

    return table_states, pkey_indices, incoming_relations, task.entity_table


def stringify_value(value) -> str:
    if pd.isna(value):
        return "null"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    return str(value)


def step_neighbors(
    table_states: Dict[str, TableState],
    pkey_indices: Dict[str, ExactMatchIndex],
    incoming_relations: Dict[str, List[Tuple[str, str, ExactMatchIndex]]],
    current_table: str,
    current_row: int,
    rng: np.random.Generator,
) -> List[Tuple[str, int, str]]:
    neighbors: List[Tuple[str, int, str]] = []
    state = table_states[current_table]

    for fk_col, parent_table in state.fkey_cols.items():
        fk_value = state.columns[fk_col][current_row]
        parent_index = pkey_indices.get(parent_table)
        if parent_index is None:
            continue
        parent_row = parent_index.find_one(fk_value)
        if parent_row is None:
            continue
        edge_token = f"edge:out:{current_table}.{fk_col}->{parent_table}"
        neighbors.append((parent_table, parent_row, edge_token))

    if state.pkey_col is not None:
        current_pkey = state.columns[state.pkey_col][current_row]
        for child_table, fk_col, relation_index in incoming_relations[current_table]:
            child_row = relation_index.sample_one(current_pkey, rng)
            if child_row is None:
                continue
            edge_token = f"edge:in:{child_table}.{fk_col}->{current_table}"
            neighbors.append((child_table, child_row, edge_token))

    return neighbors


def sample_walk_tokens(
    table_states: Dict[str, TableState],
    pkey_indices: Dict[str, ExactMatchIndex],
    incoming_relations: Dict[str, List[Tuple[str, str, ExactMatchIndex]]],
    entity_table: str,
    entity_row: int,
    rng: np.random.Generator,
) -> List[str]:
    tokens = [f"table:{entity_table}", "anchor:start"]
    current_table = entity_table
    current_row = entity_row

    for _ in range(max(int(args.walk_length), 0)):
        neighbors = step_neighbors(
            table_states,
            pkey_indices,
            incoming_relations,
            current_table,
            current_row,
            rng,
        )
        if not neighbors:
            break
        next_table, next_row, edge_token = neighbors[int(rng.integers(0, len(neighbors)))]
        tokens.append(edge_token)
        tokens.append(f"table:{next_table}")
        current_table = next_table
        current_row = next_row

    return tokens


def ngram_features(tokens: Sequence[str], max_ngram: int) -> List[str]:
    features: List[str] = []
    seq_len = len(tokens)
    max_ngram = max(int(max_ngram), 1)
    for n in range(1, max_ngram + 1):
        if seq_len < n:
            break
        for start in range(0, seq_len - n + 1):
            features.append(" ".join(tokens[start : start + n]))
    return features


def build_entity_features(
    table_states: Dict[str, TableState],
    pkey_indices: Dict[str, ExactMatchIndex],
    incoming_relations: Dict[str, List[Tuple[str, str, ExactMatchIndex]]],
    entity_table: str,
    entity_rows: Sequence[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[List[str]]]:
    documents: List[List[str]] = []
    saved_walks: List[List[str]] = []
    for entity_row in entity_rows:
        entity_tokens: List[str] = []
        walk_strings: List[str] = []
        for _ in range(max(int(args.walks_per_entity), 1)):
            walk_tokens = sample_walk_tokens(
                table_states,
                pkey_indices,
                incoming_relations,
                entity_table,
                int(entity_row),
                rng,
            )
            entity_tokens.extend(ngram_features(walk_tokens, args.ngram_max))
            walk_strings.append(" | ".join(walk_tokens))
        documents.append(entity_tokens)
        saved_walks.append(walk_strings)
    hasher = FeatureHasher(n_features=int(args.hash_dim), input_type="string", alternate_sign=False)
    matrix = hasher.transform(documents).astype(np.float32)
    dense = matrix.toarray()
    dense = normalize(dense, norm="l2", copy=False)
    return dense.astype(np.float32, copy=False), saved_walks


table_states, pkey_indices, incoming_relations, entity_table_name = collect_table_states(
    args.dataset,
    args.task,
)
entity_table_state = table_states[entity_table_name]
if entity_table_state.pkey_col is None or entity_table_state.pkey_values is None:
    raise ValueError("Target entity table must have a primary key column for topology indexing.")

entity_ids = entity_table_state.pkey_values
num_entities = len(entity_ids)
if num_entities == 0:
    raise RuntimeError(f"No rows found in entity table '{entity_table_name}'.")

output_dir = Path(args.index_path) / "topology" / args.dataset / args.task
output_dir.mkdir(parents=True, exist_ok=True)

embeddings_path = output_dir / f"{args.output_prefix}_embeddings.npy"
entity_ids_path = output_dir / f"{args.output_prefix}_entity_ids.npy"
mapping_path = output_dir / f"{args.output_prefix}_mapping.json"
meta_path = output_dir / f"{args.output_prefix}_meta.json"

embeddings_mm = np.lib.format.open_memmap(
    embeddings_path,
    mode="w+",
    dtype=np.float32,
    shape=(num_entities, int(args.hash_dim)),
)
entity_id_array = np.empty(num_entities, dtype=np.int64)
write_mapping_json = num_entities <= int(args.max_mapping_json_entities)
mapping = {} if write_mapping_json else None

walks_file = None
if args.save_walks:
    walks_path = output_dir / f"{args.output_prefix}_walks.jsonl"
    walks_file = open(walks_path, "w")

rng = np.random.default_rng(args.seed)
chunk_rows = max(int(args.chunk_rows), 1)
write_offset = 0
entity_row_indices = np.arange(num_entities, dtype=np.int64)

for start in tqdm(range(0, num_entities, chunk_rows), desc="Encoding topology chunks"):
    end = min(start + chunk_rows, num_entities)
    row_chunk = entity_row_indices[start:end]
    embeddings_chunk, walks_chunk = build_entity_features(
        table_states,
        pkey_indices,
        incoming_relations,
        entity_table_name,
        row_chunk,
        rng,
    )
    chunk_size = end - start
    embeddings_mm[write_offset : write_offset + chunk_size] = embeddings_chunk

    entity_ids_chunk = entity_ids[start:end]
    entity_ids_chunk = np.asarray(entity_ids_chunk, dtype=np.int64)
    entity_id_array[write_offset : write_offset + chunk_size] = entity_ids_chunk

    if mapping is not None:
        mapping.update(
            {str(entity_id): write_offset + idx for idx, entity_id in enumerate(entity_ids_chunk)}
        )

    if walks_file is not None:
        for local_idx, entity_id in enumerate(entity_ids_chunk):
            record = {
                "entity_id": int(entity_id),
                "walks": walks_chunk[local_idx],
            }
            walks_file.write(json.dumps(record, ensure_ascii=True) + "\n")

    write_offset += chunk_size

if walks_file is not None:
    walks_file.close()
    print(f"Saved sampled walks to {walks_path}")

if write_offset != num_entities:
    raise RuntimeError(f"Expected to write {num_entities} entities, but wrote {write_offset}.")

np.save(entity_ids_path, entity_id_array)
if mapping is not None:
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)

with open(meta_path, "w") as f:
    json.dump(
        {
            "dataset": args.dataset,
            "task": args.task,
            "entity_table": entity_table_name,
            "pkey_col": entity_table_state.pkey_col,
            "num_entities": int(num_entities),
            "embedding_dim": int(args.hash_dim),
            "walk_length": int(args.walk_length),
            "walks_per_entity": int(args.walks_per_entity),
            "ngram_max": int(args.ngram_max),
            "chunk_rows": int(chunk_rows),
            "seed": int(args.seed),
            "output_prefix": args.output_prefix,
            "entity_ids_file": entity_ids_path.name,
            "mapping_json_written": mapping is not None,
            "method": "relation-aware random walks + hashed n-gram features",
        },
        f,
        indent=2,
    )

print(f"Saved entity topology embeddings to {embeddings_path}")
print(f"Saved entity id array to {entity_ids_path}")
if mapping is not None:
    print(f"Saved entity mapping to {mapping_path}")
print(f"Saved metadata to {meta_path}")
