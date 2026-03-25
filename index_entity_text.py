import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from relbench.base import Table
from relbench.datasets import get_dataset
from relbench.tasks import get_task

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - import guard for runtime environments
    raise ImportError(
        "sentence-transformers is required for index_entity_text.py. "
        "Install it with `pip install -U sentence-transformers`."
    ) from exc


parser = argparse.ArgumentParser(
    description="Create retrieval-specific entity text embeddings by serializing target entity table rows."
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
    default="entity_text",
    help="Prefix for saved embedding/mapping files",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="sentence-transformers/all-MiniLM-L6-v2",
    help="SentenceTransformer model name for row serialization embeddings",
)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument(
    "--chunk_rows",
    type=int,
    default=50000,
    help="Number of entity-table rows to serialize and encode per chunk.",
)
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="Torch device passed to SentenceTransformer. Defaults to cuda if available, else cpu.",
)
parser.add_argument(
    "--include_columns",
    type=str,
    nargs="*",
    default=None,
    help="Optional allowlist of columns to serialize. Defaults to all columns.",
)
parser.add_argument(
    "--exclude_columns",
    type=str,
    nargs="*",
    default=None,
    help="Optional blocklist of columns to skip during serialization.",
)
parser.add_argument(
    "--keep_null_columns",
    action="store_true",
    help="If set, serialize missing values instead of dropping them.",
)
parser.add_argument(
    "--save_texts",
    action="store_true",
    help="If set, also save serialized rows as jsonl for inspection/debugging.",
)
parser.add_argument(
    "--max_mapping_json_entities",
    type=int,
    default=500000,
    help="Skip writing mapping json when entity count exceeds this threshold. Use entity_ids.npy instead.",
)
args = parser.parse_args()


def get_device(device_arg: Optional[str]) -> str:
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_scalar(value) -> Optional[str]:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def select_columns(table: Table) -> List[str]:
    columns = list(table.df.columns)
    if args.include_columns:
        missing = sorted(set(args.include_columns) - set(columns))
        if missing:
            raise ValueError(f"Requested include_columns not found in entity table: {missing}")
        columns = [col for col in columns if col in args.include_columns]
    if args.exclude_columns:
        exclude_set = set(args.exclude_columns)
        columns = [col for col in columns if col not in exclude_set]
    if table.pkey_col is not None and table.pkey_col not in columns:
        columns = [table.pkey_col] + columns
    return columns


def serialize_chunk(df_chunk: pd.DataFrame, pkey_col: str) -> tuple[list[int], list[str]]:
    entity_ids: List[int] = []
    serialized_rows: List[str] = []
    columns = list(df_chunk.columns)
    for row in df_chunk.itertuples(index=False, name=None):
        row_dict = dict(zip(columns, row))
        entity_id = int(row_dict[pkey_col])
        parts = [f"table: {args.task} / entity_table: {task.entity_table}"]
        for column_name, raw_value in row_dict.items():
            value = normalize_scalar(raw_value)
            if value is None and not args.keep_null_columns:
                continue
            if value is None:
                value = "null"
            parts.append(f"{column_name}: {value}")
        entity_ids.append(entity_id)
        serialized_rows.append(" | ".join(parts))
    return entity_ids, serialized_rows


def encode_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32, copy=False)


dataset = get_dataset(args.dataset, download=True)
task = get_task(args.dataset, args.task, download=True)
entity_table = dataset.get_db().table_dict[task.entity_table]
columns = select_columns(entity_table)
if entity_table.pkey_col is None:
    raise ValueError("Target entity table must have a primary key column.")

device = get_device(args.device)
model = SentenceTransformer(args.model_name, device=device)
if hasattr(model, "max_seq_length"):
    model.max_seq_length = args.max_length

df = entity_table.df.loc[:, columns]
num_rows = len(df)
if num_rows == 0:
    raise RuntimeError(f"No rows found in entity table '{task.entity_table}'.")

output_dir = Path(args.index_path) / "text" / args.dataset / args.task
output_dir.mkdir(parents=True, exist_ok=True)

embeddings_path = output_dir / f"{args.output_prefix}_embeddings.npy"
mapping_path = output_dir / f"{args.output_prefix}_mapping.json"
meta_path = output_dir / f"{args.output_prefix}_meta.json"
entity_ids_path = output_dir / f"{args.output_prefix}_entity_ids.npy"

embedding_dim = int(model.get_sentence_embedding_dimension())
embeddings_mm = np.lib.format.open_memmap(
    embeddings_path,
    mode="w+",
    dtype=np.float32,
    shape=(num_rows, embedding_dim),
)
entity_ids_array = np.empty(num_rows, dtype=np.int64)
write_mapping_json = num_rows <= int(args.max_mapping_json_entities)
mapping = {} if write_mapping_json else None

texts_file = None
if args.save_texts:
    texts_path = output_dir / f"{args.output_prefix}_rows.jsonl"
    texts_file = open(texts_path, "w")

write_offset = 0
chunk_rows = max(int(args.chunk_rows), 1)
for start in tqdm(range(0, num_rows, chunk_rows), desc="Encoding entity chunks"):
    end = min(start + chunk_rows, num_rows)
    df_chunk = df.iloc[start:end]
    entity_ids_chunk, serialized_rows_chunk = serialize_chunk(df_chunk, entity_table.pkey_col)
    embeddings_chunk = encode_texts(serialized_rows_chunk, model)
    if embeddings_chunk.shape[0] != len(entity_ids_chunk):
        raise RuntimeError("Encoded embedding count does not match serialized row count within chunk.")

    chunk_size = len(entity_ids_chunk)
    embeddings_mm[write_offset : write_offset + chunk_size] = embeddings_chunk
    entity_ids_array[write_offset : write_offset + chunk_size] = np.asarray(entity_ids_chunk, dtype=np.int64)
    if mapping is not None:
        mapping.update(
            {str(entity_id): write_offset + idx for idx, entity_id in enumerate(entity_ids_chunk)}
        )

    if texts_file is not None:
        for entity_id, serialized in zip(entity_ids_chunk, serialized_rows_chunk):
            record = {"entity_id": entity_id, "text": serialized}
            texts_file.write(json.dumps(record, ensure_ascii=True) + "\n")

    write_offset += chunk_size

if texts_file is not None:
    texts_file.close()
    print(f"Saved serialized rows to {texts_path}")

if write_offset != num_rows:
    raise RuntimeError(f"Expected to write {num_rows} rows, but wrote {write_offset}.")

np.save(entity_ids_path, entity_ids_array)
if mapping is not None:
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
with open(meta_path, "w") as f:
    json.dump(
        {
            "dataset": args.dataset,
            "task": args.task,
            "entity_table": task.entity_table,
            "pkey_col": entity_table.pkey_col,
            "num_entities": int(num_rows),
            "embedding_dim": embedding_dim,
            "model_name": args.model_name,
            "max_length": args.max_length,
            "chunk_rows": chunk_rows,
            "columns": columns,
            "output_prefix": args.output_prefix,
            "entity_ids_file": entity_ids_path.name,
            "mapping_json_written": mapping is not None,
        },
        f,
        indent=2,
    )

print(f"Saved entity text embeddings to {embeddings_path}")
print(f"Saved entity id array to {entity_ids_path}")
if mapping is not None:
    print(f"Saved entity text mapping to {mapping_path}")
else:
    print(
        "Skipped mapping json because entity count exceeded "
        f"max_mapping_json_entities={args.max_mapping_json_entities}. "
        f"Use {entity_ids_path} for row-aligned entity ids."
    )
print(f"Saved entity text metadata to {meta_path}")
