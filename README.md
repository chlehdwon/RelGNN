# RelTS

## Storage Format (Unified)

This project uses a unified storage layout to reduce string-key bottlenecks and memory spikes.
Model checkpoints are stored in `.pt`, while retrieval intermediates are stored in `.npy/.npz`.

### 1) Snapshot Files (from `indexing.py`)

Location:
- `{index_path}/{backbone}/{dataset}/{task}/`

Files:
- `snapshot_embeddings.npy`
  - Shape: `[N, D]`
  - Content: Snapshot embeddings for all samples in unified row order.
- `snapshot_meta.npz`
  - `entity_ids`: `int64[N]`
  - `timestamps`: `int64[N]`
  - `split_offsets`: `int64[4]`
    - `[0, train_end, val_end, test_end]`
    - Ranges:
      - train: `[split_offsets[0], split_offsets[1])`
      - val: `[split_offsets[1], split_offsets[2])`
      - test: `[split_offsets[2], split_offsets[3])`
- `lookup_train.npz`, `lookup_val.npz`, `lookup_test.npz`
  - `key_sorted`: `uint64[M]` (sorted packed keys)
  - `row_ids_sorted`: `int64[M]` (local row ids aligned with `key_sorted`)

Packed key definition:
- `key = (entity_id << 32) | (timestamp & 0xFFFFFFFF)`

### 2) Entity Embedding Files (from `indexing.py`)

Files:
- `entity.pt`
  - Shape: `[num_seen_entities, E]`
  - Content: Compact per-entity embeddings.
- `entity_mapping.json`
  - Mapping: original `entity_id` (string) -> row index in `entity.pt`

### 3) CLS Retrieval Files (from `main_pretrain.py --save`)

Files:
- `cls_embeddings.npy`
  - Shape: `[N, C]`
  - Content: CLS embeddings in unified row order.
- `cls_meta.pt`
  - `entity_ids`: `int64[N]`
  - `timestamps`: `int64[N]`
- `cls_lookup.npz`
  - `key_sorted`: `uint64[N]`
  - `row_ids_sorted`: `int64[N]`
- `top5_indices.npy` (from `main_retrieve.py`)
  - Shape: `[N, 5]`
  - Content: Top-5 retrieved row ids into `cls_embeddings.npy`

