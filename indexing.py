import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm
import os
from huggingface_hub import hf_hub_download

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

from relgnn.relgnn_model import RelGNN_Model
from relgnn.model import Model
from relgnn.text_embedder import GloveTextEmbedding
from relgnn.utils import get_configs
from relgnn.atomic_routes import get_atomic_routes

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-amazon")
parser.add_argument("--task", type=str, default="user-churn")
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
    help="Backbone model type: 'rdl', 'relgnn', or 'relgt'"
)
parser.add_argument(
    "--index_path",
    type=str,
    default="/data/relts/snapshots",
    help="Root path for saving snapshot indices"
)

args = parser.parse_args()

checkpoint_path = Path(f"/data/relts/ckpts/{args.backbone}") / f"{args.dataset}_{args.task}.pth"
if not checkpoint_path.exists():
    # Fallback to HuggingFace Hub download for relgnn
    if args.backbone == "relgnn":
        checkpoint_path = Path(hf_hub_download(repo_id="tianlangchen/RelGNN", filename=f"{args.dataset}_{args.task}.pth", cache_dir=f"/data/relts/ckpts/{args.backbone}"))
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please ensure the checkpoint is available for backbone={args.backbone}")
assert checkpoint_path.exists(), f"Checkpoint not found at {checkpoint_path}. Please download the checkpoint first."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(42)

dataset: Dataset = get_dataset(args.dataset, download=True)
task: EntityTask = get_task(args.dataset, args.task, download=True)

model_config, loader_config = get_configs(args.dataset, args.task, args.backbone)

stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, "w") as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)

data, col_stats_dict = make_pkey_fkey_graph(
    dataset.get_db(),
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

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

# Get train table for entity history building
train_table = task.get_table("train")
val_table = task.get_table("val")
test_table = task.get_table("test", mask_input_cols=False)

# Create loaders for train, val and test
loader_dict: Dict[str, NeighborLoader] = {}
for split in ["train", "val", "test"]:
    table = task.get_table(split)
    table_input = get_node_train_table_input(table=table, task=task)
    entity_table = table_input.nodes[0]
    loader_dict[split] = NeighborLoader(
        data,
        num_neighbors=[int(loader_config['num_neighbors'] / 2**i) for i in range(loader_config['num_layers'])],
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        subgraph_type=loader_config['subgraph_type'],
        batch_size=loader_config['batch_size'],
        temporal_strategy="last",  # Deterministic sampling
        shuffle=False,  # Must be False for deterministic indexing
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

@torch.no_grad()
def test(loader: NeighborLoader, test_table, alpha: float = 1.0) -> np.ndarray:
    """
    Test function with entity history incorporation from test table (only past labels)
    
    Args:
        loader: NeighborLoader for the split
        test_table: Test table containing entity IDs, timestamps, and labels
        alpha: Weight for combining model prediction and history (0~1)
               final_pred = alpha * model_pred + (1-alpha) * history_mean
               
    Returns:
        Predictions as numpy array
    """
    model.eval()

    pred_list = []
    entity_list = []
    seed_time_list = []
    
    for batch in tqdm(loader):
        batch = batch.to(device)
        
        # Get entity IDs and seed times for this batch
        entity_ids = batch[task.entity_table].batch_size
        entity_node_indices = batch[task.entity_table].n_id[:entity_ids]
        seed_times = batch[task.entity_table].seed_time
        
        # Model prediction
        pred = model(
            batch,
            task.entity_table,
        )
        
        if task.task_type == TaskType.REGRESSION:
            assert clamp_min is not None
            assert clamp_max is not None
            pred = torch.clamp(pred, clamp_min, clamp_max)

        if task.task_type in [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        ]:
            pred = torch.sigmoid(pred)  # Now pred is in [0, 1]

        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
        entity_list.append(entity_node_indices.cpu())
        seed_time_list.append(seed_times.cpu())
        
    # Concatenate all predictions, entity IDs, and seed times
    all_preds = torch.cat(pred_list, dim=0).numpy()
    all_entity_indices = torch.cat(entity_list, dim=0).numpy()
    all_seed_times = torch.cat(seed_time_list, dim=0).numpy()
    
    return all_preds


@torch.no_grad()
def create_snapshot_index(loader: NeighborLoader, split_name: str, base_dir: Path):
    """
    Create snapshot index for deterministic subgraph sampling with pretrained model embeddings.
    
    Args:
        loader: NeighborLoader for the split
        split_name: Name of the split ('train', 'val', or 'test')
        base_dir: Base directory to save {split_name}.pt file
    """
    mapping = {}  # {(entity_id, timestamp): index}
    all_embeddings = []  # Collect all embeddings
    global_index = 0
    
    print(f"\nCreating snapshot index for {split_name}...")
    model.eval()
    
    for batch_idx, batch in enumerate(tqdm(loader)):
        batch_device = batch.to(device)
        
        # Get entity IDs and seed times for this batch
        entity_ids = batch_device[task.entity_table].batch_size
        entity_node_indices = batch_device[task.entity_table].n_id[:entity_ids]
        seed_times = batch_device[task.entity_table].seed_time
        
        # Get entity embeddings from pretrained model (same as test function)
        entity_embeddings = model.forward_entity(
            batch_device,
            task.entity_table,
        )  # Shape: [batch_size, channels]
        
        # For each sample in the batch
        for i in range(entity_ids):
            entity_id = entity_node_indices[i].item()
            timestamp = seed_times[i].item()
            
            # Create mapping key
            key = f"({entity_id}, {timestamp})"
            
            # Skip if already processed (shouldn't happen with deterministic loader)
            if key in mapping:
                print(f"Warning: Duplicate key {key} found!")
                continue
            
            # Save the mapping
            mapping[key] = global_index
            
            # Collect embedding
            all_embeddings.append(entity_embeddings[i].cpu())
            
            global_index += 1
    
    # Concatenate all embeddings and save as {split_name}.pt
    all_embeddings_tensor = torch.stack(all_embeddings, dim=0)  # [num_samples, channels]
    embeddings_path = base_dir / f"{split_name}.pt"
    torch.save(all_embeddings_tensor, embeddings_path)
    
    print(f"Created {global_index} embeddings for {split_name}, saved to {embeddings_path}")
    print(f"Tensor shape: {all_embeddings_tensor.shape}")
    return mapping


def save_all_snapshots(loader_dict: Dict[str, NeighborLoader], base_dir: Path):
    """
    Save snapshots and mapping for all splits.
    
    Args:
        loader_dict: Dictionary of loaders for each split
        base_dir: Base directory (will contain train/val/test subdirs)
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    
    all_mappings = {}
    
    for split_name, loader in loader_dict.items():
        mapping = create_snapshot_index(loader, split_name, base_dir)
        all_mappings[split_name] = mapping
    
    # Save the complete mapping at the task level
    mapping_path = base_dir / "mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(all_mappings, f, indent=2)
    
    print(f"\nMapping saved to {mapping_path}")
    print(f"Total snapshots: {sum(len(m) for m in all_mappings.values())}")


if args.backbone == "relgnn":
    atomic_routes_list = get_atomic_routes(data.edge_types)
    model = RelGNN_Model(
        data=data,
        col_stats_dict=col_stats_dict,
        out_channels=out_channels,
        norm="batch_norm",
        atomic_routes=atomic_routes_list,
        **model_config,
    ).to(device)
elif args.backbone == "rdl":
    model = Model(
        data=data,
        col_stats_dict=col_stats_dict,
        out_channels=out_channels,
        norm="batch_norm",
        **model_config,
    ).to(device)

state_dict = torch.load(checkpoint_path)
model.load_state_dict(state_dict)

snapshot_base_dir = Path(args.index_path) / args.backbone / args.dataset / args.task
print(f"\nSaving snapshots to {snapshot_base_dir}")

# Save all snapshots and create mapping
save_all_snapshots(loader_dict, snapshot_base_dir)

print("\nSnapshot indexing complete!")
print(f"Structure: {snapshot_base_dir}/{{train.pt,val.pt,test.pt,mapping.json}}")
