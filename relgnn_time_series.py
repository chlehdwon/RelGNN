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
from relgnn.text_embedder import GloveTextEmbedding
from relgnn.utils import get_configs
from relgnn.atomic_routes import get_atomic_routes

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--task", type=str, default="driver-top3")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("/data/starlab/relbench_examples"),
)
parser.add_argument("--checkpoint_dir", type=str, default="/data/starlab/ckpts/relgnn/")

args = parser.parse_args()

checkpoint_path = Path(args.checkpoint_dir) / f"{args.dataset}_{args.task}.pth"
if not checkpoint_path.exists():
    checkpoint_path = Path(hf_hub_download(repo_id="tianlangchen/RelGNN", filename=f"{args.dataset}_{args.task}.pth", cache_dir=args.checkpoint_dir))
assert checkpoint_path.exists(), "Checkpoint not found. Please download the checkpoint first."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(42)

dataset: Dataset = get_dataset(args.dataset, download=True)
task: EntityTask = get_task(args.dataset, args.task, download=True)

model_config, loader_config = get_configs(args.dataset, args.task)

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

# all_table = test_table.df
all_table = pd.concat([train_table.df, val_table.df, test_table.df])
# all_table = pd.concat([train_table.df, val_table.df])


# Create loaders for val and test
loader_dict: Dict[str, NeighborLoader] = {}
for split in ["val", "test"]:
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
        temporal_strategy="last",
        shuffle=split == "train",
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
    
    # If alpha < 1.0, incorporate entity history from test table
    if alpha < 1.0:
        print(f"\nIncorporating entity history with alpha={alpha}")
        print(f"Building per-sample history from test table (only past labels)")
        
        # Prepare test table data
        # test_df = all_table.df.copy()
        test_df = all_table.copy()
        entity_col = task.entity_col
        target_col = task.target_col
        time_col = task.time_col
        
        # Convert time column to timestamp (int) if it's datetime
        if pd.api.types.is_datetime64_any_dtype(test_df[time_col]):
            test_df[time_col] = test_df[time_col].astype('int64') // 10**9  # Convert to Unix timestamp
        
        final_preds = all_preds.copy()
        has_history_count = 0
        no_history_count = 0
        
        for i, (entity_id, seed_time) in enumerate(zip(all_entity_indices, all_seed_times)):
            entity_id = int(entity_id)
            seed_time = int(seed_time)
            
            # Get all past records for this entity (time < seed_time)
            past_records = test_df[
                (test_df[entity_col] == entity_id) & 
                (test_df[time_col] < seed_time)
            ]
            
            model_pred = all_preds[i]
            if len(past_records) > 0:
                # Calculate mean of past labels
                history_mean = past_records[target_col].mean()
                
                # Combine: alpha * model + (1-alpha) * history
                final_preds[i] = alpha * model_pred + (1 - alpha) * history_mean
                has_history_count += 1
            else:
                final_preds[i] = model_pred
                # No past history: use model prediction only
                no_history_count += 1
        
        print(f"Samples with past history: {has_history_count}")
        print(f"Samples without past history: {no_history_count}")
        
        return final_preds
    else:
        # No history, return model predictions only
        return all_preds

atomic_routes_list = get_atomic_routes(data.edge_types)

model = RelGNN_Model(
    data=data,
    col_stats_dict=col_stats_dict,
    out_channels=out_channels,
    norm="batch_norm",
    atomic_routes=atomic_routes_list,
    **model_config,
).to(device)

state_dict = torch.load(checkpoint_path)
model.load_state_dict(state_dict)

# Get test table with labels (mask_input_cols=False to access target column)
test_table = task.get_table("test", mask_input_cols=False)

# Test with different alpha values
print(f"\n{'='*80}")
print(f"TESTING WITH ENTITY HISTORY FROM TEST TABLE (ONLY PAST LABELS)")
print(f"{'='*80}\n")

# 1. Baseline: Model prediction only (alpha=1.0)
print(f"[1] Baseline - Model prediction only (alpha=1.0)")
test_pred_baseline = test(loader_dict["test"], test_table, alpha=1.0)
test_metrics_baseline = task.evaluate(test_pred_baseline)
print(f"Test {tune_metric}: {test_metrics_baseline[tune_metric]:.4f}\n")

# 2. History only for entities with history (alpha=0.0 for those with history)
print(f"[2] History mean only (alpha=0.0)")
test_pred_history = test(loader_dict["test"], test_table, alpha=0.0)
test_metrics_history = task.evaluate(test_pred_history)
print(f"Test {tune_metric}: {test_metrics_history[tune_metric]:.4f}\n")

# 3. Balanced combination (alpha=0.5)
print(f"[3] Balanced combination (alpha=0.5)")
test_pred_balanced = test(loader_dict["test"], test_table, alpha=0.5)
test_metrics_balanced = task.evaluate(test_pred_balanced)
print(f"Test {tune_metric}: {test_metrics_balanced[tune_metric]:.4f}\n")

# 4. Model-heavy combination (alpha=0.7)
print(f"[4] Model-heavy combination (alpha=0.7)")
test_pred_model_heavy = test(loader_dict["test"], test_table, alpha=0.7)
test_metrics_model_heavy = task.evaluate(test_pred_model_heavy)
print(f"Test {tune_metric}: {test_metrics_model_heavy[tune_metric]:.4f}\n")

# 5. History-heavy combination (alpha=0.3)
print(f"[5] History-heavy combination (alpha=0.3)")
test_pred_history_heavy = test(loader_dict["test"], test_table, alpha=0.3)
test_metrics_history_heavy = task.evaluate(test_pred_history_heavy)
print(f"Test {tune_metric}: {test_metrics_history_heavy[tune_metric]:.4f}\n")

# Summary
print(f"{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"Baseline (model only):         {test_metrics_baseline[tune_metric]:.4f}")
print(f"History only (alpha=0.0):      {test_metrics_history[tune_metric]:.4f}")
print(f"History-heavy (alpha=0.3):     {test_metrics_history_heavy[tune_metric]:.4f}")
print(f"Balanced (alpha=0.5):          {test_metrics_balanced[tune_metric]:.4f}")
print(f"Model-heavy (alpha=0.7):       {test_metrics_model_heavy[tune_metric]:.4f}")
print(f"{'='*80}\n")

