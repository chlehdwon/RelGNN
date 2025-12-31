"""
Entity Time Series Dataset Builder

This module loads the pre-computed embeddings from indexing.py and builds
entity-level time series sequences for training temporal models.

Structure:
    - Loads embeddings from {index_path}/{dataset}/{task}/{split}.pt
    - Loads mapping from {index_path}/{dataset}/{task}/mapping.json
    - Creates entity sequences: entity_id -> [(timestamp, embedding, label), ...]
"""

import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import argparse

from relbench.base import EntityTask
from relbench.modeling.utils import to_unix_time
from relbench.datasets import get_dataset
from relbench.tasks import get_task


class EntityTimeSeriesBuilder:
    """
    Builds entity-level time series data from indexed snapshots.
    
    This class:
    1. Loads pre-computed embeddings from indexing.py output
    2. Matches embeddings with task labels via (entity_id, timestamp) keys
    3. Organizes data by entity_id for time series modeling
    """
    
    def __init__(
        self,
        index_path: str,
        dataset_name: str,
        task_name: str,
        task: EntityTask,
    ):
        """
        Args:
            index_path: Root path where snapshots are saved (e.g., /data/relts/snapshots)
            dataset_name: Dataset name (e.g., rel-amazon)
            task_name: Task name (e.g., user-churn)
            task: EntityTask object for accessing labels
        """
        self.index_path = Path(index_path)
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.task = task
        
        # Path to snapshot directory: {index_path}/{dataset}/{task}/
        self.snapshot_dir = self.index_path / dataset_name / task_name
        
        # Load mapping: {split_name: {(entity_id, timestamp): index}}
        self.mapping = self._load_mapping()
        
        # Load embeddings: {split_name: tensor of shape (num_samples, embed_dim)}
        self.embeddings = self._load_embeddings()
        
        # Build entity sequences for each split
        # entity_sequences: {split -> {entity_id -> sequence}}
        # split_indices: {split -> {entity_id -> {'train_end': int, 'val_end': int}}}
        self.entity_sequences, self.split_indices = self._build_entity_sequences()
    
    def _load_mapping(self) -> Dict[str, Dict[str, int]]:
        """Load the mapping.json file"""
        mapping_path = self.snapshot_dir / "mapping.json"
        
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"Mapping file not found at {mapping_path}. "
                f"Please run indexing.py first to generate snapshots."
            )
        
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        
        return mapping
    
    def _load_embeddings(self) -> Dict[str, torch.Tensor]:
        """Load embedding tensors for each split"""
        embeddings = {}
        
        for split in ['train', 'val', 'test']:
            embed_path = self.snapshot_dir / f"{split}.pt"
            
            if not embed_path.exists():
                raise FileNotFoundError(
                    f"Embedding file not found at {embed_path}. "
                    f"Please run indexing.py first."
                )
            
            embeddings[split] = torch.load(embed_path)
        
        return embeddings
    
    def _build_entity_sequences(self) -> Tuple[Dict[str, Dict[int, List[Tuple[float, np.ndarray, float]]]], Dict[str, Dict[int, Dict[str, int]]]]:
        """
        Build cumulative entity sequences for each split.
        
        For each split, the sequence includes all previous splits:
        - train: train only
        - val: train + val
        - test: train + val + test
        
        Returns:
            Tuple of:
            - entity_sequences: Dict[split_name -> Dict[entity_id -> List[(timestamp, embedding, label)]]]
            - split_indices: Dict[split_name -> Dict[entity_id -> {'train_end': int, 'val_end': int}]]
                             where indices represent the end position (exclusive) of each split
        """
        entity_sequences = {}
        split_indices = {}
        cumulative_sequences = defaultdict(list)
        
        # Track cumulative lengths for each entity after each split
        train_end_indices = defaultdict(int)
        val_end_indices = defaultdict(int)
        
        # Build cumulative sequences progressively
        for split in ['train', 'val', 'test']:
            # Get table and embeddings for this split
            table = self.task.get_table(split, mask_input_cols=False)
            split_mapping = self.mapping[split]
            split_embeddings = self.embeddings[split]
            
            # Temporary storage for this split's events
            split_events = defaultdict(list)
            
            # Convert timestamps using the same method as indexing.py
            unix_timestamps = to_unix_time(table.df[self.task.time_col])
            
            # Extract columns as arrays (faster than iterrows)
            entity_ids = table.df[self.task.entity_col].values
            labels = table.df[self.task.target_col].values
            
            # Process all rows
            for i in range(len(table.df)):
                entity_id = int(entity_ids[i])
                timestamp = int(unix_timestamps[i])
                label = float(labels[i])
                
                # Create key to lookup in mapping
                key = f"({entity_id}, {timestamp})"
                
                # Get embedding index
                if key not in split_mapping:
                    continue
                
                embed_idx = split_mapping[key]
                embedding = split_embeddings[embed_idx].numpy()
                
                # Add to this split's events
                split_events[entity_id].append((timestamp, embedding, label))
            
            # Add this split's events to cumulative sequences
            for entity_id, events in split_events.items():
                cumulative_sequences[entity_id].extend(events)
            
            # Update end indices BEFORE storing (for current split)
            if split == 'train':
                for entity_id in cumulative_sequences:
                    train_end_indices[entity_id] = len(cumulative_sequences[entity_id])
            elif split == 'val':
                for entity_id in cumulative_sequences:
                    val_end_indices[entity_id] = len(cumulative_sequences[entity_id])
            
            # Sort cumulative sequences by timestamp and save
            current_cumulative = {}
            current_split_indices = {}
            
            for entity_id, events in cumulative_sequences.items():
                sorted_events = sorted(events, key=lambda x: x[0])
                current_cumulative[entity_id] = sorted_events
                
                # Store split boundary indices (now properly updated)
                current_split_indices[entity_id] = {
                    'train_end': train_end_indices[entity_id],
                    'val_end': val_end_indices[entity_id],
                }
            
            entity_sequences[split] = current_cumulative
            split_indices[split] = current_split_indices
        
        return entity_sequences, split_indices
    
    def get_entity_sequences(self, split: str) -> Tuple[Dict[int, List[Tuple[float, np.ndarray, float]]], Dict[int, Dict[str, int]]]:
        """
        Get entity sequences and split indices for a specific split.
        
        Args:
            split: 'train', 'val', or 'test'
        
        Returns:
            Tuple of:
            - Dict mapping entity_id -> List of (timestamp, embedding, label) tuples
            - Dict mapping entity_id -> {'train_end': int, 'val_end': int}
        """
        return self.entity_sequences[split], self.split_indices[split]
    
    def save_entity_sequences(self, output_path: str):
        """
        Save entity sequences and split indices to disk for faster loading later.
        
        Args:
            output_path: Path to save the sequences
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split, sequences in self.entity_sequences.items():
            # Convert to saveable format
            save_data = {
                'sequences': {},
                'split_indices': self.split_indices[split],
            }
            
            for entity_id, events in sequences.items():
                # Unpack all events at once (single pass)
                timestamps, embeddings, labels = zip(*events)
                
                save_data['sequences'][entity_id] = {
                    'timestamps': list(timestamps),
                    'embeddings': np.stack(embeddings),
                    'labels': list(labels),
                }
            
            # Save as pytorch file
            save_path = output_path / f"{split}_sequences.pt"
            torch.save(save_data, save_path)
    

class ARSampleDataset(Dataset):
    """
    PyTorch Dataset for auto-regressive samples.
    
    Each sample contains:
        - entity_id: int
        - input_timestamps: (window_size,) - left-aligned, padded with zeros on the right
        - input_embeddings: (window_size, embed_dim) - left-aligned, padded with zeros on the right
        - input_labels: (window_size,) - left-aligned, padded with zeros on the right
        - input_mask: (window_size,) - True for valid positions, False for padding
        - target_timestamp: float
        - target_embedding: (embed_dim,)
        - target_label: float
    
    Note: Left-aligned padding ensures that position 0 always contains valid data,
          which prevents NaN issues in causal attention where position 0 can only attend to itself.
    """
    
    def __init__(self, ar_samples: List[Dict]):
        """
        Args:
            ar_samples: List of AR samples from create_autoregressive_samples
        """
        self.samples = ar_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get one AR sample.
        
        Returns:
            Dict with tensors for input and target
        """
        sample = self.samples[idx]
        
        return {
            'entity_id': torch.tensor(sample['entity_id'], dtype=torch.long),
            'input_timestamps': torch.tensor(sample['input_timestamps'], dtype=torch.float32),
            'input_embeddings': torch.tensor(sample['input_embeddings'], dtype=torch.float32),
            'input_labels': torch.tensor(sample['input_labels'], dtype=torch.float32),
            'input_mask': torch.tensor(sample['input_mask'], dtype=torch.bool),
            'target_timestamp': torch.tensor(sample['target_timestamp'], dtype=torch.float32),
            'target_embedding': torch.tensor(sample['target_embedding'], dtype=torch.float32),
            'target_label': torch.tensor(sample['target_label'], dtype=torch.float32),
        }


def create_ar_dataloaders(
    entity_sequences: Dict[str, Dict[int, List[Tuple[float, np.ndarray, float]]]],
    split_indices: Dict[str, Dict[int, Dict[str, int]]],
    window_size: int = 10,
    batch_size: int = 32,
    num_workers: int = 0,
    min_input_length: int = 0,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for auto-regressive training.
    
    Args:
        entity_sequences: Dict[split -> Dict[entity_id -> cumulative sequence]]
        split_indices: Dict[split -> Dict[entity_id -> {'train_end': int, 'val_end': int}]]
        window_size: Maximum number of past timesteps to use as input
        batch_size: Batch size
        num_workers: Number of dataloader workers
        min_input_length: Minimum number of valid inputs required (default: 0)
                         Applied to all splits to exclude cold-start samples
    
    Returns:
        Dict with 'train', 'val', 'test' DataLoaders
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        # Create AR samples using split-specific targets only
        ar_samples = create_autoregressive_samples(
            entity_sequences[split],
            split_indices[split],
            split_name=split,
            window_size=window_size,
            min_input_length=min_input_length,
        )
        
        # Create dataset
        dataset = ARSampleDataset(ar_samples)
        
        # Create dataloader
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    
    return dataloaders


def create_strict_ar_dataloaders(
    entity_sequences: Dict[str, Dict[int, List[Tuple[float, np.ndarray, float]]]],
    split_indices: Dict[str, Dict[int, Dict[str, int]]],
    window_size: int = 10,
    batch_size: int = 32,
    num_workers: int = 0,
    min_input_length: int = 0,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for strict auto-regressive training.
    
    This version strictly enforces temporal split boundaries:
    - train: Input from train history only
    - val: Input from train split only (no val data in inputs)
    - test: Input from train + val splits only (no test data in inputs)
    
    Args:
        entity_sequences: Dict[split -> Dict[entity_id -> cumulative sequence]]
        split_indices: Dict[split -> Dict[entity_id -> {'train_end': int, 'val_end': int}]]
        window_size: Maximum number of past timesteps to use as input
        batch_size: Batch size
        num_workers: Number of dataloader workers
        min_input_length: Minimum number of valid inputs required (default: 0)
                         Applied to all splits to exclude cold-start samples
    
    Returns:
        Dict with 'train', 'val', 'test' DataLoaders
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        # Create strict AR samples with temporal boundaries
        ar_samples = create_strict_autoregressive_samples(
            entity_sequences[split],
            split_indices[split],
            split_name=split,
            window_size=window_size,
            min_input_length=min_input_length,
        )
        
        # Create dataset
        dataset = ARSampleDataset(ar_samples)
        
        # Create dataloader
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    
    return dataloaders


def create_autoregressive_samples(
    entity_sequences: Dict[int, List[Tuple[float, np.ndarray, float]]],
    split_indices: Dict[int, Dict[str, int]],
    split_name: str,
    window_size: int = 10,
    min_input_length: int = 0,
) -> List[Dict]:
    """
    Create auto-regressive training samples using sliding window with padding.
    
    For each entity's cumulative sequence, create samples where:
    - Input: up to `window_size` previous timesteps from entire history (all previous splits)
    - Target: current timestep from the current split only
    - If input length is shorter than window_size, pad with zeros and create mask
    - Samples with fewer than `min_input_length` valid inputs are excluded
    
    Args:
        entity_sequences: Dict[entity_id -> cumulative List[(timestamp, embedding, label)]]
        split_indices: Dict[entity_id -> {'train_end': int, 'val_end': int}]
        split_name: Name of current split ('train', 'val', or 'test')
        window_size: Maximum number of past timesteps to use as input
        min_input_length: Minimum number of valid input timesteps required (default: 1)
    
    Returns:
        List of samples, each containing:
            - entity_id: int
            - input_timestamps: (window_size,) - left-aligned, padded on right if necessary
            - input_embeddings: (window_size, embed_dim) - left-aligned, padded on right if necessary
            - input_labels: (window_size,) - left-aligned, padded on right if necessary
            - input_mask: (window_size,) - True for valid positions, False for padding
            - target_timestamp: float
            - target_embedding: (embed_dim,)
            - target_label: float
    """
    samples = []
    
    for entity_id, sequence in entity_sequences.items():
        seq_len = len(sequence)
        
        # Skip empty sequences
        if seq_len < 1:
            continue
        
        # Get embedding dimension from first event in sequence
        embed_dim = sequence[0][1].shape[0]
        
        # Determine target range for this split
        indices = split_indices[entity_id]
        if split_name == 'train':
            target_start = 0
            target_end = indices['train_end']
        elif split_name == 'val':
            target_start = indices['train_end']
            target_end = indices['val_end']
        else:  # test
            target_start = indices['val_end']
            target_end = seq_len
        
        # Create samples only for targets in this split's range
        for target_idx in range(target_start, target_end):
            # Input: all previous timesteps up to target_idx (max window_size)
            # This includes data from previous splits!
            start_idx = max(0, target_idx - window_size)
            input_seq = sequence[start_idx:target_idx]
            
            # Extract input data
            input_len = len(input_seq)
            
            # Skip samples with insufficient input length
            if input_len < min_input_length:
                continue
            
            # Target: position target_idx (guaranteed to be in current split)
            target = sequence[target_idx]
            
            # Create padded arrays
            input_timestamps = np.zeros(window_size, dtype=np.float32)
            input_embeddings = np.zeros((window_size, embed_dim), dtype=np.float32)
            input_labels = np.zeros(window_size, dtype=np.float32)
            input_mask = np.zeros(window_size, dtype=bool)
            
            # Fill with actual data (left-aligned: padding on the right)
            # This ensures position 0 always has valid data if input_len > 0
            if input_len > 0:
                # Unpack all at once (single pass through input_seq)
                ts, embs, lbls = zip(*input_seq)
                input_timestamps[:input_len] = ts
                input_embeddings[:input_len] = np.stack(embs)
                input_labels[:input_len] = lbls
                input_mask[:input_len] = True
            
            # Extract target data
            target_timestamp = target[0]
            target_embedding = target[1]
            target_label = target[2]
            
            samples.append({
                'entity_id': entity_id,
                'input_timestamps': input_timestamps,
                'input_embeddings': input_embeddings,
                'input_labels': input_labels,
                'input_mask': input_mask,
                'target_timestamp': target_timestamp,
                'target_embedding': target_embedding,
                'target_label': target_label,
            })
    
    return samples


def create_strict_autoregressive_samples(
    entity_sequences: Dict[int, List[Tuple[float, np.ndarray, float]]],
    split_indices: Dict[int, Dict[str, int]],
    split_name: str,
    window_size: int = 10,
    min_input_length: int = 0,
) -> List[Dict]:
    """
    Create strict auto-regressive training samples with temporal split boundaries.
    
    This version strictly enforces that inputs only come from allowed historical splits:
    - train: Input from train history only (before target), Target from train
    - val: Input from train split only, Target from val
    - test: Input from train + val splits only, Target from test
    
    This prevents any data leakage across split boundaries.
    
    Args:
        entity_sequences: Dict[entity_id -> cumulative List[(timestamp, embedding, label)]]
        split_indices: Dict[entity_id -> {'train_end': int, 'val_end': int}]
        split_name: Name of current split ('train', 'val', or 'test')
        window_size: Maximum number of past timesteps to use as input
        min_input_length: Minimum number of valid input timesteps required (default: 0)
    
    Returns:
        List of samples, each containing:
            - entity_id: int
            - input_timestamps: (window_size,) - left-aligned, padded on right if necessary
            - input_embeddings: (window_size, embed_dim) - left-aligned, padded on right if necessary
            - input_labels: (window_size,) - left-aligned, padded on right if necessary
            - input_mask: (window_size,) - True for valid positions, False for padding
            - target_timestamp: float
            - target_embedding: (embed_dim,)
            - target_label: float
    """
    samples = []
    
    for entity_id, sequence in entity_sequences.items():
        seq_len = len(sequence)
        
        # Skip empty sequences
        if seq_len < 1:
            continue
        
        # Get embedding dimension from first event in sequence
        embed_dim = sequence[0][1].shape[0]
        
        # Determine target range and input range for this split
        indices = split_indices[entity_id]
        
        if split_name == 'train':
            # Train: targets from train, inputs from train (before target)
            target_start = 0
            target_end = indices['train_end']
            input_end = indices['train_end']  # Can use train history
        elif split_name == 'val':
            # Val: targets from val, inputs from train only
            target_start = indices['train_end']
            target_end = indices['val_end']
            input_end = indices['train_end']  # Only train history allowed
        else:  # test
            # Test: targets from test, inputs from train + val
            target_start = indices['val_end']
            target_end = seq_len
            input_end = indices['val_end']  # Only train + val history allowed
        
        # Create samples only for targets in this split's range
        for target_idx in range(target_start, target_end):
            # Strict input constraint: only use history up to input_end
            if split_name == 'train':
                # For train, use history before target within train split
                available_input_end = target_idx
            else:
                # For val/test, use all allowed history (train only for val, train+val for test)
                available_input_end = min(input_end, target_idx)
            
            # Extract input sequence from allowed range
            start_idx = max(0, available_input_end - window_size)
            input_seq = sequence[start_idx:available_input_end]
            
            # Extract input data
            input_len = len(input_seq)
            
            # Skip samples with insufficient input length
            if input_len < min_input_length:
                continue
            
            # Target: position target_idx (guaranteed to be in current split)
            target = sequence[target_idx]
            
            # Create padded arrays
            input_timestamps = np.zeros(window_size, dtype=np.float32)
            input_embeddings = np.zeros((window_size, embed_dim), dtype=np.float32)
            input_labels = np.zeros(window_size, dtype=np.float32)
            input_mask = np.zeros(window_size, dtype=bool)
            
            # Fill with actual data (left-aligned: padding on the right)
            if input_len > 0:
                # Unpack all at once (single pass through input_seq)
                ts, embs, lbls = zip(*input_seq)
                input_timestamps[:input_len] = ts
                input_embeddings[:input_len] = np.stack(embs)
                input_labels[:input_len] = lbls
                input_mask[:input_len] = True
            
            # Extract target data
            target_timestamp = target[0]
            target_embedding = target[1]
            target_label = target[2]
            
            samples.append({
                'entity_id': entity_id,
                'input_timestamps': input_timestamps,
                'input_embeddings': input_embeddings,
                'input_labels': input_labels,
                'input_mask': input_mask,
                'target_timestamp': target_timestamp,
                'target_embedding': target_embedding,
                'target_label': target_label,
            })
    
    return samples

