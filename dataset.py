"""
Entity Time Series Dataset Builder

This module loads the pre-computed embeddings from indexing.py and builds
entity-level time series sequences for training temporal models.

Structure:
    - Loads embeddings from {index_path}/{dataset}/{task}/{split}.pt
    - Loads mapping from {index_path}/{dataset}/{task}/mapping.json
    - Creates entity sequences: entity_id -> [(timestamp, embedding, label), ...]
    - Loads entity embeddings from entity.pt for per-entity features
"""

import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, IterableDataset
import argparse
import random
from tqdm import tqdm

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
        backbone: str = "relgnn",
        use_random_embedding: bool = False,
    ):
        """
        Args:
            index_path: Root path where snapshots are saved (e.g., /data/relts/snapshots)
            dataset_name: Dataset name (e.g., rel-amazon)
            task_name: Task name (e.g., user-churn)
            task: EntityTask object for accessing labels
            backbone: Backbone model type (e.g., relgnn, rdl, relgt)
            use_random_embedding: If True, use random embeddings instead of pretrained embeddings
        """
        self.index_path = Path(index_path)
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.task = task
        self.backbone = backbone
        self.use_random_embedding = use_random_embedding
        
        # Path to snapshot directory: {index_path}/{backbone}/{dataset}/{task}/
        self.snapshot_dir = self.index_path / backbone / dataset_name / task_name
        
        # Load mapping: {split_name: {(entity_id, timestamp): index}}
        self.mapping = self._load_mapping()
        
        # Load embeddings: {split_name: tensor of shape (num_samples, embed_dim)}
        self.embeddings = self._load_embeddings()

        # Load entity embeddings: tensor + id mapping
        self.entity_embeddings, self.entity_id_to_index = self._load_entity_embeddings()
        
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
        
        if self.use_random_embedding:
            # Generate random embeddings instead of loading pretrained ones
            # First, load one embedding file to get the shape
            embed_path = self.snapshot_dir / "train.pt"
            if embed_path.exists():
                sample_embedding = torch.load(embed_path)
                embed_dim = sample_embedding.shape[1]
            else:
                # Fallback: try to get dimension from any available split
                for split in ['train', 'val', 'test']:
                    embed_path = self.snapshot_dir / f"{split}.pt"
                    if embed_path.exists():
                        sample_embedding = torch.load(embed_path)
                        embed_dim = sample_embedding.shape[1]
                        break
                else:
                    raise FileNotFoundError(
                        f"No embedding file found to determine dimension. "
                        f"Please run indexing.py first."
                    )
            
            for split in ['train', 'val', 'test']:
                num_samples = len(self.mapping[split])
                random_embeddings = torch.from_numpy(
                    np.random.randn(num_samples, embed_dim).astype(np.float32)
                )
                embeddings[split] = random_embeddings
        else:
            # Load pretrained embeddings
            for split in ['train', 'val', 'test']:
                embed_path = self.snapshot_dir / f"{split}.pt"
                
                if not embed_path.exists():
                    raise FileNotFoundError(
                        f"Embedding file not found at {embed_path}. "
                        f"Please run indexing.py first."
                    )
                
                embeddings[split] = torch.load(embed_path)
        
        return embeddings

    def _load_entity_embeddings(self) -> Tuple[torch.Tensor, Dict[int, int]]:
        """Load per-entity embeddings saved by indexing.py"""
        embedding_path = self.snapshot_dir / "entity.pt"
        mapping_path = self.snapshot_dir / "entity_mapping.json"

        if not embedding_path.exists() or not mapping_path.exists():
            raise FileNotFoundError(
                f"Entity embeddings not found at {embedding_path} and {mapping_path}. "
                f"Please run indexing.py first to generate entity embeddings."
            )

        entity_embeddings = torch.load(embedding_path, map_location="cpu")
        with open(mapping_path, "r") as f:
            entity_mapping = json.load(f)

        entity_id_to_index = {int(k): int(v) for k, v in entity_mapping.items()}

        return entity_embeddings, entity_id_to_index

    def get_entity_embedding(self, entity_id: int) -> Optional[torch.Tensor]:
        """Return the per-entity embedding tensor if available."""
        idx = self.entity_id_to_index.get(entity_id)
        if idx is None:
            return None
        return self.entity_embeddings[idx]
    
    def _build_entity_sequences(self) -> Tuple[Dict[str, Dict[int, List[Tuple[float, np.ndarray, float, np.ndarray]]]], Dict[str, Dict[int, Dict[str, int]]]]:
        """
        Build cumulative entity sequences for each split.
        
        For each split, the sequence includes all previous splits:
        - train: train only
        - val: train + val
        - test: train + val + test
        
        Returns:
            Tuple of:
            - entity_sequences: Dict[split_name -> Dict[entity_id -> List[(timestamp, embedding, label, entity_embedding)]]]
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
            for i in tqdm(range(len(table.df)), desc=f"Building sequences ({split})"):
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
                
                # Get entity embedding
                entity_embedding = self.get_entity_embedding(entity_id)
                
                # Add to this split's events (now includes entity_embedding)
                split_events[entity_id].append((timestamp, embedding, label, entity_embedding))
            
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
    
    def get_entity_sequences(self, split: str) -> Tuple[Dict[int, List[Tuple[float, np.ndarray, float, np.ndarray]]], Dict[int, Dict[str, int]]]:
        """
        Get entity sequences and split indices for a specific split.
        
        Args:
            split: 'train', 'val', or 'test'
        
        Returns:
            Tuple of:
            - Dict mapping entity_id -> List of (timestamp, embedding, label, entity_embedding) tuples
            - Dict mapping entity_id -> {'train_end': int, 'val_end': int}
        """
        return self.entity_sequences[split], self.split_indices[split]
    
def _augment_sequence_with_retrieval(
    input_seq: List[Tuple[float, np.ndarray, float, np.ndarray]],
    target: Tuple[float, np.ndarray, float, np.ndarray],
    retrieval_manager,
    top_k: int,
    window_size: int,
) -> List[Tuple[float, np.ndarray, float, np.ndarray]]:
    """
    Augment short input sequences using retrieval results.

    This only adds up to (window_size - len(input_seq)) items and keeps
    the final sequence sorted by timestamp.
    
    NOTE: This function is kept for backward compatibility but is inefficient.
    Use _batch_augment_sequences_with_retrieval for better performance.
    """
    if retrieval_manager is None or top_k <= 0:
        return input_seq

    if len(input_seq) >= window_size:
        return input_seq

    needed = window_size - len(input_seq)
    k = min(top_k, needed)
    device = retrieval_manager.device

    query = {
        "target_embedding": torch.tensor(target[1], dtype=torch.float32).unsqueeze(0).to(device),
        "target_timestamp": torch.tensor([target[0]], dtype=torch.float32).to(device),
        "target_entity_embedding": torch.tensor(target[3], dtype=torch.float32).unsqueeze(0).to(device),
    }

    retrieved = retrieval_manager.retrieve(query, k=k)
    if not retrieved or len(retrieved[0]) == 0:
        return input_seq

    augmented = list(input_seq)
    target_timestamp = float(target[0])
    target_entity_emb = target[3]  # entity embedding from target
    for item in retrieved[0]:
        timestamp = float(item["timestamp"])
        if timestamp > target_timestamp:
            continue
        embedding = item["embedding"]
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.numpy()
        # Use target's entity embedding for retrieved items
        augmented.append((timestamp, embedding, float(item["label"]), target_entity_emb))

    augmented = sorted(augmented, key=lambda x: x[0])
    if len(augmented) > window_size:
        augmented = augmented[-window_size:]

    return augmented


def _batch_augment_sequences_with_retrieval(
    samples_with_metadata: List[Tuple[Dict, List[Tuple[float, np.ndarray, float, np.ndarray]], Tuple[float, np.ndarray, float, np.ndarray]]],
    retrieval_manager,
    top_k: int,
    window_size: int,
    batch_size: int = 256,
) -> List[Dict]:
    """
    Batch version of sequence augmentation using retrieval.
    
    This function processes multiple samples at once for better efficiency.
    
    Args:
        samples_with_metadata: List of (sample_dict, input_seq, target) tuples
        retrieval_manager: RetrievalManager instance
        top_k: Number of neighbors to retrieve
        window_size: Maximum sequence length
        batch_size: Batch size for retrieval queries
    
    Returns:
        List of completed sample dictionaries with augmented sequences
    """
    if retrieval_manager is None or top_k <= 0:
        return [sample for sample, _, _ in samples_with_metadata]
    
    device = retrieval_manager.device
    completed_samples = []
    
    # Separate samples that need augmentation from those that don't
    samples_needing_augmentation = []
    augmentation_indices = []
    
    for idx, (sample, input_seq, target) in enumerate(samples_with_metadata):
        if len(input_seq) < window_size:
            samples_needing_augmentation.append((idx, sample, input_seq, target))
            augmentation_indices.append(idx)
        else:
            completed_samples.append((idx, sample))
    
    if not samples_needing_augmentation:
        return [sample for sample, _, _ in samples_with_metadata]
    
    # Process in batches
    all_augmented_samples = []
    
    for batch_start in tqdm(
        range(0, len(samples_needing_augmentation), batch_size),
        desc="Augmenting sequences (batch retrieval)",
        leave=False,
    ):
        batch_end = min(batch_start + batch_size, len(samples_needing_augmentation))
        batch = samples_needing_augmentation[batch_start:batch_end]
        
        # Prepare batch queries
        target_embeddings = []
        target_timestamps = []
        target_entity_embeddings = []
        k_values = []
        
        for _, sample, input_seq, target in batch:
            needed = window_size - len(input_seq)
            k = min(top_k, needed)
            k_values.append(k)
            target_embeddings.append(target[1])
            target_timestamps.append(target[0])
            target_entity_embeddings.append(target[3])
        
        # Batch retrieval
        max_k = max(k_values)
        query_batch = {
            "target_embedding": torch.tensor(np.stack(target_embeddings), dtype=torch.float32).to(device),
            "target_timestamp": torch.tensor(target_timestamps, dtype=torch.float32).to(device),
            "target_entity_embedding": torch.tensor(np.stack(target_entity_embeddings), dtype=torch.float32).to(device),
        }
        
        retrieved_batch = retrieval_manager.retrieve(query_batch, k=max_k)
        
        # Augment each sample with its retrieved results
        for batch_idx, (orig_idx, sample, input_seq, target) in enumerate(batch):
            retrieved = retrieved_batch[batch_idx]
            k = k_values[batch_idx]
            
            augmented = list(input_seq)
            target_timestamp = float(target[0])
            target_entity_emb = target[3]  # entity embedding from target
            
            # Add retrieved items (up to k items)
            added_count = 0
            for item in retrieved:
                if added_count >= k:
                    break
                    
                timestamp = float(item["timestamp"])
                if timestamp > target_timestamp:
                    continue
                    
                embedding = item["embedding"]
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().numpy()
                    
                # Use target's entity embedding for retrieved items
                augmented.append((timestamp, embedding, float(item["label"]), target_entity_emb))
                added_count += 1
            
            # Sort and truncate
            augmented = sorted(augmented, key=lambda x: x[0])
            if len(augmented) > window_size:
                augmented = augmented[-window_size:]
            
            # Update sample with augmented sequence
            input_len = len(augmented)
            embed_dim = sample['input_embeddings'].shape[1]
            entity_embed_dim = sample['input_entity_embeddings'].shape[1]
            
            input_timestamps = np.zeros(window_size, dtype=np.float32)
            input_embeddings = np.zeros((window_size, embed_dim), dtype=np.float32)
            input_labels = np.zeros(window_size, dtype=np.float32)
            input_entity_embeddings = np.zeros((window_size, entity_embed_dim), dtype=np.float32)
            input_mask = np.zeros(window_size, dtype=bool)
            
            if input_len > 0:
                ts, embs, lbls, ent_embs = zip(*augmented)
                input_timestamps[:input_len] = ts
                input_embeddings[:input_len] = np.stack(embs)
                input_labels[:input_len] = lbls
                input_entity_embeddings[:input_len] = np.stack(ent_embs)
                input_mask[:input_len] = True
            
            sample['input_timestamps'] = input_timestamps
            sample['input_embeddings'] = input_embeddings
            sample['input_labels'] = input_labels
            sample['input_entity_embeddings'] = input_entity_embeddings
            sample['input_mask'] = input_mask
            
            all_augmented_samples.append((orig_idx, sample))
    
    # Merge augmented and non-augmented samples in original order
    all_samples = completed_samples + all_augmented_samples
    all_samples.sort(key=lambda x: x[0])
    
    return [sample for _, sample in all_samples]
    

class ARSampleDataset(Dataset):
    """
    PyTorch Dataset for auto-regressive samples.
    
    Each sample contains:
        - entity_id: int
        - input_timestamps: (window_size,) - left-aligned, padded with zeros on the right
        - input_embeddings: (window_size, embed_dim) - left-aligned, padded with zeros on the right
        - input_labels: (window_size,) - left-aligned, padded with zeros on the right
        - input_entity_embeddings: (window_size, entity_embed_dim) - left-aligned, padded with zeros on the right
        - input_mask: (window_size,) - True for valid positions, False for padding
        - target_timestamp: float
        - target_embedding: (embed_dim,)
        - target_label: float
        - target_entity_embedding: (entity_embed_dim,)
    
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

        def _to_tensor(x, dtype):
            if isinstance(x, torch.Tensor):
                return x.detach().clone().to(dtype)
            return torch.tensor(x, dtype=dtype)

        return {
            'entity_id': _to_tensor(sample['entity_id'], torch.long),
            'input_timestamps': _to_tensor(sample['input_timestamps'], torch.float32),
            'input_embeddings': _to_tensor(sample['input_embeddings'], torch.float32),
            'input_labels': _to_tensor(sample['input_labels'], torch.float32),
            'input_entity_embeddings': _to_tensor(sample['input_entity_embeddings'], torch.float32),
            'input_mask': _to_tensor(sample['input_mask'], torch.bool),
            'target_timestamp': _to_tensor(sample['target_timestamp'], torch.float32),
            'target_embedding': _to_tensor(sample['target_embedding'], torch.float32),
            'target_label': _to_tensor(sample['target_label'], torch.float32),
            'target_entity_embedding': _to_tensor(sample['target_entity_embedding'], torch.float32),
        }


class RandomARSampleDataset(IterableDataset):
    """
    PyTorch IterableDataset for auto-regressive samples with random sampling per epoch.
    
    This dataset generates random AR samples on-the-fly for each epoch, providing
    different samples every time the DataLoader iterates.
    
    Each sample contains:
        - entity_id: int
        - input_timestamps: (window_size,) - left-aligned, padded with zeros on the right
        - input_embeddings: (window_size, embed_dim) - left-aligned, padded with zeros on the right
        - input_labels: (window_size,) - left-aligned, padded with zeros on the right
        - input_entity_embeddings: (window_size, entity_embed_dim) - left-aligned, padded with zeros on the right
        - input_mask: (window_size,) - True for valid positions, False for padding
        - target_timestamp: float
        - target_embedding: (embed_dim,)
        - target_label: float
        - target_entity_embedding: (entity_embed_dim,)
    """
    
    def __init__(
        self,
        entity_sequences: Dict[int, List[Tuple[float, np.ndarray, float, np.ndarray]]],
        split_indices: Dict[int, Dict[str, int]],
        split_name: str,
        window_size: int = 10,
        min_input_length: int = 0,
        samples_per_epoch: Optional[int] = None,
        seed: Optional[int] = None,
        retrieval_manager=None,
        top_k: int = 0,
        retrieval_batch_size: int = 256,
    ):
        """
        Args:
            entity_sequences: Dict[entity_id -> cumulative List[(timestamp, embedding, label, entity_embedding)]]
            split_indices: Dict[entity_id -> {'train_end': int, 'val_end': int}]
            split_name: Name of current split ('train', 'val', or 'test')
            window_size: Maximum number of past timesteps to use as input
            min_input_length: Minimum number of valid input timesteps required
            samples_per_epoch: Number of samples to generate per epoch. If None, generates all possible samples.
            seed: Random seed for reproducibility (None for random)
            retrieval_manager: Optional retrieval manager for sequence augmentation
            top_k: Number of neighbors to retrieve for augmentation
            retrieval_batch_size: Batch size for retrieval operations
        """
        self.entity_sequences = entity_sequences
        self.split_indices = split_indices
        self.split_name = split_name
        self.window_size = window_size
        self.min_input_length = min_input_length
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.retrieval_manager = retrieval_manager
        self.top_k = top_k
        self.retrieval_batch_size = retrieval_batch_size
        
        # Pre-compute all possible sample indices for efficient random sampling
        self._precompute_sample_indices()
    
    def __len__(self) -> int:
        """
        Return the number of samples per epoch.
        
        If samples_per_epoch is set, return that value.
        Otherwise, return the total number of possible samples.
        """
        if self.samples_per_epoch is not None:
            return min(self.samples_per_epoch, len(self.sample_indices))
        return len(self.sample_indices)
    
    def _precompute_sample_indices(self):
        """Pre-compute all possible (entity_id, target_idx) pairs for random sampling."""
        self.sample_indices = []
        
        for entity_id, sequence in self.entity_sequences.items():
            seq_len = len(sequence)
            if seq_len < 1:
                continue
            
            indices = self.split_indices[entity_id]
            
            # Determine target range for this split
            if self.split_name == 'train':
                target_start = 0
                target_end = indices['train_end']
            elif self.split_name == 'val':
                target_start = indices['train_end']
                target_end = indices['val_end']
            else:  # test
                target_start = indices['val_end']
                target_end = seq_len
            
            # Add all valid target indices
            for target_idx in range(target_start, target_end):
                available_input_end = target_idx
                
                start_idx = max(0, available_input_end - self.window_size)
                input_len = available_input_end - start_idx
                
                if input_len >= self.min_input_length:
                    self.sample_indices.append((entity_id, target_idx))
    
    def __iter__(self):
        """
        Generate random AR samples for this epoch.
        
        Returns:
            Iterator over sample dictionaries
        """
        # Set random seed for this epoch if provided
        if self.seed is not None:
            random.seed(self.seed + hash(str(self.sample_indices[:10])) % 1000000)
        
        # Randomly sample indices
        if self.samples_per_epoch is not None and self.samples_per_epoch < len(self.sample_indices):
            sampled_indices = random.sample(self.sample_indices, self.samples_per_epoch)
        else:
            sampled_indices = self.sample_indices.copy()
            random.shuffle(sampled_indices)
        
        # Generate all samples first (for augmentation)
        samples_with_metadata = []
        for entity_id, target_idx in sampled_indices:
            sample, input_seq, target = self._create_sample_with_metadata(entity_id, target_idx)
            if sample is not None:
                samples_with_metadata.append((sample, input_seq, target))
        
        # Apply batch augmentation if needed
        if self.retrieval_manager is not None and self.top_k > 0:
            samples = _batch_augment_sequences_with_retrieval(
                samples_with_metadata,
                self.retrieval_manager,
                self.top_k,
                self.window_size,
                batch_size=self.retrieval_batch_size,
            )
        else:
            samples = [sample for sample, _, _ in samples_with_metadata]

        def _to_tensor(x, dtype):
            if isinstance(x, torch.Tensor):
                return x.detach().clone().to(dtype)
            return torch.tensor(x, dtype=dtype)

        # Yield samples as tensors
        for sample in samples:
            yield {
                'entity_id': _to_tensor(sample['entity_id'], torch.long),
                'input_timestamps': _to_tensor(sample['input_timestamps'], torch.float32),
                'input_embeddings': _to_tensor(sample['input_embeddings'], torch.float32),
                'input_labels': _to_tensor(sample['input_labels'], torch.float32),
                'input_entity_embeddings': _to_tensor(sample['input_entity_embeddings'], torch.float32),
                'input_mask': _to_tensor(sample['input_mask'], torch.bool),
                'target_timestamp': _to_tensor(sample['target_timestamp'], torch.float32),
                'target_embedding': _to_tensor(sample['target_embedding'], torch.float32),
                'target_label': _to_tensor(sample['target_label'], torch.float32),
                'target_entity_embedding': _to_tensor(sample['target_entity_embedding'], torch.float32),
            }
    
    def _create_sample(self, entity_id: int, target_idx: int) -> Optional[Dict]:
        """
        Create a single AR sample for given entity and target index.
        
        NOTE: This method is kept for backward compatibility but is less efficient
        when using retrieval. Use _create_sample_with_metadata + batch augmentation instead.
        """
        sequence = self.entity_sequences[entity_id]
        seq_len = len(sequence)
        
        if seq_len < 1:
            return None
        
        embed_dim = sequence[0][1].shape[0]
        indices = self.split_indices[entity_id]
        
        # Determine input range
        available_input_end = target_idx
        
        # Extract input sequence
        start_idx = max(0, available_input_end - self.window_size)
        input_seq = sequence[start_idx:available_input_end]
        input_len = len(input_seq)
        
        # Target
        target = sequence[target_idx]

        # Augment short sequences using retrieval (single-sample, inefficient)
        if input_len < self.window_size and self.retrieval_manager is not None:
            input_seq = _augment_sequence_with_retrieval(
                input_seq=input_seq,
                target=target,
                retrieval_manager=self.retrieval_manager,
                top_k=self.top_k,
                window_size=self.window_size,
            )
            input_len = len(input_seq)

        if input_len < self.min_input_length:
            return None
        
        # Get entity embedding dimension
        entity_embed_dim = sequence[0][3].shape[0]
        
        # Create padded arrays
        input_timestamps = np.zeros(self.window_size, dtype=np.float32)
        input_embeddings = np.zeros((self.window_size, embed_dim), dtype=np.float32)
        input_labels = np.zeros(self.window_size, dtype=np.float32)
        input_entity_embeddings = np.zeros((self.window_size, entity_embed_dim), dtype=np.float32)
        input_mask = np.zeros(self.window_size, dtype=bool)
        
        # Fill with actual data (left-aligned)
        if input_len > 0:
            ts, embs, lbls, ent_embs = zip(*input_seq)
            input_timestamps[:input_len] = ts
            input_embeddings[:input_len] = np.stack(embs)
            input_labels[:input_len] = lbls
            input_entity_embeddings[:input_len] = np.stack(ent_embs)
            input_mask[:input_len] = True
        
        return {
            'entity_id': entity_id,
            'input_timestamps': input_timestamps,
            'input_embeddings': input_embeddings,
            'input_labels': input_labels,
            'input_entity_embeddings': input_entity_embeddings,
            'input_mask': input_mask,
            'target_timestamp': target[0],
            'target_embedding': target[1],
            'target_label': target[2],
            'target_entity_embedding': target[3],
        }
    
    def _create_sample_with_metadata(self, entity_id: int, target_idx: int) -> Tuple[Optional[Dict], List[Tuple[float, np.ndarray, float, np.ndarray]], Tuple[float, np.ndarray, float, np.ndarray]]:
        """
        Create a single AR sample with metadata for batch augmentation.
        
        Returns:
            Tuple of (sample_dict, input_seq, target) or (None, [], ())
        """
        sequence = self.entity_sequences[entity_id]
        seq_len = len(sequence)
        
        if seq_len < 1:
            return None, [], ()
        
        embed_dim = sequence[0][1].shape[0]
        entity_embed_dim = sequence[0][3].shape[0]
        indices = self.split_indices[entity_id]
        
        # Determine input range
        available_input_end = target_idx
        
        # Extract input sequence (no augmentation yet)
        start_idx = max(0, available_input_end - self.window_size)
        input_seq = sequence[start_idx:available_input_end]
        input_len = len(input_seq)
        
        # Target
        target = sequence[target_idx]

        # Check minimum length before augmentation
        if input_len < self.min_input_length:
            return None, [], ()
        
        # Create padded arrays (without augmentation)
        input_timestamps = np.zeros(self.window_size, dtype=np.float32)
        input_embeddings = np.zeros((self.window_size, embed_dim), dtype=np.float32)
        input_labels = np.zeros(self.window_size, dtype=np.float32)
        input_entity_embeddings = np.zeros((self.window_size, entity_embed_dim), dtype=np.float32)
        input_mask = np.zeros(self.window_size, dtype=bool)
        
        # Fill with actual data (left-aligned)
        if input_len > 0:
            ts, embs, lbls, ent_embs = zip(*input_seq)
            input_timestamps[:input_len] = ts
            input_embeddings[:input_len] = np.stack(embs)
            input_labels[:input_len] = lbls
            input_entity_embeddings[:input_len] = np.stack(ent_embs)
            input_mask[:input_len] = True
        
        sample = {
            'entity_id': entity_id,
            'input_timestamps': input_timestamps,
            'input_embeddings': input_embeddings,
            'input_labels': input_labels,
            'input_entity_embeddings': input_entity_embeddings,
            'input_mask': input_mask,
            'target_timestamp': target[0],
            'target_embedding': target[1],
            'target_label': target[2],
            'target_entity_embedding': target[3],
        }
        
        return sample, input_seq, target


def create_ar_dataloaders(
    entity_sequences: Dict[str, Dict[int, List[Tuple[float, np.ndarray, float, np.ndarray]]]],
    split_indices: Dict[str, Dict[int, Dict[str, int]]],
    window_size: int = 10,
    batch_size: int = 32,
    num_workers: int = 0,
    min_input_length: int = 0,
    retrieval_manager=None,
    top_k: int = 0,
    retrieval_batch_size: int = 256,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for auto-regressive training.
    
    Args:
        entity_sequences: Dict[split -> Dict[entity_id -> cumulative sequence]]
        split_indices: Dict[split -> Dict[entity_id -> {'train_end': int, 'val_end': int}]]
        window_size: Maximum number of past timesteps to use as input
        batch_size: Batch size for training
        num_workers: Number of dataloader workers
        min_input_length: Minimum number of valid inputs required (default: 0)
                         Applied to all splits to exclude cold-start samples
        retrieval_manager: Optional retrieval manager for sequence augmentation
        top_k: Number of neighbors to retrieve for augmentation
        retrieval_batch_size: Batch size for retrieval operations (default: 256)
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
            retrieval_manager=retrieval_manager,
            top_k=top_k,
            retrieval_batch_size=retrieval_batch_size,
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


def create_random_ar_dataloaders(
    entity_sequences: Dict[str, Dict[int, List[Tuple[float, np.ndarray, float, np.ndarray]]]],
    split_indices: Dict[str, Dict[int, Dict[str, int]]],
    window_size: int = 10,
    batch_size: int = 32,
    num_workers: int = 0,
    min_input_length: int = 0,
    samples_per_epoch: Optional[int] = None,
    seed: Optional[int] = None,
    retrieval_manager=None,
    top_k: int = 0,
    retrieval_batch_size: int = 256,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for auto-regressive training with random sampling per epoch.
    
    Each epoch will generate different random samples from the entity sequences,
    providing data augmentation through random sampling.
    
    NOTE: This function uses batch retrieval for efficient augmentation.
    
    Args:
        entity_sequences: Dict[split -> Dict[entity_id -> cumulative sequence]]
        split_indices: Dict[split -> Dict[entity_id -> {'train_end': int, 'val_end': int}]]
        window_size: Maximum number of past timesteps to use as input
        batch_size: Batch size for training
        num_workers: Number of dataloader workers
        min_input_length: Minimum number of valid inputs required (default: 0)
        samples_per_epoch: Number of samples to generate per epoch. If None, generates all possible samples.
        seed: Random seed for reproducibility (None for random)
        retrieval_manager: Optional retrieval manager for sequence augmentation
        top_k: Number of neighbors to retrieve for augmentation
        retrieval_batch_size: Batch size for retrieval operations
    
    Returns:
        Dict with 'train', 'val', 'test' DataLoaders
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        # Create random AR dataset
        dataset = RandomARSampleDataset(
            entity_sequences[split],
            split_indices[split],
            split_name=split,
            window_size=window_size,
            min_input_length=min_input_length,
            samples_per_epoch=samples_per_epoch,
            seed=seed,
            retrieval_manager=retrieval_manager,
            top_k=top_k,
            retrieval_batch_size=retrieval_batch_size,
        )
        
        # Create dataloader (no shuffle needed for IterableDataset)
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # IterableDataset handles randomness internally
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    
    return dataloaders


def create_autoregressive_samples(
    entity_sequences: Dict[int, List[Tuple[float, np.ndarray, float, np.ndarray]]],
    split_indices: Dict[int, Dict[str, int]],
    split_name: str,
    window_size: int = 10,
    min_input_length: int = 0,
    retrieval_manager=None,
    top_k: int = 0,
    retrieval_batch_size: int = 256,
) -> List[Dict]:
    """
    Create auto-regressive training samples using sliding window with padding.
    
    For each entity's cumulative sequence, create samples where:
    - Input: up to `window_size` previous timesteps from entire history (all previous splits)
    - Target: current timestep from the current split only
    - If input length is shorter than window_size, pad with zeros and create mask
    - Samples with fewer than `min_input_length` valid inputs are excluded
    
    Args:
        entity_sequences: Dict[entity_id -> cumulative List[(timestamp, embedding, label, entity_embedding)]]
        split_indices: Dict[entity_id -> {'train_end': int, 'val_end': int}]
        split_name: Name of current split ('train', 'val', or 'test')
        window_size: Maximum number of past timesteps to use as input
        min_input_length: Minimum number of valid input timesteps required (default: 1)
        retrieval_manager: Optional retrieval manager for sequence augmentation
        top_k: Number of neighbors to retrieve for augmentation
        retrieval_batch_size: Batch size for retrieval operations
    Returns:
        List of samples, each containing:
            - entity_id: int
            - input_timestamps: (window_size,) - left-aligned, padded on right if necessary
            - input_embeddings: (window_size, embed_dim) - left-aligned, padded on right if necessary
            - input_labels: (window_size,) - left-aligned, padded on right if necessary
            - input_entity_embeddings: (window_size, entity_embed_dim) - left-aligned, padded on right if necessary
            - input_mask: (window_size,) - True for valid positions, False for padding
            - target_timestamp: float
            - target_embedding: (embed_dim,)
            - target_label: float
            - target_entity_embedding: (entity_embed_dim,)
    """
    # Phase 1: Create all samples without augmentation
    samples_with_metadata = []
    for entity_id, sequence in tqdm(
        entity_sequences.items(),
        desc=f"Creating AR samples ({split_name})",
        total=len(entity_sequences),
    ):
        seq_len = len(sequence)
        
        # Skip empty sequences
        if seq_len < 1:
            continue
        
        # Get embedding dimension from first event in sequence
        embed_dim = sequence[0][1].shape[0]
        entity_embed_dim = sequence[0][3].shape[0]
        
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
            
            # Target: position target_idx (guaranteed to be in current split)
            target = sequence[target_idx]

            # Skip samples with insufficient input length
            if input_len < min_input_length:
                continue
            
            # Create padded arrays
            input_timestamps = np.zeros(window_size, dtype=np.float32)
            input_embeddings = np.zeros((window_size, embed_dim), dtype=np.float32)
            input_labels = np.zeros(window_size, dtype=np.float32)
            input_entity_embeddings = np.zeros((window_size, entity_embed_dim), dtype=np.float32)
            input_mask = np.zeros(window_size, dtype=bool)
            
            # Fill with actual data (left-aligned: padding on the right)
            # This ensures position 0 always has valid data if input_len > 0
            if input_len > 0:
                # Unpack all at once (single pass through input_seq)
                ts, embs, lbls, ent_embs = zip(*input_seq)
                input_timestamps[:input_len] = ts
                input_embeddings[:input_len] = np.stack(embs)
                input_labels[:input_len] = lbls
                input_entity_embeddings[:input_len] = np.stack(ent_embs)
                input_mask[:input_len] = True
            
            # Extract target data
            target_timestamp = target[0]
            target_embedding = target[1]
            target_label = target[2]
            target_entity_embedding = target[3]
            
            sample = {
                'entity_id': entity_id,
                'input_timestamps': input_timestamps,
                'input_embeddings': input_embeddings,
                'input_labels': input_labels,
                'input_entity_embeddings': input_entity_embeddings,
                'input_mask': input_mask,
                'target_timestamp': target_timestamp,
                'target_embedding': target_embedding,
                'target_label': target_label,
                'target_entity_embedding': target_entity_embedding,
            }
            
            # Store with metadata for potential augmentation
            samples_with_metadata.append((sample, input_seq, target))

    # Phase 2: Batch augmentation if needed
    if retrieval_manager is not None and top_k > 0:
        samples = _batch_augment_sequences_with_retrieval(
            samples_with_metadata,
            retrieval_manager,
            top_k,
            window_size,
            batch_size=retrieval_batch_size,
        )
    else:
        samples = [sample for sample, _, _ in samples_with_metadata]
    
    return samples

