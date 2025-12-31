import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Any
from torch import Tensor
from torch_geometric.typing import NodeType
from torch_geometric.nn import PositionalEncoding


class HeteroTemporalEncoder(torch.nn.Module):
    def __init__(self, node_types: List[NodeType], channels: int):
        super().__init__()

        self.encoder = PositionalEncoding(channels)
        self.lin = torch.nn.Linear(channels, channels)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.lin.reset_parameters()

    def forward(
        self,
        seed_time: Tensor,
        time: Tensor,
    ) -> Tensor:
        """
        Args:
            seed_time: (batch, seq_len) or (batch,) - reference timestamps
            time: (batch, seq_len) - timestamps to encode
        Returns:
            (batch, seq_len, channels) - encoded time features
        """
        # Ensure seed_time and time have same shape
        if seed_time.dim() == 1:
            seed_time = seed_time.unsqueeze(1).expand_as(time)
        
        batch_size, seq_len = time.shape
        
        rel_time = seed_time - time
        rel_time = rel_time / (60 * 60 * 24)  # Convert seconds to days.
        
        # PositionalEncoding expects 1D tensor (flattened)
        # Input: (batch * seq_len,) -> Output: (batch * seq_len, channels)
        rel_time_flat = rel_time.view(-1)  # (batch * seq_len,)
        encoded = self.encoder(rel_time_flat)  # (batch * seq_len, channels)
        
        # Reshape back to (batch, seq_len, channels)
        encoded = encoded.view(batch_size, seq_len, -1)
        
        return self.lin(encoded)


class LabelEmbedder(nn.Module):
    """
    Embed binary classification labels (0, 1) or mask token.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        # Embeddings for label 0, 1, and mask
        self.label_emb = nn.Embedding(2, embed_dim)  # 0, 1
        self.mask_emb = nn.Parameter(torch.randn(embed_dim))
        
    def forward(self, labels: Tensor, is_mask: Tensor = None) -> Tensor:
        """
        Args:
            labels: (batch, seq_len) - 0 or 1 for classification
            is_mask: (batch, seq_len) - True where label should be masked
        Returns:
            (batch, seq_len, embed_dim)
        """
        batch_size, seq_len = labels.shape
        device = labels.device
        
        # Get label embeddings
        emb = self.label_emb(labels.long())
        
        # Replace with mask embedding where needed
        if is_mask is not None:
            mask_emb_expanded = self.mask_emb.view(1, 1, -1).expand(batch_size, seq_len, -1)
            emb = torch.where(is_mask.unsqueeze(-1), mask_emb_expanded, emb)
        
        return emb


class RelTS_Model(nn.Module):
    """
    Relational Time Series Model for auto-regressive prediction.
    
    Combines snapshot embeddings, time encoding, and label embeddings,
    then uses a Transformer to predict the target label.
    
    Args:
        channels: Embedding dimension (from snapshot embeddings)
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        ff_dim: Feedforward dimension
        dropout: Dropout rate
        num_classes: Number of output classes (1 for binary, >1 for multiclass)
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        num_classes: int = 1,
    ):
        super().__init__()
        
        self.channels = channels
        self.num_classes = num_classes
        
        # Time and label encoders
        # Use HeteroTemporalEncoder (node_types not used in our case)
        self.temporal_encoder = HeteroTemporalEncoder(node_types=[], channels=channels)
        self.label_embedder = LabelEmbedder(channels)
        
        # Input normalization after combining embeddings
        self.input_norm = nn.LayerNorm(channels)
        
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                activation='gelu',
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                layer_norm_eps=1e-6,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(channels),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        self.temporal_encoder.reset_parameters()
        self.input_norm.reset_parameters()
        # label_embedder doesn't have reset_parameters, will use default init
        # transformer will use default init
        for module in self.classifier:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        
    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Forward pass for auto-regressive prediction.
        
        Args:
            batch: Dictionary containing:
                - input_embeddings: (batch, window_size, embed_dim) - history embeddings
                - input_timestamps: (batch, window_size) - history timestamps
                - input_labels: (batch, window_size) - history labels
                - input_mask: (batch, window_size) - True for valid, False for padding
                - target_embedding: (batch, embed_dim) - current snapshot embedding
                - target_timestamp: (batch,) - current timestamp
                - target_label: (batch,) - ground truth (not used in forward)
        
        Returns:
            logits: (batch, num_classes) - predictions for target label
        """
        batch_size = batch['input_embeddings'].size(0)
        device = batch['input_embeddings'].device
        
        # Check for cold-start samples (no valid history)
        has_context = batch['input_mask'].any(dim=1)  # (batch,) - True if any valid input
        
        # Handle all cases with unified logic
        logits = torch.zeros(batch_size, self.num_classes, device=device)
        
        if (~has_context).any():
            cold_start_logits = self.classifier(batch['target_embedding'][~has_context])
            logits[~has_context] = cold_start_logits
        
        # For samples with context: use full transformer
        if has_context.any():
            context_batch = {k: v[has_context] for k, v in batch.items() if isinstance(v, torch.Tensor)}
            context_logits = self._forward_with_context(context_batch)
            logits[has_context] = context_logits
        
        return logits
    
    def _forward_with_context(self, batch: Dict[str, Tensor]) -> Tensor:
        """Forward pass for samples with valid context."""
        batch_size = batch['input_embeddings'].size(0)
        device = batch['input_embeddings'].device
        
        # 1. Combine input sequence + target
        seq_embeddings = torch.cat([
            batch['input_embeddings'],
            batch['target_embedding'].unsqueeze(1)
        ], dim=1)  # (batch, window_size + 1, channels)
        
        seq_timestamps = torch.cat([
            batch['input_timestamps'],
            batch['target_timestamp'].unsqueeze(1)
        ], dim=1)  # (batch, window_size + 1)
        
        # For labels: historical labels + dummy for target (will be masked)
        seq_labels = torch.cat([
            batch['input_labels'],
            torch.zeros(batch_size, 1, device=device)  # Dummy, will be masked
        ], dim=1)  # (batch, window_size + 1)
        
        # 2. Create mask indicators
        # Historical: use actual validity, Target: mark as mask
        is_mask = torch.zeros(batch_size, seq_timestamps.size(1), dtype=torch.bool, device=device)
        is_mask[:, -1] = True  # Last position is target (mask label)
        
        # 3. Encode time and labels
        # Use target timestamp as seed_time (reference point)
        seed_time = batch['target_timestamp']  # (batch,)
        time_emb = self.temporal_encoder(seed_time, seq_timestamps)  # (batch, seq_len, embed_dim)
        label_emb = self.label_embedder(seq_labels, is_mask=is_mask)  # (batch, seq_len, embed_dim)
        
        # 4. Combine: snapshot + time + label (additive)
        x = seq_embeddings + time_emb + label_emb  # (batch, seq_len, embed_dim)
        
        # Apply input normalization after combining embeddings
        x = self.input_norm(x)
        
        # 5. Create padding mask for transformer
        # batch['input_mask']: True = valid data, False = padding
        seq_mask = torch.cat([
            batch['input_mask'],
            torch.ones(batch_size, 1, dtype=torch.bool, device=device)  # Target always valid
        ], dim=1)  # (batch, window_size + 1)
        
        # PyTorch transformer convention: True = ignore (padding), False = attend
        padding_mask = ~seq_mask  # (batch, window_size + 1)
        
        # 6. Create causal mask
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )  # Upper triangular = True (masked)
        
        # 7. Apply transformer
        h = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )  # (batch, seq_len, embed_dim)
        
        # 8. Predict from last position (target)
        # Note: Target position can see all history but not its own label (masked)
        target_repr = h[:, -1, :]  # (batch, embed_dim)
        
        logits = self.classifier(target_repr)  # (batch, num_classes)
        
        return logits


class RelGNN_Head(nn.Module):
    """
    Baseline model: Predict directly from snapshot embedding only.
    No temporal encoding, no label encoding, no sequence modeling.
    Equivalent to RelGNN's self.head for ablation comparison.
    """
    def __init__(
        self,
        channels: int,
        num_classes: int = 1,
        dropout: float = 0.1,
        use_relgnn_head: bool = False,
    ):
        super().__init__()
        
        self.channels = channels
        self.num_classes = num_classes
        self.use_relgnn_head = use_relgnn_head
        
        if use_relgnn_head:
            # Simple linear head (exactly matching RelGNN's MLP with num_layers=1)
            # Structure: head.lins.0 (Linear layer)
            from torch_geometric.nn import MLP
            self.head = MLP(
                in_channels=channels,
                hidden_channels=None,
                out_channels=num_classes,
                num_layers=1,
                norm=None,
            )
        else:
            # More sophisticated head for standalone training
            self.head = nn.Sequential(
                nn.Linear(channels, channels),
                nn.LayerNorm(channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(channels, num_classes),
            )
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        if hasattr(self.head, 'reset_parameters'):
            self.head.reset_parameters()
        else:
            for module in self.head:
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
    
    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Forward pass using only target snapshot embedding.
        
        Args:
            batch: Dictionary containing:
                - target_embedding: (batch, channels)
                - (other fields ignored)
        
        Returns:
            logits: (batch, num_classes)
        """
        # Use only target snapshot embedding
        target_emb = batch['target_embedding']  # (batch, channels)
        
        # Direct prediction
        logits = self.head(target_emb)  # (batch, num_classes)
        
        return logits


