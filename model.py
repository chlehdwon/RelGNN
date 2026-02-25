import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from torch import Tensor
from torch_geometric.typing import NodeType
from torch_geometric.nn import PositionalEncoding


# ---------------------------------------------------------------------------
# Retriever objectives (RAG4DyG-style): contrastive loss with time decay + InfoNCE
# ---------------------------------------------------------------------------

def cltime_retriever_loss(
    anchors: Tensor,
    positives: Tensor,
    hard_negatives: Tensor,
    anchors_time: Tensor,
    positives_time: Tensor,
    negatives_time: Tensor,
    temperature: float = 0.07,
    decay_rate: float = 0.1,
) -> Tensor:
    """
    Contrastive loss with time decay for retriever training.
    Anchor should be close to positive (decayed by time diff) and far from negatives.

    Args:
        anchors: (B, C) query embeddings
        positives: (B, C) positive (same entity, earlier time) embeddings
        hard_negatives: (B, C) hard negative embeddings
        anchors_time, positives_time, negatives_time: (B,) timestamps (e.g. unix seconds)
        temperature: softmax temperature
        decay_rate: lambda_decay for exp(-decay_rate * |time_diff|)

    Returns:
        Scalar loss (cross-entropy over in-batch + hard negatives with time decay).
    """
    batch_size = anchors.size(0)
    all_embeddings = torch.cat([anchors, positives, hard_negatives], dim=0)
    similarity_matrix = F.cosine_similarity(
        all_embeddings.unsqueeze(1), all_embeddings.unsqueeze(0), dim=2
    )

    # Positive pairs: anchor-positive, apply time decay
    time_diff_pos = torch.abs(anchors_time.unsqueeze(1) - positives_time.unsqueeze(0))
    decay_factor_pos = torch.exp(-decay_rate * time_diff_pos).to(anchors.device)
    pos_similarities = (
        similarity_matrix[:batch_size, batch_size : 2 * batch_size] * decay_factor_pos
    )

    # In-batch negatives: anchor vs anchor (time decay, no self)
    time_diff_neg = torch.abs(anchors_time.unsqueeze(1) - anchors_time.unsqueeze(0))
    decay_factor_neg = torch.exp(-decay_rate * time_diff_neg).to(anchors.device)
    decay_factor_neg.fill_diagonal_(0)
    neg_similarities = similarity_matrix[:batch_size, :batch_size] * decay_factor_neg

    # Hard negatives: anchor vs hard_negatives
    time_diff_hn = torch.abs(anchors_time.unsqueeze(1) - negatives_time.unsqueeze(0))
    decay_factor_hn = torch.exp(-decay_rate * time_diff_hn).to(anchors.device)
    hard_neg_similarities = (
        similarity_matrix[:batch_size, 2 * batch_size :] * decay_factor_hn
    )

    logits = (
        torch.cat([pos_similarities, neg_similarities, hard_neg_similarities], dim=1)
        / temperature
    )
    labels = torch.arange(batch_size, device=anchors.device)
    return F.cross_entropy(logits, labels)


def mask_correlated_samples_retriever(batch_size: int, device: torch.device) -> Tensor:
    """Mask for InfoNCE: exclude self and positive pair (i, i+B) from negatives."""
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=torch.bool, device=device)
    mask.fill_diagonal_(0)
    i = torch.arange(batch_size, device=device)
    mask[i, batch_size + i] = False
    mask[batch_size + i, i] = False
    return mask


def info_nce_retriever_loss(
    z_i: Tensor,
    z_j: Tensor,
    temperature: float = 0.07,
    mask: Tensor = None,
) -> Tensor:
    """
    InfoNCE loss for augmented views (same context, two augmented embeddings).
    z_i, z_j: (B, C) normalized embeddings from two views.

    Args:
        z_i, z_j: (B, C) view embeddings
        temperature: softmax temperature
        mask: (2*B, 2*B) bool mask for negative sampling (optional, built if None)

    Returns:
        Scalar InfoNCE loss.
    """
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    sim = torch.mm(z, z.T) / temperature
    # sim[i, B+i] = sim(z_i[i], z_j[i]); symmetric so sim[B+i, i] == sim[i, B+i]
    sim_pos = torch.diag(sim, batch_size)  # (B,) positive pair similarity per sample
    positive_samples = sim_pos.repeat(2).reshape(2 * batch_size, 1)  # same value for query i and query B+i
    if mask is None:
        mask = mask_correlated_samples_retriever(batch_size, z.device)
    negative_samples = sim[mask].reshape(2 * batch_size, -1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
    logits = torch.cat([positive_samples, negative_samples], dim=1)
    return F.cross_entropy(logits, labels)


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
        entity_embed_dim: int = None,
        use_entity_embedding: bool = False,
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
        if not use_entity_embedding or entity_embed_dim is None:
            self.entity_proj = None
        elif entity_embed_dim == channels:
            self.entity_proj = nn.Identity()
        else:
            self.entity_proj = nn.Linear(entity_embed_dim, channels)

        
        # Input normalization after combining embeddings
        self.input_norm = nn.LayerNorm(channels)
        self.cls_emb = nn.Parameter(torch.zeros(1, 1, channels))
        self.sep_emb = nn.Parameter(torch.zeros(1, 1, channels))
        self.pos_encoder = PositionalEncoding(channels)
        
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
        nn.init.normal_(self.cls_emb, std=0.02)
        nn.init.normal_(self.sep_emb, std=0.02)
        self.pos_encoder.reset_parameters()
        # label_embedder doesn't have reset_parameters, will use default init
        # transformer will use default init
        if self.entity_proj is not None and hasattr(self.entity_proj, 'reset_parameters'):
            self.entity_proj.reset_parameters()
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
        
        context_batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        return self._forward_with_context(context_batch)
    
    def _encode_sequence(self, batch: Dict[str, Tensor]) -> Tensor:
        """Encode sequence and return transformer outputs."""
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
        
        # 3. Encode time and labels (fused in retrieval path to avoid duplicate encoder calls)
        seed_time = batch['target_timestamp']  # (batch,)
        use_retrieval = 'retrieved_cls_emb' in batch and 'retrieved_labels' in batch
        num_refs = 5

        # 3b. Add entity embeddings if provided
        if self.entity_proj is not None and 'input_entity_embeddings' in batch and 'target_entity_embedding' in batch:
            seq_entity_embeddings = torch.cat([
                batch['input_entity_embeddings'],
                batch['target_entity_embedding'].unsqueeze(1)
            ], dim=1)
            seq_embeddings = seq_embeddings + self.entity_proj(seq_entity_embeddings)

        if use_retrieval:
            # Fused temporal + label encoding: one call for [ref_timestamps; seq_timestamps] and [ref_labels; seq_labels]
            all_timestamps = torch.cat([
                batch['retrieved_timestamps'],
                seq_timestamps,
            ], dim=1)  # (B, 5 + window_size+1)
            all_time_emb = self.temporal_encoder(seed_time, all_timestamps)
            ref_time_emb = all_time_emb[:, :num_refs]
            time_emb = all_time_emb[:, num_refs:]
            ref_labels = batch['retrieved_labels'].long().clamp(0, 1)
            all_labels = torch.cat([ref_labels, seq_labels], dim=1)
            all_is_mask = torch.cat([
                torch.zeros(batch_size, num_refs, dtype=torch.bool, device=device),
                is_mask,
            ], dim=1)
            all_label_emb = self.label_embedder(all_labels, is_mask=all_is_mask)
            ref_label_emb = all_label_emb[:, :num_refs]
            label_emb = all_label_emb[:, num_refs:]
            ref_tokens = batch['retrieved_cls_emb'] + ref_label_emb + ref_time_emb
            retrieved_ref_mask = batch.get('retrieved_ref_mask')
            if retrieved_ref_mask is not None:
                ref_tokens = ref_tokens * retrieved_ref_mask.unsqueeze(-1).float()
            cls_emb = self.cls_emb.expand(batch_size, -1, -1)
            sep_emb = self.sep_emb.expand(batch_size, -1, -1)
            # [CLS] [ref_1] ... [ref_5] [SEP] [ctx1] ... [ctx_k] [target]
            seq_embeddings = torch.cat([
                cls_emb,
                ref_tokens,
                sep_emb,
                seq_embeddings,
            ], dim=1)
            # Single zero block for CLS+refs+SEP (1+5+1 positions) instead of three separate allocations
            zero_block = torch.zeros(batch_size, 1 + num_refs + 1, self.channels, device=device, dtype=time_emb.dtype)
            time_emb = torch.cat([zero_block, time_emb], dim=1)
            label_emb = torch.cat([zero_block, label_emb], dim=1)
            ref_valid = retrieved_ref_mask if retrieved_ref_mask is not None else torch.ones(batch_size, num_refs, dtype=torch.bool, device=device)
            seq_mask = torch.cat([
                torch.ones(batch_size, 1, dtype=torch.bool, device=device),
                ref_valid,
                torch.ones(batch_size, 1, dtype=torch.bool, device=device),
                batch['input_mask'],
                torch.ones(batch_size, 1, dtype=torch.bool, device=device),
            ], dim=1)
        else:
            # No retrieval: encode context only, then prepend [CLS]
            time_emb = self.temporal_encoder(seed_time, seq_timestamps)
            label_emb = self.label_embedder(seq_labels, is_mask=is_mask)
            cls_emb = self.cls_emb.expand(batch_size, -1, -1)
            seq_embeddings = torch.cat([cls_emb, seq_embeddings], dim=1)
            time_emb = torch.cat([torch.zeros_like(cls_emb), time_emb], dim=1)
            label_emb = torch.cat([torch.zeros_like(cls_emb), label_emb], dim=1)
            seq_mask = torch.cat([
                torch.ones(batch_size, 1, dtype=torch.bool, device=device),
                batch['input_mask'],
                torch.ones(batch_size, 1, dtype=torch.bool, device=device),
            ], dim=1)

        # 5. Combine: snapshot + time + label (additive)
        x = seq_embeddings + time_emb + label_emb

        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=device)
        pos_emb = self.pos_encoder(pos).unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_emb
        x = self.input_norm(x)

        padding_mask = ~seq_mask
        
        # # 6. Create causal mask: position i cannot attend to j when j > i
        # # Exception: position 0 (CLS) attends to all, so encode_cls gets full-sequence summary
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        causal_mask[0, :] = False  # CLS (position 0) can attend to all positions

        # 7. Apply transformer
        h = self.transformer(
            x,
            # mask=causal_mask,
            src_key_padding_mask=padding_mask
        )  # (batch, seq_len, embed_dim)
        
        return h

    def _forward_with_context(self, batch: Dict[str, Tensor]) -> Tensor:
        """Forward pass for samples with valid context."""
        h = self._encode_sequence(batch)
        
        # Predict from CLS token representation.
        # This aligns the prediction representation with encode_cls() outputs.
        cls_repr = h[:, 0, :]  # (batch, embed_dim)
        
        logits = self.classifier(cls_repr)  # (batch, num_classes)
        
        return logits

    def encode_cls(self, batch: Dict[str, Tensor]) -> Tensor:
        """Return CLS embedding for each sample."""
        h = self._encode_sequence(batch)
        return h[:, 0, :]


