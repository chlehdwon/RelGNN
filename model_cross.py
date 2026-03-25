import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional

from relbench.base import TaskType
from model import RelTS_Model


class RelTS_Cross_Model(nn.Module):
    """
    Keep base sequence path unchanged: [CLS] [ctx1] ... [target].
    Merge retrieved memory tokens into CLS through cross-attention.
    Retrieved memory can be one token per reference or a flattened set of recent context tokens.
    """

    def __init__(
        self,
        channels: int,
        task_type: Optional[TaskType] = None,
        entity_embed_dim: int = None,
        use_entity_embedding: bool = False,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        num_classes: int = 1,
        cross_heads: Optional[int] = None,
        cross_dropout: float = 0.1,
        use_ref_time_label: bool = True,
        freeze_backbone: bool = True,
        train_base_classifier: bool = False,
    ):
        super().__init__()
        self.base_model = RelTS_Model(
            channels=channels,
            task_type=task_type,
            entity_embed_dim=entity_embed_dim,
            use_entity_embedding=use_entity_embedding,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            num_classes=num_classes,
        )

        heads = int(cross_heads) if cross_heads is not None else int(num_heads)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=cross_dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(channels)
        self.cross_ff = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(cross_dropout),
            nn.Linear(channels, channels),
        )
        self.cross_ff_norm = nn.LayerNorm(channels)
        self.gate_proj = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Linear(channels, 1),
        )
        nn.init.constant_(self.gate_proj[-1].bias, -2.0)

        self.use_ref_time_label = bool(use_ref_time_label)

        if freeze_backbone:
            self.freeze_backbone(train_base_classifier=train_base_classifier)

    def freeze_backbone(self, train_base_classifier: bool = False):
        for p in self.base_model.parameters():
            p.requires_grad = False
        if train_base_classifier:
            for p in self.base_model.classifier.parameters():
                p.requires_grad = True

    def load_base_state_dict(self, state_dict: Dict[str, Tensor]):
        self.base_model.load_state_dict(state_dict, strict=True)

    def _build_context_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        context_batch = {
            k: v
            for k, v in batch.items()
            if isinstance(v, torch.Tensor) and not k.startswith("retrieved_")
        }
        return context_batch

    def _get_retrieved_memory(self, batch: Dict[str, Tensor]) -> Optional[Tensor]:
        if "retrieved_ref_tokens" in batch:
            return batch["retrieved_ref_tokens"]
        return batch.get("retrieved_cls_emb")

    def _build_ref_tokens(self, batch: Dict[str, Tensor]) -> Tensor:
        ref_tokens = self._get_retrieved_memory(batch)
        if ref_tokens is None:
            raise ValueError("Retrieved memory tokens are required for cross-attention.")

        if self.use_ref_time_label and (
            "retrieved_labels" in batch and "retrieved_timestamps" in batch
        ):
            bsz, k = batch["retrieved_labels"].shape
            ref_mask_for_label = ~batch["retrieved_ref_mask"] if "retrieved_ref_mask" in batch else torch.zeros(
                bsz, k, dtype=torch.bool, device=ref_tokens.device
            )
            ref_time_emb = self.base_model.temporal_encoder(
                batch["target_timestamp"],
                batch["retrieved_timestamps"],
            )
            ref_label_emb = self.base_model.label_embedder(
                batch["retrieved_labels"].float(),
                is_mask=ref_mask_for_label,
            )
            ref_tokens = ref_tokens + ref_time_emb + ref_label_emb

        if "retrieved_ref_mask" in batch:
            ref_tokens = ref_tokens * batch["retrieved_ref_mask"].unsqueeze(-1).to(ref_tokens.dtype)
        return ref_tokens

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        context_batch = self._build_context_batch(batch)
        h = self.base_model._encode_sequence(context_batch)
        cls_repr = h[:, 0, :]

        retrieved_memory = self._get_retrieved_memory(batch)
        if retrieved_memory is None:
            return self.base_model.classifier(cls_repr)

        ref_mask = batch.get("retrieved_ref_mask")
        if ref_mask is None:
            ref_mask = torch.ones(
                retrieved_memory.shape[:2],
                dtype=torch.bool,
                device=retrieved_memory.device,
            )
        valid_any = ref_mask.any(dim=1)
        if not bool(valid_any.any().item()):
            return self.base_model.classifier(cls_repr)

        fused_cls = cls_repr.clone()
        ref_tokens = self._build_ref_tokens(batch)

        idx = torch.nonzero(valid_any, as_tuple=False).view(-1)
        q = cls_repr[idx].unsqueeze(1)
        mem = ref_tokens[idx]
        kpm = ~ref_mask[idx]

        cross_out, _ = self.cross_attn(
            q,
            mem,
            mem,
            key_padding_mask=kpm,
            need_weights=False,
        )
        cross_out = cross_out.squeeze(1)
        cross_out = self.cross_norm(cls_repr[idx] + cross_out)
        cross_out = self.cross_ff_norm(cross_out + self.cross_ff(cross_out))

        gate = torch.sigmoid(self.gate_proj(torch.cat([cls_repr[idx], cross_out], dim=-1)))
        fused_cls[idx] = cls_repr[idx] + gate * cross_out

        return self.base_model.classifier(fused_cls)

    def encode_cls(self, batch: Dict[str, Tensor]) -> Tensor:
        context_batch = self._build_context_batch(batch)
        h = self.base_model._encode_sequence(context_batch)
        return h[:, 0, :]
