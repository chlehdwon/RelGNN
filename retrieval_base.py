import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import PositionalEncoding
import faiss


class RelativeTimeEncoder(torch.nn.Module):
    """
    Stable relative time encoder using sinusoidal positional encoding.
    """
    def __init__(self, channels: int, seconds_per_day: float = 60 * 60 * 24):
        super().__init__()
        self.encoder = PositionalEncoding(channels)
        self.lin = torch.nn.Linear(channels, channels)
        self.seconds_per_day = seconds_per_day

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, rel_time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rel_time: (batch, 1) or (batch, seq_len) or (batch,)
        Returns:
            (batch, channels) or (batch, seq_len, channels)
        """
        if rel_time.dim() == 1:
            rel_time_flat = rel_time
            rel_shape = rel_time.shape
        else:
            rel_shape = rel_time.shape
            rel_time_flat = rel_time.view(-1)

        rel_time_flat = rel_time_flat / self.seconds_per_day
        encoded = self.encoder(rel_time_flat)

        if len(rel_shape) == 1:
            return self.lin(encoded)

        encoded = encoded.view(*rel_shape, -1)
        if rel_shape[1] == 1:
            encoded = encoded.squeeze(1)
        return self.lin(encoded)


class BaseRetrievalManager:
    """
    Shared retrieval logic for indexing and querying.
    """
    def __init__(self, model, device, optimizer, use_random_retrieval: bool = False):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.use_random_retrieval = use_random_retrieval

        self.memory_bank = None
        self.faiss_index = None
        self.metadata = []
        self.faiss_resources = None

    def _encode_features(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _prepare_query_time(self, query_batch, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _format_retrieved_item(self, item: dict) -> dict:
        return item

    def build_index(self, loader):
        self.model.eval()
        embeddings = []
        self.metadata = []

        with torch.no_grad():
            for batch in loader:
                x = batch['embedding'].to(self.device)
                t = batch['timestamp'].to(self.device).view(-1, 1)

                feat = self._encode_features(x, t)
                feat = F.normalize(feat, dim=1)
                embeddings.append(feat.cpu())

                x_cpu = x.cpu()
                t_cpu = t.cpu()
                labels = batch.get('label', None)
                labels_cpu = labels.cpu() if labels is not None else None
                e_ids = batch['entity_id'] if 'entity_id' in batch else torch.zeros(len(x))

                for i in range(len(x)):
                    metadata_item = {
                        'embedding': x_cpu[i],
                        'timestamp': t_cpu[i].item(),
                        'entity_id': e_ids[i].item(),
                    }
                    if labels_cpu is not None:
                        metadata_item['label'] = labels_cpu[i].item()
                    self.metadata.append(metadata_item)

        self.memory_bank = torch.cat(embeddings, dim=0)

        emb_np = self.memory_bank.numpy().astype("float32", copy=False)
        self.faiss_index = faiss.IndexFlatIP(emb_np.shape[1])
        self.faiss_index.add(emb_np)
        if torch.cuda.is_available():
            try:
                if hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0:
                    self.faiss_resources = faiss.StandardGpuResources()
                    gpu_id = torch.cuda.current_device()
                    self.faiss_index = faiss.index_cpu_to_gpu(self.faiss_resources, gpu_id, self.faiss_index)
            except Exception as exc:
                print(f"[retrieval] GPU FAISS unavailable, using CPU. Reason: {exc}")

    def retrieve(self, query_batch, k=5, alpha=0.5):
        self.model.eval()
        x = query_batch['target_embedding']

        if self.memory_bank is None or len(self.metadata) == 0:
            raise ValueError("Memory bank is empty. Run build_index() first.")

        if self.use_random_retrieval:
            num_items = len(self.metadata)
            k_eff = min(k, num_items)
            if k_eff <= 0:
                return [[] for _ in range(x.shape[0])]
            retrieved_data = []
            for _ in range(x.shape[0]):
                replace = k_eff > num_items
                indices = np.random.choice(num_items, size=k_eff, replace=replace)
                batch_retrieved = [self._format_retrieved_item(self.metadata[int(idx)]) for idx in indices]
                retrieved_data.append(batch_retrieved)
            return retrieved_data

        t = self._prepare_query_time(query_batch, x)
        with torch.no_grad():
            q_feat = self._encode_features(x, t)
            q_feat = F.normalize(q_feat, dim=1)

            num_items = len(self.metadata)
            k_eff = min(k, num_items)
            if k_eff <= 0:
                return [[] for _ in range(x.shape[0])]

            if self.faiss_index is None:
                raise RuntimeError("FAISS index is missing. Run build_index() first.")

            q_np = q_feat.cpu().numpy().astype("float32", copy=False)
            context_scores, indices = self.faiss_index.search(q_np, k_eff)

            retrieved_data = []
            for batch_indices in indices:
                batch_retrieved = [self._format_retrieved_item(self.metadata[int(idx)]) for idx in batch_indices]
                retrieved_data.append(batch_retrieved)
            return retrieved_data
