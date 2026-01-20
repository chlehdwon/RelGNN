import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from retrieval_base import RelativeTimeEncoder, BaseRetrievalManager

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    Based on: https://arxiv.org/abs/2004.11362
    """
    def __init__(self, temperature=0.5):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, embed_dim] - Normalized embeddings
            labels: [batch_size] - Class labels for each sample
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Reshape labels to [batch_size, 1]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        
        # Mask: mask[i][j] = 1 if sample i and j have the same label (Positive Pair), else 0
        mask = torch.eq(labels, labels.T).float().to(device)

        # Normalize features (L2 norm) ensures dot product equals cosine similarity
        features = F.normalize(features, dim=1)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        # Numerical stability: subtract max per row
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast (the diagonal)
        # We don't want the model to learn that a sample is similar to itself (trivial)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute Log-Probability
        exp_logits = torch.exp(logits) * logits_mask
        # Denominator: Sum of exp(logits) for all other samples (negatives + positives)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # Final Loss: Scaled by temperature ratio
        loss = - mean_log_prob_pos
        loss = loss.mean()

        return loss


class Contextretrieval(nn.Module):
    """
    Encoder Network: (Entity_Context + Time) -> Retrieval Embedding
    This maps a specific entity state at a specific time to a vector space.
    """
    def __init__(self, input_dim, embed_dim=128, projection_dim=128):
        super(Contextretrieval, self).__init__()
        self.time_encoder = RelativeTimeEncoder(channels=16)
        
        # Main Encoder
        # Combines context features and time features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 16, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Projection Head (Used ONLY during SupCon Training)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim)
        )

    def forward(self, x, t):
        """
        Args:
            x: Context features [Batch, Input_Dim]
            t: Time features [Batch, 1]
        Returns:
            feat: Representation for retrieval [Batch, Embed_Dim]
            proj: Projection for SupCon loss [Batch, Proj_Dim]
        """
        t_emb = self.time_encoder(t)
        combined = torch.cat([x, t_emb], dim=1)
        
        feat = self.encoder(combined)
        proj = self.head(feat)
        return feat, proj


class RetrievalManager(BaseRetrievalManager):
    """
    Manages the lifecycle of the retrieval: Training, Indexing, and Searching.
    """
    def __init__(
        self,
        input_dim,
        device,
        lr=1e-3,
        embed_dim=128,
        use_random_retrieval: bool = False,
    ):
        model = Contextretrieval(input_dim, embed_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        super().__init__(model=model, device=device, optimizer=optimizer, use_random_retrieval=use_random_retrieval)
        self.criterion = SupConLoss().to(device)
        self.use_faiss = True

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        steps = 0
        
        for batch in tqdm(loader, desc="[retrieval] Training"):
            x = batch['embedding'].to(self.device)
            y = batch['label'].to(self.device).view(-1)
            t = batch['timestamp'].to(self.device).view(-1, 1)

            y = (y > 0.5).float()
            
            pos_indices = (y == 1).nonzero(as_tuple=True)[0]
            neg_indices = (y == 0).nonzero(as_tuple=True)[0]

            if len(pos_indices) < 2: continue

            # Balanced sampling: cap both classes to the same count
            num_per_class = min(len(pos_indices), len(neg_indices))
            if num_per_class < 2:
                continue
            
            # Sample positives if they are more than needed
            if len(pos_indices) > num_per_class:
                pos_perm = torch.randperm(len(pos_indices))[:num_per_class]
                sampled_pos_indices = pos_indices[pos_perm]
            else:
                sampled_pos_indices = pos_indices
            
            # Sample negatives if they are more than needed
            if len(neg_indices) > num_per_class:
                neg_perm = torch.randperm(len(neg_indices))[:num_per_class]
                sampled_neg_indices = neg_indices[neg_perm]
            else:
                sampled_neg_indices = neg_indices
            
            keep_indices = torch.cat([sampled_pos_indices, sampled_neg_indices], dim=0)
            
            x, t, y = x[keep_indices], t[keep_indices], y[keep_indices]

            # --- 3. Train Step ---
            _, proj = self.model(x, t)
            loss = self.criterion(proj, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
        return total_loss / (steps + 1e-6)

    def _encode_features(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        feat, _ = self.model(x, t)
        return feat

    def _prepare_query_time(self, query_batch, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(query_batch['target_timestamp'], dtype=x.dtype).view(-1, 1).to(self.device)