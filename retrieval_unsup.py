import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from retrieval_base import RelativeTimeEncoder, BaseRetrievalManager

class InfoNCELoss(nn.Module):
    """
    Unsupervised Contrastive Loss (SimCSE Style).
    Maximizes agreement between two views of the same sample,
    while repelling other samples in the batch.
    """
    def __init__(self, temperature=0.07): # SimCSE usually uses smaller temp
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        Args:
            features: [batch_size, 2, embed_dim] 
                      dim 1 contains (view1, view2)
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Flatten: [2*batch, dim] -> [view1_0, view1_1..., view2_0, view2_1...]
        # But for easier implementation, let's look at view1 vs view2
        z1 = F.normalize(features[:, 0, :], dim=1)
        z2 = F.normalize(features[:, 1, :], dim=1)
        
        # Cosine Similarity Matrix: [batch, batch]
        # sim[i, j] = cos(z1[i], z2[j])
        # Diagonal (i=j) should be high (Positive), Off-diagonal should be low (Negative)
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature
        
        # Labels are simply the indices [0, 1, 2, ..., batch-1]
        labels = torch.arange(batch_size).long().to(device)
        
        # Cross Entropy Loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss

class ContextRetrieval(nn.Module):
    """
    Unsupervised Encoder with Dropout for SimCSE.
    """
    def __init__(self, input_dim, embed_dim=128):
        super(ContextRetrieval, self).__init__()
        self.time_encoder = RelativeTimeEncoder(channels=32) # Time dimension
        
        # [Core of SimCSE] Dropout is ESSENTIAL here.
        # It creates the "augmentation" (noise) needed for self-supervised learning.
        self.dropout_rate = 0.1
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 32, 256), # Input + Time
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate), # <--- Positive View Generator
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x, t):
        t_emb = self.time_encoder(t)
        
        # Combine Context + Entity (x) and Time (t)
        # This makes the embedding "Time-Aware"
        combined = torch.cat([x, t_emb], dim=1)
        
        # Encode
        feat = self.encoder(combined)
        return feat

class RetrievalManager(BaseRetrievalManager):
    def __init__(
        self,
        input_dim,
        device,
        lr=1e-4,
        embed_dim=128,
        use_random_retrieval=False,
        use_entity_embedding: bool = True,
        **kwargs
    ):
        model = ContextRetrieval(input_dim, embed_dim)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        super().__init__(
            model=model,
            device=device,
            optimizer=optimizer,
            use_random_retrieval=use_random_retrieval,
            use_entity_embedding=use_entity_embedding,
        )
        self.criterion = InfoNCELoss(temperature=0.07).to(device)

    def train_epoch(self, loader):
        """
        Unsupervised Training Loop (SimCSE)
        """
        self.model.train()
        total_loss = 0
        steps = 0
        
        for batch in tqdm(loader, desc="[retrieval] Training (SimCSE)"):
            context_emb = batch['embedding'].to(self.device)
            if self.use_entity_embedding and 'entity_embedding' in batch:
                entity_emb = batch['entity_embedding'].to(self.device)
                x = torch.cat([context_emb, entity_emb], dim=1)
            else:
                x = context_emb
            t = batch['timestamp'].to(self.device).view(-1, 1)
            
            # --- SimCSE Logic ---
            # Pass the SAME input twice.
            # Due to Dropout inside the model, z1 and z2 will be slightly different.
            # We want the model to map these two "noisy views" to the same point.
            
            z1 = self.model(x, t) # View 1
            z2 = self.model(x, t) # View 2
            
            # Stack for Loss: [Batch, 2, Dim]
            features = torch.stack([z1, z2], dim=1)
            
            loss = self.criterion(features)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
        return total_loss / (steps + 1e-6)

    def _encode_features(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x, t)

    def _prepare_query_time(self, query_batch, x: torch.Tensor) -> torch.Tensor:
        if isinstance(query_batch['target_timestamp'], torch.Tensor):
            return query_batch['target_timestamp'].view(-1, 1).to(self.device)
        return torch.tensor(query_batch['target_timestamp']).view(-1, 1).to(self.device)

    def _format_retrieved_item(self, item: dict) -> dict:
        item_with_dummy_label = item.copy()
        item_with_dummy_label['label'] = 0.0
        return item_with_dummy_label