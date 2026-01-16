import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    Based on: https://arxiv.org/abs/2004.11362
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

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
        
        # Compute similarity logits: (Batch, Batch)
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

        # Compute Mean Log-Likelihood over Positive Pairs
        # Denominator: Number of positives for each anchor (avoid division by zero)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # Final Loss: Scaled by temperature ratio
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


class ContextRetriever(nn.Module):
    """
    Encoder Network: (Entity_Context + Time) -> Retrieval Embedding
    This maps a specific entity state at a specific time to a vector space.
    """
    def __init__(self, input_dim, embed_dim=128, projection_dim=128):
        super(ContextRetriever, self).__init__()
        
        # Time Encoding (Scalar Time -> Vector)
        # Since 'timestamp' is usually a large float (Unix time), 
        # consider normalization before passing here or use sinusoidal encoding.
        # Here we use a simple MLP for demonstration.
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
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
        # Projects embedding to a space where contrastive loss is calculated.
        # This head is usually discarded during inference/retrieval.
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


class RetrievalManager:
    """
    Manages the lifecycle of the Retriever: Training, Indexing, and Searching.
    """
    def __init__(self, input_dim, device, embed_dim=128):
        self.device = device
        self.model = ContextRetriever(input_dim, embed_dim).to(device)
        self.criterion = SupConLoss().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        self.memory_bank = None  # Stores cached embeddings for retrieval
        self.metadata = []       # Stores metadata (label, time, raw_embedding) corresponding to the bank

    def train_epoch(self, loader):
        """
        Runs one epoch of Supervised Contrastive Pre-training.
        """
        self.model.train()
        total_loss = 0
        steps = 0
        
        for batch in tqdm(loader, desc="[Retriever] Pre-training"):
            # The loader provides window sequences: (Batch, Window, Dim).
            # For contrastive learning, we treat every single timestep as an independent sample.
            # So we flatten the batch: (Batch * Window, Dim).
            
            # 1. Flatten Input History
            # input_embeddings: (B, W, D) -> (B*W, D)
            hist_x = batch['input_embeddings'].to(self.device).view(-1, batch['input_embeddings'].shape[-1])
            hist_t = batch['input_timestamps'].to(self.device).view(-1, 1)
            hist_y = batch['input_labels'].to(self.device).view(-1)
            hist_mask = batch['input_mask'].to(self.device).view(-1)

            # 2. Include Target (Current State) in Training
            # The target state also has a label, so it's valuable for learning the pattern.
            curr_x = batch['target_embedding'].to(self.device)
            curr_t = batch['target_timestamp'].to(self.device).view(-1, 1)
            curr_y = batch['target_label'].to(self.device).view(-1)
            curr_mask = torch.ones_like(curr_y, dtype=torch.bool) # Target is always valid

            # 3. Concatenate History + Target
            x = torch.cat([hist_x, curr_x], dim=0)
            t = torch.cat([hist_t, curr_t], dim=0)
            y = torch.cat([hist_y, curr_y], dim=0)
            mask = torch.cat([hist_mask, curr_mask], dim=0)

            # 4. Filter Valid Samples only (Remove padding)
            valid_idx = mask.bool()
            # We need at least 2 samples to calculate contrastive loss
            if valid_idx.sum() < 2: continue 

            x, t, y = x[valid_idx], t[valid_idx], y[valid_idx]

            # Forward pass
            _, proj = self.model(x, t)
            
            # Compute Loss
            # This pushes samples with different 'y' apart and pulls samples with same 'y' together.
            loss = self.criterion(proj, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
        return total_loss / (steps + 1e-6)

    def build_index(self, loader):
        """
        Encodes all valid events in the training set and stores them in memory.
        This effectively creates the 'Knowledge Base' for RAG.
        """
        self.model.eval()
        embeddings = []
        self.metadata = []
        
        print("[Retriever] Building Index...")
        with torch.no_grad():
            for batch in tqdm(loader, desc="Indexing"):
                # We want to index: Historical events + Target events
                # Ensure that only TRAIN data is passed here to avoid data leakage.
                
                # Flatten Logic (Same as training)
                hist_x = batch['input_embeddings'].to(self.device).view(-1, batch['input_embeddings'].shape[-1])
                hist_t = batch['input_timestamps'].to(self.device).view(-1, 1)
                hist_y = batch['input_labels'].to(self.device).view(-1)
                hist_mask = batch['input_mask'].to(self.device).view(-1)
                
                curr_x = batch['target_embedding'].to(self.device)
                curr_t = batch['target_timestamp'].to(self.device).view(-1, 1)
                curr_y = batch['target_label'].to(self.device).view(-1)
                curr_mask = torch.ones_like(curr_y, dtype=torch.bool)

                x = torch.cat([hist_x, curr_x], dim=0)
                t = torch.cat([hist_t, curr_t], dim=0)
                y = torch.cat([hist_y, curr_y], dim=0)
                mask = torch.cat([hist_mask, curr_mask], dim=0)
                
                # Filter valid
                valid_idx = mask.bool()
                if valid_idx.sum() == 0: continue

                x, t, y = x[valid_idx], t[valid_idx], y[valid_idx]

                # Inference to get embeddings
                feat, _ = self.model(x, t)
                feat = F.normalize(feat, dim=1) # Normalize for Cosine Similarity search

                embeddings.append(feat.cpu())
                
                # Store Metadata 
                # We store the raw embedding/label/time to inject them back into the main model later.
                x_cpu = x.cpu()
                t_cpu = t.cpu()
                y_cpu = y.cpu()
                
                for i in range(len(y)):
                    self.metadata.append({
                        'embedding': x_cpu[i], # Raw RelGNN embedding (not the retriever's encoded vector)
                        'timestamp': t_cpu[i].item(),
                        'label': y_cpu[i].item()
                    })
        
        # Concatenate all embeddings into a single tensor (Memory Bank)
        self.memory_bank = torch.cat(embeddings, dim=0)
        print(f"[Retriever] Index Built. Total vectors: {self.memory_bank.shape[0]}")

    def retrieve(self, query_batch, k=5):
        """
        Retrieves top-k similar context for the given query batch.
        
        Args:
            query_batch: Dict containing 'target_embedding', 'target_timestamp' (Current state)
            k: Number of neighbors to retrieve
            
        Returns:
            retrieved_data: List of List of dicts (metadata for k neighbors per sample)
        """
        self.model.eval()
        x = query_batch['target_embedding']
        t = query_batch['target_timestamp'].view(-1, 1)
        
        with torch.no_grad():
            # Encode Query (Current state)
            q_feat, _ = self.model(x, t)
            q_feat = F.normalize(q_feat, dim=1)
            
            # Similarity Search via Matrix Multiplication
            # (Batch, Dim) @ (Memory, Dim).T -> (Batch, Memory)
            # Scores represent Cosine Similarity
            sim_scores = torch.mm(q_feat, self.memory_bank.to(self.device).T)
            
            # Get Top-K indices
            _, indices = torch.topk(sim_scores, k, dim=1)
            
            # Retrieve corresponding metadata
            retrieved_data = []
            for batch_idx in range(indices.shape[0]):
                batch_retrieved = []
                for neighbor_idx in indices[batch_idx]:
                    # Fetch metadata from RAM list using the index
                    batch_retrieved.append(self.metadata[neighbor_idx.item()])
                retrieved_data.append(batch_retrieved)
                
            return retrieved_data