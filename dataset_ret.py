from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm

class RetrievalDataset(Dataset):
    def __init__(self, entity_sequences):
        self.samples = []
        
        print("[RetrievalDataset] Flattening data for retrieval training...")
        for entity_id, sequence in tqdm(entity_sequences.items(), desc="Preparing Data"):
            for item in sequence:

                ts, emb, label, entity_emb = item

                if not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb, dtype=torch.float32)
                if not isinstance(entity_emb, torch.Tensor):
                    entity_emb = torch.tensor(entity_emb, dtype=torch.float32)
                label = float(label)
                ts = float(ts)

                self.samples.append({
                    'embedding': emb,
                    'label': label,
                    'timestamp': ts,
                    'entity_embedding': entity_emb,
                    'entity_id': entity_id
                })
        print(f"[RetrievalDataset] Total unique samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def retrieval_collate_fn(batch_list):
    batch = {
        'embedding': torch.stack([b['embedding'] for b in batch_list]),
        'label': torch.tensor([b['label'] for b in batch_list]),
        'timestamp': torch.tensor([b['timestamp'] for b in batch_list]),
        'entity_embedding': torch.stack([b['entity_embedding'] for b in batch_list]),
    }
    return batch