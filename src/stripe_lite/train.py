from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from .data import iter_time_buckets
from .utils import set_seed
from tqdm import tqdm

class BucketDataset(Dataset):
    def __init__(self, df, neg_per_pos=3, device="cpu"):
        self.df = df
        self.neg_per_pos = neg_per_pos
        self.device = device
        self.nodes = np.unique(np.concatenate([df["u"].values, df["v"].values]))
        self.rels  = np.unique(df["r"].values)

    def sample_negatives(self, batch):
        B = len(batch)
        negs = []
        for _ in range(self.neg_per_pos * B):
            u = int(np.random.choice(self.nodes))
            v = int(np.random.choice(self.nodes))
            if u==v: 
                continue
            r = int(np.random.choice(self.rels))
            is_add = int(np.random.rand() < 0.5)
            negs.append((u,v,r,is_add))
        return negs

    def make_tensor_batch(self, frame, with_labels=True):
        u = torch.tensor(frame["u"].values, dtype=torch.long, device=self.device)
        v = torch.tensor(frame["v"].values, dtype=torch.long, device=self.device)
        r = torch.tensor(frame["r"].values, dtype=torch.long, device=self.device)
        is_add = torch.tensor(frame["is_add"].values, dtype=torch.long, device=self.device)
        batch = {"u":u,"v":v,"r":r,"is_add":is_add}
        if with_labels:
            # positifs
            labels_pos = torch.ones(len(frame), dtype=torch.float32, device=self.device)
            # négatifs
            neg = self.sample_negatives(frame)
            if len(neg):
                u_n,v_n,r_n,a_n = zip(*neg)
                u_n = torch.tensor(u_n, dtype=torch.long, device=self.device)
                v_n = torch.tensor(v_n, dtype=torch.long, device=self.device)
                r_n = torch.tensor(r_n, dtype=torch.long, device=self.device)
                a_n = torch.tensor(a_n, dtype=torch.long, device=self.device)
                labels_neg = torch.zeros(len(neg), dtype=torch.float32, device=self.device)
                batch = {
                    "u": torch.cat([u, u_n]),
                    "v": torch.cat([v, v_n]),
                    "r": torch.cat([r, r_n]),
                    "is_add": torch.cat([is_add, a_n]),
                    "label": torch.cat([labels_pos, labels_neg]),
                }
            else:
                batch["label"] = labels_pos
        return batch

def train_loop(df_train, df_val, model, optimizer, device, bucket_seconds=60, epochs=10, patience=3):
    set_seed(42)
    best_auc = -1.0
    best_state = None
    patience_left = patience

    from .evaluate import roc_auc_stream

    for epoch in range(1, epochs+1):
        model.train()
        h_mem = model.init_memory(device)
        pbar = tqdm(iter_time_buckets(df_train, bucket_seconds), desc=f"train@{epoch}")
        total_loss = 0.0; steps = 0

        for bucket in pbar:
            ds = BucketDataset(bucket, device=device)
            batch = ds.make_tensor_batch(bucket, with_labels=True)
            logits, loss, h_mem = model.forward_bucket(batch, h_mem)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()); steps += 1
            if steps % 10 == 0:
                pbar.set_postfix(loss=f"{total_loss/steps:.4f}")

        with torch.no_grad():
            val_auc = roc_auc_stream(df_val, model, device, bucket_seconds=bucket_seconds)

        print(f"[epoch {epoch}] train_loss={total_loss/max(steps,1):.4f}  val_auc≈{val_auc:.4f}")

        # early stopping simple
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu() for k,v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc
