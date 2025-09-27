import torch
import torch.nn as nn
import torch.nn.functional as F

class StripeLiteModel(nn.Module):
    """
    Modèle "lite" inspiré STRIPE:
      - embeddings statiques (noeuds & relations)
      - mémoire dynamique via GRU sur sommation de messages par bucket temporel
      - score de plausibilité bilinéaire avec tête add/remove
    """
    def __init__(self, num_nodes, num_rels, d_node=64, d_rel=32, d_hid=128):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_rels  = num_rels
        self.node_emb  = nn.Embedding(num_nodes, d_node)
        self.rel_emb   = nn.Embedding(num_rels, d_rel)
        self.proj_u    = nn.Linear(d_node, d_hid, bias=False)
        self.proj_v    = nn.Linear(d_node, d_hid, bias=False)
        self.proj_r    = nn.Linear(d_rel,  d_hid, bias=False)

        # Mémoire par noeud (agrégée)
        self.gru = nn.GRU(d_hid, d_hid, batch_first=True)

        # Deux têtes: add/remove
        self.head_add = nn.Linear(d_hid, 1)
        self.head_rem = nn.Linear(d_hid, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        for m in [self.proj_u, self.proj_v, self.proj_r, self.head_add, self.head_rem]:
            if hasattr(m, "weight"):
                nn.init.xavier_uniform_(m.weight)

    def init_memory(self, device):
        # h0 pour GRU (1 layer, batch=1)
        return torch.zeros(1, 1, self.gru.hidden_size, device=device)

    def encode_triplet(self, u_idx, v_idx, r_idx):
        u = self.proj_u(self.node_emb(u_idx))
        v = self.proj_v(self.node_emb(v_idx))
        r = self.proj_r(self.rel_emb(r_idx))
        h = F.relu(u + v + r)
        return h  # (B, d_hid)

    def update_memory(self, h_mem, h_batch):
        # h_batch: (B, d_hid) -> GRU attend (B, T=1, d_hid)
        inp = h_batch.unsqueeze(1)
        out, h_new = self.gru(inp, h_mem)
        return h_new  # (1,1,d_hid)

    def plaus_logit(self, u, v, r, is_add):
        """
        u,v,r: (B,)
        is_add: bool tensor (B,)
        """
        h = self.encode_triplet(u,v,r)  # (B, d_hid)
        logit_add = self.head_add(h).squeeze(-1)
        logit_rem = self.head_rem(h).squeeze(-1)
        return torch.where(is_add, logit_add, logit_rem)

    def forward_bucket(self, batch, h_mem):
        """
        batch: dict avec tensors u,v,r,is_add
        Retourne: loss (si labels fournis), h_mem_new
        """
        u, v = batch["u"], batch["v"]
        r = batch["r"]
        is_add = batch["is_add"].bool()
        logits = self.plaus_logit(u,v,r,is_add)
        labels = batch.get("label", None)
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        # mise à jour mémoire avec l'agrégation du bucket
        h = self.encode_triplet(u,v,r).mean(dim=0, keepdim=True)  # (1,d_hid)
        h_mem_new = self.update_memory(h_mem, h)  # (1,1,d_hid)
        return logits, loss, h_mem_new
