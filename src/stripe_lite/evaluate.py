import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from .data import iter_time_buckets

def roc_auc_stream(df, model, device, bucket_seconds=60, neg_per_pos=3):
    """Proxy AUC: score pos vs négatifs échantillonnés à la volée."""
    model.eval()
    scores, labels = [], []
    h_mem = model.init_memory(device)
    nodes = np.unique(np.concatenate([df["u"].values, df["v"].values]))
    rels  = np.unique(df["r"].values)

    with torch.no_grad():
        for bucket in iter_time_buckets(df, bucket_seconds):
            # pos
            u = torch.tensor(bucket["u"].values, dtype=torch.long, device=device)
            v = torch.tensor(bucket["v"].values, dtype=torch.long, device=device)
            r = torch.tensor(bucket["r"].values, dtype=torch.long, device=device)
            a = torch.tensor(bucket["is_add"].values, dtype=torch.long, device=device)
            logits, _, h_mem = model.forward_bucket({"u":u,"v":v,"r":r,"is_add":a}, h_mem)
            scores.extend(torch.sigmoid(logits).detach().cpu().tolist())
            labels.extend([1]*len(bucket))

            # neg
            n_neg = neg_per_pos * len(bucket)
            if n_neg>0:
                u_n = torch.tensor(np.random.choice(nodes, size=n_neg), dtype=torch.long, device=device)
                v_n = torch.tensor(np.random.choice(nodes, size=n_neg), dtype=torch.long, device=device)
                r_n = torch.tensor(np.random.choice(rels,  size=n_neg), dtype=torch.long, device=device)
                a_n = torch.tensor(np.random.randint(0,2,size=n_neg), dtype=torch.long, device=device)
                logits_n, _, h_mem = model.forward_bucket({"u":u_n,"v":v_n,"r":r_n,"is_add":a_n}, h_mem)
                scores.extend(torch.sigmoid(logits_n).detach().cpu().tolist())
                labels.extend([0]*n_neg)

    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")

def calibrate_thresholds(df_val, model, device, bucket_seconds=60, quantile=0.98):
    """Calibre deux seuils séparés (add/remove) sur validation."""
    model.eval()
    h_mem = model.init_memory(device)
    add_scores, rem_scores = [], []
    with torch.no_grad():
        for bucket in iter_time_buckets(df_val, bucket_seconds):
            u = torch.tensor(bucket["u"].values, dtype=torch.long, device=device)
            v = torch.tensor(bucket["v"].values, dtype=torch.long, device=device)
            r = torch.tensor(bucket["r"].values, dtype=torch.long, device=device)
            a = torch.tensor(bucket["is_add"].values, dtype=torch.long, device=device)
            logits, _, h_mem = model.forward_bucket({"u":u,"v":v,"r":r,"is_add":a}, h_mem)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            mask_add = (bucket["is_add"].values == 1)
            add_scores.extend(probs[mask_add])
            rem_scores.extend(probs[~mask_add])

    tau_add = float(np.quantile(add_scores, quantile)) if len(add_scores) else 0.5
    tau_rem = float(np.quantile(rem_scores, quantile)) if len(rem_scores) else 0.5
    return tau_add, tau_rem

def evaluate_stream(df_test_aug, model, device, tau_add:float, tau_rem:float, bucket_seconds=60):
    """
    Si df_test_aug contient 'anomaly' (bool), on calcule AUROC, F1 avec seuils.
    """
    model.eval()
    scores, labels = [], []
    preds = []
    h_mem = model.init_memory(device)
    with torch.no_grad():
        for bucket in iter_time_buckets(df_test_aug, bucket_seconds):
            u = torch.tensor(bucket["u"].values, dtype=torch.long, device=device)
            v = torch.tensor(bucket["v"].values, dtype=torch.long, device=device)
            r = torch.tensor(bucket["r"].values, dtype=torch.long, device=device)
            a = torch.tensor(bucket["is_add"].values, dtype=torch.long, device=device)
            logits, _, h_mem = model.forward_bucket({"u":u,"v":v,"r":r,"is_add":a}, h_mem)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            for p, add_flag in zip(probs, bucket["is_add"].values):
                scores.append(p)
                thr = tau_add if add_flag==1 else tau_rem
                preds.append(1 if p < thr else 0)  # petit score => anomalie
            if "anomaly" in bucket.columns:
                labels.extend(bucket["anomaly"].astype(int).tolist())

    metrics = {}
    if labels:
        from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support
        try:
            metrics["auroc"] = float(roc_auc_score(labels, [1-s for s in scores]))  # 1-score => "anomaly score"
        except Exception:
            metrics["auroc"] = float("nan")
        metrics["f1"] = float(f1_score(labels, preds))
        p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        metrics.update({"precision":float(p), "recall":float(r)})
    return metrics, preds, scores
