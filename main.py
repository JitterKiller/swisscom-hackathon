import argparse, os, torch
from stripe_lite import (
    load_events_df, index_dataframe, temporal_split,
    inject_synthetic_anomalies,
    StripeLiteModel, train_loop,
    calibrate_thresholds, evaluate_stream
)
from stripe_lite.utils import save_checkpoint, load_checkpoint

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def cmd_train(args):
    df = load_events_df(args.csv)
    df, node2idx, rel2idx = index_dataframe(df)
    df_train, df_val, df_test = temporal_split(df, (1-args.val_ratio-args.test_ratio, args.val_ratio, args.test_ratio))

    device = get_device()
    model = StripeLiteModel(num_nodes=len(node2idx), num_rels=len(rel2idx),
                            d_node=args.d_node, d_rel=args.d_rel, d_hid=args.d_hid).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model, best_auc = train_loop(df_train, df_val, model, optim, device,
                                 bucket_seconds=args.batch_seconds,
                                 epochs=args.epochs, patience=args.patience)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "best.pt")
    save_checkpoint(ckpt_path, {"state_dict":model.state_dict(),
                                "num_nodes":len(node2idx),"num_rels":len(rel2idx)})
    print(f"Saved: {ckpt_path} (val AUC≈{best_auc:.4f})")

def cmd_eval(args):
    df = load_events_df(args.csv)
    df, node2idx, rel2idx = index_dataframe(df)
    _, df_val, df_test = temporal_split(df, (1-args.val_ratio-args.test_ratio, args.val_ratio, args.test_ratio))

    device = get_device()
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model = StripeLiteModel(ckpt["num_nodes"], ckpt["num_rels"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    # Calibration sur val
    tau_add, tau_rem = calibrate_thresholds(df_val, model, device,
                                            bucket_seconds=args.batch_seconds, quantile=0.98)
    print(f"Calibrated thresholds (add/rem): {tau_add:.6f} {tau_rem:.6f}")

    # Injection anomalies sur test
    df_test_aug = inject_synthetic_anomalies(df_test, num_anoms=int(args.inject_pct*len(df_test)))
    metrics, preds, scores = evaluate_stream(df_test_aug, model, device, tau_add, tau_rem,
                                             bucket_seconds=args.batch_seconds)
    print("Metrics on test_aug:", metrics)

def cmd_score_one(args):
    import pandas as pd
    device = get_device()
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model = StripeLiteModel(ckpt["num_nodes"], ckpt["num_rels"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # charger df complet pour l'historique
    df = load_events_df(args.csv)
    df, _, _ = index_dataframe(df)
    # histoire strictement avant t
    t = pd.to_datetime(args.timestamp)
    df_hist = df[df["timestamp"] < t].copy()
    df_cand = {
        "u": int(args.u), "v": int(args.v), "r": int(args.r),
        "is_add": 1 if args.is_add.lower()=="add" else 0
    }

    # stream histoire pour mettre à jour la mémoire
    h_mem = model.init_memory(device)
    from stripe_lite.data import iter_time_buckets
    with torch.no_grad():
        for bucket in iter_time_buckets(df_hist, args.batch_seconds):
            u = torch.tensor(bucket["u"].values, dtype=torch.long, device=device)
            v = torch.tensor(bucket["v"].values, dtype=torch.long, device=device)
            r = torch.tensor(bucket["r"].values, dtype=torch.long, device=device)
            a = torch.tensor(bucket["is_add"].values, dtype=torch.long, device=device)
            _, _, h_mem = model.forward_bucket({"u":u,"v":v,"r":r,"is_add":a}, h_mem)

        # score du candidat
        u = torch.tensor([df_cand["u"]], dtype=torch.long, device=device)
        v = torch.tensor([df_cand["v"]], dtype=torch.long, device=device)
        r = torch.tensor([df_cand["r"]], dtype=torch.long, device=device)
        a = torch.tensor([df_cand["is_add"]], dtype=torch.long, device=device)
        logits, _, _ = model.forward_bucket({"u":u,"v":v,"r":r,"is_add":a}, h_mem)
        prob = torch.sigmoid(logits).item()
        print(f"Plausibility prob={prob:.4f} (higher=more normal)")

def build_argparser():
    p = argparse.ArgumentParser(description="STRIPE-Lite Edge Anomaly Detection")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--csv", required=True)
    pt.add_argument("--val_ratio", type=float, default=0.15)
    pt.add_argument("--test_ratio", type=float, default=0.15)
    pt.add_argument("--epochs", type=int, default=20)
    pt.add_argument("--patience", type=int, default=4)
    pt.add_argument("--batch_seconds", type=int, default=60)
    pt.add_argument("--neg_per_pos", type=int, default=3)
    pt.add_argument("--d_node", type=int, default=64)
    pt.add_argument("--d_rel", type=int, default=32)
    pt.add_argument("--d_hid", type=int, default=128)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--out_dir", default="artifacts")

    pe = sub.add_parser("eval")
    pe.add_argument("--csv", required=True)
    pe.add_argument("--checkpoint", required=True)
    pe.add_argument("--val_ratio", type=float, default=0.15)
    pe.add_argument("--test_ratio", type=float, default=0.15)
    pe.add_argument("--batch_seconds", type=int, default=60)
    pe.add_argument("--inject_pct", type=float, default=0.02)
    pe.add_argument("--tau_add", type=float, default=0.5)  # (sera recalibré)
    pe.add_argument("--tau_rem", type=float, default=0.5)

    ps = sub.add_parser("score-one")
    ps.add_argument("--csv", required=True)
    ps.add_argument("--checkpoint", required=True)
    ps.add_argument("--batch_seconds", type=int, default=60)
    ps.add_argument("--timestamp", required=True, help="ISO datetime of candidate event")
    ps.add_argument("--u", type=int, required=True)
    ps.add_argument("--v", type=int, required=True)
    ps.add_argument("--r", type=int, required=True)
    ps.add_argument("--is_add", choices=["add","remove"], required=True)

    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "score-one":
        cmd_score_one(args)
