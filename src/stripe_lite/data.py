import pandas as pd
import numpy as np
from typing import Tuple, Dict, Iterator

REQ_COLS = ["src","dst","label","timestamp","event_type"]

def load_events_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = set(REQ_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    # normalise
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["event_type"] = df["event_type"].str.lower().str.strip()
    if not set(df["event_type"].unique()).issubset({"add","remove","deleted","del","rm"} | {"rem","delete"}):
        pass
    df["event_type"] = df["event_type"].replace({"deleted":"remove","del":"remove","rm":"remove","rem":"remove","delete":"remove"})
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def index_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
    nodes = pd.Index(pd.unique(df[["src","dst"]].values.ravel()))
    rels  = pd.Index(df["label"].unique())
    node2idx = {n:i for i,n in enumerate(nodes)}
    rel2idx  = {r:i for i,r in enumerate(rels)}
    out = df.copy()
    out["u"] = out["src"].map(node2idx)
    out["v"] = out["dst"].map(node2idx)
    out["r"] = out["label"].map(rel2idx)
    out["is_add"] = (out["event_type"] == "add").astype(int)
    out["_ts"] = out["timestamp"].astype("int64") // 10**9  # secondes
    return out, node2idx, rel2idx

def temporal_split(df: pd.DataFrame, ratios=(0.7,0.15,0.15)):
    a,b,c = ratios
    assert abs(a+b+c - 1.0) < 1e-9
    n = len(df)
    n_train = int(a*n)
    n_val   = int(b*n)
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val   = df.iloc[n_train:n_train+n_val].reset_index(drop=True)
    df_test  = df.iloc[n_train+n_val:].reset_index(drop=True)
    return df_train, df_val, df_test

def iter_time_buckets(df: pd.DataFrame, bucket_seconds: int) -> Iterator[pd.DataFrame]:
    """Yield groups of events with same time-window order; preserves chronology."""
    if len(df)==0: return
    start = df["_ts"].iloc[0] // bucket_seconds * bucket_seconds
    end   = df["_ts"].iloc[-1]
    cur = start
    idx = 0
    while cur <= end:
        nxt = cur + bucket_seconds
        mask = (df["_ts"] >= cur) & (df["_ts"] < nxt)
        batch = df[mask]
        if len(batch):
            yield batch
        cur = nxt
        idx += 1
