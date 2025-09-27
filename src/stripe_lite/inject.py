import numpy as np
import pandas as pd

def inject_synthetic_anomalies(
    df_test: pd.DataFrame,
    num_anoms: int | None = None,
    p_modes = {"dup_add":0.5, "impossible_remove":0.5, "rel_flip":0.0},
    seed: int = 123
) -> pd.DataFrame:
    """
    Injecte des anomalies plausibles:
      - dup_add: on ADD un (u,v,r) déjà actif (doublon)
      - impossible_remove: on REMOVE un (u,v,r) non actif
      - rel_flip: on ADD (u,v) avec une relation différente r'
    Hypothèse: df_test est chronologique et indexé (u,v,r,is_add,_ts).
    """
    rng = np.random.default_rng(seed)
    df = df_test.sort_values("_ts").reset_index(drop=True).copy()
    df["anomaly"] = False

    if num_anoms is None:
        num_anoms = max(1, int(0.02 * len(df)))  # 2%

    # État d'activité
    active = set()
    for _, row in df.iterrows():
        key = (int(row["u"]), int(row["v"]), int(row["r"]))
        if row["is_add"] == 1:
            active.add(key)
        else:
            active.discard(key)

    nodes = np.unique(np.concatenate([df["u"].values, df["v"].values]))
    rels  = np.unique(df["r"].values)
    synthetic = []

    pm = p_modes
    choices = ["dup_add","impossible_remove","rel_flip"]
    probs = [pm.get("dup_add",0), pm.get("impossible_remove",0), pm.get("rel_flip",0)]
    probs = np.array(probs, dtype=float)
    probs = probs / probs.sum() if probs.sum()>0 else np.array([1,0,0])

    ts_min, ts_max = df["_ts"].min(), df["_ts"].max()

    for _ in range(num_anoms):
        mode = rng.choice(choices, p=probs)
        t = int(rng.integers(ts_min, ts_max+1))
        if mode == "dup_add" and active:
            u,v,r = list(active)[rng.integers(0, len(active))]
            is_add = 1
        elif mode == "impossible_remove":
            # tire un triple aléatoire qui *n'est pas* actif
            for _try in range(100):
                u = int(rng.choice(nodes))
                v = int(rng.choice(nodes))
                if u==v: 
                    continue
                r = int(rng.choice(rels))
                if (u,v,r) not in active:
                    break
            is_add = 0
        else:  # rel_flip
            # recycle une arête existante mais change r'
            base = df.iloc[int(rng.integers(0,len(df)))]
            u, v = int(base["u"]), int(base["v"])
            r_candidates = [x for x in rels if x != int(base["r"])]
            r = int(rng.choice(r_candidates)) if r_candidates else int(base["r"])
            is_add = 1

        synthetic.append({"u":u,"v":v,"r":r,"_ts":t,"is_add":is_add,"anomaly":True})

    df_syn = pd.DataFrame(synthetic)
    df_aug = pd.concat([df, df_syn], ignore_index=True).sort_values("_ts").reset_index(drop=True)
    return df_aug
