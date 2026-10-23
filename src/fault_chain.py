"""Action-dependent Markov chain of DHT22 health, estimated from the log.
Usage: python fault_chain.py
Requires steps.parquet (rl_prep.py).
Outputs: fault_chain.csv (P^(a)(H' | H) with counts), fault_onset.csv (onset and recovery
probabilities by action with day-bootstrap intervals), fault_context.csv

Health of a 10-min step: H = 0 healthy (no failure code), 1 degraded (failure code in at most
half of the readings), 2 failed (more than half).
"""
import numpy as np
import pandas as pd

SEED, NBOOT, ALPHA = 2026, 2000, 0.5
NAMES = ["healthy", "degraded", "failed"]
g = pd.read_parquet("steps.parquet")
t = g.dropna(subset=["H1"]).copy(); t["H1"] = t["H1"].astype(int)


def matrices(df):
    N = np.zeros((2, 3, 3))
    np.add.at(N, (df["a"].to_numpy(int), df["H"].to_numpy(int), df["H1"].to_numpy(int)), 1)
    return N, (N + ALPHA) / (N + ALPHA).sum(2, keepdims=True)


N, P = matrices(t)
rows = []
for a in (0, 1):
    for i in range(3):
        for j in range(3):
            rows.append({"action": a, "from": NAMES[i], "to": NAMES[j], "count": int(N[a, i, j]), "p": P[a, i, j]})
pd.DataFrame(rows).to_csv("fault_chain.csv", index=False)


def quantities(df):
    """Onset within one step and within three steps (30 min) from a healthy step, and recovery
    from a failed step, by action."""
    out = {}
    for a in (0, 1):
        h = df[(df["H"] == 0) & (df["a"] == a)]
        out[f"onset1_a{a}"] = (h["H1"] > 0).mean() if len(h) else np.nan
        h3 = h.dropna(subset=["H2", "H3"])
        out[f"onset3_a{a}"] = ((h3[["H1", "H2", "H3"]] > 0).any(axis=1)).mean() if len(h3) else np.nan
        fl = df[(df["H"] == 2) & (df["a"] == a)]
        out[f"recover1_a{a}"] = (fl["H1"] == 0).mean() if len(fl) else np.nan
        out[f"n_healthy_a{a}"] = len(h); out[f"n_healthy3_a{a}"] = len(h3); out[f"n_failed_a{a}"] = len(fl)
    return out


est = quantities(t)
rng = np.random.default_rng(SEED)
days = t["day"].unique(); by_day = {d: t[t["day"] == d] for d in days}
boot = pd.DataFrame([quantities(pd.concat([by_day[d] for d in rng.choice(days, len(days), replace=True)]))
                     for _ in range(NBOOT)])
res = []
for q in ("onset1", "onset3", "recover1"):
    diff = boot[f"{q}_a1"] - boot[f"{q}_a0"]
    res.append({"quantity": q, "a0": est[f"{q}_a0"], "a1": est[f"{q}_a1"], "n_a0": est[f"n_{'failed' if q == 'recover1' else 'healthy3' if q == 'onset3' else 'healthy'}_a0"],
                "n_a1": est[f"n_{'failed' if q == 'recover1' else 'healthy3' if q == 'onset3' else 'healthy'}_a1"],
                "diff": est[f"{q}_a1"] - est[f"{q}_a0"], "diff_lo": diff.quantile(.025), "diff_hi": diff.quantile(.975),
                "p_diff_positive": (diff > 0).mean(), "replicates": int(diff.notna().sum())})
res = pd.DataFrame(res); res.to_csv("fault_onset.csv", index=False)

# Common-cause check: if an electrical glitch produced both a low soil reading (hence a decision)
# and a DHT22 failure, failures should also follow dry readings that carry no decision.
def onset_by_dry(df):
    h = df[(df["H"] == 0) & (df["a"] == 0)].dropna(subset=["H2", "H3"])
    on3 = (h[["H1", "H2", "H3"]] > 0).any(axis=1)
    on1 = h["H1"] > 0
    dry = h["f"] > 0
    return {"onset1_dry": on1[dry].mean(), "onset1_nodry": on1[~dry].mean(),
            "onset3_dry": on3[dry].mean(), "onset3_nodry": on3[~dry].mean(),
            "n_dry": int(dry.sum()), "n_nodry": int((~dry).sum())}


cc = onset_by_dry(t)
bcc = pd.DataFrame([onset_by_dry(pd.concat([by_day[d] for d in rng.choice(days, len(days), replace=True)]))
                    for _ in range(NBOOT)])
d3 = bcc["onset3_dry"] - bcc["onset3_nodry"]
cc.update({"diff3": cc["onset3_dry"] - cc["onset3_nodry"], "diff3_lo": d3.quantile(.025), "diff3_hi": d3.quantile(.975),
           "p_diff3_positive": (d3 > 0).mean()})
# Dry readings by health state: a failed air sensor coincides with low soil readings?
for h in range(3):
    z = g[g["H"] == h]
    cc[f"share_dry_reading_{NAMES[h]}"] = (z["f"] > 0).mean()
    cc[f"mean_dry_share_{NAMES[h]}"] = z["f"].mean()
pd.Series(cc).to_csv("fault_commoncause.csv", header=["value"])
print(pd.Series(cc).round(4).to_string())

ctx = pd.DataFrame([{"health": NAMES[h], "steps": int((g["H"] == h).sum()), "decision_steps": int(((g["H"] == h) & (g["a"] == 1)).sum()),
                     "dry_reading_steps": int(((g["H"] == h) & (g["f"] > 0)).sum())} for h in range(3)])
ctx.to_csv("fault_context.csv", index=False)
print(pd.DataFrame(rows).pivot_table(index=["action", "from"], columns="to", values="p").round(3))
print(res.round(4).to_string(index=False)); print(ctx.to_string(index=False))
