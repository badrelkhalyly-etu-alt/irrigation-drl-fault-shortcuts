"""Diagnostics that decide what an MDP built from the log can support.
Usage: python diagnostics.py
Requires steps.parquet and transitions.parquet (rl_prep.py).
Outputs: diag_behaviour.csv, diag_coverage.csv, diag_effect.csv, diag_markov.csv
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from common_rl import NS, BANDS, state_label, M_EDGES

SEED, NBOOT = 2026, 2000
g = pd.read_parquet("steps.parquet")
t = pd.read_parquet("transitions.parquet")
g["band"] = np.digitize(g["m"], M_EDGES); t["band"] = np.digitize(t["m"], M_EDGES)

# 1. Behaviour of the logged controller.
rows = []
for b, name in enumerate(BANDS):
    z = g[g["band"] == b]
    rows.append({"band": name, "steps": len(z), "dry_steps": int((z["f"] > 0).sum()), "decision_steps": int(z["a"].sum()),
                 "p_decision_given_dry": z.loc[z["f"] > 0, "a"].mean() if (z["f"] > 0).any() else np.nan,
                 "mean_dry_share_when_dry": z.loc[z["f"] > 0, "f"].mean()})
z = g[g["f"] > 0]
rows.append({"band": "all", "steps": len(g), "dry_steps": len(z), "decision_steps": int(g["a"].sum()),
             "p_decision_given_dry": z["a"].mean(), "mean_dry_share_when_dry": z["f"].mean()})
beh = pd.DataFrame(rows); beh.to_csv("diag_behaviour.csv", index=False)
print(beh.round(3).to_string(index=False))

# 2. Coverage of state-action pairs.
C = pd.crosstab(t["s"], t["a"]).reindex(range(NS), fill_value=0)
cov = pd.DataFrame({"state": range(NS), "label": [state_label(s) for s in range(NS)],
                    "n_a0": C.get(0, 0).to_numpy(), "n_a1": C.get(1, 0).to_numpy()})
cov.to_csv("diag_coverage.csv", index=False)

# 3. Effect of a logged decision, compared within state (20 tabular states).
t["d1"] = t["m1"] - t["m"]; t["d3"] = t["m3"] - t["m"]
OUTCOMES = {"d1": "moisture change, next step", "f1": "dry share, next step",
            "d3": "moisture change, 3 steps", "f3": "dry share, 3 steps ahead"}


def within_state(df, y):
    df = df.dropna(subset=[y])
    m = df.groupby(["s", "a"])[y].agg(["mean", "size"]).unstack()
    both = m.dropna()
    if both.empty:
        return np.nan, np.nan
    diff = both[("mean", 1)] - both[("mean", 0)]
    w = both[("size", 1)]
    naive = df.loc[df["a"] == 1, y].mean() - df.loc[df["a"] == 0, y].mean()
    return float((diff * w).sum() / w.sum()), float(naive)


rng = np.random.default_rng(SEED)
days = t["day"].unique()
by_day = {d: t[t["day"] == d] for d in days}
eff = []
for y, label in OUTCOMES.items():
    est, naive = within_state(t, y)
    boot = []
    for _ in range(NBOOT):
        sample = pd.concat([by_day[d] for d in rng.choice(days, len(days), replace=True)])
        boot.append(within_state(sample, y)[0])
    boot = np.array(boot); boot = boot[~np.isnan(boot)]
    n1 = int(t.dropna(subset=[y])["a"].sum())
    eff.append({"outcome": y, "label": label, "within_state": est, "naive": naive, "lo": np.quantile(boot, .025),
                "hi": np.quantile(boot, .975), "decision_transitions": n1, "replicates": len(boot)})
eff = pd.DataFrame(eff); eff.to_csv("diag_effect.csv", index=False)
print(eff.round(4).to_string(index=False))

# 4. Markov order: does the past add to (m, f, a) when predicting the next step?
u = t[t["prev_exists"]].dropna(subset=["m_prev", "f_prev"]).copy()
u["sin"] = np.sin(2 * np.pi * u["hour"] / 24); u["cos"] = np.cos(2 * np.pi * u["hour"] / 24)
SETS = {"M0: m, f, a": ["m", "f", "a"],
        "M1: + previous decision": ["m", "f", "a", "ap"],
        "M2: + previous m and f": ["m", "f", "a", "ap", "m_prev", "f_prev"],
        "M3: + hour of day": ["m", "f", "a", "ap", "m_prev", "f_prev", "sin", "cos"]}
mk = []
for target in ("d1", "f1"):
    for name, cols in SETS.items():
        for mname, make in (("linear", lambda: LinearRegression()),
                            ("boosting", lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=SEED))):
            pred = np.zeros(len(u))
            for d in u["day"].unique():
                tr, te = u["day"] != d, u["day"] == d
                pred[te.to_numpy()] = make().fit(u.loc[tr, cols], u.loc[tr, target]).predict(u.loc[te, cols])
            mk.append({"target": target, "features": name, "model": mname, "MAE": mean_absolute_error(u[target], pred), "n": len(u)})
# 5. Do valid air readings carry information on the soil once (m, f, a) is known?
v = t[t["H"] == 0].dropna(subset=["hum", "tair"]).copy()
for target in ("d1", "f1"):
    for name, cols in {"A0: m, f, a": ["m", "f", "a"], "A1: + humidity": ["m", "f", "a", "hum"],
                       "A2: + humidity, air temperature": ["m", "f", "a", "hum", "tair"]}.items():
        for mname, make in (("linear", lambda: LinearRegression()),
                            ("boosting", lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=SEED))):
            pred = np.zeros(len(v))
            for d in v["day"].unique():
                tr, te = v["day"] != d, v["day"] == d
                pred[te.to_numpy()] = make().fit(v.loc[tr, cols], v.loc[tr, target]).predict(v.loc[te, cols])
            mk.append({"target": target, "features": name, "model": mname, "MAE": mean_absolute_error(v[target], pred), "n": len(v)})
mk = pd.DataFrame(mk); mk.to_csv("diag_markov.csv", index=False)
print(mk.pivot_table(index=["target", "features"], columns="model", values="MAE").round(4).to_string())
