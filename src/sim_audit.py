"""Audit of the simulators: how large is the effect of a decision in each model, and why do the
fixed policies score on different scales in the tabular MDP and in the k-NN simulator?
Usage: python sim_audit.py
Requires transitions.parquet, fault_chain.csv.
Outputs: audit_effect.csv, audit_policies.csv

Models compared (soil part only; the air sensor plays no role here):
  replay k=10 (main), replay k=5, replay k=30, band-constrained k=10, null-effect (a decision
  replays transitions without a decision), and the tabular MDP.
"""
import numpy as np
import pandas as pd
from common_rl import TabularMDP, NS, NF, M_EDGES, BANDS, DAY_STEPS, GAMMA, MAIN_LAMBDA, features
from common_rl import FaultPOMDP
from shortcut_core import fault_matrices, baselines, evaluate, EVAL_EPISODES, EVAL_SEED

t = pd.read_parquet("transitions.parquet")
P_rep = fault_matrices()["repaired"]
SIMS = {"replay k=10": dict(k=10), "replay k=5": dict(k=5), "replay k=30": dict(k=30),
        "band-constrained k=10": dict(k=10, band_constrained=True), "null effect k=10": dict(k=10, null_effect=True)}
sims = {name: FaultPOMDP(t, P_rep, start_healthy=True, **kw) for name, kw in SIMS.items()}
tab = TabularMDP(t)

# 1. One-step effect of a decision on the expected next dry share, at every logged dry step.
dry = t[t["f"] > 0].copy()
st = np.column_stack([dry["m"], dry["f"], dry["hum_fill"], dry["tair_fill"], np.zeros(len(dry))])
band = np.digitize(dry["m"], M_EDGES)
rows = []


def expected_f1(sim, a):
    out = np.zeros(len(st))
    for idx, model, nb in sim.neighbours(st, np.full(len(st), a)):
        out[idx] = model[2][nb].mean(1)
    return out


effects = {name: expected_f1(sim, 1) - expected_f1(sim, 0) for name, sim in sims.items()}
effects["tabular MDP"] = tab.F[dry["s"].to_numpy(), 1] - tab.F[dry["s"].to_numpy(), 0]
# Where do the a = 1 neighbours of the main simulator come from?
nn1, _, _, _, _ = sims["replay k=10"].models[(-1, 1)]
nn0 = sims["replay k=10"].models[(-1, 0)][0]
d1, i1 = nn1.kneighbors(features(st[:, 0], st[:, 1]))
d0, _ = nn0.kneighbors(features(st[:, 0], st[:, 1]))
dec = t[t["a"] == 1]
other_band = (np.digitize(dec["m"].to_numpy()[i1], M_EDGES) != band[:, None]).mean(1)
wetter = (dec["m"].to_numpy()[i1] > dry["m"].to_numpy()[:, None] + 10).mean(1)
for b in list(range(len(BANDS))) + ["all"]:
    sel = np.ones(len(dry), bool) if b == "all" else band == b
    row = {"band": "all" if b == "all" else BANDS[b], "dry_steps": int(sel.sum())}
    for name, e in effects.items():
        row[f"effect: {name}"] = e[sel].mean()
    row["mean distance to a=1 neighbours"] = d1[sel].mean()
    row["mean distance to a=0 neighbours"] = d0[sel].mean()
    row["share of a=1 neighbours from another band"] = other_band[sel].mean()
    row["share of a=1 neighbours >10 points wetter"] = wetter[sel].mean()
    rows.append(row)
eff = pd.DataFrame(rows); eff.to_csv("audit_effect.csv", index=False)

# 2. Fixed policies in every model, on a common footing.
pols = baselines(t)
mu = tab.mu
pi_beh = np.where(tab.C.sum(1) > 0, tab.C[:, 1] / np.maximum(tab.C.sum(1), 1), 0.0)
TAB_PI = {"never": np.zeros(NS), "any dry reading": ((np.arange(NS) % NF) > 0).astype(float),
          "logged controller (clone)": pi_beh, "tabular plan": tab.solve(MAIN_LAMBDA)}


def tabular_days(pi, episodes=EVAL_EPISODES, seed=EVAL_SEED):
    """Undiscounted 144-step days in the tabular chain, from the empirical state distribution."""
    rng = np.random.default_rng(seed)
    s = rng.choice(NS, size=episodes, p=mu); dry_tot = dec_tot = 0.0
    cumP = tab.P.cumsum(2)
    for _ in range(DAY_STEPS):
        a = (rng.random(episodes) < pi[s]).astype(int)
        dry_tot += tab.F[s, a].sum(); dec_tot += a.sum()
        s = np.minimum((rng.random(episodes)[:, None] > cumP[s, a]).sum(1), NS - 1)
    n = episodes * DAY_STEPS
    return {"dry_share": dry_tot / n, "decisions_per_day": dec_tot / episodes, "reward": -(dry_tot + MAIN_LAMBDA * dec_tot) / n}


prow = []
for name, pi in TAB_PI.items():
    j = tab.evaluate(pi, MAIN_LAMBDA)
    prow.append({"policy": name, "model": "tabular MDP, discounted J", "reward": j["J"], "dry_share": j["dry_share"],
                 "decisions_per_day": j["decisions_per_day"]})
    prow.append({"policy": name, "model": "tabular MDP, 144-step days", **tabular_days(pi)})
for sname, sim in sims.items():
    for name, pol in pols.items():
        r = evaluate(pol, sim)
        prow.append({"policy": name, "model": sname, "reward": r["reward"], "dry_share": r["dry_share"],
                     "decisions_per_day": r["decisions_per_day"], "deficit": r["deficit"]})
pol_df = pd.DataFrame(prow); pol_df.to_csv("audit_policies.csv", index=False)
pd.set_option("display.width", 220)
print(eff.round(4).T.to_string())
print(pol_df.pivot_table(index="model", columns="policy", values="dry_share").round(4).to_string())
print(pol_df.pivot_table(index="model", columns="policy", values="decisions_per_day").round(2).to_string())
