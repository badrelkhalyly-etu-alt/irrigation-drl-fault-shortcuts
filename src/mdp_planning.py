"""Planning on the tabular MDP estimated from the log, and three ways to evaluate the plan.
Usage: python mdp_planning.py
Requires transitions.parquet (rl_prep.py).
Outputs: mdp_policies.csv, mdp_insample.csv, mdp_bootstrap.csv, mdp_bootstrap_draws.csv,
         mdp_stability.csv, mdp_crossfit.csv

Reward per step: -(share of dry readings in the next step) - lambda * decision.
Policies: never irrigate; the logged controller (empirical P(a = 1 | s)); irrigate whenever a dry
reading appears; value iteration (VI) on the estimated MDP, restricted to supported actions.
"""
import numpy as np
import pandas as pd
from common_rl import TabularMDP, NS, LAMBDAS, MAIN_LAMBDA, state_label, NF

SEED, NBOOT = 2026, 500
t = pd.read_parquet("transitions.parquet")
full = TabularMDP(t)
pi_beh = np.where(full.C.sum(1) > 0, full.C[:, 1] / np.maximum(full.C.sum(1), 1), 0.0)
dry_state = (np.arange(NS) % NF) > 0
FIXED = {"never": np.zeros(NS), "logged controller": pi_beh, "irrigate on any dry reading": dry_state.astype(float)}

# Policies and in-sample values.
pol_rows, ins_rows, vi = [], [], {}
for lam in LAMBDAS:
    vi[lam] = full.solve(lam)
    for s in range(NS):
        pol_rows.append({"lambda": lam, "state": s, "label": state_label(s), "n_a0": int(full.C[s, 0]),
                         "n_a1": int(full.C[s, 1]), "supported": bool(full.support[s]), "p_logged": pi_beh[s], "vi": int(vi[lam][s])})
    for name, pi in {**FIXED, "value iteration": vi[lam]}.items():
        ins_rows.append({"lambda": lam, "policy": name, **full.evaluate(pi, lam)})
pd.DataFrame(pol_rows).to_csv("mdp_policies.csv", index=False)
ins = pd.DataFrame(ins_rows); ins.to_csv("mdp_insample.csv", index=False)
print(ins.round(4).to_string(index=False))

# Day bootstrap: rebuild the MDP from resampled days, evaluate the fixed full-data policies,
# and re-solve to see how stable the optimal policy is.
rng = np.random.default_rng(SEED)
days = t["day"].unique()
by_day = {d: t[t["day"] == d] for d in days}
draws, stab = [], np.zeros((len(LAMBDAS), NS))
for b in range(NBOOT):
    mb = TabularMDP(pd.concat([by_day[d] for d in rng.choice(days, len(days), replace=True)]))
    for i, lam in enumerate(LAMBDAS):
        row = {"rep": b, "lambda": lam}
        for name, pi in {**FIXED, "value iteration": vi[lam]}.items():
            row[name] = mb.evaluate(pi, lam)["J"]
        row["re-solved VI"] = mb.evaluate(mb.solve(lam), lam)["J"]
        stab[i] += mb.solve(lam)
        draws.append(row)
dr = pd.DataFrame(draws); dr.to_csv("mdp_bootstrap_draws.csv", index=False)
summ = []
for lam in LAMBDAS:
    z = dr[dr["lambda"] == lam]
    for base in ("never", "logged controller", "irrigate on any dry reading"):
        diff = z["value iteration"] - z[base]
        summ.append({"lambda": lam, "comparison": f"VI - {base}", "in_sample": float(ins[(ins["lambda"] == lam) & (ins.policy == "value iteration")].J.iloc[0]
                     - ins[(ins["lambda"] == lam) & (ins.policy == base)].J.iloc[0]),
                     "median": diff.median(), "lo": diff.quantile(.025), "hi": diff.quantile(.975), "p_positive": (diff > 0).mean()})
summ = pd.DataFrame(summ); summ.to_csv("mdp_bootstrap.csv", index=False)
print(summ.round(5).to_string(index=False))
st = pd.DataFrame(stab / NBOOT, columns=[state_label(s) for s in range(NS)]); st.insert(0, "lambda", LAMBDAS)
st.to_csv("mdp_stability.csv", index=False)

# Cross-fitting over two folds of days: plan on one fold, evaluate on the model of the other.
cf = []
for k in (0, 1):
    train, test = TabularMDP(t[t["fold"] == k]), TabularMDP(t[t["fold"] != k])
    for lam in LAMBDAS:
        pi = train.solve(lam)
        cf.append({"plan_fold": k, "lambda": lam, "policy": "value iteration", "irrigated_states": int(pi.sum()),
                   "J_plan_fold": train.evaluate(pi, lam)["J"], **{f"heldout_{key}": v for key, v in test.evaluate(pi, lam).items()}})
        for name, p in FIXED.items():
            cf.append({"plan_fold": k, "lambda": lam, "policy": name, "irrigated_states": int((p > 0).sum()),
                       "J_plan_fold": train.evaluate(p, lam)["J"], **{f"heldout_{key}": v for key, v in test.evaluate(p, lam).items()}})
cf = pd.DataFrame(cf); cf.to_csv("mdp_crossfit.csv", index=False)
print(cf.round(4).to_string(index=False))
