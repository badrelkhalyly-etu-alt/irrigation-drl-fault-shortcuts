"""Shortcut detection and remedies for double DQN in a POMDP with action-dependent sensor faults.
Usage: python drl_shortcut.py
Requires transitions.parquet (rl_prep.py) and fault_chain.csv (fault_chain.py).
Outputs: shortcut_eval.csv, shortcut_csi.csv, shortcut_curve.csv, shortcut_per_seed.csv,
shortcut_stats.csv (mean, IQM, bootstrap intervals over 15 seeds), shortcut_tests.csv

Agents (double DQN, 15 seeds each, lambda = MAIN_LAMBDA), all trained in the source regime unless stated:
  full        sees (m, f, humidity, air temperature), failure codes included
  mask        sees (m, f) only
  ffill       air readings carrying the failure code are replaced by the last valid ones
  sc          SC-DQN: full input + consistency penalty (beta = 1) between the clean and the
              faulted version of each stored observation
  sc+RAE      SC-DQN with reliability-aware exploration (epsilon x 0.2 when a failure code is shown)
  full+RAE    plain DQN with reliability-aware exploration
  sc-noAD     SC-DQN trained with action-independent faults (pooled chain)
  sc-b0.1, sc-b10   SC-DQN with consistency weight 0.1 and 10
The simulator replays logged soil transitions (k nearest neighbours on (m, f), same action).
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from shortcut_core import make_regimes, run_experiment

CONFIGS = {
    "full": dict(mode="full", beta=0.0, rho=0.0, train="source"),
    "mask": dict(mode="mask", beta=0.0, rho=0.0, train="source"),
    "ffill": dict(mode="ffill", beta=0.0, rho=0.0, train="source"),
    "sc": dict(mode="full", beta=1.0, rho=0.0, train="source"),
    "sc+RAE": dict(mode="full", beta=1.0, rho=0.8, train="source"),
    "full+RAE": dict(mode="full", beta=0.0, rho=0.8, train="source"),
    "sc-noAD": dict(mode="full", beta=1.0, rho=0.0, train="action-independent"),
    "sc-b0.1": dict(mode="full", beta=0.1, rho=0.0, train="source"),
    "sc-b10": dict(mode="full", beta=10.0, rho=0.0, train="source"),
}


def main():
    t = pd.read_parquet("transitions.parquet")
    ev, cs, per_seed = run_experiment(t, make_regimes(t), CONFIGS, "shortcut", curve_configs=("full", "sc"))

    # One-sided Mann-Whitney tests between independent sets of seeds.
    tests = []
    for a, b, metric, alt in [("sc", "full", "spread", "less"), ("sc", "full", "mean_abs_dq", "less"),
                              ("sc", "full", "worst", "greater"), ("sc", "full", "coupled", "greater"),
                              ("sc", "full+RAE", "mean_abs_dq", "less"), ("full+RAE", "full", "flip_rate", "greater"),
                              ("ffill", "sc", "worst", "two-sided"), ("sc-b0.1", "sc", "source", "two-sided")]:
        x = per_seed.loc[per_seed.config == a, metric]; y = per_seed.loc[per_seed.config == b, metric]
        u, p = mannwhitneyu(x, y, alternative=alt)
        tests.append({"a": a, "b": b, "metric": metric, "alternative": alt, "U": u, "p": p,
                      "median_a": x.median(), "median_b": y.median()})
    tests = pd.DataFrame(tests); tests.to_csv("shortcut_tests.csv", index=False)
    pd.set_option("display.width", 220)
    st = pd.read_csv("shortcut_stats.csv")
    print(st.pivot_table(index="config", columns="metric", values="mean").round(4).to_string())
    print(tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
