"""Robustness check: the main agents retrained in a band-constrained simulator.
Usage: python drl_robustness.py
Requires transitions.parquet and fault_chain.csv.
Outputs: robust_eval.csv, robust_csi.csv, robust_per_seed.csv, robust_stats.csv, robust_tests.csv

In the band-constrained simulator a decision draws its outcome only from logged decisions in the
same moisture band; in bands with fewer than three logged decisions a decision has no effect.
This removes the borrowing of outcomes from decisions logged in wetter soil (see sim_audit.py).
The band-constrained simulator is about three times slower, so this check uses 10 seeds.
"""
import pandas as pd
from scipy.stats import mannwhitneyu
from shortcut_core import make_regimes, run_experiment

CONFIGS = {
    "full": dict(mode="full", beta=0.0, rho=0.0, train="source"),
    "mask": dict(mode="mask", beta=0.0, rho=0.0, train="source"),
    "ffill": dict(mode="ffill", beta=0.0, rho=0.0, train="source"),
    "sc": dict(mode="full", beta=1.0, rho=0.0, train="source"),
}

def main():
    t = pd.read_parquet("transitions.parquet")
    ev, cs, per_seed = run_experiment(t, make_regimes(t, band_constrained=True), CONFIGS, "robust", seeds=tuple(range(10)))
    tests = []
    for metric, alt in (("spread", "less"), ("mean_abs_dq", "less"), ("worst", "greater")):
        x = per_seed.loc[per_seed.config == "sc", metric]; y = per_seed.loc[per_seed.config == "full", metric]
        u, p = mannwhitneyu(x, y, alternative=alt)
        tests.append({"a": "sc", "b": "full", "metric": metric, "alternative": alt, "U": u, "p": p,
                      "median_a": x.median(), "median_b": y.median()})
    pd.DataFrame(tests).to_csv("robust_tests.csv", index=False)
    print(pd.read_csv("robust_stats.csv").pivot_table(index="config", columns="metric", values="mean").round(4).to_string())
    print(pd.DataFrame(tests).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
