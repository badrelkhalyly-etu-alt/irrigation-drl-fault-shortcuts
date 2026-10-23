"""Cleaning and 10-min decision steps.
Usage: python rl_prep.py greenhouse_dataset.csv
Outputs: steps.parquet (one row per 10-min step with at least 3 readings), transitions.parquet
(steps whose next step exists), prep_summary.csv

Cleaning is the one of the companion leakage-aware study: labeled rows with a readable timestamp
and five numeric sensor values (53,189 rows, 208 decisions). Duplicates are kept: on a time grid
they are repeated measurements.
"""
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from common_rl import STEP_MIN, DRY, state_index, day_folds

FEATURES = ["Soil Percent", "Soil Raw", "Suhu Tanah", "Suhu Udara", "Humidity"]
MIN_READINGS = 3


def parse_ts(s):
    for fmt in ("%Y-%m-%d_%H-%M-%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(s), fmt)
        except ValueError:
            pass
    return pd.NaT


raw = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "greenhouse_dataset.csv", sep=";")
d = raw[raw["Label"].notna()].copy()
d[FEATURES] = d[FEATURES].apply(pd.to_numeric, errors="coerce")
d["ts"] = pd.to_datetime(d["Timestamp"].map(parse_ts))
d = d.dropna(subset=FEATURES + ["ts"]).sort_values("ts", kind="mergesort")
d["Label"] = d["Label"].astype(int)
d["dry"] = d["Soil Percent"] <= DRY
d["dht_fail"] = (d["Humidity"] == 0) & (d["Suhu Udara"].round(1) == 0.2)   # DHT22 failure code
d["hum_ok"] = d["Humidity"].where(~d["dht_fail"]); d["tair_ok"] = d["Suhu Udara"].where(~d["dht_fail"])
assert d.loc[d["Label"] == 1, "Soil Percent"].max() <= DRY

d["bin"] = d["ts"].dt.floor(f"{STEP_MIN}min")
g = d.groupby("bin").agg(n=("Label", "size"), m=("Soil Percent", "median"), f=("dry", "mean"),
                         lo=("Soil Percent", "min"), a=("Label", "max"), n_dec=("Label", "sum"),
                         tsoil=("Suhu Tanah", "median"), dht_fail=("dht_fail", "mean"),
                         hum=("hum_ok", "median"), tair=("tair_ok", "median"))
g = g[g["n"] >= MIN_READINGS].copy()
# DHT22 health of the step: 0 healthy (no failure code), 1 degraded (<= half), 2 failed (> half).
g["H"] = np.select([g["dht_fail"] == 0, g["dht_fail"] <= 0.5], [0, 1], 2)
g["day"] = g.index.strftime("%Y-%m-%d")
# Climate of the step as the air sensor would have read it when working; failed steps take the
# day median of valid readings (used only as the hidden climate of the simulator).
for c in ("hum", "tair"):
    g[f"{c}_fill"] = g[c].fillna(g.groupby("day")[c].transform("median")).fillna(g[c].median())
step = pd.Timedelta(f"{STEP_MIN}min")
prev = g.reindex(g.index - step)
g["ap"] = prev["a"].fillna(0).astype(int).to_numpy()          # decision in the previous step
g["prev_exists"] = prev["m"].notna().to_numpy()
g["m_prev"] = prev["m"].to_numpy(); g["f_prev"] = prev["f"].to_numpy()
for k in (1, 2, 3):
    nx = g.reindex(g.index + k * step)
    g[f"m{k}"] = nx["m"].to_numpy(); g[f"f{k}"] = nx["f"].to_numpy(); g[f"H{k}"] = nx["H"].to_numpy()
nx = g.reindex(g.index + step)
g["hum1"] = nx["hum_fill"].to_numpy(); g["tair1"] = nx["tair_fill"].to_numpy()
g = g.reset_index().rename(columns={"bin": "t"})
g["hour"] = g["t"].dt.hour + g["t"].dt.minute / 60
g["s"] = state_index(g["m"], g["f"])
folds = day_folds(g)
g["fold"] = g["day"].map(folds)
g.to_parquet("steps.parquet", index=False)

t = g.dropna(subset=["m1"]).copy()
t["s1"] = state_index(t["m1"], t["f1"])
t.to_parquet("transitions.parquet", index=False)

summary = {
    "clean_readings": len(d), "decisions": int(d["Label"].sum()),
    "steps": len(g), "decision_steps": int(g["a"].sum()), "transitions": len(t),
    "decision_transitions": int(t["a"].sum()), "days": g["day"].nunique(),
    "steps_with_dry_reading": int((g["f"] > 0).sum()),
    "decision_steps_with_median_ge_50": int(((g["a"] == 1) & (g["m"] >= 50)).sum()),
    "steps_with_median_lt_50": int((g["m"] < 50).sum()),
    "mean_dry_share": float(g["f"].mean()),
    "steps_healthy": int((g["H"] == 0).sum()), "steps_degraded": int((g["H"] == 1).sum()), "steps_failed": int((g["H"] == 2).sum()),
    "decision_steps_failed_or_degraded": int(((g["a"] == 1) & (g["H"] > 0)).sum()),
    "fold0_days": int((pd.Series(folds) == 0).sum()), "fold1_days": int((pd.Series(folds) == 1).sum()),
    "fold0_decision_steps": int(g.loc[g["fold"] == 0, "a"].sum()), "fold1_decision_steps": int(g.loc[g["fold"] == 1, "a"].sum()),
    "fold0_transitions": int((t["fold"] == 0).sum()), "fold1_transitions": int((t["fold"] == 1).sum()),
}
pd.Series(summary).to_csv("prep_summary.csv", header=["value"])
print(pd.Series(summary).to_string())
print("folds:", dict(sorted(folds.items())))
