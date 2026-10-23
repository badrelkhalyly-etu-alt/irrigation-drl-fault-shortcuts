"""Run every step of the paper from the raw CSV.

Usage (from the repository root):
    python reproduce.py

Every generated file goes to outputs/. The raw CSV is expected at
data/greenhouse_dataset.csv (see data/README.md).
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATA = ROOT / "data" / "greenhouse_dataset.csv"

STEPS = [
    ["rl_prep.py", "{data}"],     # cleaning, 10-min decision steps, transitions, day folds
    ["diagnostics.py"],           # controller behaviour, coverage, decision effect, Markov order
    ["fault_chain.py"],           # action-dependent Markov chain of DHT22 health, common-cause check
    ["mdp_planning.py"],          # tabular MDP on the soil state: reference policy, day bootstrap
    ["sim_audit.py"],             # effect of a decision and fixed policies in every soil model
    ["drl_shortcut.py"],          # DQN variants (15 seeds) in the fault POMDP, four sensor regimes, shortcut index
    ["drl_robustness.py"],        # main agents retrained in the band-constrained simulator
    ["figures_rl.py"],
]


def main():
    if not DATA.exists():
        sys.exit(f"Missing {DATA}. Download it first (see data/README.md).")
    OUT.mkdir(exist_ok=True)
    for step in STEPS:
        cmd = [sys.executable, str(ROOT / "src" / step[0])] + [str(DATA) if a == "{data}" else a for a in step[1:]]
        t0 = time.time()
        print(">>", " ".join(step), flush=True)
        subprocess.run(cmd, cwd=OUT, check=True)
        print(f"   {time.time() - t0:.0f} s", flush=True)
    print("\nDone. Results are in", OUT)


if __name__ == "__main__":
    main()
