# Detecting and stress-testing action-dependent sensor shortcuts in deep RL for irrigation

Code for the paper:

> **Detecting and Stress-Testing Action-Dependent Sensor Shortcuts in Deep Reinforcement Learning for Smart Irrigation**
> B. El Khalyly, A. Belkhadim, Z. Ibn Batouta and A. Yalid, 2026 (under review).

In the public IoT greenhouse log used here, the DHT22 air sensor often writes a failure code (humidity 0 %, air temperature 0.2 °C), and 46 of the 71 ten-minute steps with an irrigation decision fall in steps where the sensor had failed. If the pump disturbs the sensor, the observations carry a trace of the controller's own past actions, and an RL agent can learn to use that trace instead of the soil state. This is the copycat / policy-confounding mechanism, written here by hardware. The repository:

1. builds 10-min decision steps from the raw log and checks what an MDP built from it can support;
2. estimates a **Markov chain of sensor health** (healthy / degraded / failed) **whose transition matrix depends on the decision**, with day-bootstrap intervals and a **common-cause check**;
3. **audits the replay simulator**: how large the effect of a decision is in each soil model, and where the simulator extrapolates;
4. trains **double DQN** agents (numpy, identical on every machine) with **15 seeds**. It scores them under four sensor regimes, measures their reliance on the failure code with a **counterfactual shortcut index (CSI)**, and compares remedies: masking, last-valid imputation and a consistency penalty (SC-DQN). It reports means, IQMs and bootstrap intervals, and runs Mann–Whitney tests;
5. repeats the main comparison in a **band-constrained simulator** (10 seeds).

## Pipeline

| Script | What it does | Paper |
|---|---|---|
| `src/rl_prep.py` | cleaning (53,189 readings, 208 decisions), 10-min steps (784), transitions (753), DHT22 health per step | Table 2 |
| `src/diagnostics.py` | controller behaviour, coverage, within-state decision effect, Markov order, value of air readings | Table 3, Fig. 1 |
| `src/fault_chain.py` | action-dependent transition matrices of DHT22 health; onset, recovery and common-cause check with day-bootstrap intervals | Table 4, Fig. 2 |
| `src/mdp_planning.py` | tabular soil MDP, reference plan by policy iteration, day bootstrap | Fig. 3 |
| `src/sim_audit.py` | effect of a decision and fixed policies in the tabular model and in five replay simulators | Tables 5–6 |
| `src/drl_shortcut.py` | nine DQN variants × 15 seeds, four sensor regimes, CSI, seed statistics and tests | Tables 7–8, Figs. 4–6 |
| `src/drl_robustness.py` | four main agents × 10 seeds in the band-constrained simulator | Table 9, Sec. 7.4 |
| `src/figures_rl.py` | Figs. 1–6 | |
| `src/common_rl.py`, `src/shortcut_core.py` | MDP, replay simulators, fault POMDP, numpy MLP, training, evaluation, statistics | |

## Run it in Google Colab or Jupyter

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uy5kX4N-8XSk_6vVdfhokh6TRjWVJBCG)

Colab notebook: <https://colab.research.google.com/drive/1uy5kX4N-8XSk_6vVdfhokh6TRjWVJBCG> (the same notebook can also be opened from this repository: [open from GitHub](https://colab.research.google.com/github/badrelkhalyly-etu-alt/irrigation-drl-fault-shortcuts/blob/main/reproduce_colab_drl_irrigation.ipynb)).

`reproduce_colab_drl_irrigation.ipynb` runs everything and ends with a table that compares every number of the paper with the current run (✅ / ❌).
- On Colab, upload `greenhouse_dataset.csv` in the Files panel first; the notebook reads it from `/content`. The first cell installs the pinned versions and restarts the session once; then choose *Run all* again.
- Locally, the copy in `data/` is used.
- Run time: the main experiment trains 135 networks and the robustness check 40. Trainings run in parallel on all CPU cores. This takes about 15 min on a 14-core laptop; on a free Colab CPU (2 cores), expect 2 to 3 hours. The robustness check can be skipped (`RUN_ROBUSTNESS = False`); its checks then show ❌.

## Quick start (command line)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python reproduce.py
```

Outputs go to `outputs/`; our own run is in `results/`.

## Main results (from `results/`)

**Fault chain.** After a decision taken with a healthy sensor, a failure begins within 30 min in 21.4 % of cases (3/14), against 5.5 % without a decision (32/585). The 95 % day-bootstrap interval of the difference is [−5.2, 82.0] points. Without a decision, healthy steps with a dry soil reading are not followed by more failures than those without one (5.49 % vs 5.46 %), so a glitch acting through the soil reading is not supported. A glitch that triggers the decision itself cannot be excluded. Air readings add nothing to the prediction of soil moisture once soil state and decision are known.

**Simulator audit.** Averaged over logged dry steps, a decision lowers the expected next dry share by 0.033 in the main replay simulator, 0.021 in the tabular model and 0.0097 within state in the log. In the driest band, the simulator borrows decision outcomes from wetter soil. The band-constrained simulator removes this.

**15 seeds, reward per step (×10⁻³): mean over seeds; worst regime (IQM); spread; CSI ΔQ**

| Agent | Coupled | Worst (IQM) | Spread | CSI ΔQ |
|---|---|---|---|---|
| DQN | −28.9 | −28.9 | 3.6 | 0.151 |
| SC-DQN (β = 1) | −25.1 | −26.0 | 1.6 | 0.016 |
| DQN + last valid | −26.4 | −25.5 | 1.6 | – |
| DQN + masking | −27.1 | −25.0 | 0.0 | – |
| Tabular plan (soil only) | −26.1 | −26.1 | 0.0 | – |

SC-DQN against DQN, one-sided Mann–Whitney tests: CSI p < 0.001; spread p = 0.0004; coupled regime p = 0.0003; worst regime p = 0.003. Last-valid imputation and masking are as robust as SC-DQN (last valid against SC-DQN, worst regime: p = 0.56, two-sided). Reliability-aware exploration and training with the action-dependent chain add nothing measurable.

**Band-constrained simulator, 10 seeds.** The plain DQN still depends on the fault regime (spread 5.1, CSI 0.164). SC-DQN reduces both (spread 2.8, p = 0.032; CSI 0.019, p = 0.0001). Last-valid imputation is the best learned agent there (worst regime −33.6), level with the tabular plan (−33.4).

## Reproducibility notes

- Verified on Windows 11, Python 3.11.9, with the versions in `requirements.txt`.
- Runs are bit-identical on every machine. The simulators and the networks use element-wise numpy operations only. The network does not call BLAS, whose summation order depends on the processor. Nearest-neighbour ties, which are frequent because many logged states share the same (m, f), are broken by row order. A tree-based search would break them differently on Windows and Linux. We checked that the results do not change when numpy's vector instruction sets or the OpenBLAS kernel are switched, and that they do not depend on the number of parallel processes (`N_JOBS`, default: all cores).
- `Label` is the controller's logged decision; the log has no water volumes or crop measurements. The reward (share of readings at or below 49 % in the next step, minus a cost per decision) is a modelling choice; the paper also reports the shortfall of the median below 60 %.

## Data

See [`data/README.md`](data/README.md).

## Citation

Please cite the paper (reference to be updated on publication) and the dataset:

```
I. Keok, "IoT Greenhouse Irrigation Sensor Dataset," Mendeley Data, V1, 2026, doi: 10.17632/2tvpsht4s4.1.
```

## License

Code: MIT (see `LICENSE`). Data: CC BY 4.0, from the original authors.
