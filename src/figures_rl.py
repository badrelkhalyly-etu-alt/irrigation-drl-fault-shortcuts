"""Figures of the paper.
Usage: python figures_rl.py  (after the other scripts)
Output: figures/fig1_behaviour.png ... figures/fig6_csi_curves.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common_rl import BANDS, FLEVELS, NF, LAMBDAS

OUT = Path("figures"); OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
LABEL = {"full": "DQN", "mask": "DQN + masking", "ffill": "DQN + last valid", "sc": "SC-DQN",
         "sc+RAE": "SC-DQN + RAE", "full+RAE": "DQN + RAE", "sc-noAD": "SC-DQN, action-indep. training",
         "sc-b0.1": "SC-DQN β=0.1", "sc-b10": "SC-DQN β=10"}
ORDER = list(LABEL)
REG = ["source", "repaired", "action-independent", "coupled"]

# Fig. 1 - logged controller.
beh = pd.read_csv("diag_behaviour.csv"); beh = beh[beh.band != "all"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.6), dpi=220)
x = np.arange(len(beh))
a1.bar(x - .2, beh.dry_steps, .4, color="0.7", label="steps with a dry reading")
a1.bar(x + .2, beh.decision_steps, .4, color="C3", label="steps with a logged decision")
for i, p in enumerate(beh.p_decision_given_dry):
    a1.text(i, max(beh.dry_steps.iloc[i], beh.decision_steps.iloc[i]) + 3, f"{100 * p:.0f}%", ha="center", fontsize=6.5)
a1.set_xticks(x); a1.set_xticklabels(beh.band); a1.set_xlabel("step median soil moisture (%)"); a1.set_ylabel("10-min steps")
a1.legend(frameon=False, fontsize=6.5)
ctx = pd.read_csv("fault_context.csv")
a2.bar(np.arange(3) - .2, ctx.dry_reading_steps, .4, color="0.7", label="steps with a dry reading")
a2.bar(np.arange(3) + .2, ctx.decision_steps, .4, color="C3", label="steps with a logged decision")
for i, r in ctx.iterrows():
    a2.text(i + .2, r.decision_steps + 3, f"{100 * r.decision_steps / r.steps:.0f}% of\n{r.steps} steps", ha="center", fontsize=6)
a2.set_xticks(range(3)); a2.set_xticklabels(ctx.health); a2.set_xlabel("DHT22 health of the step"); a2.legend(frameon=False, fontsize=6.5)
fig.tight_layout(); fig.savefig(OUT / "fig1_behaviour.png", bbox_inches="tight"); plt.close(fig)

# Fig. 2 - fault chain: onset after a decision, onset after a dry reading without a decision, recovery.
on = pd.read_csv("fault_onset.csv").set_index("quantity")
cc = pd.read_csv("fault_commoncause.csv", index_col=0)["value"]
groups = [("failure onset,\nnext step", on.loc["onset1", "a0"], on.loc["onset1", "a1"], on.loc["onset1", "n_a0"], on.loc["onset1", "n_a1"], ("no decision", "decision")),
          ("failure onset,\nwithin 30 min", on.loc["onset3", "a0"], on.loc["onset3", "a1"], on.loc["onset3", "n_a0"], on.loc["onset3", "n_a1"], ("no decision", "decision")),
          ("onset within 30 min,\nno decision", cc["onset3_nodry"], cc["onset3_dry"], cc["n_nodry"], cc["n_dry"], ("no dry\nreading", "dry\nreading")),
          ("recovery,\nnext step", on.loc["recover1", "a0"], on.loc["recover1", "a1"], on.loc["recover1", "n_a0"], on.loc["recover1", "n_a1"], ("no decision", "decision"))]
fig, ax = plt.subplots(figsize=(6.2, 2.6), dpi=220)
for i, (lab, v0, v1, n0, n1, (l0, l1)) in enumerate(groups):
    c0, c1 = ("C0", "C3") if l1 == "decision" else ("0.6", "C1")
    ax.bar(i - .18, 100 * v0, .36, color=c0); ax.bar(i + .18, 100 * v1, .36, color=c1)
    ax.text(i - .18, 100 * v0 + .8, f"{l0}\nn={int(n0)}", ha="center", fontsize=5.2)
    ax.text(i + .18, 100 * v1 + .8, f"{l1}\nn={int(n1)}", ha="center", fontsize=5.2)
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=7); ax.set_ylabel("probability (%)")
ax.set_ylim(0, 100 * on.loc["onset3", "a1"] + 8)
fig.tight_layout(); fig.savefig(OUT / "fig2_fault_chain.png", bbox_inches="tight"); plt.close(fig)

# Fig. 3 - tabular plan on the soil MDP (reference policy).
pol = pd.read_csv("mdp_policies.csv")
p0 = pol[pol["lambda"] == LAMBDAS[0]].sort_values("state")
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), dpi=220, gridspec_kw={"width_ratios": [1, 1.4]})
cov = p0.n_a1.to_numpy().reshape(len(BANDS), NF).T; cov0 = p0.n_a0.to_numpy().reshape(len(BANDS), NF).T
axes[0].imshow(cov, cmap="Reds", aspect="auto")
for i in range(NF):
    for j in range(len(BANDS)):
        axes[0].text(j, i, f"{cov[i, j]}/{cov0[i, j]}", ha="center", va="center", fontsize=6)
axes[0].set_xticks(range(len(BANDS))); axes[0].set_xticklabels(BANDS); axes[0].set_yticks(range(NF)); axes[0].set_yticklabels(FLEVELS)
axes[0].set_xlabel("moisture band (%)"); axes[0].set_title("logged transitions (a=1 / a=0)", fontsize=8)
mat = np.array([pol[pol["lambda"] == lam].sort_values("state").vi.to_numpy() for lam in LAMBDAS])
axes[1].imshow(mat, cmap="Greens", vmin=0, vmax=1.4, aspect="auto")
for i in range(len(LAMBDAS)):
    for j in range(mat.shape[1]):
        if mat[i, j]:
            axes[1].text(j, i, "I", ha="center", va="center", fontsize=7)
for b in range(1, len(BANDS)):
    axes[1].axvline(b * NF - .5, color="k", lw=.6)
axes[1].set_yticks(range(len(LAMBDAS))); axes[1].set_yticklabels([f"λ={l}" for l in LAMBDAS])
axes[1].set_xticks([b * NF + 1 for b in range(len(BANDS))]); axes[1].set_xticklabels(BANDS)
axes[1].set_xlabel("moisture band (%); within each band: f=0 | 0<f≤0.1 | f>0.1")
axes[1].set_title("optimal tabular policy: irrigate (I)", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig3_policy.png", bbox_inches="tight"); plt.close(fig)

# Fig. 4 - decision rate on dry steps with and without a visible failure code (source regime).
ev = pd.read_csv("shortcut_eval.csv"); ev = ev[ev.seed >= 0]
src = ev[ev.regime == "source"].groupby("config")[["p_decide_dry_clean", "p_decide_dry_fault"]].agg(["mean", "std"]).reindex(ORDER)
fig, ax = plt.subplots(figsize=(6.6, 2.6), dpi=220)
x = np.arange(len(ORDER))
for k, (c, col, nm) in enumerate((("p_decide_dry_clean", "C0", "air sensor working"), ("p_decide_dry_fault", "C3", "failure code shown"))):
    mu, sd = 100 * src[(c, "mean")], 100 * src[(c, "std")]
    ax.bar(x + (k - .5) * .38, mu, .38, yerr=[np.minimum(mu, sd), np.minimum(100 - mu, sd)], capsize=1.5, color=col, label=nm, error_kw={"lw": .6})
ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in ORDER], rotation=25, ha="right", fontsize=6.5)
ax.set_ylabel("P(decision | dry reading) (%)"); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig4_shortcut.png", bbox_inches="tight"); plt.close(fig)

# Fig. 5 - reward under the four sensor regimes: mean and 95% bootstrap interval over 15 seeds.
stt = pd.read_csv("shortcut_stats.csv").set_index(["config", "metric"])
fig, ax = plt.subplots(figsize=(6.8, 2.8), dpi=220)
for k, reg in enumerate(REG):
    m = np.array([stt.loc[(c, reg), "mean"] for c in ORDER]); lo = np.array([stt.loc[(c, reg), "mean_lo"] for c in ORDER])
    hi = np.array([stt.loc[(c, reg), "mean_hi"] for c in ORDER])
    ax.errorbar(x + (k - 1.5) * .17, 1000 * m, yerr=[1000 * (m - lo), 1000 * (hi - m)], fmt="o", ms=3, capsize=1.5, lw=.7, label=reg)
base = pd.read_csv("shortcut_eval.csv"); base = base[(base.seed < 0) & (base.regime == "source")].set_index("config")["reward"]
for nm, col in (("tabular plan", "C2"), ("any dry reading", "C1")):
    ax.axhline(1000 * base[nm], color=col, ls=":", lw=.8)
    ax.text(len(ORDER) - .45, 1000 * base[nm], nm, fontsize=5.5, va="center", color=col)
ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in ORDER], rotation=25, ha="right", fontsize=6.5)
ax.set_ylabel("mean reward per step (× 10⁻³)"); ax.legend(frameon=False, fontsize=6.5, ncol=4, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "fig5_regimes.png", bbox_inches="tight"); plt.close(fig)

# Fig. 6 - counterfactual Q-gap (mean, 95% bootstrap interval) and learning curves.
RAW = [c for c in ORDER if c not in ("mask", "ffill")]
cu = pd.read_csv("shortcut_curve.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.6), dpi=220)
xr = np.arange(len(RAW))
m = np.array([stt.loc[(c, "mean_abs_dq"), "mean"] for c in RAW])
lo = np.array([stt.loc[(c, "mean_abs_dq"), "mean_lo"] for c in RAW]); hi = np.array([stt.loc[(c, "mean_abs_dq"), "mean_hi"] for c in RAW])
a1.bar(xr, m, .6, yerr=[m - lo, hi - m], capsize=1.5, color="C0", error_kw={"lw": .6})
a1.set_xticks(xr); a1.set_xticklabels([LABEL[c] for c in RAW], rotation=35, ha="right", fontsize=6)
a1.set_ylabel("CSI$_{ΔQ}$ (mean |ΔQ|)")
for (cfg, col) in (("full", "C3"), ("sc", "C2")):
    z = cu[cu.config == cfg].groupby("iteration").reward
    a2.plot(z.mean().index, 1000 * z.mean(), color=col, lw=1.2, label=LABEL[cfg])
    a2.fill_between(z.mean().index, 1000 * z.quantile(.25), 1000 * z.quantile(.75), color=col, alpha=.2)
a2.set_xlabel("training iteration"); a2.set_ylabel("reward per step, source (× 10⁻³)"); a2.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig6_csi_curves.png", bbox_inches="tight"); plt.close(fig)
print("figures written to", OUT.resolve())
