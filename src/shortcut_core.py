"""Shared code of the DQN experiments: sensor regimes, agents, training, evaluation, shortcut index,
and seed-level statistics (mean, interquartile mean, bootstrap intervals)."""
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from common_rl import (FaultPOMDP, MLP, TabularMDP, GAMMA, MAIN_LAMBDA, DAY_STEPS, features, obs_features,
                       faulted, state_index)

LAM = MAIN_LAMBDA
SEEDS = tuple(range(15))
N_ENV, ITERS, BATCH, SYNC, WARMUP = 16, 3000, 64, 100, 200
EVAL_EPISODES, EVAL_SEED, CSI_EPISODES = 400, 7, 100
DEFICIT_LEVEL = 60.0           # secondary outcome: mean shortfall of the step median below 60 %
NAMES = ["healthy", "degraded", "failed"]
REGIME_NAMES = ["source", "repaired", "action-independent", "coupled"]


def fault_matrices(path="fault_chain.csv"):
    fc = pd.read_csv(path)
    P = np.zeros((2, 3, 3)); N = np.zeros((2, 3, 3))
    for r in fc.itertuples():
        P[r.action, NAMES.index(r._2), NAMES.index(r.to)] = r.p
        N[r.action, NAMES.index(r._2), NAMES.index(r.to)] = r.count
    pooled = (N.sum(0) + 0.5) / (N.sum(0) + 0.5).sum(1, keepdims=True)
    P_ind = np.stack([pooled, pooled])
    P_rep = np.zeros((2, 3, 3)); P_rep[:, :, 0] = 1.0
    P_cpl = P.copy(); P_cpl[1, 0] = [0.4, 0.1, 0.5]
    return {"source": P, "repaired": P_rep, "action-independent": P_ind, "coupled": P_cpl}


def make_regimes(t, **sim_kw):
    Ps = fault_matrices()
    return {name: FaultPOMDP(t, Ps[name], start_healthy=(name == "repaired"), **sim_kw) for name in REGIME_NAMES}


class Agent:
    """Turns observations into network inputs; keeps the last valid air readings for 'ffill'."""

    def __init__(self, mode, net=None):
        self.mode, self.net, self.last = mode, net, None

    def reset(self, obs, st):
        self.last = st[:, 2:4].copy()

    def inputs(self, obs, fault):
        if self.mode == "mask":
            return features(obs[:, 0], obs[:, 1])
        if self.mode == "ffill":
            o = obs.copy()
            self.last[~fault] = obs[~fault, 2:4]
            o[fault, 2:4] = self.last[fault]
            return obs_features(o)
        return obs_features(obs)

    def counterfactual(self, st):
        if self.mode == "mask":
            x = features(st[:, 0], st[:, 1]); return x, x
        return obs_features(st[:, :4]), obs_features(faulted(st[:, :4]))

    def greedy(self, x, f):
        q = self.net.predict(x); q[:, 1] = np.where(f > 0, q[:, 1], -np.inf)
        return q.argmax(1)


def train(cfg, seed, regimes, curve=None, tag=None):
    """Double DQN; cfg: mode, beta (consistency weight), rho (reliability-aware exploration), train regime."""
    rng = np.random.default_rng(seed)
    env = regimes[cfg["train"]]
    din = 2 if cfg["mode"] == "mask" else 4
    net, tgt = MLP([din, 64, 64, 2], rng), MLP([din, 64, 64, 2], rng); tgt.copy_from(net)
    ag = Agent(cfg["mode"], net)
    cap = N_ENV * ITERS
    X = np.zeros((cap, din)); X1 = np.zeros((cap, din)); XC = np.zeros((cap, din)); XF = np.zeros((cap, din))
    A = np.zeros(cap, int); R = np.zeros(cap); F1 = np.zeros(cap); n = 0
    for it in range(ITERS):
        if it % DAY_STEPS == 0:
            st = env.reset(N_ENV, rng); obs, fault = env.observe(st, rng); ag.reset(obs, st)
            x = ag.inputs(obs, fault)
        eps = max(0.05, 1 - it / (0.5 * ITERS)) * (1 - cfg["rho"] * fault)
        a = ag.greedy(x, obs[:, 1])
        explore = rng.random(N_ENV) < eps
        a[explore] = rng.integers(0, 2, explore.sum())
        a[obs[:, 1] <= 0] = 0
        xc, xf = ag.counterfactual(st)
        st1 = env.step(st, a, rng); obs1, fault1 = env.observe(st1, rng); x1 = ag.inputs(obs1, fault1)
        sl = slice(n, n + N_ENV)
        X[sl], A[sl], R[sl], X1[sl], F1[sl], XC[sl], XF[sl] = x, a, -st1[:, 1] - LAM * a, x1, st1[:, 1], xc, xf
        n += N_ENV; st, obs, fault, x = st1, obs1, fault1, x1
        if n >= WARMUP:
            i = rng.integers(0, n, BATCH)
            q_on = net.predict(X1[i]); q_on[:, 1] = np.where(F1[i] > 0, q_on[:, 1], -np.inf)
            q_t = tgt.predict(X1[i])[np.arange(BATCH), q_on.argmax(1)]
            pair = (XC[i], XF[i]) if cfg["beta"] > 0 else None
            net.update(X[i], A[i], R[i] + GAMMA * q_t, pair=pair, beta=cfg["beta"])
        if it % SYNC == 0:
            tgt.copy_from(net)
        if curve is not None and (it + 1) % 500 == 0:
            curve.append({"config": tag, "seed": seed, "iteration": it + 1, **evaluate(ag, regimes["source"], 100)})
    return ag


def evaluate(policy, env, episodes=EVAL_EPISODES):
    """policy: an Agent, or a function (obs, rng) -> actions that ignores the air channels."""
    rng = np.random.default_rng(EVAL_SEED)
    st = env.reset(episodes, rng); obs, fault = env.observe(st, rng)
    if isinstance(policy, Agent):
        policy.reset(obs, st)
    tot = dict.fromkeys(["r", "dry", "dec", "deficit", "dec_fault", "dry_fault", "dec_clean", "dry_clean"], 0.0)
    for _ in range(DAY_STEPS):
        if isinstance(policy, Agent):
            a = policy.greedy(policy.inputs(obs, fault), obs[:, 1])
        else:
            a = policy(obs, rng).astype(int)
        a[obs[:, 1] <= 0] = 0
        dry = obs[:, 1] > 0
        tot["dec_fault"] += (a * (dry & fault)).sum(); tot["dry_fault"] += (dry & fault).sum()
        tot["dec_clean"] += (a * (dry & ~fault)).sum(); tot["dry_clean"] += (dry & ~fault).sum()
        st = env.step(st, a, rng); obs, fault = env.observe(st, rng)
        tot["r"] += (-st[:, 1] - LAM * a).sum(); tot["dry"] += st[:, 1].sum(); tot["dec"] += a.sum()
        tot["deficit"] += np.maximum(0, DEFICIT_LEVEL - st[:, 0]).sum()
    n = episodes * DAY_STEPS
    return {"reward": tot["r"] / n, "dry_share": tot["dry"] / n, "decisions_per_day": tot["dec"] / episodes,
            "deficit": tot["deficit"] / n,
            "p_decide_dry_fault": tot["dec_fault"] / max(tot["dry_fault"], 1),
            "p_decide_dry_clean": tot["dec_clean"] / max(tot["dry_clean"], 1)}


def csi(ag, env):
    """Counterfactual shortcut index along the agent's own greedy trajectories in env."""
    rng = np.random.default_rng(EVAL_SEED + 1)
    st = env.reset(CSI_EPISODES, rng); obs, fault = env.observe(st, rng); ag.reset(obs, st)
    flips = dq = n_dry = n_all = 0.0
    for _ in range(DAY_STEPS):
        xc, xf = ag.counterfactual(st)
        qc, qf = ag.net.predict(xc), ag.net.predict(xf)
        dq += np.abs(qc - qf).sum(); n_all += 2 * len(st)
        dry = st[:, 1] > 0
        flips += (qc[dry].argmax(1) != qf[dry].argmax(1)).sum(); n_dry += dry.sum()
        a = ag.greedy(ag.inputs(obs, fault), obs[:, 1]); a[obs[:, 1] <= 0] = 0
        st = env.step(st, a, rng); obs, fault = env.observe(st, rng)
    return {"flip_rate": flips / max(n_dry, 1), "mean_abs_dq": dq / n_all}


def baselines(t):
    dry_t = t[t["f"] > 0]
    clone = LogisticRegression().fit(features(dry_t["m"], dry_t["f"]), dry_t["a"])
    pi_vi = TabularMDP(t).solve(LAM)
    return {
        "never": lambda o, rng: np.zeros(len(o), int),
        "any dry reading": lambda o, rng: (o[:, 1] > 0).astype(int),
        "logged controller (clone)": lambda o, rng: (rng.random(len(o)) < np.where(
            o[:, 1] > 0, clone.predict_proba(features(o[:, 0], o[:, 1]))[:, 1], 0)).astype(int),
        "tabular plan": lambda o, rng: pi_vi[state_index(o[:, 0], o[:, 1])].round().astype(int),
    }


_WORKER = {}


def _init_worker(regimes):
    _WORKER["regimes"] = regimes


def _train_and_score(job):
    """One (config, seed): train, score in every regime, shortcut index. Each job has its own random
    generators, so the result does not depend on which process runs it."""
    name, cfg, seed, with_curve = job
    regimes = _WORKER["regimes"]
    curve = [] if with_curve else None
    ag = train(cfg, seed, regimes, curve, name)
    rows = [{"config": name, "seed": seed, "regime": reg, **evaluate(ag, env)} for reg, env in regimes.items()]
    return rows, {"config": name, "seed": seed, **csi(ag, regimes["source"])}, curve or []


def run_experiment(t, regimes, configs, prefix, curve_configs=(), seeds=SEEDS):
    """Train every config with every seed, score in every regime, compute the shortcut index in the
    source regime, and write <prefix>_eval.csv, <prefix>_csi.csv, <prefix>_curve.csv, <prefix>_stats.csv.
    The trainings run in N_JOBS processes (default: all cores); results are identical for any N_JOBS."""
    rows, csis, curve = [], [], []
    for name, pol in baselines(t).items():
        for reg, env in regimes.items():
            rows.append({"config": name, "seed": -1, "regime": reg, **evaluate(pol, env)})
    jobs = [(name, cfg, seed, name in curve_configs) for name, cfg in configs.items() for seed in seeds]
    n_jobs = int(os.environ.get("N_JOBS", os.cpu_count() or 1))
    if n_jobs > 1:
        pool = ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker, initargs=(regimes,))
        results = pool.map(_train_and_score, jobs)
    else:
        _init_worker(regimes); pool = None
        results = map(_train_and_score, jobs)
    for k, (r, c, cv) in enumerate(results):
        rows += r; csis.append(c); curve += cv
        if k == len(jobs) - 1 or jobs[k + 1][0] != jobs[k][0]:
            print(prefix, jobs[k][0], "done", flush=True)
    if pool is not None:
        pool.shutdown()
    ev = pd.DataFrame(rows); ev.to_csv(f"{prefix}_eval.csv", index=False)
    cs = pd.DataFrame(csis); cs.to_csv(f"{prefix}_csi.csv", index=False)
    pd.DataFrame(curve).to_csv(f"{prefix}_curve.csv", index=False)

    # Seed-level quantities: reward per regime, worst regime, spread, shortcut index.
    piv = ev[ev.seed >= 0].pivot_table(index=["config", "seed"], columns="regime", values="reward")
    piv["worst"] = piv[REGIME_NAMES].min(1); piv["spread"] = piv[REGIME_NAMES].max(1) - piv[REGIME_NAMES].min(1)
    per_seed = piv.reset_index().merge(cs, on=["config", "seed"])
    per_seed.to_csv(f"{prefix}_per_seed.csv", index=False)
    rng = np.random.default_rng(2026)
    stats = []
    for name in configs:
        z = per_seed[per_seed.config == name]
        for metric in REGIME_NAMES + ["worst", "spread", "mean_abs_dq", "flip_rate"]:
            stats.append({"config": name, "metric": metric, **seed_stats(z[metric], rng)})
    pd.DataFrame(stats).to_csv(f"{prefix}_stats.csv", index=False)
    return ev, cs, per_seed


def iqm(x):
    x = np.sort(np.asarray(x, float)); n = len(x); lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return x[lo:hi].mean()


def seed_stats(values, rng, nboot=10000):
    """Mean, IQM and 95% percentile-bootstrap intervals over seeds."""
    v = np.asarray(values, float)
    idx = rng.integers(0, len(v), (nboot, len(v)))
    bm = v[idx].mean(1); bi = np.array([iqm(r) for r in v[idx]])
    return {"mean": v.mean(), "sd": v.std(ddof=1), "mean_lo": np.quantile(bm, .025), "mean_hi": np.quantile(bm, .975),
            "iqm": iqm(v), "iqm_lo": np.quantile(bi, .025), "iqm_hi": np.quantile(bi, .975), "n": len(v)}
