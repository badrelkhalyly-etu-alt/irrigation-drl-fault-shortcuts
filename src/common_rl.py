"""Shared pieces: decision steps, tabular MDP, k-NN simulator, a small numpy MLP, policies.

The state of a 10-min step is (m, f): the median soil-moisture reading of the step and the share of
its readings at or below 49 % (dry readings). diagnostics.py shows that the previous decision, the
previous step and the hour add nothing to the prediction of the next step given (m, f, a).
Everything is numpy/scikit-learn so that runs are identical on a laptop and on Colab.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

STEP_MIN = 10
DRY = 49            # a reading <= 49 % is "dry"; the controller never logs a decision above it
GAMMA = 0.95        # 10-min steps: effective horizon of about 20 steps (3.3 h)
LAMBDAS = (0.0, 0.02, 0.05, 0.1, 0.2)
MAIN_LAMBDA = 0.05
M_EDGES = (60, 70, 80, 90)          # moisture bands: <60, 60-70, 70-80, 80-90, >=90
F_EDGES = (1e-9, 0.1)               # dry share: 0, (0, 0.1], > 0.1
NF = len(F_EDGES) + 1
NS = (len(M_EDGES) + 1) * NF        # 15 states
MIN_SUPPORT = 3                     # a = 1 is allowed in a state only if logged at least this often
ALPHA = 1.0                         # shrinkage of action-specific estimates toward the state average
DAY_STEPS = 24 * 60 // STEP_MIN     # 144 steps per simulated day
BANDS = ["<60", "60-70", "70-80", "80-90", ">=90"]
FLEVELS = ["f=0", "0<f<=0.1", "f>0.1"]


def state_index(m, f):
    f = np.asarray(f, float)
    return np.digitize(m, M_EDGES) * NF + np.digitize(f, F_EDGES, right=True)


def state_label(s):
    return f"{BANDS[s // NF]} | {FLEVELS[s % NF]}"


def day_folds(steps):
    """Two folds of days, alternating after sorting days by number of decision steps."""
    cnt = steps.groupby("day")["a"].sum().reset_index().sort_values(["a", "day"], ascending=[False, True])
    return {day: i % 2 for i, day in enumerate(cnt["day"])}


# --------------------------------------------------------------------------- tabular MDP
class TabularMDP:
    """Transition kernel and expected next dry share estimated from logged transitions.
    Action-specific estimates are shrunk toward the state average with weight ALPHA, so an
    action never logged in a state is assumed to change nothing there."""

    def __init__(self, trans, alpha=ALPHA, min_support=MIN_SUPPORT):
        s, a, s1, f1 = (trans[c].to_numpy() for c in ("s", "a", "s1", "f1"))
        N = np.zeros((NS, 2, NS)); Fs = np.zeros((NS, 2)); C = np.zeros((NS, 2))
        np.add.at(N, (s, a, s1), 1); np.add.at(Fs, (s, a), f1); np.add.at(C, (s, a), 1)
        pooled, pooledF = N.sum(1), Fs.sum(1)
        self.P = np.zeros((NS, 2, NS)); self.F = np.zeros((NS, 2))
        for k in range(NS):
            n = pooled[k].sum()
            base = pooled[k] / n if n else np.eye(NS)[k]
            bf = pooledF[k] / n if n else 0.0
            for act in (0, 1):
                self.P[k, act] = (N[k, act] + alpha * base) / (C[k, act] + alpha)
                self.F[k, act] = (Fs[k, act] + alpha * bf) / (C[k, act] + alpha)
        self.C = C
        self.support = C[:, 1] >= min_support
        self.mu = np.bincount(s, minlength=NS) / len(s)

    def evaluate(self, pi, lam, mu=None):
        """pi[s] = probability of irrigating in s. Returns the normalized discounted value J
        (per-step scale), the discounted dry share and the decisions per day."""
        mu = self.mu if mu is None else mu
        Pp = (1 - pi)[:, None] * self.P[:, 0] + pi[:, None] * self.P[:, 1]
        Fp = (1 - pi) * self.F[:, 0] + pi * self.F[:, 1]
        occ = (1 - GAMMA) * mu @ np.linalg.inv(np.eye(NS) - GAMMA * Pp)
        dry, rate = occ @ Fp, occ @ pi
        return {"J": -(dry + lam * rate), "dry_share": dry, "decisions_per_day": rate * DAY_STEPS}

    def solve(self, lam, max_iter=100):
        """Optimal policy restricted to supported actions, by policy iteration (exact evaluation
        at each step; it reaches the same fixed point as value iteration)."""
        pi = np.zeros(NS)
        for _ in range(max_iter):
            Pp = (1 - pi)[:, None] * self.P[:, 0] + pi[:, None] * self.P[:, 1]
            Rp = -((1 - pi) * self.F[:, 0] + pi * (self.F[:, 1] + lam))
            V = np.linalg.solve(np.eye(NS) - GAMMA * Pp, Rp)
            Q0 = -self.F[:, 0] + GAMMA * self.P[:, 0] @ V
            Q1 = np.where(self.support, -self.F[:, 1] - lam + GAMMA * self.P[:, 1] @ V, -np.inf)
            new = (Q1 > Q0 + 1e-12).astype(float)
            if np.array_equal(new, pi):
                break
            pi = new
        return pi


# --------------------------------------------------------------------------- features, simulator
def features(m, f):
    return np.column_stack([(np.asarray(m, float) - 75.0) / 15.0, np.asarray(f, float) * 4.0])


class KNNSim:
    """Next-step model that replays logged transitions: given (state, action) it draws one of the
    k nearest logged transitions with that action and applies its moisture change and next dry share."""

    def __init__(self, trans, k=10):
        self.start = trans[["m", "f"]].to_numpy(float)
        self.nn, self.dm, self.f1 = {}, {}, {}
        for act in (0, 1):
            t = trans[trans["a"] == act]
            self.nn[act] = NearestNeighbors(n_neighbors=min(k, len(t))).fit(features(t["m"], t["f"]))
            self.dm[act] = (t["m1"] - t["m"]).to_numpy()
            self.f1[act] = t["f1"].to_numpy()

    def step(self, st, a, rng):
        new = st.copy()
        for act in (0, 1):
            idx = np.where(a == act)[0]
            if len(idx) == 0:
                continue
            nb = self.nn[act].kneighbors(features(st[idx, 0], st[idx, 1]), return_distance=False)
            pick = nb[np.arange(len(idx)), rng.integers(0, nb.shape[1], len(idx))]
            new[idx, 0] = np.clip(st[idx, 0] + self.dm[act][pick], 0, 100)
            new[idx, 1] = self.f1[act][pick]
        return new

    def reset(self, n, rng):
        return self.start[rng.integers(0, len(self.start), n)].copy()

    def run(self, policy, lam, episodes, rng, steps=DAY_STEPS):
        """Simulate `episodes` days in parallel; return per-step averages."""
        st = self.reset(episodes, rng)
        tot_r = tot_dry = tot_a = 0.0
        for _ in range(steps):
            a = policy(st, rng).astype(int)
            a[st[:, 1] <= 0] = 0                         # no logged decision without a dry reading
            st = self.step(st, a, rng)
            tot_dry += st[:, 1].sum(); tot_a += a.sum(); tot_r += (-st[:, 1] - lam * a).sum()
        n = episodes * steps
        return {"reward": tot_r / n, "dry_share": tot_dry / n, "decisions_per_day": tot_a / episodes}


# --------------------------------------------------------------------------- numpy MLP with Adam
class MLP:
    def __init__(self, sizes, rng, lr=5e-4):
        self.W = [rng.normal(0, np.sqrt(2 / i), (i, o)) for i, o in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(o) for o in sizes[1:]]
        self.lr, self.t = lr, 0
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]

    def forward(self, x):
        acts = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.maximum(x, 0)
            acts.append(x)
        return x, acts

    def predict(self, x):
        return self.forward(x)[0]

    def backward(self, acts, g):
        """Gradients of the parameters for an output gradient g."""
        gW, gb = [], []
        for i in range(len(self.W) - 1, -1, -1):
            gW.insert(0, acts[i].T @ g); gb.insert(0, g.sum(0))
            if i:
                g = (g @ self.W[i].T) * (acts[i] > 0)
        return gW + gb

    def adam(self, grads):
        self.t += 1
        for j, (p, gr) in enumerate(zip(self.W + self.b, grads)):
            self.m[j] = 0.9 * self.m[j] + 0.1 * gr
            self.v[j] = 0.999 * self.v[j] + 0.001 * gr ** 2
            p -= self.lr * (self.m[j] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[j] / (1 - 0.999 ** self.t)) + 1e-8)

    def update(self, x, a, target, pair=None, beta=0.0):
        """One Adam step on the Huber loss of Q(x, a) against target. With pair = (x_clean, x_fault)
        and beta > 0, adds beta * mean over the batch of sum_a (Q(x_clean, a) - Q(x_fault, a))^2."""
        out, acts = self.forward(x)
        err = out[np.arange(len(a)), a] - target
        g = np.zeros_like(out)
        g[np.arange(len(a)), a] = np.clip(err, -1, 1) / len(a)
        grads = self.backward(acts, g)
        if pair is not None and beta > 0:
            qc, ac = self.forward(pair[0]); qf, af = self.forward(pair[1])
            dq = 2 * beta * (qc - qf) / len(qc)
            gc, gf = self.backward(ac, dq), self.backward(af, -dq)
            grads = [x0 + x1 + x2 for x0, x1, x2 in zip(grads, gc, gf)]
        self.adam(grads)
        return float(np.mean(err ** 2))

    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]; self.b = [b.copy() for b in other.b]


def q_masked(net, st):
    q = net.predict(features(st[:, 0], st[:, 1]))
    q[:, 1] = np.where(st[:, 1] > 0, q[:, 1], -np.inf)
    return q


# --------------------------------------------------------------------------- POMDP with action-dependent sensor faults
HUM_FAIL, TAIR_FAIL = 0.0, 0.2      # what the logger writes when the DHT22 read fails


class FaultPOMDP:
    """Hidden state: soil moisture m, dry share f, air humidity and temperature, DHT22 health H
    (0 healthy, 1 degraded, 2 failed). Soil dynamics replay logged transitions (k nearest
    neighbours on (m, f) with the same action); the air climate follows the chosen neighbour; H
    follows a Markov chain whose matrix depends on the action, P[a, H, H'].
    The agent sees (m, f, humidity, air temperature) where the two air readings carry the
    failure code when H = 2, and with probability 1/2 when H = 1."""

    def __init__(self, trans, P_fault, k=10, start_healthy=False, band_constrained=False, null_effect=False):
        """band_constrained: neighbours are searched within the moisture band of the state; where a
        band holds fewer than MIN_SUPPORT logged decisions, a decision behaves like no decision.
        null_effect: a decision always behaves like no decision (a simulator with no action effect)."""
        self.P = np.asarray(P_fault, float)
        self.cum = self.P.cumsum(2)
        self.start = trans[["m", "f", "hum_fill", "tair_fill", "H"]].to_numpy(float)
        if start_healthy:
            self.start[:, 4] = 0
        self.band_constrained = band_constrained
        bands = np.digitize(trans["m"], M_EDGES)
        self.models = {}
        for b in (range(len(M_EDGES) + 1) if band_constrained else [-1]):
            for act in (0, 1):
                sel = (trans["a"] == act) & ((bands == b) if b >= 0 else True)
                if act == 1 and (null_effect or sel.sum() < MIN_SUPPORT):
                    sel = (trans["a"] == 0) & ((bands == b) if b >= 0 else True)
                t = trans[sel]
                self.models[(b, act)] = (NearestNeighbors(n_neighbors=min(k, len(t))).fit(features(t["m"], t["f"])),
                                         (t["m1"] - t["m"]).to_numpy(), t["f1"].to_numpy(), t["hum1"].to_numpy(), t["tair1"].to_numpy())

    def reset(self, n, rng):
        return self.start[rng.integers(0, len(self.start), n)].copy()

    def neighbours(self, st, a):
        """Indices of the k nearest logged transitions for each (state, action), with the model used."""
        out = []
        band = np.digitize(st[:, 0], M_EDGES) if self.band_constrained else np.full(len(st), -1)
        for (b, act), model in self.models.items():
            idx = np.where((a == act) & (band == b))[0]
            if len(idx):
                out.append((idx, model, model[0].kneighbors(features(st[idx, 0], st[idx, 1]), return_distance=False)))
        return out

    def step(self, st, a, rng):
        new = st.copy()
        for idx, (nn, dm, f1, h1, t1), nb in self.neighbours(st, a):
            pick = nb[np.arange(len(idx)), rng.integers(0, nb.shape[1], len(idx))]
            new[idx, 0] = np.clip(st[idx, 0] + dm[pick], 0, 100)
            new[idx, 1] = f1[pick]; new[idx, 2] = h1[pick]; new[idx, 3] = t1[pick]
        u = rng.random(len(st))
        cum = self.cum[a, st[:, 4].astype(int)]
        new[:, 4] = (u[:, None] > cum).sum(1).clip(0, 2)
        return new

    @staticmethod
    def observe(st, rng):
        """Observed (m, f, humidity, air temperature) and the boolean 'failure code shown'."""
        fault = (st[:, 4] == 2) | ((st[:, 4] == 1) & (rng.random(len(st)) < 0.5))
        obs = st[:, :4].copy()
        obs[fault, 2] = HUM_FAIL; obs[fault, 3] = TAIR_FAIL
        return obs, fault


def obs_features(obs):
    """Scaled network input: (m, f, humidity, air temperature)."""
    return np.column_stack([(obs[:, 0] - 75.0) / 15.0, obs[:, 1] * 4.0, (obs[:, 2] - 80.0) / 20.0, (obs[:, 3] - 28.0) / 5.0])


def faulted(obs):
    out = obs.copy(); out[:, 2] = HUM_FAIL; out[:, 3] = TAIR_FAIL
    return out


# --------------------------------------------------------------------------- policies on simulator states
def p_never(st, rng):
    return np.zeros(len(st), int)


def p_any_dry(st, rng):
    return (st[:, 1] > 0).astype(int)


def p_tabular(pi):
    return lambda st, rng: pi[state_index(st[:, 0], st[:, 1])].round().astype(int)


def p_behaviour(clf):
    """Stochastic clone of the logged controller: logistic model of P(a = 1 | m, f) on dry steps."""
    def f(st, rng):
        p = np.zeros(len(st)); dry = st[:, 1] > 0
        if dry.any():
            p[dry] = clf.predict_proba(features(st[dry, 0], st[dry, 1]))[:, 1]
        return (rng.random(len(st)) < p).astype(int)
    return f


def p_net(net):
    return lambda st, rng: q_masked(net, st).argmax(1)
