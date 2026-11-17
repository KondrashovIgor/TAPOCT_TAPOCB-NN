"""Paired, scenario-specific routing algorithm-selection experiment.

Four arms, in the checkpoint order MatNet, NLNS, lkh3, Christofides.
The hidden alpha/beta/gamma are used only by the experiment's feedback oracle.
The learners see geometry, the selected arm, and its scalar client response.
Requires numpy, pandas, scikit-learn, torch and the existing tapoct_fast.py.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ALGORITHMS = ("MatNet", "NLNS", "lkh3", "Christofides")
COMPETITORS = ("LKH-3", "MatNet", "Regression", "NeuralUCB", "NeuralTS", "NeuralRBMLE-PC")
POLICIES = ("TAPOCB-NN",) + COMPETITORS
TAPOCB_POLICY_RULE_VERSION = "rho_gamma_cumulative_v1"
FEATURE_NAMES = (
    "area_km2", "centroid_lat", "centroid_lon", "box_width_km",
    "box_height_km", "normalized_lat", "normalized_lon",
    "distance_from_center_km", "avg_route_time_s", "num_points",
    "cluster_fraction",
)


def features_for_polygon(pid: str, properties: Mapping, aug_map: Mapping) -> np.ndarray:
    """Same feature order as Supportive_NN_integrated.ipynb."""
    p, row = properties[pid], aug_map[pid]
    D = np.asarray(row["distance matrix"], dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1] or not np.isfinite(D).all():
        raise ValueError(f"{pid}: invalid route-time matrix")
    x = np.array((p["area_km2"], p["centroid_lat"], p["centroid_lon"],
                  p["bounding_box"]["width_km"], p["bounding_box"]["height_km"],
                  p["relative_position"]["normalized_lat"],
                  p["relative_position"]["normalized_lon"],
                  p["relative_position"]["distance_from_center_km"],
                  D.mean(), p["num_points"], float(row["cluster"]) / 7), dtype=float)
    if not np.isfinite(x).all():
        raise ValueError(f"{pid}: nonfinite features")
    return x


def original_polygon_id(pid: str, aug_map: Mapping) -> str:
    if not aug_map[pid].get("augmented", False):
        return pid
    if not pid.startswith("AUG--AUG-"):
        raise ValueError(f"Unrecognized augmented polygon ID: {pid}")
    source = "AUG--" + pid[len("AUG--AUG-"):].rsplit("-", 1)[0]
    if source not in aug_map or aug_map[source].get("augmented", False):
        raise ValueError(f"Original polygon missing for {pid}: {source}")
    return source


def grouped_partition(ids: Sequence[str], aug_map: Mapping, seed: int = 42):
    """80/20 outer and 80/20 development split, identical to supportive NN."""
    from sklearn.model_selection import GroupShuffleSplit
    ids = list(dict.fromkeys(ids))
    groups = np.asarray([original_polygon_id(p, aug_map) for p in ids])
    if len(set(groups)) < 10:
        raise ValueError("At least ten distinct original polygons are required")
    idx = np.arange(len(ids))
    dev, test = next(GroupShuffleSplit(n_splits=1, test_size=.20,
                                       random_state=seed).split(idx, groups=groups))
    train_i, val_i = next(GroupShuffleSplit(n_splits=1, test_size=.20,
                                            random_state=seed+1).split(dev, groups=groups[dev]))
    train, val = dev[train_i], dev[val_i]
    sets = [set(groups[k]) for k in (train, val, test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    return [ids[i] for i in train], [ids[i] for i in val], [ids[i] for i in test]


def sequence_with_repetitions(ids: Sequence[str], properties: Mapping, *,
                              seed: int = 42, max_repeats: int = 50) -> list[str]:
    """Central, small polygons receive higher independent repeat probabilities."""
    if not 1 <= max_repeats <= 50:
        raise ValueError("max_repeats must be within [1, 50]")
    ids = list(dict.fromkeys(ids))
    if not ids:
        raise ValueError("Empty held-out polygon set")
    dist = np.array([properties[p]["relative_position"]["distance_from_center_km"] for p in ids])
    size = np.array([properties[p]["num_points"] for p in ids])
    if not np.isfinite(dist).all() or not np.isfinite(size).all() or (size < 1).any():
        raise ValueError("Invalid center distance or polygon size")
    centrality = 1 - np.argsort(np.argsort(dist, kind="stable"), kind="stable") / max(len(ids)-1, 1)
    smallness = 1 - np.argsort(np.argsort(size, kind="stable"), kind="stable") / max(len(ids)-1, 1)
    frequency_p = .05 + .75 * (.5 * centrality + .5 * smallness)
    rng = np.random.default_rng(seed)
    counts = 1 + rng.binomial(max_repeats - 1, frequency_p)
    seq = [pid for pid, count in zip(ids, counts) for _ in range(int(count))]
    rng.shuffle(seq)
    return seq


def build_context_scaler(train_ids, properties, aug_map):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit(np.stack([features_for_polygon(p, properties, aug_map)
                                          for p in train_ids]))


def standard_context(pid, properties, aug_map, scaler):
    return scaler.transform(features_for_polygon(pid, properties, aug_map)[None, :])[0]


def make_logged_feedback(train_ids, results, preference_map, *, seed=42,
                         max_samples=None, calls_per_arm=2):
    """Shared measured teacher budget: two observations per arm and polygon.

    Triplets contain historical cost and CT only; feedback is recomputed for the
    current scenario and never supplied as a client preference vector to a learner.
    All four baselines see the SAME historical log and never see test polygons.
    """
    import tapoct_fast as tf
    rng = np.random.default_rng(seed)
    ids = list(dict.fromkeys(train_ids))
    rng.shuffle(ids)
    if max_samples is not None:
        ids = ids[:max_samples]
    logged = []
    for pid in ids:
        if pid not in results or pid not in preference_map:
            raise KeyError(f"No triplets/preferences for training polygon {pid}")
        result = results[pid]
        if tuple(result["algorithms"]) != ALGORITHMS:
            raise ValueError(f"{pid}: solver order differs from checkpoint")
        spec = tf.TenderSpec(**result["spec"])
        oracle = tf.ClientFeedbackOracle(preference_map[pid], spec)
        for k, name in enumerate(ALGORITHMS):
            choices = [t for t in result["triplets"] if t["algorithm"] == name][:calls_per_arm]
            if len(choices) != calls_per_arm:
                raise ValueError(f"{pid}: fewer than {calls_per_arm} measured {name} calls")
            for t in choices:
                feedback = oracle(spec.price, float(t["route_cost"]), float(t["ct"]),
                                  bool(t["valid_route"]))
                logged.append((pid, k, float(feedback) / 5))
    rng.shuffle(logged)
    return logged


class RegressionSelector:
    """Separate ridge regression per arm; observed-arm-only online updates.

    See Degroote et al., A Regression-Based Methodology for Online Algorithm Selection (SoCS 2018),
    p. 39, Algorithm 2 (greedy online selection strategy). Ridge is a concrete
    regression estimator; epsilon=0 recovers the paper's greedy mapping.
    """
    def __init__(self, dimension=11, ridge=1.0, epsilon=0.0, seed=42):
        self.dimension = dimension + 1
        self.A = np.tile(np.eye(self.dimension)[None] * ridge, (4, 1, 1))
        self.b = np.zeros((4, self.dimension))
        self.n = np.zeros(4, dtype=int)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)

    def select(self, x):
        z = np.append(np.asarray(x, float), 1)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(4))
        estimates = np.array([z @ np.linalg.solve(self.A[k], self.b[k]) for k in range(4)])
        return int(np.argmax(estimates))

    def update(self, x, arm, normalized_reward):
        z = np.append(np.asarray(x, float), 1)
        self.A[arm] += np.outer(z, z)
        self.b[arm] += z * float(normalized_reward)
        self.n[arm] += 1


class NeuralSelector:
    """Full network Jacobian and covariance for UCB, TS, or RBMLE-PC.

    Online network fitting uses a configurable recent replay minibatch rather
    than exact all-history gradient descent at every tender; state this practical
    approximation in the manuscript. The uncertainty matrices are full, not
    diagonal or last-layer-only. All rewards passed here are in [0, 1].
    """
    def __init__(self, method, *, dimension=11, width=16, ridge=1.0,
                 exploration=.15, seed=42, lr=1e-3, train_every=8,
                 gradient_steps=2, batch_size=32):
        import torch
        from torch import nn
        if method not in ("NeuralUCB", "NeuralTS", "NeuralRBMLE-PC"):
            raise ValueError(method)
        self.torch, self.method, self.width = torch, method, width
        self.model = nn.Sequential(nn.Linear(dimension + 4, width), nn.ReLU(),
                                   nn.Linear(width, 1))
        torch.manual_seed(seed)
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        self.theta0 = torch.nn.utils.parameters_to_vector(self.model.parameters()).detach().clone()
        self.p = self.theta0.numel()
        self.ridge, self.exploration = ridge, exploration
        self.Ainv = np.eye(self.p, dtype=np.float64) / ridge
        self.lr, self.train_every = lr, train_every
        self.gradient_steps, self.batch_size = gradient_steps, batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.rng = np.random.default_rng(seed)
        self.history = []

    def _input(self, x, k):
        y = np.r_[np.asarray(x, dtype=np.float32), np.eye(4, dtype=np.float32)[k]]
        return self.torch.from_numpy(y)

    def _prediction_and_gradient(self, x, k):
        self.model.eval()
        inp = self._input(x, k)
        f = self.model(inp).reshape(())
        grads = self.torch.autograd.grad(f, tuple(self.model.parameters()))
        g = self.torch.cat([v.reshape(-1) for v in grads]).detach().numpy().astype(np.float64)
        return float(f.detach()), g / math.sqrt(self.width)

    def _corrected_prediction(self, inp, shift):
        # NeuralRBMLE-PC, Hung et al. (AAAI 2023), p. 7947, Algorithm 2
        # and Eq. (15): theta_bar = theta + alpha(t) Z^{-1} grad(f)/m.
        from torch.func import functional_call
        vector = self.torch.nn.utils.parameters_to_vector(self.model.parameters()).detach()
        corrected = vector + self.torch.from_numpy(shift).to(vector.dtype)
        params = {}
        offset = 0
        for name, p in self.model.named_parameters():
            params[name] = corrected[offset:offset + p.numel()].reshape_as(p)
            offset += p.numel()
        with self.torch.no_grad():
            return float(functional_call(self.model, params, (inp,)).reshape(()))

    def select(self, x):
        scores = []
        t = len(self.history) + 1
        for k in range(4):
            mean, g = self._prediction_and_gradient(x, k)
            var = max(float(g @ self.Ainv @ g), 0)
            if self.method == "NeuralUCB":
                # Zhou et al., ICML 2020, p. 3, Algorithm 1, lines 5-8.
                score = mean + self.exploration * math.sqrt(var)
            elif self.method == "NeuralTS":
                # Zhang et al., ICLR 2021, p. 3, Algorithm 1, lines 4-8.
                score = self.rng.normal(mean, self.exploration * math.sqrt(self.ridge * var))
            else:
                # Alpha(t)=c sqrt(t); factor 1/m is already in scaled g.
                alpha = self.exploration * math.sqrt(t)
                shift = alpha * (self.Ainv @ g) / math.sqrt(self.width)
                score = self._corrected_prediction(self._input(x, k), shift)
            scores.append(score)
        return int(np.argmax(scores))

    def _fit_recent(self, epochs):
        if not self.history:
            return
        self.model.train()
        recent = self.history[-self.batch_size:]
        data = self.torch.stack([self._input(x, k) for x, k, _ in recent])
        target = self.torch.tensor([r for _, _, r in recent], dtype=self.torch.float32)
        for _ in range(epochs):
            self.optimizer.zero_grad()
            pred = self.model(data).reshape(-1)
            # Center regularization on theta0: NeuralTS Eq. (2.3), p. 2;
            # NeuralUCB Algorithm 2, p. 4; NeuralRBMLE-PC Eq. (14), p. 7947.
            theta = self.torch.nn.utils.parameters_to_vector(self.model.parameters())
            loss = (pred - target).square().mean() / 2
            loss += self.ridge * (theta - self.theta0).square().sum() / (2 * max(len(self.history), 1))
            loss.backward()
            self.optimizer.step()
        self.model.eval()

    def update(self, x, arm, normalized_reward):
        if not 0 <= normalized_reward <= 1:
            raise ValueError("Neural baseline reward must be normalized to [0, 1]")
        # UCB: pre-update g (Algorithm 1 line 11). TS and PC: post-fit g
        # (NeuralTS Algorithm 1 line 10; NeuralRBMLE-PC Algorithm 2 line 10).
        pre_g = self._prediction_and_gradient(x, arm)[1] if self.method == "NeuralUCB" else None
        self.history.append((np.asarray(x, dtype=np.float32).copy(), int(arm), float(normalized_reward)))
        if len(self.history) % self.train_every == 0:
            self._fit_recent(self.gradient_steps)
        g = pre_g if pre_g is not None else self._prediction_and_gradient(x, arm)[1]
        q = self.Ainv @ g
        self.Ainv -= np.outer(q, q) / (1 + g @ q)
        self.Ainv = (self.Ainv + self.Ainv.T) / 2

    def prefit(self, observations, epochs=20):
        """Offline warm start, with only one observed arm reward per polygon."""
        self.history.extend((np.asarray(x, dtype=np.float32).copy(), int(k), float(r))
                            for x, k, r in observations)
        # Cycle over ALL training polygons; online updates below use only the
        # most recent minibatch. No validation/test polygon is seen here.
        self.model.train()
        gen = self.torch.Generator().manual_seed(42)
        for _ in range(epochs):
            for indices in self.torch.randperm(len(self.history), generator=gen).split(self.batch_size):
                batch = [self.history[int(i)] for i in indices]
                x = self.torch.stack([self._input(z, k) for z, k, _ in batch])
                y = self.torch.tensor([r for _, _, r in batch], dtype=self.torch.float32)
                self.optimizer.zero_grad()
                pred = self.model(x).reshape(-1)
                theta = self.torch.nn.utils.parameters_to_vector(self.model.parameters())
                loss = (pred - y).square().mean() / 2
                loss += self.ridge * (theta - self.theta0).square().sum() / (2 * len(self.history))
                loss.backward()
                self.optimizer.step()
        self.model.eval()
        self.Ainv = np.eye(self.p) / self.ridge
        for x, k, _ in self.history:
            _, g = self._prediction_and_gradient(x, k)
            q = self.Ainv @ g
            self.Ainv -= np.outer(q, q) / (1 + g @ q)
        self.Ainv = (self.Ainv + self.Ainv.T) / 2

    def export(self):
        return {"method": self.method, "width": self.width,
                "ridge": self.ridge, "exploration": self.exploration,
                "lr": self.lr, "train_every": self.train_every,
                "gradient_steps": self.gradient_steps, "batch_size": self.batch_size,
                "model": self.model.state_dict(), "theta0": self.theta0,
                "Ainv": self.Ainv.copy(), "history": self.history,
                "optimizer": self.optimizer.state_dict(),
                "rng_state": copy.deepcopy(self.rng.bit_generator.state)}

    @classmethod
    def restore(cls, state, *, seed=42):
        obj = cls(state["method"], width=state["width"], ridge=state["ridge"],
                  exploration=state["exploration"], lr=state["lr"],
                  train_every=state["train_every"],
                  gradient_steps=state["gradient_steps"],
                  batch_size=state["batch_size"], seed=seed)
        obj.model.load_state_dict(state["model"])
        # Older checkpoints do not carry optimizer/RNG state; they still load.
        if "optimizer" in state:
            obj.optimizer.load_state_dict(state["optimizer"])
        if "rng_state" in state:
            obj.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        obj.theta0 = state["theta0"].clone()
        obj.Ainv = np.asarray(state["Ainv"]).copy()
        obj.history = [(np.asarray(x).copy(), k, r) for x, k, r in state["history"]]
        return obj


def train_logged_selectors(logged, properties, aug_map, *, seed=42,
                           scaler=None, pretrain_epochs=20, **neural_kwargs):
    """Fit four competitors on exactly the same observed-arm-only log."""
    if scaler is None:
        scaler = build_context_scaler([pid for pid, _, _ in logged], properties, aug_map)
    observations = [(standard_context(pid, properties, aug_map, scaler), arm, reward)
                    for pid, arm, reward in logged]
    selectors = {"Regression": RegressionSelector(seed=seed)}
    for x, k, reward in observations:
        selectors["Regression"].update(x, k, reward)
    for name in COMPETITORS[3:]:
        selectors[name] = NeuralSelector(name, seed=seed, **neural_kwargs)
        selectors[name].prefit(observations, epochs=pretrain_epochs)
    return selectors, scaler


class CourierPolicyNetwork:
    """Read the same 11-feature checkpoint emitted by Supportive_NN_integrated."""
    def __init__(self, checkpoint_path):
        import torch
        from torch import nn
        self.torch = torch
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if tuple(ck["algorithms"]) != ALGORITHMS or ck["input_dim"] != len(FEATURE_NAMES):
            raise ValueError("NN checkpoint algorithm or feature order mismatch")
        if tuple(ck["feature_names"]) != FEATURE_NAMES:
            raise ValueError("NN checkpoint feature order differs from portfolio_sota")
        h = ck["hidden_dim"]
        self.model = nn.Sequential(nn.Linear(11, h), nn.LeakyReLU(), nn.LayerNorm(h), nn.Dropout(.15),
                                   nn.Linear(h, h), nn.LeakyReLU(), nn.LayerNorm(h), nn.Dropout(.15),
                                   nn.Linear(h, 4))
        # Supportive checkpoint is PolygonAlgNN with a .network prefix.
        state = {k.removeprefix("network."): v for k, v in ck["model_state_dict"].items()}
        self.model.load_state_dict(state)
        self.model.eval()
        self.mean = np.asarray(ck["scaler_mean"], dtype=np.float32)
        self.scale = np.asarray(ck["scaler_scale"], dtype=np.float32)
        self.scenario = tuple(ck["scenario"])

    def predict(self, raw_features):
        x = (np.asarray(raw_features, dtype=np.float32) - self.mean) / self.scale
        with self.torch.no_grad():
            p = self.torch.softmax(self.model(self.torch.as_tensor(x)), dim=-1)
            return p.cpu().numpy().reshape(4)


class TAPOCBMemory:
    """Cumulative deterministic shift plus the Eq. (18)-(19) injection gate.

    ``is_win`` in this offline experiment is the paired score >= LKH-3 score,
    not an observed market-auction victory. One entry is maintained per polygon.
    The deterministic update (+.15 on a win, -.20 on a loss) remains the
    operational model used previously; only the exploration rule follows
    the revised rho/gamma formulation. A missing NN warm-start uses a uniform
    prior supplied by the caller.
    """
    def __init__(self, rho=.15, *, allowed_arms=None, injection_enabled=True,
                 dirichlet_alpha=.5):
        if not 0 <= rho <= .5 or dirichlet_alpha <= 0:
            raise ValueError("rho must be in [0, .5] and Dirichlet alpha positive")
        self.rho = float(rho)
        self.injection_enabled = bool(injection_enabled)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.allowed = (np.ones(len(ALGORITHMS), dtype=bool) if allowed_arms is None
                        else np.asarray(allowed_arms, dtype=bool).copy())
        if self.allowed.shape != (len(ALGORITHMS),) or not self.allowed.any():
            raise ValueError("allowed_arms must leave at least one solver enabled")
        self.entries = {}
        self.uniform_fallbacks = 0

    def _project(self, policy):
        p = np.asarray(policy, dtype=float).copy()
        if p.shape != (len(ALGORITHMS),) or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError("Invalid TAPOCB prior or updated policy")
        p[~self.allowed] = 0.
        total = p.sum()
        if total <= 0:
            self.uniform_fallbacks += 1
            p[self.allowed] = 1. / self.allowed.sum()
        else:
            p /= total
        return p

    def _shift(self, policy, arm, win):
        """Compute the next deterministic policy hat from the latest feedback."""
        if not 0 <= arm < len(ALGORITHMS) or not self.allowed[arm]:
            raise ValueError("Feedback refers to a disabled or invalid arm")
        shifted = policy.copy()
        if win:
            shifted[arm] = min(shifted[arm] + .15, .95)
        else:
            shifted[arm] = max(shifted[arm] - .20, .02)
        return self._project(shifted)

    def get_policy(self, pid, mlp_policy, rng):
        if pid not in self.entries:
            initial = self._project(mlp_policy)
            self.entries[pid] = {"policy_hat": initial, "history": [],
                                 "processed": 0, "wins_vs_lkh": 0}
        entry = self.entries[pid]
        history = entry["history"]
        while entry["processed"] < len(history):
            arm, _, win = history[entry["processed"]]
            entry["policy_hat"] = self._shift(entry["policy_hat"], arm, win)
            entry["wins_vs_lkh"] += int(win)
            entry["processed"] += 1
        t = len(history)
        gamma = self.rho * entry["wins_vs_lkh"] / (t + 1)
        entry["last_gamma"] = gamma
        entry["last_injection_weight"] = 0.
        entry["last_injection_applied"] = False
        # The first tender uses its NN (or uniform) prior without history.
        if not t or not self.injection_enabled or self.rho == 0:
            return entry["policy_hat"].copy()
        if rng.random() >= self.rho:
            return entry["policy_hat"].copy()
        noise = np.zeros(len(ALGORITHMS), dtype=float)
        noise[self.allowed] = rng.dirichlet(
            np.full(self.allowed.sum(), self.dirichlet_alpha))
        weight = self.rho - gamma
        entry["last_injection_weight"] = weight
        entry["last_injection_applied"] = True
        return self._project((1. - weight) * entry["policy_hat"] + weight * noise)

    def record_outcome(self, pid, arm, reward, is_win):
        if pid not in self.entries:
            raise KeyError(f"{pid}: get_policy must precede feedback")
        if not 0 <= arm < len(ALGORITHMS) or not self.allowed[arm]:
            raise ValueError("Feedback refers to a disabled or invalid arm")
        self.entries[pid]["history"].append((int(arm), float(reward), bool(is_win)))


def _sync_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def precompute_depot_rows(ids: Sequence[str], aug_map: Mapping,
                          depot_row_fn: Callable[[Sequence[Any]], Sequence[float]] | None = None):
    """Resolve depot->order seconds once per polygon, as in TAPOCT payload.

    Prefer an existing OSRM depot row for augmented polygons. If absent, call
    depot_row_fn(full_map_ids) against the notebook's ordered distance index.
    Missing OD pairs (get_distance default 1e9) fail instead of contaminating
    route costs. The resulting rows can be reused across all eight scenarios.
    """
    rows = {}
    for pid in dict.fromkeys(ids):
        record = aug_map[pid]
        raw = record.get("depot row")
        if raw is None:
            if depot_row_fn is None:
                raise ValueError(f"{pid}: no depot row; pass depot_row_fn from the notebook")
            raw = depot_row_fn(list(record["full map"]))
        row = np.asarray(raw, dtype=float)
        n = len(record["full map"])
        if row.shape != (n,) or not np.isfinite(row).all() or (row < 0).any() or (row >= 1e9).any():
            raise ValueError(f"{pid}: invalid depot->order seconds or missing OD lookup")
        rows[pid] = row
    return rows


def warmup_solver_bundle(solvers: Mapping[str, Callable], pid: str, aug_map: Mapping,
                         *, depot_rows: Mapping | None = None, depot_row_fn=None):
    """Use one real, suitably sized OSRM matrix; keep warmup outside CT."""
    import tapoct_fast as tf
    row = aug_map[pid]
    orders = [0] + list(row["full map"])
    depot_row = (depot_rows[pid] if depot_rows is not None
                 else precompute_depot_rows([pid], aug_map, depot_row_fn)[pid])
    matrix = tf.build_state_matrix(row["distance matrix"], depot_row)
    for name in ALGORITHMS:
        route, _ = solvers[name](orders, orders[0], matrix)
        if route is None or len(route) != len(orders):
            raise RuntimeError(f"Warmup failed for {name} on {pid}")
        _sync_cuda()


def measure_route(pid: str, arm: int, *, aug_map: Mapping, solvers: Mapping,
                  cache=None, repeat_index=0, depot_rows: Mapping | None = None,
                  depot_row_fn=None):
    """Execute selected solver, validate open ATSP route, and time CUDA work."""
    import tapoct_fast as tf
    key = (pid, int(arm), int(repeat_index))
    if cache is not None and key in cache:
        return cache[key]
    row = aug_map[pid]
    if row.get("distance provenance", {}).get("matrix_unit") != "s":
        raise ValueError(f"{pid}: distance matrix must represent seconds")
    orders = [0] + list(row["full map"])
    if len(set(orders)) != len(orders):
        raise ValueError(f"{pid}: duplicate order IDs or depot ID")
    depot_row = (depot_rows[pid] if depot_rows is not None
                 else precompute_depot_rows([pid], aug_map, depot_row_fn)[pid])
    matrix = tf.build_state_matrix(row["distance matrix"], depot_row)
    _sync_cuda()
    t0 = time.perf_counter()
    route, _ = solvers[ALGORITHMS[arm]](orders, 0, matrix)
    _sync_cuda()
    elapsed = time.perf_counter() - t0
    if route is None or len(route) != len(orders) or route[0] != 0 or set(route) != set(orders):
        raise ValueError(f"{pid}/{ALGORITHMS[arm]}: invalid route")
    # Verify objective from the common matrix; do not trust solver-specific
    # reported cost or accidentally include the artificial return-to-depot edge.
    cost = float(tf.route_cost_open(route, {p: i for i, p in enumerate(orders)}, matrix))
    if not np.isfinite(cost) or cost < 0:
        raise ValueError(f"{pid}/{ALGORITHMS[arm]}: invalid route cost")
    measured = (cost, elapsed)
    if cache is not None:
        cache[key] = measured
    return measured


def _spec_for(pid, epos_results, dynamic_config=None):
    import tapoct_fast as tf
    spec = dict(epos_results[pid]["spec"])
    if dynamic_config:
        spec.update({k: dynamic_config[k] for k in ("U", "price", "CT_SA", "time_window")})
    return tf.TenderSpec(**spec)


def run_scenario(sequence, scenario, *, aug_map, properties, epos_results,
                 preference_map, tapocb_checkpoint, warmstart_selectors,
                 context_scaler, solvers, dynamic_config=None, seed=42,
                 reuse_routes=False, route_draws=1, route_cache=None,
                 depot_rows: Mapping | None = None, depot_row_fn=None):
    """Online paired evaluation; each learner sees its OWN selected-arm reward.

    reuse_routes=True caches measured (cost, CT), enabling practical sensitivity
    sweeps, but reports measured solver CT, not wall-clock time of cache lookup.
    This is common-outcome replay, not a fresh runtime measurement per tender.
    """
    import tapoct_fast as tf
    if route_draws < 1:
        raise ValueError("route_draws must be positive")
    if depot_rows is None:
        depot_rows = precompute_depot_rows(sequence, aug_map, depot_row_fn)
    model = CourierPolicyNetwork(tapocb_checkpoint)
    if model.scenario != tuple(scenario):
        raise ValueError(f"Checkpoint scenario {model.scenario} differs from {scenario}")
    # torch is stored on NeuralSelector as a module reference. deepcopy of that
    # object raises TypeError: cannot pickle 'module' object. Reconstruct neural
    # learners from explicit state, preserving their optimizer and RNG, and
    # copy the small NumPy regression learner normally. The caller's warm-start
    # objects remain unchanged across scenarios and repeated smoke runs.
    # Use the declared method name rather than isinstance: notebooks may hold
    # objects created before importlib.reload(portfolio_sota).
    selectors = {name: (NeuralSelector.restore(selector.export())
                        if name in COMPETITORS[3:]
                        else copy.deepcopy(selector))
                 for name, selector in warmstart_selectors.items()}
    if set(selectors) != set(COMPETITORS[2:]):
        raise ValueError("Four portfolio baselines are required")
    action_seed, exploration_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(action_seed)
    exploration_rng = np.random.default_rng(exploration_seed)
    memory = TAPOCBMemory()
    cache = (route_cache if route_cache is not None else {}) if reuse_routes else None
    rows = []
    for step, pid in enumerate(sequence):
        if pid not in preference_map:
            raise KeyError(f"Missing scenario preference vector for {pid}")
        spec = _spec_for(pid, epos_results, dynamic_config)
        oracle = tf.ClientFeedbackOracle(preference_map[pid], spec)
        raw_x = features_for_polygon(pid, properties, aug_map)
        x = context_scaler.transform(raw_x[None, :])[0]
        # Decisions are made BEFORE the client's offer response is revealed.
        decisions = {}
        t0 = time.perf_counter()
        prior = model.predict(raw_x)
        p = memory.get_policy(pid, prior, exploration_rng)
        if not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
            raise ValueError(f"{pid}: invalid TAPOCB-NN policy")
        p = p / p.sum()
        decisions["TAPOCB-NN"] = (int(rng.choice(4, p=p)), time.perf_counter() - t0)
        decisions["LKH-3"] = (2, 0.0)
        decisions["MatNet"] = (0, 0.0)
        for name, learner in selectors.items():
            t0 = time.perf_counter()
            k = learner.select(x)
            decisions[name] = (k, time.perf_counter() - t0)
        outcomes = {}
        for name in POLICIES:
            k, select_ct = decisions[name]
            cost, route_ct = measure_route(pid, k, aug_map=aug_map,
                                            solvers=solvers, cache=cache,
                                            repeat_index=(step % route_draws if reuse_routes else step),
                                            depot_rows=depot_rows)
            ct = route_ct + select_ct
            reward = float(oracle(spec.price, cost, ct))
            outcomes[name] = reward
            rows.append({"scenario": "/".join(scenario), "step": step,
                         "polygon_id": pid, "cluster": aug_map[pid]["cluster"],
                         "method": name, "arm": ALGORITHMS[k], "route_cost_s": cost,
                         "solver_ct_s": route_ct, "selection_ct_s": select_ct,
                         "total_ct_s": ct, "client_reward": reward,
                         "rho": memory.rho if name == "TAPOCB-NN" else np.nan,
                         "gamma_t": (memory.entries[pid]["last_gamma"]
                                     if name == "TAPOCB-NN" else np.nan),
                         "injection_weight": (memory.entries[pid]["last_injection_weight"]
                                              if name == "TAPOCB-NN" else np.nan),
                         "injection_applied": (memory.entries[pid]["last_injection_applied"]
                                               if name == "TAPOCB-NN" else False)})
        # Exact historical proxy used in the notebook: tie counts as memory win.
        # This is a paired CLIENT-SCORE comparison, not an auction win.
        arm = decisions["TAPOCB-NN"][0]
        memory.record_outcome(pid, arm, outcomes["TAPOCB-NN"],
                              outcomes["TAPOCB-NN"] >= outcomes["LKH-3"])
        for name, learner in selectors.items():
            learner.update(x, decisions[name][0], outcomes[name] / 5)
    import pandas as pd
    return pd.DataFrame(rows)


def summarize_sensitivity(raw, *, aug_map=None, bootstrap_samples=1000,
                          ci_level=.95, seed=42):
    """Wide per-cluster and per-scenario paired reward tables.

    top_1 maximizes the SUM of client scores over the same sequence; ties are
    broken lexicographically. Pairwise win rate counts ties as half a win.
    A share is (mean_TAPOCB - mean_competitor)/abs(mean_competitor).

    Percentile intervals resample original polygons as blocks, preserving all
    their augmented instances and repeated tenders. Both compared policies use
    identical bootstrap draws. These are conditional descriptive intervals for
    fixed policies and the chosen top_1, not retraining or selection-adjusted CIs.
    """
    import pandas as pd
    if bootstrap_samples < 1 or not 0 < ci_level < 1:
        raise ValueError("bootstrap_samples must be positive and ci_level in (0, 1)")
    required = {"scenario", "step", "polygon_id", "cluster", "method", "client_reward"}
    if not required <= set(raw.columns):
        raise ValueError(f"Missing columns: {required - set(raw.columns)}")
    if raw.duplicated(["scenario", "step", "method"]).any():
        raise ValueError("Duplicate method/outcome per paired tender")
    wide = raw.pivot(index=["scenario", "step", "polygon_id", "cluster"],
                     columns="method", values="client_reward")
    if wide.isna().any().any() or set(wide.columns) != set(POLICIES):
        raise ValueError("Every tender must have seven valid, paired method outcomes")
    rng = np.random.default_rng(seed)

    def paired_intervals(frame, winner):
        """Block-bootstrap tender-level means/medians on identical source draws."""
        pid = frame.index.get_level_values("polygon_id")
        source = [original_polygon_id(p, aug_map) for p in pid] if aug_map is not None else list(pid)
        _, group = np.unique(source, return_inverse=True)
        n_groups = int(group.max()) + 1
        if n_groups < 2:
            # A one-source cluster cannot supply a meaningful sampling interval.
            return n_groups, {key: np.nan for key in (
                "avg_TAPOCB-NN_reward_ci_low", "avg_TAPOCB-NN_reward_ci_high",
                "median_TAPOCB-NN_reward_ci_low", "median_TAPOCB-NN_reward_ci_high",
                "avg_top_1_reward_ci_low", "avg_top_1_reward_ci_high",
                "median_top_1_reward_ci_low", "median_top_1_reward_ci_high")}

        names = ("TAPOCB-NN", winner)
        values = frame[list(names)].to_numpy(dtype=float)
        sizes = np.bincount(group, minlength=n_groups)
        sums = np.stack([np.bincount(group, weights=values[:, j], minlength=n_groups)
                         for j in range(2)], axis=1)
        sorted_indices = [np.argsort(values[:, j], kind="stable") for j in range(2)]
        boot_means = np.empty((bootstrap_samples, 2), dtype=float)
        boot_medians = np.empty((bootstrap_samples, 2), dtype=float)
        probability = np.full(n_groups, 1.0 / n_groups)
        for begin in range(0, bootstrap_samples, 128):
            end = min(begin + 128, bootstrap_samples)
            counts = rng.multinomial(n_groups, probability, size=end - begin)
            total = counts @ sizes
            boot_means[begin:end] = (counts @ sums) / total[:, None]
            for j, order in enumerate(sorted_indices):
                cumulative = np.cumsum(counts[:, group[order]], axis=1)
                lower_pos = (total - 1) // 2
                upper_pos = total // 2
                lower_idx = (cumulative > lower_pos[:, None]).argmax(axis=1)
                upper_idx = (cumulative > upper_pos[:, None]).argmax(axis=1)
                sorted_values = values[order, j]
                boot_medians[begin:end, j] = (
                    sorted_values[lower_idx] + sorted_values[upper_idx]) / 2

        quantiles = [(1 - ci_level) / 2, (1 + ci_level) / 2]
        mean_ci = np.quantile(boot_means, quantiles, axis=0)
        median_ci = np.quantile(boot_medians, quantiles, axis=0)
        intervals = {}
        for j, name in enumerate(("TAPOCB-NN", "top_1")):
            intervals[f"avg_{name}_reward_ci_low"] = float(mean_ci[0, j])
            intervals[f"avg_{name}_reward_ci_high"] = float(mean_ci[1, j])
            intervals[f"median_{name}_reward_ci_low"] = float(median_ci[0, j])
            intervals[f"median_{name}_reward_ci_high"] = float(median_ci[1, j])
        return n_groups, intervals

    def aggregate(frame, scenario, cluster=None):
        means = frame.mean(axis=0)
        winner = sorted(COMPETITORS, key=lambda c: (-frame[c].sum(), c))[0]
        t, b = float(means["TAPOCB-NN"]), float(means[winner])
        median_t = float(frame["TAPOCB-NN"].median())
        median_b = float(frame[winner].median())
        w = frame["TAPOCB-NN"].to_numpy() - frame[winner].to_numpy()
        row = {"scenario": scenario, "n_tenders": len(frame),
               "avg_TAPOCB-NN_reward": t, "top_1": winner,
               "avg_top_1_reward": b,
               "avg_delta_TAPOCB-NN_from_top_1_units": t-b,
               "avg_delta_TAPOCB-NN_from_top_1_share": (t-b)/abs(b) if b else np.nan,
               "median_TAPOCB-NN_reward": median_t,
               "median_top_1_reward": median_b,
               "median_delta_TAPOCB-NN_from_top_1_units": median_t-median_b,
               "median_delta_TAPOCB-NN_from_top_1_share": (
                   (median_t-median_b)/abs(median_b) if median_b else np.nan),
               "TAPOCB-NN_win_rate": float((np.sum(w > 0) + .5*np.sum(w == 0)) / len(w)),
               "top_1_per_scenario_by_win_rate": float((np.sum(w < 0) + .5*np.sum(w == 0)) / len(w))}
        n_groups, intervals = paired_intervals(frame, winner)
        row["n_original_polygons"] = n_groups
        row.update(intervals)
        if cluster is not None:
            row["cluster"] = cluster
            row["delta_TAPOCB_from_best_units"] = t-b
            row["delta_TAPOCB_from_best_share"] = (t-b)/abs(b) if b else np.nan
        for name in COMPETITORS:
            baseline = float(means[name]); gap = t - baseline
            row[f"avg_{name}_reward"] = baseline
            row[f"median_{name}_reward"] = float(frame[name].median())
            row[f"delta_from_TAPOCB_units_{name}"] = baseline - t
            row[f"delta_from_TAPOCB_share_{name}"] = (baseline-t)/abs(t) if t else np.nan
            row[f"delta_TAPOCB_from_{name}_units"] = gap
            row[f"delta_TAPOCB_from_{name}_share"] = gap/abs(baseline) if baseline else np.nan
        return row

    cluster_rows = [aggregate(frame.droplevel(["scenario", "cluster"]), scenario, cluster)
                    for (scenario, cluster), frame in wide.groupby(level=["scenario", "cluster"])]
    scenario_rows = [aggregate(frame.droplevel("scenario"), scenario)
                     for scenario, frame in wide.groupby(level="scenario")]
    return pd.DataFrame(cluster_rows), pd.DataFrame(scenario_rows)
