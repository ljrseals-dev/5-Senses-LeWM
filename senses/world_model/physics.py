"""
Ground symbolic rules in physics the world model has (or has not) learned.

Pipeline (numpy only; the eye is duck-typed, see below):
  1. Fit a linear probe latent -> position on training frames, score it on
     held-out frames (does the latent even encode the physical quantity?).
  2. Estimate the true dynamics *from the labelled data* (displacement = A·a + g,
     wall bounds) — no access to the simulator is needed.
  3. Query the model counterfactually: encode a frame, predict under an action,
     decode with the probe, and compare to step 2. One score in [0, 1] per law.
  4. Check surprise: implausible (teleporting) transitions should cost more
     prediction error than real ones.
The resulting facts/rules go into the classical expert system with provenance.

Eye interface (numpy in/out): ``encode_frames(frames (B,3,H,W) float32) -> (B,D)``
and ``predict_latents(z (B,D), a (B,A)) -> (B,D)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from body_of_secrets.truth import TruthStatus
from .toy_env import to_float


# ---------------------------------------------------------------- probe
@dataclass
class LinearProbe:
    W: np.ndarray
    b: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return ((Z - self.mu) / self.sigma) @ self.W + self.b


def fit_ridge_probe(Z: np.ndarray, Y: np.ndarray, alpha: float = 1e-2) -> LinearProbe:
    mu, sigma = Z.mean(0), Z.std(0) + 1e-8
    Zs = (Z - mu) / sigma
    ym = Y.mean(0)
    A = Zs.T @ Zs + alpha * len(Z) * np.eye(Z.shape[1])
    W = np.linalg.solve(A, Zs.T @ (Y - ym))
    return LinearProbe(W, ym, mu, sigma)


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean(0)) ** 2).sum()) + 1e-12
    return 1.0 - ss_res / ss_tot


# --------------------------------------------------------------- results
def _status_for(score: float) -> TruthStatus:
    if not np.isfinite(score):
        return TruthStatus.UNKNOWN
    if score >= 0.8:
        return TruthStatus.PLAUSIBLE      # learned-model evidence never exceeds PLAUSIBLE
    if score >= 0.5:
        return TruthStatus.SPECULATIVE
    return TruthStatus.UNKNOWN


@dataclass
class LawResult:
    name: str
    description: str
    score: float                           # nan = not enough data to test
    measured: Dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> TruthStatus:
        return _status_for(self.score)


@dataclass
class PhysicsReport:
    probe_r2: float
    n_train: int
    n_test: int
    laws: List[LawResult]
    data_dynamics: Dict[str, Any]

    @property
    def min_law_score(self) -> float:
        s = [l.score for l in self.laws if np.isfinite(l.score)]
        return float(min(s)) if s else float("nan")

    def assert_into(self, expert, source: str = "eyes:physics-probe") -> None:
        expert.assert_fact("probe_r2", float(self.probe_r2), source=source,
                           status=_status_for(self.probe_r2))
        for l in self.laws:
            if np.isfinite(l.score):
                expert.assert_fact(f"law_{l.name}_score", float(l.score), source=source,
                                   status=l.status)
        if np.isfinite(self.min_law_score):
            expert.assert_fact("min_law_score", self.min_law_score, source=source,
                               status=_status_for(self.min_law_score))

    def to_dict(self) -> Dict[str, Any]:
        return {"probe_r2": self.probe_r2, "n_train": self.n_train, "n_test": self.n_test,
                "min_law_score": self.min_law_score, "data_dynamics": self.data_dynamics,
                "laws": [{"name": l.name, "description": l.description, "score": l.score,
                          "status": l.status.name, "measured": l.measured} for l in self.laws]}


# -------------------------------------------------------------- grounder
def estimate_dynamics(states: np.ndarray, actions: np.ndarray) -> Dict[str, Any]:
    """Fit disp = A a + g on transitions away from the walls; estimate wall bounds."""
    lo = states.reshape(-1, 2).min(0)
    hi = states.reshape(-1, 2).max(0)
    s0, s1 = states[:, :-1].reshape(-1, 2), states[:, 1:].reshape(-1, 2)
    a = actions.reshape(-1, 2)
    eps = 1e-4
    free = np.all((s0 > lo + eps) & (s0 < hi - eps) & (s1 > lo + eps) & (s1 < hi - eps), axis=1)
    X = np.concatenate([a[free], np.ones((free.sum(), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(X, (s1 - s0)[free], rcond=None)     # (3, 2)
    return {"A": coef[:2].T, "g": coef[2], "lo": lo, "hi": hi, "n_free": int(free.sum())}


class PhysicsGrounder:
    def __init__(self, eye, seed: int = 0, max_samples: int = 256):
        self.eye = eye
        self.rng = np.random.default_rng(seed)
        self.max_samples = max_samples

    def _encode(self, obs: np.ndarray, batch: int = 256) -> np.ndarray:
        flat = obs.reshape(-1, *obs.shape[-3:])
        out = [self.eye.encode_frames(to_float(flat[i:i + batch])) for i in range(0, len(flat), batch)]
        return np.concatenate(out).reshape(*obs.shape[:2], -1)

    def _disp(self, probe, z0, pos0, a) -> np.ndarray:
        z1 = self.eye.predict_latents(z0, np.broadcast_to(a, (len(z0), a.shape[-1])).astype(np.float32))
        return probe.predict(z1) - pos0

    def run(self, train: Dict[str, np.ndarray], test: Dict[str, np.ndarray]) -> PhysicsReport:
        Ztr = self._encode(train["obs"]).reshape(-1, self.eye_latent_dim(train))
        Ytr = train["states"].reshape(-1, 2)
        probe = fit_ridge_probe(Ztr, Ytr)
        Zte = self._encode(test["obs"])
        D = Zte.shape[-1]
        Yte = test["states"].reshape(-1, 2)
        probe_r2 = r2_score(Yte, probe.predict(Zte.reshape(-1, D)))

        dyn = estimate_dynamics(np.concatenate([train["states"], test["states"]]),
                                np.concatenate([train["actions"], test["actions"]]))
        A, g, lo, hi = dyn["A"], dyn["g"], dyn["lo"], dyn["hi"]
        k = float(np.mean(np.abs(np.diag(A)))) + 1e-9

        z_all, s_all = Zte.reshape(-1, D), Yte
        probe_pos = probe.predict(z_all)       # position as *decoded*, so errors don't cancel
        laws: List[LawResult] = []

        def pick(mask: np.ndarray) -> np.ndarray:
            idx = np.flatnonzero(mask)
            if len(idx) > self.max_samples:
                idx = self.rng.choice(idx, self.max_samples, replace=False)
            return idx

        # 1) action response (interior samples)
        interior = np.all((s_all > lo + 1.5 * k) & (s_all < hi - 1.5 * k), axis=1)
        idx = pick(interior)
        if len(idx) >= 8:
            errs = []
            for ax in range(2):
                for sgn in (1.0, -1.0):
                    a = np.zeros(2, np.float32); a[ax] = sgn
                    d = self._disp(probe, z_all[idx], probe_pos[idx], a)
                    exp = A @ a + g
                    errs.append(np.linalg.norm(d - exp, axis=1).mean() / (np.linalg.norm(A @ a) + 1e-9))
            laws.append(LawResult("action_response", "action a shifts position by A·a (+drift)",
                                  float(np.clip(1 - np.mean(errs), 0, 1)),
                                  {"mean_relative_error": float(np.mean(errs)), "n": int(len(idx))}))
        else:
            laws.append(LawResult("action_response", "action a shifts position by A·a (+drift)", float("nan")))

        # 2) drift / gravity (zero action)
        if len(idx) >= 8:
            d0 = self._disp(probe, z_all[idx], probe_pos[idx], np.zeros(2, np.float32)).mean(0)
            laws.append(LawResult("drift", "with no action the object drifts by g per step",
                                  float(np.clip(1 - np.linalg.norm(d0 - g) / k, 0, 1)),
                                  {"model_drift": d0.tolist(), "data_drift": g.tolist()}))
        else:
            laws.append(LawResult("drift", "with no action the object drifts by g per step", float("nan")))

        # 3) walls block motion
        errs, n_w = [], 0
        for ax in range(2):
            for sgn, wall in ((1.0, hi[ax]), (-1.0, lo[ax])):
                dist = (wall - s_all[:, ax]) * sgn
                near = (dist > 0.2 * k) & (dist < 0.8 * k)        # push would overshoot the wall
                widx = pick(near)
                if len(widx) < 4:
                    continue
                a = np.zeros(2, np.float32); a[ax] = sgn
                d = self._disp(probe, z_all[widx], probe_pos[widx], a)[:, ax]
                free_move = (A @ a + g)[ax]
                exp = np.where(sgn > 0, np.minimum(free_move, dist[widx]),
                               np.maximum(free_move, -dist[widx]))
                errs.append(np.abs(d - exp) / k)
                n_w += len(widx)
        laws.append(LawResult("wall_blocking", "objects cannot pass the boundary",
                              float(np.clip(1 - np.concatenate(errs).mean(), 0, 1)) if errs else float("nan"),
                              {"n": n_w}))

        # 4) surprise detects implausible transitions
        laws.append(self._surprise_law(test, k))

        # if the probe cannot read position at all, the counterfactual scores mean nothing
        if probe_r2 < 0.5:
            for l in laws:
                l.measured["note"] = "probe R² < 0.5: latent does not linearly encode position"
        return PhysicsReport(float(probe_r2), int(len(Ztr)), int(len(Yte)), laws,
                             {"A": A.tolist(), "g": g.tolist(), "lo": lo.tolist(),
                              "hi": hi.tolist(), "n_free": dyn["n_free"]})

    def eye_latent_dim(self, data) -> int:
        return int(self.eye.encode_frames(to_float(data["obs"][0, :1])).shape[-1])

    def _surprise_law(self, test: Dict[str, np.ndarray], k: float) -> LawResult:
        obs, acts, st = test["obs"], test["actions"], test["states"]
        n, T1 = obs.shape[:2]
        if n * (T1 - 1) < 16:
            return LawResult("surprise_detects_violation", "teleports cost more than real motion", float("nan"))
        ti = self.rng.integers(0, n, self.max_samples)
        tt = self.rng.integers(0, T1 - 1, self.max_samples)
        f0 = to_float(obs[ti, tt]); f1 = to_float(obs[ti, tt + 1]); a = acts[ti, tt]
        z0, z1 = self.eye.encode_frames(f0), self.eye.encode_frames(f1)
        err_real = ((self.eye.predict_latents(z0, a) - z1) ** 2).mean(1)
        # implausible: the "next" frame is a distant, unrelated position
        pool_i = self.rng.integers(0, n, self.max_samples * 4)
        pool_t = self.rng.integers(0, T1, self.max_samples * 4)
        far = np.linalg.norm(st[pool_i, pool_t][:, None] - st[ti, tt + 1][None], axis=-1) > 3 * k
        sel = [int(np.flatnonzero(far[:, j])[0]) if far[:, j].any() else -1 for j in range(len(ti))]
        keep = np.array([s >= 0 for s in sel])
        if keep.sum() < 8:
            return LawResult("surprise_detects_violation", "teleports cost more than real motion", float("nan"))
        sel_arr = np.array(sel)[keep]
        z_bad = self.eye.encode_frames(to_float(obs[pool_i[sel_arr], pool_t[sel_arr]]))
        err_bad = ((self.eye.predict_latents(z0[keep], a[keep]) - z_bad) ** 2).mean(1)
        er = err_real[keep]
        auc = float(((err_bad[:, None] > er[None]).mean() + 0.5 * (err_bad[:, None] == er[None]).mean()))
        return LawResult("surprise_detects_violation", "teleports cost more than real motion", auc,
                         {"auc": auc, "mean_err_real": float(er.mean()), "mean_err_teleport": float(err_bad.mean())})
