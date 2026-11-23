"""Numpy fake 'eyes' for testing the physics grounder without torch."""
import numpy as np


class OracleEye:
    """Latent = fixed linear mix of the ball centroid; dynamics = the true BallWorld laws."""

    def __init__(self, cfg, noise=0.0, seed=0):
        self.cfg, self.noise = cfg, noise
        rng = np.random.default_rng(seed)
        self.M = rng.standard_normal((8, 3))
        self.pinv = np.linalg.pinv(self.M)
        self.rng = rng

    def _centroid(self, f):
        w = f[:, 0]
        S = w.shape[-1]
        g = np.arange(S)
        tot = w.sum((1, 2)) + 1e-9
        return np.stack([(w.sum(1) * g).sum(-1) / tot, (w.sum(2) * g).sum(-1) / tot], 1) / (S - 1)

    def encode_frames(self, f):
        c = self._centroid(f)
        z = np.concatenate([c, np.ones((len(c), 1))], 1) @ self.M.T
        return z + self.noise * self.rng.standard_normal(z.shape)

    def _pos(self, z):
        return (z @ self.pinv.T)[:, :2]

    def predict_latents(self, z, a):
        c = self.cfg
        p = np.clip(self._pos(z) + c.action_scale * np.clip(a, -1, 1) + [c.drift_x, c.drift_y], c.lo, c.hi)
        return np.concatenate([p, np.ones((len(p), 1))], 1) @ self.M.T


class IdentityEye(OracleEye):
    """Reads position fine but predicts 'nothing moves' (ignores actions and drift)."""

    def predict_latents(self, z, a):
        return z
