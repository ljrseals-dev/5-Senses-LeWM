import os
import tempfile
import unittest

import numpy as np

from senses.world_model import toy_env as te
from tests.fakes import OracleEye


class TestBallWorld(unittest.TestCase):
    def test_laws(self):
        cfg = te.BallWorldConfig(drift_y=0.02)
        env = te.BallWorld(cfg, seed=0)
        env.reset([0.5, 0.5])
        p = env.step([1.0, 0.0])
        np.testing.assert_allclose(p, [0.5 + cfg.action_scale, 0.5 + 0.02])      # action + drift
        p = env.step([100.0, 0.0])                                               # actions are clipped
        np.testing.assert_allclose(p[0], 0.5 + 2 * cfg.action_scale)
        env.reset([cfg.hi, cfg.hi])
        p = env.step([1.0, 1.0])
        np.testing.assert_allclose(p, [cfg.hi, cfg.hi])                          # walls block

    def test_render_centroid_matches_state(self):
        cfg = te.BallWorldConfig()
        pos = np.array([[0.2, 0.3], [0.8, 0.9], [cfg.lo, cfg.lo]])
        frames = te.render_batch(pos, cfg)
        self.assertEqual(frames.shape, (3, 3, 64, 64))
        np.testing.assert_allclose(OracleEye(cfg)._centroid(frames), pos, atol=2e-3)

    def test_deterministic_and_roundtrip(self):
        cfg = te.BallWorldConfig()
        a = te.collect(4, 10, cfg, seed=7)
        b = te.collect(4, 10, cfg, seed=7)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        self.assertTrue((a["states"] >= cfg.lo - 1e-6).all() and (a["states"] <= cfg.hi + 1e-6).all())
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.npz")
            te.save_dataset(p, a, cfg)
            a2, cfg2 = te.load_dataset(p)
        self.assertEqual(cfg, cfg2)
        np.testing.assert_array_equal(a["obs"], a2["obs"])

    def test_data_contains_walls_and_zero_actions(self):
        d = te.collect(40, 24, te.BallWorldConfig(), seed=0)
        cfg = te.BallWorldConfig()
        self.assertTrue((np.abs(d["actions"]).sum(-1) == 0).any())
        at_wall = ((d["states"] <= cfg.lo + 1e-6) | (d["states"] >= cfg.hi - 1e-6)).any(-1)
        self.assertGreater(at_wall.mean(), 0.02)


class TestConfig(unittest.TestCase):
    def test_env_overrides_and_defaults(self):
        from senses.config import load_config
        c = load_config(None, env={"DEVICE": "cuda", "LATENT_DIM": "64", "LAMBDA_REG": "0.5",
                                   "LEWM_CHECKPOINT": "x.pt"})
        self.assertEqual((c["training"]["device"], c["model"]["latent_dim"], c["model"]["lambda_reg"],
                          c["checkpoint"]), ("cuda", 64, 0.5, "x.pt"))
        c = load_config("configs/default.yaml", env={})
        self.assertEqual(c["model"]["encoder"]["channels"], [32, 64, 128])


if __name__ == "__main__":
    unittest.main()
