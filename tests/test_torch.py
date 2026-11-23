"""Torch-dependent tests (skipped automatically when torch is not installed)."""
import os
import tempfile
import unittest

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

from senses.world_model import toy_env as te


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestTorch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        from senses.world_model.jepa import LeWMStyle
        self.model = LeWMStyle(latent_dim=32, action_dim=2, n_projections=32, image_size=64)

    def test_sigreg_prefers_gaussian_over_collapse(self):
        from senses.world_model.sigreg import SIGReg
        s = SIGReg(n_projections=64)
        gauss = s(torch.randn(256, 32)).item()
        collapsed = s(torch.randn(1, 32).expand(256, -1) + 0.001 * torch.randn(256, 32)).item()
        self.assertLess(gauss, collapsed)

    def test_shapes_and_finite_loss(self):
        x = torch.rand(4, 3, 64, 64); a = torch.zeros(4, 2)
        total, zp, zn, pred, reg = self.model(x, a, x)
        self.assertEqual(tuple(zp.shape), (4, 32))
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertTrue(all(p.grad is not None for p in self.model.encoder.parameters()))

    def test_encoder_keeps_position_information(self):
        cfg = te.BallWorldConfig()
        f = torch.from_numpy(te.render_batch(np.array([[0.2, 0.2], [0.8, 0.8]]), cfg))
        z = self.model.encode(f)
        self.assertGreater((z[0] - z[1]).abs().max().item(), 1e-4)

    def test_planner_result_and_determinism(self):
        from senses.world_model.planner import LatentCEMPlanner
        z0, zg = torch.randn(32), torch.randn(32)
        p = LatentCEMPlanner(self.model, horizon=4, n_candidates=16, n_elite=4, n_iters=3, seed=5)
        r1, r2 = p.plan(z0, zg), p.plan(z0, zg)
        self.assertEqual(tuple(r1.actions.shape), (4, 2))
        self.assertTrue(torch.equal(r1.actions, r2.actions))
        self.assertEqual(len(r1.history), 3)
        self.assertLessEqual(r1.cost, r1.history[0]["all_mean_cost"] + 1e-6)
        acts, cost = r1                                               # tuple-style unpacking still works
        import json; json.dumps(r1.to_dict())

    def test_adapter_checkpoint_roundtrip_and_surprise(self):
        from senses.world_model.adapter import WorldModelEye
        eye = WorldModelEye(self.model)
        cfg = te.BallWorldConfig()
        d = te.collect(4, 6, cfg, seed=0)
        f0, f1, a = d["obs"][:, 0], d["obs"][:, 1], d["actions"][:, 0]
        eye.calibrate(f0, a, f1)
        z = eye.surprise_z(f0, a, f1)
        self.assertEqual(z.shape, (4,))
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "m.pt"); eye.save_checkpoint(p)
            eye2 = WorldModelEye.from_checkpoint(p)
        np.testing.assert_allclose(eye.encode_frames(te.to_float(f0)), eye2.encode_frames(te.to_float(f0)), atol=1e-6)
        np.testing.assert_allclose(z, eye2.surprise_z(f0, a, f1), atol=1e-5)

    def test_callable_backend_plugs_in(self):
        from senses.world_model.adapter import CallableBackend, WorldModelEye
        be = CallableBackend(lambda o: o.flatten(1)[:, :8], lambda z, a: z + a.sum(-1, keepdim=True), 8, 2)
        eye = WorldModelEye(be)
        res = eye.plan(torch.rand(3, 64, 64), torch.rand(3, 64, 64), horizon=3, n_candidates=8, n_elite=2, n_iters=2, seed=0)
        self.assertEqual(tuple(res.actions.shape), (3, 2))

    def test_short_training_lowers_loss_and_grounds_physics(self):
        from senses.world_model.adapter import WorldModelEye
        from senses.world_model.data import PairDataset
        from senses.world_model.jepa import LeWMStyle
        from senses.world_model.trainer import train_model
        torch.manual_seed(0)
        cfg = te.BallWorldConfig(image_size=32)
        d = te.collect(40, 16, cfg, seed=0)
        ds = PairDataset(d["obs"], d["actions"])
        m = LeWMStyle(latent_dim=16, action_dim=2, n_projections=32, image_size=32, channels=[16, 32])
        hist = train_model(m, ds, steps=300, batch_size=32, lr=1e-3, log_every=50, log_fn=None)
        self.assertLess(hist[-1]["pred"] + 0.1 * hist[-1]["reg"], hist[0]["pred"] + 0.1 * hist[0]["reg"])
        from senses.world_model.physics import PhysicsGrounder
        sl = lambda a, b: {k: v[a:b] for k, v in d.items()}
        rep = PhysicsGrounder(WorldModelEye(m), seed=0).run(sl(0, 30), sl(30, 40))
        self.assertTrue(np.isfinite(rep.probe_r2))   # real R² is reported by scripts/train_toy.py


if __name__ == "__main__":
    unittest.main()
