import unittest
import numpy as np

from senses.world_model import toy_env as te
from senses.world_model.physics import PhysicsGrounder, fit_ridge_probe, r2_score, estimate_dynamics
from body_of_secrets import ClassicalBrain
from tests.fakes import OracleEye, IdentityEye


def _data(seed=0, n=60):
    cfg = te.BallWorldConfig()
    d = te.collect(n, 24, cfg, seed=seed)
    cut = int(0.7 * n)
    sl = lambda a, b: {k: v[a:b] for k, v in d.items()}
    return cfg, sl(0, cut), sl(cut, n)


class TestPhysics(unittest.TestCase):
    def test_estimate_dynamics_recovers_truth(self):
        cfg, tr, te_ = _data()
        dyn = estimate_dynamics(tr["states"], tr["actions"])
        np.testing.assert_allclose(np.diag(dyn["A"]), cfg.action_scale, atol=1e-4)
        np.testing.assert_allclose(dyn["g"], [cfg.drift_x, cfg.drift_y], atol=1e-4)
        np.testing.assert_allclose(dyn["lo"], cfg.lo, atol=1e-4)

    def test_probe(self):
        rng = np.random.default_rng(0)
        Y = rng.uniform(0, 1, (400, 2))
        Z = Y @ rng.standard_normal((2, 10)) + 0.01 * rng.standard_normal((400, 10))
        p = fit_ridge_probe(Z[:300], Y[:300])
        self.assertGreater(r2_score(Y[300:], p.predict(Z[300:])), 0.99)

    def test_oracle_scores_high(self):
        cfg, tr, te_ = _data()
        rep = PhysicsGrounder(OracleEye(cfg), seed=1).run(tr, te_)
        self.assertGreater(rep.probe_r2, 0.99)
        scores = {l.name: l.score for l in rep.laws}
        for name in ("action_response", "drift", "wall_blocking", "surprise_detects_violation"):
            self.assertGreater(scores[name], 0.9, (name, scores))
        self.assertGreater(rep.min_law_score, 0.9)

    def test_no_dynamics_model_is_caught(self):
        cfg, tr, te_ = _data()
        rep = PhysicsGrounder(IdentityEye(cfg), seed=1).run(tr, te_)
        s = {l.name: l.score for l in rep.laws}
        self.assertLess(s["action_response"], 0.3, s)
        self.assertLess(rep.min_law_score, 0.5)

    def test_grounding_drives_rules(self):
        cfg, tr, te_ = _data()
        good = ClassicalBrain()
        good.ingest_physics(PhysicsGrounder(OracleEye(cfg), seed=1).run(tr, te_))
        names = [f.rule for f in good.infer()]
        self.assertIn("plan-grounded", names)
        self.assertNotIn("plan-ungrounded", names)
        f = [x for x in good.infer() if x.rule == "plan-grounded"][0]
        self.assertEqual(f.status.name, "PLAUSIBLE")        # learned evidence never exceeds PLAUSIBLE

        bad = ClassicalBrain()
        bad.ingest_physics(PhysicsGrounder(IdentityEye(cfg), seed=1).run(tr, te_))
        names = [f.rule for f in bad.infer()]
        self.assertIn("plan-ungrounded", names)
        self.assertNotIn("plan-grounded", names)

    def test_report_serialises(self):
        import json
        cfg, tr, te_ = _data(n=30)
        rep = PhysicsGrounder(OracleEye(cfg), seed=1).run(tr, te_)
        json.dumps(rep.to_dict())


if __name__ == "__main__":
    unittest.main()
