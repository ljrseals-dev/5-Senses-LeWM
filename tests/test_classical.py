import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

from body_of_secrets import AssociativeMemory, ClassicalBrain, ExpertSystem, TruthStatus
from body_of_secrets.cli import main as cli_main
from body_of_secrets.expert import parse_when
from body_of_secrets.truth import weakest


class TestMemory(unittest.TestCase):
    def test_recalls_noisy_cue(self):
        rng = np.random.default_rng(0)
        mem = AssociativeMemory()
        pats = [rng.standard_normal(256) for _ in range(8)]
        for i, p in enumerate(pats):
            mem.store(p, f"p{i}")
        self.assertFalse(mem.overloaded)
        for i, p in enumerate(pats):
            cue = p.copy()
            flip = rng.choice(256, 25, replace=False)          # ~10% of bits wrong
            cue[flip] *= -1
            r = mem.recall(cue)
            self.assertEqual(r.label, f"p{i}")
            self.assertTrue(r.recognized and r.converged)
            self.assertGreaterEqual(r.overlap, 0.95)

    def test_unrelated_cue_not_recognized(self):
        rng = np.random.default_rng(1)
        mem = AssociativeMemory()
        for _ in range(5):
            mem.store(rng.standard_normal(256))
        self.assertFalse(mem.recall(rng.standard_normal(256)).recognized)

    def test_empty_recall(self):
        r = AssociativeMemory(dim=8).recall(np.ones(8))
        self.assertIsNone(r.label)
        self.assertFalse(r.recognized)

    def test_fixed_dim_projects_instead_of_truncating(self):
        rng = np.random.default_rng(2)
        mem = AssociativeMemory(dim=64)
        a, b = rng.standard_normal(192), rng.standard_normal(192)
        # a and b agree on the first 64 entries but differ elsewhere -> truncation would confuse them
        b[:64] = a[:64]
        mem.store(a, "a"); mem.store(b, "b")
        self.assertEqual(mem.recall(a).label, "a")
        self.assertEqual(mem.recall(b).label, "b")

    def test_zero_input_is_valid(self):
        mem = AssociativeMemory()
        mem.store(np.zeros(16), "zeros")
        self.assertTrue(mem.recall(np.zeros(16)).recognized)

    def test_save_load(self):
        rng = np.random.default_rng(3)
        mem = AssociativeMemory()
        p = rng.standard_normal(64)
        mem.store(p, "x")
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "m.npz")
            mem.save(f)
            m2 = AssociativeMemory.load(f)
        self.assertEqual(m2.recall(p).label, "x")


class TestExpert(unittest.TestCase):
    def test_parse(self):
        cl = parse_when("a > 3 and b <= c")
        self.assertEqual([(c.fact, c.op, c.rhs) for c in cl], [("a", ">", 3.0), ("b", "<=", "c")])
        with self.assertRaises(ValueError):
            parse_when("surprise high")

    def test_forward_chaining_with_provenance(self):
        e = ExpertSystem()
        e.add_rule("r", "x >= 2", "do it", status="OFFICIAL")
        self.assertEqual(e.infer(), [])                            # fact missing -> cannot fire
        e.assert_fact("x", 5, source="user", status=TruthStatus.SPECULATIVE)
        (f,) = e.infer()
        self.assertEqual(f.status, TruthStatus.SPECULATIVE)       # weakest of rule + premises
        self.assertIn("src=user", f.explain())
        e.assert_fact("x", 1)
        self.assertEqual(e.infer(), [])

    def test_fact_to_fact_comparison(self):
        e = ExpertSystem()
        e.add_rule("g", "d <= tol", "stop")
        e.assert_fact("d", 0.1); e.assert_fact("tol", 0.5)
        self.assertEqual(len(e.infer()), 1)

    def test_keyword_matching_has_no_substring_bug(self):
        e = ExpertSystem()
        e.add_rule("goal", "latent close to goal embedding", "halt",
                   description="goal reached")
        self.assertIsNone(e.query_rule("please stop"))             # old code matched 'to' inside 'stop'
        self.assertIn("No rule matched", e.query("please stop"))
        self.assertIsNotNone(e.query_rule("was the goal reached?"))

    def test_weakest(self):
        self.assertEqual(weakest(TruthStatus.OFFICIAL, TruthStatus.PLAUSIBLE), TruthStatus.PLAUSIBLE)

    def test_audit_export(self):
        e = ExpertSystem()
        e.add_rule("r", "x > 0", "go"); e.assert_fact("x", 1); e.infer()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.jsonl"); e.export_audit(p)
            with open(p, encoding="utf-8") as fh:
                self.assertIn('"rule": "r"', fh.read())


class TestBrain(unittest.TestCase):
    def test_ingest_goal_surprise(self):
        b = ClassicalBrain()
        rng = np.random.default_rng(0)
        goal = rng.standard_normal(192)
        b.set_goal(goal, tolerance=0.5)
        b.ingest_sensory(goal + 0.01, label="near", surprise_z=0.2)
        names = [f.rule for f in b.infer()]
        self.assertIn("goal-reached", names)
        self.assertNotIn("visual-surprise", names)
        b.ingest_sensory(rng.standard_normal(192), label="far", surprise_z=6.0)
        names = [f.rule for f in b.infer()]
        self.assertIn("visual-surprise", names)
        self.assertNotIn("goal-reached", names)

    def test_full_dimension_memory(self):
        b = ClassicalBrain()
        b.ingest_sensory(np.random.default_rng(0).standard_normal(192))
        self.assertEqual(b.memory.dim, 192)                         # no silent truncation to 64

    def test_ask_paths(self):
        b = ClassicalBrain()
        self.assertEqual(b.ask("What do I see?"), "Brain has no sensory data yet.")
        b.ingest_sensory(np.random.default_rng(0).standard_normal(64), label="seen")
        ans = b.ask("What do I see?")
        self.assertIn("seen", ans); self.assertIn("overlap=", ans)

    def test_save_load(self):
        b = ClassicalBrain()
        b.ingest_sensory(np.random.default_rng(0).standard_normal(64), label="seen")
        with tempfile.TemporaryDirectory() as d:
            b.save(d)
            b2 = ClassicalBrain(); b2.load(d)
        self.assertEqual(len(b2.memory), 1)
        self.assertIn("last_latent_norm", b2.expert.facts)


class TestCLI(unittest.TestCase):
    def _run(self, *argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_main(list(argv))
        return rc, buf.getvalue()

    def test_commands(self):
        rc, out = self._run("status"); self.assertEqual(rc, 0); self.assertIn("memory_patterns", out)
        rc, out = self._run("rules"); self.assertIn("visual-surprise", out)
        rc, out = self._run("ask", "Is there high surprise?"); self.assertIn("visual-surprise", out)
        rc, out = self._run("infer"); self.assertIn("No rule fired", out)


if __name__ == "__main__":
    unittest.main()
