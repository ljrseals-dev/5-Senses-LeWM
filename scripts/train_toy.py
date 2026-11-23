#!/usr/bin/env python3
"""
Train the educational LeWM-style Eyes on BallWorld, then:
  1. check the learned latent against the labelled physics (probe + counterfactual laws),
  2. ground the classical expert rules in those measurements,
  3. calibrate surprise and test it on real vs. teleporting transitions,
  4. plan in latent space and *verify the plan in the true simulator*,
and write an auditable JSONL trail.

    python scripts/train_toy.py [--steps N] [--device cuda] [--out runs/toy]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from body_of_secrets.brain import ClassicalBrain                      # noqa: E402
from senses.audit import append_jsonl                                 # noqa: E402
from senses.config import load_config                                 # noqa: E402
from senses.world_model import toy_env as te                          # noqa: E402
from senses.world_model.adapter import WorldModelEye                  # noqa: E402
from senses.world_model.data import PairDataset                       # noqa: E402
from senses.world_model.jepa import LeWMStyle                         # noqa: E402
from senses.world_model.physics import PhysicsGrounder                # noqa: E402
from senses.world_model.trainer import train_model                    # noqa: E402


def ensure_data(cfg, path, regen, seed):
    if regen or not os.path.exists(path):
        d = cfg["data"]
        env_cfg = te.BallWorldConfig(image_size=d["image_size"])
        print(f"Generating {d['n_trajectories']} trajectories -> {path}")
        data = te.collect(d["n_trajectories"], d["trajectory_length"], env_cfg, seed=seed)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        te.save_dataset(path, data, env_cfg)
    return te.load_dataset(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--device")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out")
    ap.add_argument("--data", help="path to a BallWorld .npz (generated if missing)")
    ap.add_argument("--regen", action="store_true", help="regenerate the dataset")
    args = ap.parse_args(argv)

    cfg = load_config(args.config if os.path.exists(args.config) else None)
    tr = cfg["training"]
    steps = args.steps or tr["max_steps"]
    bs = args.batch_size or tr["batch_size"]
    seed = tr["seed"] if args.seed is None else args.seed
    out = args.out or tr["out_dir"]
    dev = args.device or tr["device"]
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; using CPU")
        dev = "cpu"
    os.makedirs(out, exist_ok=True)
    audit = os.path.join(out, "audit.jsonl")
    torch.manual_seed(seed)

    path = args.data or os.path.join(cfg["data"]["toy_trajectories"], "ballworld.npz")
    data, env_cfg = ensure_data(cfg, path, args.regen, seed)
    cfg["data"]["image_size"] = env_cfg.image_size
    n = len(data["obs"])
    n_test = max(2, int(cfg["data"]["test_fraction"] * n))
    train = {k: v[: n - n_test] for k, v in data.items()}
    test = {k: v[n - n_test:] for k, v in data.items()}
    print(f"Device: {dev} | train traj: {len(train['obs'])} | held-out traj: {len(test['obs'])}")

    model = LeWMStyle.from_config(cfg)
    print(f"Model parameters: {model.num_parameters():,}")
    ds = PairDataset(train["obs"], train["actions"])
    history = train_model(model, ds, steps=steps, batch_size=bs, lr=tr["lr"], device=dev,
                          log_every=tr["log_every"], seed=seed)
    eye = WorldModelEye(model, device=dev)

    # ---- calibrate surprise on (a subset of) normal training transitions
    sub = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:1024]
    eye.calibrate(ds.obs_t[sub].numpy().astype(np.float32) / 255, ds.actions[sub].numpy(),
                  ds.obs_tp1[sub].numpy().astype(np.float32) / 255)
    ckpt = os.path.join(out, "lewm_toy.pt")
    eye.save_checkpoint(ckpt)
    print(f"Saved checkpoint -> {ckpt}")

    # ---- 1) probe the learned physics against labelled held-out data
    report = PhysicsGrounder(eye, seed=seed).run(train, test)
    print(f"\nPhysics grounding (probe R² on held-out frames = {report.probe_r2:.3f})")
    for l in report.laws:
        print(f"  {l.name:30s} score={l.score:.3f}  [{l.status.name}]")
    append_jsonl(audit, {"event": "physics_report", **report.to_dict()})

    # ---- 2) ground the symbolic rules
    brain = ClassicalBrain()
    brain.ingest_physics(report)
    print("\nRules fired after grounding:")
    fired = brain.infer()
    for f in fired or []:
        print("  " + f.explain())
    if not fired:
        print("  (none)")

    # ---- 3) surprise: real vs. teleporting transition
    ti, tt = 0, 3
    f0, f1 = test["obs"][ti, tt], test["obs"][ti, tt + 1]
    a = test["actions"][ti, tt][None]
    far_i = int(np.argmax(np.linalg.norm(test["states"][:, 1] - test["states"][ti, tt + 1], axis=1)))
    f_bad = test["obs"][far_i, 1]
    z_ok = float(eye.surprise_z(f0, a, f1)[0])
    z_bad = float(eye.surprise_z(f0, a, f_bad)[0])
    print(f"\nSurprise z-score: real transition = {z_ok:.2f} | teleport = {z_bad:.2f}")
    brain.ingest_sensory(eye.encode(f_bad)[0], label="teleport", surprise_z=z_bad)
    for f in brain.infer():
        if f.rule == "visual-surprise":
            print("  " + f.explain())
    append_jsonl(audit, {"event": "surprise", "z_real": z_ok, "z_teleport": z_bad})

    # ---- 4) plan in latent space, verify in the true simulator
    pl = cfg["planner"]
    H = pl["horizon"]
    errs = {"plan": [], "zero": [], "random": []}
    rng = np.random.default_rng(seed)
    for i in range(min(len(test["obs"]), 20)):
        t0 = 2
        if t0 + H >= test["obs"].shape[1]:
            break
        start, goal = test["obs"][i, t0], test["obs"][i, t0 + H]
        res = eye.plan(start, goal, audit_path=audit, horizon=H, n_candidates=pl["n_candidates"],
                       n_elite=pl["n_elite"], n_iters=pl["n_iters"], seed=pl["seed"] + i)
        goal_pos = test["states"][i, t0 + H]
        for name, acts in (("plan", res.actions.cpu().numpy()), ("zero", np.zeros((H, 2))),
                           ("random", rng.uniform(-1, 1, (H, 2)))):
            env = te.BallWorld(env_cfg)
            env.reset(test["states"][i, t0])
            for a_t in acts:
                env.step(a_t)
            errs[name].append(float(np.linalg.norm(env.pos - goal_pos)))
    print(f"\nLatent plans executed in the true simulator ({len(errs['plan'])} start/goal pairs):")
    for k, v in errs.items():
        print(f"  mean final distance to goal, {k:7s}: {np.mean(v):.4f}")
    append_jsonl(audit, {"event": "plan_eval", **{k: float(np.mean(v)) for k, v in errs.items()}})

    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump({"history": history, "physics": report.to_dict(),
                   "plan_eval": {k: float(np.mean(v)) for k, v in errs.items()}}, fh, indent=2)
    brain.save(os.path.join(out, "brain"))
    print(f"\nAudit trail: {audit}")


if __name__ == "__main__":
    main()
