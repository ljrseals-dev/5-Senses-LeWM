#!/usr/bin/env python3
"""Demo: classical brain + world-model Eyes. Uses a trained checkpoint if one exists."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from body_of_secrets.brain import ClassicalBrain          # noqa: E402
from senses.config import load_config                     # noqa: E402
from senses.world_model import toy_env as te              # noqa: E402
from senses.world_model.adapter import WorldModelEye      # noqa: E402


def main():
    print("=== 5-Senses-LeWM integration demo ===\n")
    cfg = load_config("configs/default.yaml" if os.path.exists("configs/default.yaml") else None)
    ckpt = cfg["checkpoint"] or os.path.join(cfg["training"]["out_dir"], "lewm_toy.pt")
    if os.path.exists(ckpt):
        eye = WorldModelEye.from_checkpoint(ckpt, device=cfg["training"]["device"])
        print(f"Loaded trained Eyes from {ckpt}")
    else:
        eye = WorldModelEye(device=cfg["training"]["device"])
        print("No checkpoint found (run `make train` first): using an UNTRAINED model, so the "
              "numbers below are only a plumbing check.")

    env = te.BallWorld(te.BallWorldConfig(image_size=eye.model.image_size), seed=0)
    brain = ClassicalBrain()
    pos = env.reset()
    frame = env.render()
    z = eye.encode(frame)
    print(f"Latent shape from Eyes: {tuple(z.shape)}")

    goal = te.BallWorld(env.cfg).render(np.array([0.8, 0.8]))
    brain.set_goal(eye.encode(goal)[0], tolerance=0.5)
    brain.ingest_sensory(z[0], label="demo_obs")
    print("Brain status:", brain.status())
    print("Brain answer:", brain.ask("What do I see?"))
    print("Brain answer:", brain.ask("Is there high surprise?"))

    plan = eye.plan(frame, goal, horizon=cfg["planner"]["horizon"], seed=0)
    print(f"Plan: cost {plan.initial_cost:.3f} -> {plan.cost:.3f} (lower is closer to the goal)")
    print("\nDemo finished.")


if __name__ == "__main__":
    main()
