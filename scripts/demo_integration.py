#!/usr/bin/env python3
"""Minimal demo: classical brain + world-model eye."""

from __future__ import annotations

import torch
from senses.world_model.adapter import WorldModelEye
from body_of_secrets.brain import ClassicalBrain


def main():
    print("=== 5-Senses-LeWM integration demo ===\n")

    eye = WorldModelEye(device="cpu")
    brain = ClassicalBrain()

    # fake observation (batch of 1, 3-channel 64×64)
    obs = torch.randn(1, 3, 64, 64)
    z = eye.encode(obs)
    print(f"Latent shape from Eyes: {tuple(z.shape)}")

    # classical reasoning can now use the embedding
    brain.ingest_sensory(z, label="demo_obs")
    print("Brain status:", brain.status())
    print("Brain answer:", brain.ask("What do I see?"))
    print("Brain answer:", brain.ask("Is there high surprise?"))

    # also show a one-step prediction
    action = torch.zeros(1, 2)
    z_next = eye.predict_next(z, action)
    print(f"Predicted next latent shape: {tuple(z_next.shape)}")

    print("\nDemo finished successfully.")


if __name__ == "__main__":
    main()