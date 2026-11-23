# 5-Senses-LeWM

**The Eyes module for BODY OF SECRETS / 5-Senses.**

This repository contains an experimental visual-sensing layer built around a small, educational LeWorldModel-style JEPA. Its role is to encode observations, predict latent transitions conditioned on actions, and expose prediction error for downstream anomaly triage. It is a module within the larger system, not a replacement for the classical Brain, Memory, or Archive.

## System boundary

- **Eyes (this repository):** image observations, latent representations, action-conditioned prediction, planning experiments, and transition-error signals.
- **Brain:** interprets signals and applies transparent expert rules.
- **Memory:** stores/retrieves representations; it must not treat an embedding as a verified factual claim.
- **Archive:** stores documents and source-backed evidence. Visual predictions do not become archive facts without independent validation and provenance.

The Eyes module must report model outputs as predictions, not observations or verified facts. A high prediction error is an investigation signal, not proof of an anomaly or cause.

## Research basis

Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., & Balestriero, R. (2026). *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*. arXiv:2603.19312. https://arxiv.org/abs/2603.19312

Official implementation: https://github.com/lucas-maes/le-wm

This repository is an independent educational implementation, not the official LeWorldModel code or a reproduction of its full training pipeline. Use the upstream implementation and its documented checkpoints for research-grade experiments.

## Current capabilities and limits

| Component | Current status | Important boundary |
|---|---|---|
| CNN encoder | Implemented | Expects floating-point RGB tensors shaped (B, 3, H, W) |
| Action predictor | Implemented | Fixed action dimension; no environment adapter included |
| SIGReg | Educational approximation | Random projection sketch; not claimed numerically identical to upstream |
| Latent CEM | Implemented | Optimizes latent goal distance; no safety or task constraints |
| Eye adapter | Implemented | Exposes embeddings, predictions, and prediction-error score |
| Training data | Toy generator only | Synthetic noise is not physical-world training data |
| Checkpoints | Not included | Fresh model weights are untrained until a user trains them |

The toy trainer is a smoke-test/example, not evidence that the model has learned real-world physics. A useful deployed Eye requires representative, correctly aligned observation/action/next-observation trajectories, held-out evaluation, and a versioned checkpoint with provenance.

## Install

Python 3.10 or newer is required.

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[dev]"
    python -m pytest -q

## Run examples

    python scripts/demo_integration.py
    python scripts/train_toy.py

The demo uses randomly initialized weights and synthetic observations. It demonstrates API wiring only; it does not demonstrate visual recognition, learned physics, or validated anomaly detection.

## API sketch

    import torch
    from senses.world_model.adapter import WorldModelEye

    eye = WorldModelEye(device="cpu")
    observation = torch.zeros(1, 3, 64, 64)
    latent = eye.encode(observation)
    action = torch.zeros(1, eye.model.action_dim)
    predicted_next_latent = eye.predict_next(latent, action)

WorldModelEye.surprise(obs_t, action, obs_tp1) returns transition prediction MSE. SIGReg is a training-time batch-distribution regularizer and is excluded from this per-transition score.

## License and attribution

MIT; see LICENSE. See CITATION.cff for citation metadata. Cite the original LeWorldModel paper when building on its research ideas.
