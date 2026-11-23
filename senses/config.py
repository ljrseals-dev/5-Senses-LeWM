"""Config loading: defaults <- YAML file <- environment variables (see .env.example)."""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Mapping, Optional

DEFAULTS: Dict[str, Any] = {
    "model": {"latent_dim": 192, "action_dim": 2, "lambda_reg": 0.1, "n_projections": 256,
              "detach_target": False,
              "encoder": {"channels": [32, 64, 128]}, "predictor": {"hidden": 256}},
    "training": {"batch_size": 32, "lr": 3.0e-4, "max_steps": 2000, "device": "cpu",
                 "log_every": 50, "seed": 0, "out_dir": "runs/toy"},
    "planner": {"horizon": 8, "n_candidates": 64, "n_elite": 8, "n_iters": 4, "seed": 0},
    "data": {"image_size": 64, "toy_trajectories": "data/toy_trajectories",
             "n_trajectories": 160, "trajectory_length": 24, "test_fraction": 0.2},
    "checkpoint": None,
}


def _merge(base: Dict[str, Any], over: Mapping[str, Any]) -> Dict[str, Any]:
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Optional[str] = None, env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        import yaml
        with open(path, encoding="utf-8") as fh:
            _merge(cfg, yaml.safe_load(fh) or {})
    env = os.environ if env is None else env
    if env.get("DEVICE"):
        cfg["training"]["device"] = env["DEVICE"]
    if env.get("LATENT_DIM"):
        cfg["model"]["latent_dim"] = int(env["LATENT_DIM"])
    if env.get("LAMBDA_REG"):
        cfg["model"]["lambda_reg"] = float(env["LAMBDA_REG"])
    if env.get("LEWM_CHECKPOINT"):
        cfg["checkpoint"] = env["LEWM_CHECKPOINT"]
    return cfg
