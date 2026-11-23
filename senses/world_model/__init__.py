"""
Educational LeWorldModel-style JEPA world model (Eyes).

Inspired by:
Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., & Balestriero, R. (2026).
LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.
arXiv:2603.19312. https://arxiv.org/abs/2603.19312

Official code: https://github.com/lucas-maes/le-wm

Submodules load lazily: ``toy_env`` and ``physics`` need only numpy; the rest need torch.
"""

import importlib

_LAZY = {
    "LeWMStyle": "jepa", "SimpleEncoder": "jepa", "ARPredictor": "jepa",
    "SIGReg": "sigreg", "EppsPulley": "sigreg",
    "WorldModelEye": "adapter", "CallableBackend": "adapter",
    "LatentCEMPlanner": "planner", "PlanResult": "planner",
    "BallWorld": "toy_env", "BallWorldConfig": "toy_env",
    "PhysicsGrounder": "physics", "PhysicsReport": "physics",
}
__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
