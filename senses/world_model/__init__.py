"""
Educational LeWorldModel-style JEPA world model (Eyes).

Inspired by:
Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., & Balestriero, R. (2026).
LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.
arXiv:2603.19312. https://arxiv.org/abs/2603.19312

Official code: https://github.com/lucas-maes/le-wm
"""

from .jepa import LeWMStyle, SimpleEncoder, ARPredictor
from .sigreg import SIGReg, EppsPulley
from .adapter import WorldModelEye
from .planner import LatentCEMPlanner

__all__ = [
    "LeWMStyle",
    "SimpleEncoder",
    "ARPredictor",
    "SIGReg",
    "EppsPulley",
    "WorldModelEye",
    "LatentCEMPlanner",
]