#!/usr/bin/env python3
"""
Small offline Q&A bot about JEPA / LeWorldModel / SIGReg / Epps-Pulley and this repo.

Still deterministic keyword retrieval (no LLM, no network) — but with aliases, scored matching,
multi-topic answers, and a fallback to the classical brain's rules.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from body_of_secrets.brain import ClassicalBrain  # noqa: E402

# topic -> (aliases, answer)
TOPICS: Dict[str, Tuple[List[str], str]] = {
    "jepa": (["jepa", "joint-embedding", "joint embedding"],
             "JEPA = Joint-Embedding Predictive Architecture (LeCun, 2022): predict the embedding of the "
             "target (e.g. the next frame) instead of its pixels or tokens."),
    "leworldmodel": (["leworldmodel", "lewm", "le-wm", "world model", "worldmodel"],
                     "LeWorldModel (Maes, Le Lidec, Scieur, LeCun & Balestriero, 2026) is a JEPA that trains "
                     "end-to-end from raw pixels with two loss terms — next-embedding prediction and a Gaussian "
                     "regulariser (SIGReg) — with no stop-gradient or EMA. The paper reports ~15M parameters on "
                     "one GPU. arXiv:2603.19312; code: https://github.com/lucas-maes/le-wm"),
    "sigreg": (["sigreg", "regularizer", "regulariser", "isotropic"],
               "SIGReg (Sketched Isotropic Gaussian Regularizer) projects embeddings onto random 1-D directions "
               "and applies a normality test (Epps–Pulley) to each, pushing the latent distribution toward an "
               "isotropic Gaussian. This is what prevents collapse. senses/world_model/sigreg.py is an "
               "educational implementation."),
    "epps": (["epps", "pulley", "normality", "characteristic function"],
             "Epps–Pulley (1983) tests normality by comparing the sample's empirical characteristic function "
             "with that of a standard normal, as a weighted L2 distance."),
    "collapse": (["collapse", "collapsed", "degenerate"],
                 "Representation collapse: the encoder maps everything to (almost) the same vector, so "
                 "prediction is trivial but the embedding is useless. SIGReg is the anti-collapse term."),
    "hopfield": (["hopfield", "associative", "memory"],
                 "body_of_secrets/memory.py is a binary Hopfield network: Hebbian storage, asynchronous recall, "
                 "reliable capacity ≈ 0.138·dim patterns. Recall reports overlap, Hamming distance and energy."),
    "planner": (["planner", "planning", "cem", "cross-entropy", "plan"],
                "The planner (senses/world_model/planner.py) runs CEM in latent space: sample action sequences, "
                "roll them out with the predictor, keep those ending nearest the goal embedding. Every plan "
                "returns its cost history and parameters so it can be audited (and is checked against the true "
                "simulator in scripts/train_toy.py)."),
    "grounding": (["grounding", "ground", "probe", "physics", "laws"],
                  "Physics grounding (senses/world_model/physics.py): fit a linear probe latent→position, then "
                  "ask the model counterfactual questions (what does action a do? does an unforced object drift? "
                  "do walls block?) and compare against dynamics estimated from the labelled data. The scores "
                  "become facts with truth status (never above PLAUSIBLE) that the expert rules reason over."),
    "surprise": (["surprise", "anomaly", "implausible"],
                 "Surprise = latent prediction error on a transition. WorldModelEye.calibrate() records its "
                 "distribution on normal transitions; surprise_z() z-scores new ones, and the 'visual-surprise' "
                 "rule fires when z is high. The paper also evaluates surprise on physically implausible events."),
    "citation": (["citation", "cite", "reference"],
                 "Maes et al. (2026). LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture "
                 "from Pixels. arXiv:2603.19312.\nEpps & Pulley (1983). A test for normality based on the "
                 "empirical characteristic function. Biometrika."),
}

_brain: Optional[ClassicalBrain] = None


def respond(user_input: str, brain: Optional[ClassicalBrain] = None) -> str:
    text = user_input.lower().strip()
    if not text:
        return "Ask about: " + ", ".join(sorted(TOPICS)) + "."
    if text in {"help", "topics", "?"}:
        return "I know about: " + ", ".join(sorted(TOPICS)) + ". Type 'quit' to leave."

    scored = []
    for name, (aliases, answer) in TOPICS.items():
        hits = sum(1 for a in aliases if re.search(r"(?<![a-z])" + re.escape(a), text))
        if hits:
            scored.append((hits, name, answer))
    if scored:
        scored.sort(key=lambda t: (-t[0], t[1]))
        return "\n\n".join(a for _, _, a in scored[:2])

    b = brain or _brain or ClassicalBrain()
    if b.expert.query_rule(user_input) is not None:
        return b.expert.query(user_input)
    return "I don't know that one. Try: " + ", ".join(sorted(TOPICS)) + "."


def main():
    global _brain
    _brain = ClassicalBrain()
    print("5-Senses-LeWM Q&A (offline, keyword retrieval). Type 'help' for topics, 'quit' to exit.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if user.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break
        print("Bot:", respond(user, _brain), "\n")


if __name__ == "__main__":
    main()
