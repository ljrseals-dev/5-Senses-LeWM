#!/usr/bin/env python3
"""
Simple educational chatbot about JEPA / LeWorldModel / SIGReg / Epps-Pulley.
Intentionally under-developed: keyword matching only, no real NLP, no memory.
"""

from __future__ import annotations

from typing import Dict

KNOWLEDGE: Dict[str, str] = {
    "jepa": (
        "JEPA = Joint-Embedding Predictive Architecture (Yann LeCun, 2022). "
        "It predicts abstract embeddings of the future/target instead of pixels or tokens."
    ),
    "leworldmodel": (
        "LeWorldModel (LeWM) is a 2026 JEPA world model by Maes, Le Lidec, Scieur, LeCun & Balestriero. "
        "It trains stably end-to-end from pixels with only two loss terms: prediction + SIGReg. "
        "Paper: arXiv:2603.19312. Code: https://github.com/lucas-maes/le-wm"
    ),
    "sigreg": (
        "SIGReg = Sketched Isotropic Gaussian Regularizer. "
        "It projects embeddings onto many random 1-D directions and applies the Epps-Pulley "
        "normality test on each projection, forcing the latent distribution toward N(0, I)."
    ),
    "epps": (
        "Epps-Pulley (1983) is a normality test based on the empirical characteristic function. "
        "It measures the weighted L2 distance between the sample ECF and the CF of a standard normal."
    ),
    "collapse": (
        "Representation collapse happens when the encoder maps every input to (almost) the same vector. "
        "Prediction becomes trivial but the embeddings become useless. SIGReg prevents this."
    ),
    "help": (
        "I know about: jepa, leworldmodel, sigreg, epps, collapse, citation. "
        "Type 'quit' or 'exit' to leave."
    ),
    "citation": (
        "Maes et al. (2026). LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels. "
        "arXiv:2603.19312\n"
        "Epps & Pulley (1983). A test for normality based on the empirical characteristic function. Biometrika."
    ),
}


def respond(user_input: str) -> str:
    text = user_input.lower().strip()
    if not text:
        return "Say something about JEPA, SIGReg, Epps-Pulley, etc."

    for key, answer in KNOWLEDGE.items():
        if key in text:
            return answer

    return (
        "I only understand a few keywords right now (jepa, leworldmodel, sigreg, epps, collapse, citation, help). "
        "This chatbot is intentionally under-developed."
    )


def main():
    print("Educational JEPA / LeWorldModel chatbot (very basic version)")
    print("Type 'help' for keywords, 'quit' to exit.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if user.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break

        print("Bot:", respond(user))
        print()


if __name__ == "__main__":
    main()