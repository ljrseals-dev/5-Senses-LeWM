# 5-Senses-LeWM

**Classical AI laboratory (inspired by 1980s research) augmented with a modern pixel-based world model.**

This repository extends the design philosophy of [ljrseals-dev/5-Senses](https://github.com/ljrseals-dev/5-Senses) by adding a sensory **“Eyes”** layer based on **LeWorldModel**.

Classical components remain offline-first, transparent, provenance-aware and lightweight.  
The world model learns physics from raw pixels with a stable two-term objective (prediction + SIGReg).

---

## Research Citation

The world-model component is based on:

> Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., & Balestriero, R. (2026).  
> *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*.  
> arXiv:2603.19312. https://arxiv.org/abs/2603.19312

Official implementation: [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) (MIT License)

Please cite the original paper if you use or build upon the world-model ideas in this repository.

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026},
  eprint={2603.19312},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2603.19312}
}
```

A `CITATION.cff` file is included so GitHub can show a “Cite this repository” button.

---

## Overview

| Layer | Technology | Role |
|-------|------------|------|
| Classical core | Hopfield-style associative memory, rule engine, provenance | Transparent reasoning, offline-first |
| Eyes (world model) | Simplified LeWM-style JEPA | Raw-pixel physics prediction & latent planning |
| Bridge | `WorldModelEye` adapter | Feeds latent embeddings into the classical brain |

- Total educational model size stays small (~few million parameters).
- Official LeWorldModel (~15 M parameters, single-GPU) is the recommended production path.
- This repo supplies a clean integration layer + educational implementation so you can experiment immediately.

---

## Quick start

```bash
# create & activate a virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

pip install -e ".[dev]"

# classical status
python -m body_of_secrets.cli status

# end-to-end demo (classical brain + world-model eye)
python scripts/demo_integration.py

# toy training loop
python scripts/train_toy.py
```

Or simply:

```bash
make install
make demo
make train
```

---

## Using the real LeWorldModel

```bash
# install official stack (see lucas-maes/le-wm README)
uv pip install stable-worldmodel[train,env]
# or follow the official instructions

# then point senses/world_model/adapter.py at a checkpoint
# or replace the educational model with the official one
```

---

## Directory layout

```text
5-Senses-LeWM/
├── README.md
├── LICENSE                    # MIT
├── CITATION.cff
├── pyproject.toml
├── .env.example
├── Makefile
├── body_of_secrets/           # classical core (minimal stubs)
│   ├── __init__.py
│   ├── brain.py
│   ├── memory.py
│   ├── expert.py
│   └── cli.py
├── senses/
│   ├── __init__.py
│   └── world_model/
│       ├── __init__.py
│       ├── jepa.py            # simplified LeWM-style model
│       ├── sigreg.py          # SIGReg + Epps-Pulley
│       ├── planner.py         # latent CEM planner
│       └── adapter.py         # bridge to classical brain
├── configs/
│   └── default.yaml
├── scripts/
│   ├── train_toy.py
│   └── demo_integration.py
└── data/
    └── toy_trajectories/      # placeholder
```

---

## Philosophy

Teach the system to *see* and predict physics (world model) while keeping reasoning transparent and classical.

- Classical core (Hopfield-style associative memory, rule engine, provenance, offline-first) stays light and auditable.
- World model is treated as an optional “Eyes” plugin: it can run on its own or feed latent physics embeddings into the classical brain.
- Ground symbolic expert rules in learned physical laws extracted from video or image data.

---

## Key technical notes

### Training objective (educational simplification of LeWM)

\[
\mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda \cdot \text{SIGReg}(Z)
\]

- \(\mathcal{L}_{\text{pred}}\): MSE between predicted next embedding and actual next embedding.
- **SIGReg** (Sketched Isotropic Gaussian Regularizer): projects embeddings onto many random 1-D directions and applies a differentiable Epps–Pulley-style normality statistic so the latent distribution stays close to an isotropic Gaussian. This is the principal anti-collapse mechanism.

The educational implementation in `senses/world_model/` is deliberately simple and fully end-to-end.  
For production performance, use the official code and checkpoints from Maes et al. (2026).

### Planning

At inference the planner rolls out candidate action sequences entirely in latent space (Cross-Entropy Method / CEM) and selects the sequence whose final predicted embedding is closest to a goal embedding.

---

## License

MIT (see `LICENSE`).

The educational code is independent of the official LeWorldModel repository.  
Always cite the original paper when the world-model ideas are used.

---

## Acknowledgements

- Original 5-Senses / BODY OF SECRETS classical laboratory: [ljrseals-dev/5-Senses](https://github.com/ljrseals-dev/5-Senses)
- LeWorldModel research & official code: Maes, Le Lidec, Scieur, LeCun, Balestriero (2026) — [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)
- JEPA conceptual foundation: Yann LeCun, *A Path Towards Autonomous Machine Intelligence* (2022)