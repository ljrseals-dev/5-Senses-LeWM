# 5-Senses-LeWM

**Classical AI laboratory (inspired by 1980s research) augmented with a modern pixel-based world model.**

This repository extends [ljrseals-dev/5-Senses](https://github.com/ljrseals-dev/5-Senses) with a sensory **"Eyes"** layer based on **LeWorldModel**, and uses what the Eyes learn to *ground* the classical expert rules in measured physics.

Classical components stay offline-first, transparent and provenance-aware. The world model is an optional plugin.

---

## Research Citation

The world-model component is based on:

> Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., & Balestriero, R. (2026).
> *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*.
> arXiv:2603.19312. https://arxiv.org/abs/2603.19312

Official implementation: [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) (MIT License)

```
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

Please cite the paper if you build on the world-model ideas. A `CITATION.cff` is included.

---

## Overview

| Layer | Technology | Role |
| --- | --- | --- |
| Classical core (`body_of_secrets/`, numpy only) | Hopfield memory, forward-chaining rule engine, truth-status provenance | Transparent reasoning, offline-first |
| Eyes (`senses/world_model/`) | Educational LeWM-style JEPA (~1.9M params) | Raw-pixel physics prediction, surprise, latent planning |
| Bridge | `WorldModelEye`, `PhysicsGrounder` | Latents into memory; measured physical laws into expert facts |

The official LeWorldModel (~15M parameters, single GPU, per the paper) is the production path; the educational model here exists so everything runs and is testable on a laptop.

### What the Eyes do

1. **Predict physics from pixels.** Encoder + action-conditioned predictor, trained end-to-end with `prediction MSE + λ·SIGReg` and no stop-gradient (`detach_target: false`, as in the paper).
2. **Detect implausible events.** `calibrate()` records the prediction-error distribution on normal transitions; `surprise_z()` z-scores new ones.
3. **Plan in latent space (auditable).** CEM planner returns a `PlanResult` with cost history, parameters and seed; plans can be appended to a JSONL audit trail.
4. **Ground the expert rules.** `PhysicsGrounder` fits a linear probe latent→position, then asks the model counterfactual questions and compares them with dynamics estimated from the labelled data:
   - `action_response` — does action *a* move the object by *A·a*?
   - `drift` — does an unforced object drift (gravity)?
   - `wall_blocking` — are objects stopped at the boundary?
   - `surprise_detects_violation` — do teleports cost more than real motion?

   Scores become facts with a truth status (never above `PLAUSIBLE`: learned-model evidence is not consensus). Rules such as `plan-grounded` (`min_law_score >= 0.8`) and `plan-ungrounded` (`min_law_score < 0.5`) then fire from them.

---

## Quick start

```
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

make test        # unit tests (torch tests skip if torch is missing)
make train       # trains on BallWorld, grounds physics, calibrates surprise, plans, verifies plans in the simulator
make demo        # classical brain + Eyes (uses runs/toy/lewm_toy.pt if present)

python -m body_of_secrets.cli status
python -m body_of_secrets.cli ask "Is there high surprise?"
python scripts/chatbot.py
```

`make train` writes `runs/toy/` (checkpoint, `metrics.json`, `audit.jsonl`, brain state). Settings come from `configs/default.yaml`, overridden by the environment variables in `.env.example` (`DEVICE`, `LATENT_DIM`, `LAMBDA_REG`, `LEWM_CHECKPOINT`), overridden by CLI flags (`--steps`, `--device`, `--out`, ...).

### Your own data

* **Labelled trajectories** (obs `(n, T+1, 3, H, W)` uint8, actions, states): save with `toy_env.save_dataset`-style `.npz` and pass `--data`. Probing needs the labels.
* **Unlabelled image folders**: `data.frames_from_folder(...)` + `data.pairs_from_frames(...)` (needs `pip install pillow`). Without recorded actions the model learns passive dynamics only, and physics probing is not possible. Decode video to frames first (e.g. with ffmpeg).

---

## Using the real LeWorldModel

The official code builds on `stable-worldmodel` / `stable-pretraining`; its checkpoints ship as `<name>_object.ckpt` (loaded via the `stable_worldmodel` API) and `<name>_weight.ckpt` (state_dict).

```
uv pip install stable-worldmodel[train,env]      # see lucas-maes/le-wm
```

`senses.world_model.adapter.WorldModelEye` accepts any *backend* with `latent_dim`, `action_dim`, `encode(obs) -> z` and `predict(z, action) -> z`. Wrap an official model with `CallableBackend(encode_fn, predict_fn, latent_dim, action_dim)` and the surprise, planning, and grounding code works unchanged. **That glue is not written or tested here**: check the official `jepa.py` for the exact encode/predict signatures (it may expect history windows and its own preprocessing). `save_checkpoint`/`from_checkpoint` are for the educational model only; they load weights-only (no pickle code execution).

---

## Directory layout

```
5-Senses-LeWM/
├── body_of_secrets/           # classical core (numpy only)
│   ├── truth.py               # OFFICIAL | CONSENSUS | PLAUSIBLE | SPECULATIVE | UNKNOWN
│   ├── memory.py              # Hopfield memory (recall with overlap/energy, save/load)
│   ├── expert.py              # rules, facts, forward chaining, provenance, audit
│   ├── brain.py               # goal, surprise, physics facts, persistence
│   └── cli.py                 # status | rules | infer | ask  (stdlib argparse)
├── senses/
│   ├── config.py  audit.py
│   └── world_model/
│       ├── jepa.py  sigreg.py  planner.py  adapter.py
│       ├── toy_env.py         # BallWorld physics + dataset generator (numpy)
│       ├── physics.py         # probe + counterfactual law checks (numpy)
│       ├── data.py  trainer.py
├── configs/default.yaml  scripts/  tests/  data/toy_trajectories/
```

---

## Status and limits

* The numpy side (classical core, toy world, physics grounding, CLI, config) is covered by unit tests that pass.
* The torch side (`jepa`, `sigreg`, `planner`, `adapter`, `data`, `trainer`, `scripts/train_toy.py`) has tests in `tests/test_torch.py`, but these have **not been run by the author of this revision** (no torch in the authoring sandbox). Run `make test` first; treat the first `make train` as the real check. Whether a given training run reaches high law scores is an empirical question the grounding report answers.
* BallWorld is a deliberately simple first-order world (action, drift, walls). High scores there say nothing about real video.
* Hopfield memory binarises latents by sign; capacity is about 0.138·dim patterns.

---

## License

MIT (see `LICENSE`). The educational code is independent of the official LeWorldModel repository. Always cite the original paper when the world-model ideas are used.

## Acknowledgements

- Original 5-Senses / BODY OF SECRETS classical laboratory: [ljrseals-dev/5-Senses](https://github.com/ljrseals-dev/5-Senses)
- LeWorldModel research & official code: Maes, Le Lidec, Scieur, LeCun, Balestriero (2026) — [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)
- JEPA conceptual foundation: Yann LeCun, *A Path Towards Autonomous Machine Intelligence* (2022)
