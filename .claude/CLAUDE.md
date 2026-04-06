# Lewis Experiment

Measuring superadditive composition of independently trained vision models through learned connective tissue.

## Core Thesis
When independently trained models are composed through learned cross-attention connectors,
the resulting system exhibits superadditive performance: I₃ > 0 (three-way interaction term).

## Tech Stack
- Python 3.11+, PyTorch, timm, transformers, HuggingFace datasets
- Target hardware: NVIDIA GH200 (141GB unified memory)
- Models: DINOv2 ViT-S/14 (22M), SigLIP ViT-B/16 (86M), MAE ViT-B/16 (86M) — all frozen

## Project Structure
```
lewis/
  models.py       — Model loading + feature extraction (frozen)
  connectors.py   — Cross-attention connector architecture
  dataset.py      — GQA data loading + question type classification
  config.py       — All 39 experimental conditions
  train.py        — Training loop (connectors + task head only)
  evaluate.py     — Compute I₃, pairwise terms, all metrics
  utils.py        — Logging, seeding, device setup
run_all.py        — Main script: loops all conditions on GH200
analyze.py        — Post-hoc analysis: tables, plots, statistical tests
```

## Quick Commands
- `python -m pytest tests/ -v` — run tests
- `python run_all.py --dry-run` — validate pipeline without training
- `python run_all.py --device cuda` — full experiment run
- `python analyze.py results/` — generate analysis from saved results

## Key Design Decisions
- See docs/EXPERIMENT.md for full experimental design
- See docs/DECISIONS.md for architectural choices
- All models loaded ONCE, kept in memory across all 39 conditions
- Each condition: independent connector init + task head init + train + eval + save
