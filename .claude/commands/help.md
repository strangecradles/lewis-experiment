# Lewis Experiment — Environment Guide

## What This Project Does
Measures whether composing independently trained vision models (DINOv2, SigLIP, MAE)
through learned cross-attention connectors produces superadditive performance on GQA.

## Available Commands
- `/project:help` — this guide
- `/project:tasks` — manage TODO items
- `/project:run` — run the experiment

## Key Files
- `lewis/config.py` — all 39 experimental conditions
- `lewis/connectors.py` — the learned connective tissue
- `run_all.py` — main entry point
- `docs/EXPERIMENT.md` — full experimental design

## Architecture
Three frozen vision models → pairwise cross-attention connectors → concatenated CLS → task head → GQA answer
