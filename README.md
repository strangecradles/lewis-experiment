# Lewis Experiment: Superadditive Composition of Vision Models

Testing whether independently-trained vision models (DINOv2, SigLIP, MAE) exhibit **superadditive** performance when composed via learned connectors.

## Core Hypothesis

When you compose three diverse vision models with cross-attention connectors, the whole is **greater than the sum of its parts**. Mathematically:

```
I₃(A,B,C) = P(ABC) - [P(AB) + P(AC) + P(BC)] + [P(A) + P(B) + P(C)] > 0
```

In plain English: the composed system answers questions that **none of the component models can answer alone**.

## Quick Start

### Local Development
```bash
python run_all.py --device mps --dry-run --num-train 1000 --num-eval 100
```

### Remote GPU (Vast.ai B200, ~5 hours, ~$20)
```bash
# See VASTAI_QUICKSTART.md for detailed setup
git clone https://github.com/strangecradles/lewis-experiment.git
cd lewis-experiment
python run_all.py --device cuda --results-dir results/
python analyze.py results/
```

## Project Structure

```
lewis/
  models.py      — ModelBank: DINOv2, SigLIP, MAE (frozen)
  connectors.py  — Cross-attention connectors, ConnectorBank, TaskHead
  dataset.py     — GQA loading + question capability classification
  config.py      — 63 experimental conditions (7 main + 6 controls × 3 seeds)
  train.py       — Training loop (connectors only, models frozen)
  evaluate.py    — I₃ computation, all derived metrics
  utils.py       — Seeding, device, logging, result I/O

run_all.py       — Main orchestration: load models once, loop 63 conditions
analyze.py       — Post-hoc analysis, plots, statistical tests

.claude/         — Claude Code environment (CLAUDE.md, commands, rules)
```

## Architecture

### Vision Models (Frozen)
- **DINOv2 ViT-S/14** (22M params) → spatial structure, boundaries
- **SigLIP ViT-B/16** (86M params) → semantic categories, language-aligned
- **MAE ViT-B/16** (86M params) → texture, color, material properties

### Connectors (Trained)
- **Cross-attention connectors** (bidirectional, 2-layer, 4 heads)
- Each connector: ~5M params, learns how to enrich one model's representations with another's
- **TaskHead**: 2-layer MLP (512 hidden) from concatenated CLS tokens → 1500 answer logits

### Experimental Conditions (63 total)

**Main (21):**
- Single: A, B, C
- Pairs: AB, AC, BC
- Triplet: ABC
- × 3 random seeds

**Controls (42):**
- Concatenation (no connectors): all 7 subsets
- MLP connectors (same param count): AB, AC, BC, ABC
- Single large model: DINOv2 ViT-B/14
- Ensemble (3 independent models, logit averaging)
- Diversity control: 3× DINOv2 copies with different seeds
- × 3 seeds where applicable

## Key Metrics

### Interaction Terms
- **I₃(A,B,C)**: Three-way interaction (main hypothesis)
- **I₂(A,B)**, **I₂(A,C)**, **I₂(B,C)**: Pairwise interactions
- Computed via inclusion-exclusion principle

### Derived Metrics
- **delta_scale**: P(ABC composed) - P(single large) — does composition beat scaling?
- **delta_ensemble**: P(ABC composed) - P(ABC ensemble) — do learned connectors beat logit averaging?
- **delta_lewis**: I₃(with connectors) - I₃(concatenation only) — does learning connectors help?
- **delta_diversity**: I₃(diverse trio) - I₃(DINOv2 copies) — does diversity drive superadditivity?

### Per-Question-Type
- Spatial (left, right, above, below, position)
- Semantic (what, which, how many, type, category)
- Material (texture, color, composition, made of)
- Compound (requiring 2-3 capabilities)

## Usage

### Full experiment (B200, ~5-6 hours)
```bash
python run_all.py --device cuda --results-dir results/
```

### Dry run (validate pipeline, ~2 min)
```bash
python run_all.py --device cuda --dry-run --num-train 1000 --num-eval 100
```

### Monitor training
```bash
tail -f results/training.log
nvidia-smi -l 1
```

### Analyze results
```bash
python analyze.py results/
# Outputs:
#   results/metrics_summary.json
#   results/figures/*.png (publication-ready plots)
```

## Expected Results

**Based on Lewis hypothesis:**
- **I₃ > 0** for all 7 main conditions (superadditive)
- **I₃ largest** for compound questions (requiring all 3 capabilities)
- **delta_lewis > 0** (connectors outperform concatenation)
- **delta_scale > 0** (composition beats single large model)
- **delta_diversity > 0** (diversity drives superadditivity)

**Training:**
- ~5-10 min per condition (connectors only, models frozen)
- Early stopping typically kicks in at epoch 3-5
- Final accuracy ranges: 55-75% depending on question type

## Paper Connection

This experiment tests the **Lewis hypothesis** from recent work on AI agent composition and credit assignment:

> *"When you combine agents with complementary capabilities and connect them via learned attention mechanisms, emergent behaviors arise that neither agent exhibits alone."*

Key papers:
- "Social Environment Design" (ICML 2024)
- "Transcendence" (NeurIPS 2024)
- Large Legislative Models (arXiv 2410.08345)

## Isara

This connects directly to Isara's core mission:

- **Coordination**: Cross-attention connectors are a learned coordination protocol
- **Credit assignment**: I₃ terms quantify how much each model contributes to emergent capability
- **Scaling**: Isara manages 1000+ agents; this is a controlled experiment on 3

## Hardware Requirements

### Minimum
- 1x GPU with 24GB VRAM (A100, H100, RTX 4090)
- 8 CPU cores, 32GB RAM
- ~50GB disk (models + data + results)

### Recommended
- 1x B200 or H200 (179GB or 141GB VRAM) — overkill, but fast
- Vast.ai B200 @ $3.81/hr (allocated only during training)

## Troubleshooting

See **VASTAI_QUICKSTART.md** for remote GPU setup and debugging.

**Common issues:**
- GQA dataset download failures → pre-download with `datasets load_dataset('gqa', 'balanced')`
- CUDA OOM → check model loading, shouldn't happen with 24GB+
- Hanging on condition loop → check disk space, results/ directory writable

## Development

### Run tests
```bash
python -m pytest tests/
```

### Add a new experimental condition
Edit `lewis/config.py` → add entry to `get_all_conditions()` → rerun `run_all.py`

### Extend metrics
Edit `lewis/evaluate.py` → add function to `compute_all_metrics()` → update `analyze.py`

## Next Steps (Post-Experiment)

1. **Verify hypothesis**: Check `results/metrics_summary.json` — is I₃ > 0?
2. **Analyze by question type**: Which capabilities drive superadditivity?
3. **Scale to 100+ agents**: Can this pattern hold with larger agent swarms?
4. **Learn connector structure**: What does the learned attention matrix look like?
5. **Publish**: Draft paper on emergent composition in diverse agent swarms

---

**GitHub**: https://github.com/strangecradles/lewis-experiment

**Author**: Ashton Perlroth (Yale Physics, Isara Internship 2026)

**Status**: Ready for GPU execution. Tested on M1/M2 macOS (mps), pending B200 validation.
