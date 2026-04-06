# Lewis Experiment — Deployment Checklist

## ✅ Repository Ready

**GitHub**: https://github.com/strangecradles/lewis-experiment

**Status**: All code committed and pushed to `main` branch

```
d60ab02 Complete .gitignore
cbd6c89 Add comprehensive README
f3f1251 Add Vast.ai quickstart guide  
8982eb3 Initial Lewis experiment
```

## ✅ Code Structure

| File | Lines | Purpose |
|------|-------|---------|
| `lewis/models.py` | 344 | ModelBank: frozen vision models |
| `lewis/connectors.py` | 464 | Cross-attention connector architecture |
| `lewis/dataset.py` | 467 | GQA loading + question classification |
| `lewis/config.py` | 314 | All 63 experimental conditions |
| `lewis/train.py` | 283 | Training loop (AdamW, cosine, early stop) |
| `lewis/evaluate.py` | 356 | I₃ metrics + all derived metrics |
| `lewis/utils.py` | 200 | Logging, seeding, device detection |
| `run_all.py` | 389 | Main orchestration |
| `analyze.py` | 461 | Post-hoc analysis + plots |
| **Total** | **3,878** | **Production-ready** |

## ✅ Tested Locally

- ✅ All imports pass
- ✅ Forward pass works (B200 simulation)
- ✅ 63 conditions generate correctly
- ✅ Question classifier working
- ✅ Utils functions (logging, device detection, seeding)

## ✅ Documentation

- `README.md` — overview, architecture, usage
- `VASTAI_QUICKSTART.md` — step-by-step remote GPU setup
- `.claude/docs/EXPERIMENT.md` — full experimental design

## 🚀 To Execute on B200 (Vast.ai)

### 1. Rent GPU (5 min)
Go to https://cloud.vast.ai/create/
- Filter: B200, Alabama, 95%+ reliability
- Current best: 1x B200 @ $3.81/hr
- Click "Rent" → get SSH command

### 2. Clone & Setup (10 min)
```bash
ssh root@<vastai-ip>
cd /root
git clone https://github.com/strangecradles/lewis-experiment.git
cd lewis-experiment
pip install torch --index-url https://download.pytorch.org/whl/cu121 -q
pip install open_clip_torch timm peft accelerate trl datasets wandb -q
```

### 3. Run Experiment (5-6 hours)
```bash
python run_all.py --device cuda --results-dir results/ 2>&1 | tee experiment.log
```

### 4. Download Results (5 min)
```bash
# On your Mac:
rsync -avz root@<vastai-ip>:/root/lewis-experiment/results/ ./lewis_results/
```

### 5. Analyze Locally (2 min)
```bash
cd lewis-experiment
python analyze.py results/
```

**Total time**: ~6 hours GPU + 20 min setup/analysis
**Total cost**: ~$25 (B200) + $3 (bandwidth)

## 📊 Expected Outputs

After `analyze.py`:
- `results/metrics_summary.json` — all I₃ values
- `results/figures/i3_by_question_type.png` — publication-ready bar chart
- `results/figures/interaction_heatmap.png` — all terms by question type
- `results/figures/composition_boxplot.png` — hierarchy analysis

## 🎯 Success Criteria

Hypothesis **confirmed** if:
1. ✅ I₃(ABC) > 0 for all seeds
2. ✅ I₃ largest for compound questions
3. ✅ I₃(connectors) > I₃(concatenation)
4. ✅ I₃(ABC) > I₂ pairs

## 📝 Next Steps

1. Rent B200 GPU on Vast.ai
2. Follow `VASTAI_QUICKSTART.md`
3. Run `python run_all.py --device cuda`
4. Download results
5. Run `python analyze.py results/`
6. Check `results/metrics_summary.json`
7. Write up findings

---

**Repo**: https://github.com/strangecradles/lewis-experiment  
**Ready**: YES ✅
**Estimated cost**: $25-30  
**Estimated duration**: 6 hours
