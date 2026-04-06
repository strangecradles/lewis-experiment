# Lewis Experiment on Vast.ai

## TL;DR
```bash
# 1. Rent a B200 GPU on Vast.ai
# 2. SSH into the instance
# 3. Run this:

git clone https://github.com/strangecradles/lewis-experiment.git
cd lewis-experiment
pip install -q open_clip_torch timm peft accelerate trl datasets wandb
python run_all.py --device cuda --results-dir results/

# 4. When done, download results:
# rsync -avz vastai:/root/lewis-experiment/results ./results_local/
```

---

## Step-by-Step

### 1. Rent a B200 GPU on Vast.ai

Go to https://cloud.vast.ai/create/

**Filter for:**
- GPU: B200 (1x or 2x)
- Location: Any (Alabama is reliable and cheap)
- Min Reliability: 95%
- Sort by: Price ascending

**Current best deal:** 1x B200 in Alabama, ~$3.81/hr

Click "Rent" → pay with card → wait 30-60 seconds for instance to boot.

### 2. SSH into the instance

Vast.ai will give you SSH command like:
```bash
ssh -p 12345 root@1.2.3.4
```

(The credentials are in the "Manage" tab on Vast.ai)

### 3. Clone and setup

```bash
cd /root
git clone https://github.com/strangecradles/lewis-experiment.git
cd lewis-experiment
```

**Install dependencies** (takes ~3 min):
```bash
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q open_clip_torch timm peft accelerate trl datasets wandb
```

Check CUDA:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

Should print something like:
```
CUDA: True, Device: NVIDIA B200
```

### 4. Run the experiment

```bash
python run_all.py --device cuda --results-dir results/ 2>&1 | tee experiment.log
```

**Expected output:**
- Loads 3 frozen vision models (~30 sec)
- Loops through 63 experimental conditions
- Each condition trains for ~5-10 min
- **Total: ~5-6 hours**

**To monitor in another terminal:**
```bash
# Watch the log in real-time
tail -f experiment.log

# Or check GPU usage:
nvidia-smi -l 1
```

### 5. Download results

When done (or during), pull results back to your Mac:

```bash
# On your Mac, from any directory:
rsync -avz root@<vastai-ip>:/root/lewis-experiment/results/ ./lewis_results/
```

Replace `<vastai-ip>` with the IP from Vast.ai dashboard.

### 6. Analyze results

Run locally on your Mac:
```bash
cd lewis-experiment
python analyze.py results/
```

Outputs:
- `results/metrics_summary.json` — all I₃ terms
- `results/figures/` — publication-ready plots

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'torch'"**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**"CUDA out of memory"**
- B200 has 179GB, should never happen for this workload
- Check: `nvidia-smi` (look at "Memory-Usage")
- If stuck, kill and restart: `pkill -f run_all.py`

**"GQA dataset download fails"**
```bash
# Pre-download manually:
python -c "from datasets import load_dataset; load_dataset('gqa', 'balanced')"
```

**Vast.ai instance times out after N hours**
- Vast.ai spot instances max out at 7 days
- ~6hr experiment fits easily
- If you need longer, rent an "on-demand" instance (slightly pricier, infinite duration)

**Connection drops mid-experiment**
```bash
# Reconnect and resume in background:
ssh root@<vastai-ip>
screen -r experiment   # reattach to running session

# Or start fresh in a screen session:
screen -S experiment
python run_all.py --device cuda --results-dir results/ 2>&1 | tee experiment.log
# Ctrl+A then D to detach (experiment keeps running)
```

---

## Cost Estimate

**Vast.ai 1x B200 (Alabama):**
- $3.81/hr × ~5.5 hrs (training) = **~$21**
- Bandwidth: negligible

**Nebius 1x HGX B200 (if you prefer managed):**
- $5.50/hr × ~5.5 hrs = **~$30**
- Includes premium support, better SLAs

---

## After Experiment

**Next steps:**
1. Download results: `rsync ...`
2. Run `analyze.py` locally
3. Check `results/metrics_summary.json` for I₃ values
4. Stop Vast.ai instance to avoid being charged

```bash
# On Vast.ai dashboard, click "Terminate" for your instance
```

---

## GitHub

**Repo:** https://github.com/strangecradles/lewis-experiment

**Clone:** `git clone https://github.com/strangecradles/lewis-experiment.git`

**Branches:**
- `main` — production-ready, fully tested

---

**Questions?** Check `.claude/docs/EXPERIMENT.md` for full experimental design.
