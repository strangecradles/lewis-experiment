---
tags: [experiment, architecture, validation, composition, original]
date_added: 2026-04-06
status: experimental-design
parent: "[[Learned_Connective_Architecture]]"
---

# The Lewis Experiment: Measuring Superadditive Composition of Independently Trained Models

> "In each of my friends there is something that only some other friend can fully bring out. By myself I am not large enough to call the whole man into activity; I want other lights than my own to show all his facets."
> — C.S. Lewis, *The Four Loves* (quoting Charles Lamb)

---

## 0. The Core Claim

When independently trained models are composed through learned connective tissue, the resulting system exhibits **superadditive** performance: the whole is greater than the sum of its parts, and removing any one model degrades the others' effective capabilities, not just its own contribution.

This is the Lewis claim applied to neural networks. Removing model C from the composition {A, B, C} doesn't just remove C's contribution. It removes "C's part in B" — the representations in B that only C's presence could elicit through the learned connective pathways.

**Formal definition of superadditivity:**

Let P(S) denote the performance of a composed system containing model subset S ⊆ {A, B, C}, with learned connective tissue trained for that specific subset.

The **three-way interaction term** is:

```
I₃(A,B,C) = P({A,B,C}) − [P({A,B}) + P({A,C}) + P({B,C})] + [P({A}) + P({B}) + P({C})]
```

This follows the inclusion-exclusion decomposition from combinatorics / ANOVA. If I₃ > 0, the three-way system produces capabilities that cannot be predicted from any combination of pairwise systems. That's the Lewis effect.

**Null hypothesis:** I₃ ≤ 0. Composition is at best additive — you can predict the full system's performance from the subsystems.

**Alternative hypothesis:** I₃ > 0. Composition produces emergent capabilities that exist only in the full triad.

---

## 1. Why This Experiment Matters

The current paradigm scales intelligence by scaling individual models: more parameters, more data, more compute. The alternative paradigm this experiment tests is: intelligence through composition of diverse specialists with learned interconnections.

If superadditive composition is real and measurable, it implies:

1. **A different scaling law.** Instead of scaling one model, you scale the number and diversity of composed models. The interaction terms grow combinatorially.

2. **Efficient frontiers shift.** A system of three 100M-parameter models with 50M parameters of connective tissue (350M total) could outperform a single 350M-parameter model on tasks requiring diverse reasoning — not just through routing, but through emergent interaction.

3. **Representational diversity has compounding returns.** Models trained by different organizations on different data with different objectives aren't just redundant — they're complementary in ways that a single training run can never be, because a single optimization trajectory explores one mode of the loss landscape.

4. **The Meta-Harness thesis extends to the sub-symbolic level.** If the connective tissue (harness) between sub-models is where the leverage is, then optimizing connections yields larger gains than optimizing individual components.

---

## 2. Model Selection

### Selection Criteria

The three models must be:

- **Independently trained** — different organizations or different training runs, not fine-tuned variants of the same base model
- **Same modality** (vision, for simplicity) — so we isolate the composition effect from the trivially superadditive case of combining different modalities
- **Provably different representational biases** — different training objectives induce different internal representations
- **Small enough to run frozen on a single GPU** — the entire experiment should be runnable on 1× A100 or even 1× 3090

### The Three Models

| Model | Training Objective | Representational Bias | Parameters | Source |
|-------|-------------------|----------------------|------------|--------|
| **DINOv2 ViT-S/14** | Self-supervised (self-distillation, no labels) | Spatial structure, object boundaries, part-whole relationships, layout. Strong on localization, weak on semantic categories without fine-tuning. | 22M | Meta |
| **SigLIP ViT-B/16** | Contrastive vision-language (sigmoid loss on image-text pairs) | Semantic categories, attributes, language-aligned concepts. Strong on "what is this," weak on precise spatial reasoning. | 86M | Google |
| **MAE ViT-B/16** | Masked autoencoding (reconstruct masked patches from visible patches) | Low-level texture, color, material, fine-grained visual detail. Strong on pixel-level properties, weaker on high-level semantics. | 86M | Meta |

### Why These Three

The training objectives create three distinct representational spaces:

- **DINOv2** learns by predicting one view of an image from another (self-distillation). This encourages representations that are invariant to viewpoint but preserve spatial structure. It sees *where things are* and *how they relate spatially*.

- **SigLIP** learns by aligning images with text descriptions. This encourages representations that capture the semantic content that humans describe in language. It knows *what things are called* and *what properties they have*.

- **MAE** learns by reconstructing missing patches. This encourages representations that capture low-level visual statistics — texture, color gradients, material properties, surface detail. It knows *what things look like* at the pixel level.

The key property: no single model is a superset of the others. DINOv2 can't tell you what material something is made of (MAE can). SigLIP can't tell you the precise spatial arrangement of objects (DINOv2 can). MAE can't tell you that something is a "kitchen" (SigLIP can).

---

## 3. Task Selection

### Requirements

The task must:

1. Be solvable partially by each individual model (so each contributes)
2. Contain questions that require *exactly two* of the three representational types (to measure pairwise interactions)
3. Contain questions that require *all three* representational types (to measure three-way interactions)
4. Provide question-type annotations (so you can break down performance by required capability)
5. Be cheap to evaluate (synthetic or existing benchmark)

### Primary Task: GQA (Visual Reasoning on Real Images)

GQA (Hudson & Manning, 2019) is ideal because:

- Questions are generated from scene graphs, providing structured annotations of spatial relationships, attributes, and object categories
- Questions are tagged by type: spatial, attribute, category, relational, comparative, logical
- It uses real images (Visual Genome), so models can't shortcut with synthetic artifacts
- It's widely benchmarked, providing context for results
- The dataset is large (22M questions) but evaluation is fast

**Critical question types for this experiment:**

| Question Type | Example | Primary Model | Secondary Model | Tertiary Model |
|--------------|---------|---------------|-----------------|----------------|
| Pure spatial | "Is the cup to the left or right of the plate?" | DINOv2 | — | — |
| Pure semantic | "What kind of animal is in the image?" | SigLIP | — | — |
| Pure material/texture | "What material is the table made of?" | MAE | — | — |
| Spatial + semantic | "What is to the left of the red car?" | DINOv2 | SigLIP | — |
| Semantic + material | "Is the wooden object a chair or a table?" | SigLIP | MAE | — |
| Spatial + material | "Is the shiny object above or below the matte one?" | DINOv2 | MAE | — |
| **All three** | "What material is the object to the left of the largest animal made of?" | DINOv2 | SigLIP | MAE |

The "all three" questions are where superadditive effects should appear. They require spatial reasoning (DINOv2), category/size recognition (SigLIP), and material identification (MAE) simultaneously.

### Secondary Task: CLEVR-Material (Synthetic, Controlled)

To supplement GQA with a fully controlled setting, generate a CLEVR variant that adds material/texture properties to objects:

- Standard CLEVR spatial and count questions (tests DINOv2)
- Semantic questions about shapes and sizes (tests SigLIP)
- Material questions about surface properties — metallic, matte, translucent, wooden (tests MAE)
- Compound questions requiring all three

The advantage of the synthetic task: you control exactly which capabilities each question requires, enabling precise measurement of interaction terms.

---

## 4. Architecture

### 4.1 Connector Design

For each pair of models (i, j), train a learned connector C_ij:

```
C_ij: cross-attention module where model i's patch tokens serve as keys/values
      and model j's patch tokens serve as queries (and vice versa, bidirectional)
```

Architecture per connector:
- 2-layer transformer cross-attention
- Hidden dimension: 256
- 4 attention heads
- Bidirectional: each model can query the other
- ~5M parameters per pairwise connector

For the three-way system {A, B, C}, there are three pairwise connectors: C_AB, C_AC, C_BC. Total connector parameters: ~15M.

### 4.2 Task Head

A shared task head maps from the composed representation to GQA answers:

- Input: concatenated [CLS] tokens from each active model, after connective processing
- Architecture: 2-layer MLP with GELU, hidden dimension 512
- Output: classification over GQA answer vocabulary

### 4.3 What Gets Trained

- All three vision models: **frozen**
- All connectors: **trained**
- Task head: **trained**

Total trainable parameters: ~17M (15M connectors + 2M task head)

### 4.4 Information Flow

For the full system {A, B, C}:

```
Image
  ├──→ DINOv2 (frozen) ──→ DINOv2 features (patch tokens)
  ├──→ SigLIP  (frozen) ──→ SigLIP features (patch tokens)
  └──→ MAE     (frozen) ──→ MAE features (patch tokens)
         │                        │                      │
         ├──── C_AB ──────────────┤                      │
         ├──── C_AC ──────────────┼──────────────────────┤
         │                        ├──── C_BC ────────────┤
         ▼                        ▼                      ▼
   DINOv2' [CLS]          SigLIP' [CLS]           MAE' [CLS]
         └────────────────────┼──────────────────────────┘
                              ▼
                         Task Head
                              ▼
                          Answer
```

The primed features (DINOv2', SigLIP', MAE') include information from the other models, mediated by the connectors. This is where "A's part in B" lives — in the way A's representations are transformed by their interaction with B through C_AB.

---

## 5. Experimental Conditions

### The Full Combinatorial Design

To measure superadditivity, you need ALL subsets of {A, B, C}:

| Condition | Models | Connectors | What It Measures |
|-----------|--------|------------|-----------------|
| A alone | DINOv2 | None | Individual spatial capability |
| B alone | SigLIP | None | Individual semantic capability |
| C alone | MAE | None | Individual material/texture capability |
| A+B | DINOv2 + SigLIP | C_AB | Pairwise spatial-semantic |
| A+C | DINOv2 + MAE | C_AC | Pairwise spatial-material |
| B+C | SigLIP + MAE | C_BC | Pairwise semantic-material |
| A+B+C | All three | C_AB + C_AC + C_BC | Full composition |

**Critical detail:** Each condition gets its own independently trained connectors and task head. You cannot just ablate the full system by removing a model — that would conflate the Lewis effect with the trivial effect of removing an input channel. Each subset must be trained from scratch to achieve its own optimum.

This means training **7 separate systems** (3 individuals + 3 pairs + 1 triad).

### Controls

**Control 1: Simple Concatenation Baseline**

For each subset, also train a version with no learned connectors — just concatenated [CLS] tokens from the frozen models → task head. This isolates the contribution of the connective tissue from the contribution of model diversity.

**Control 2: Single Large Model Baseline**

Train a single DINOv2 ViT-B/14 (86M params, comparable to the full composed system's ~200M total) → task head. This tests whether composition of three small models beats one medium model.

**Control 3: Parameter-Matched Single Connector**

For the pairwise conditions, also train a version with a connector that has the SAME parameter count as the cross-attention connector but uses a simple MLP on concatenated [CLS] tokens. This isolates the effect of the cross-attention architecture (representational access) from raw parameter count.

**Control 4: Ensemble Baseline**

Train three independent models (A → task head, B → task head, C → task head) and combine their output logits via learned weighted average. This is the classical ensemble — no learned composition, just output-level combination. The Lewis effect should exceed the ensemble effect.

**Control 5: Diversity Control (Critical)**

Take three DINOv2 ViT-S checkpoints trained with different random seeds — same architecture, same objective, same data. Run the full experiment (all 7 subsets + connectors) with these three near-identical models. This isolates the effect of the cross-attention architecture from the effect of representational diversity.

- If I₃ > 0 for the diverse trio (DINOv2 + SigLIP + MAE) but I₃ ≈ 0 for the three DINOv2 copies → **the diversity thesis is validated**. Superadditivity requires genuinely different representations.
- If I₃ > 0 for BOTH → the effect is about cross-attention architecture, not diversity. Still useful, but a weaker result.
- If I₃ ≈ 0 for both → composition doesn't produce emergence regardless.

This is the cleanest test of the core claim.

**Control 6: Concatenation I₃**

Compute the full interaction term I₃ for the concatenation-only versions (Control 1) across all 7 subsets. The key metric is then:

```
Δ_Lewis = I₃(with connectors) − I₃(concatenation only)
```

If Δ_Lewis > 0, the learned connective tissue creates emergent interactions beyond what simple feature access provides. If Δ_Lewis ≈ 0 but I₃(concatenation) > 0, the effect is "more features = better," not the Lewis mechanism.

---

## 6. Measurements

### 6.1 Primary Metric: Three-Way Interaction Term

```
I₃ = P(ABC) − [P(AB) + P(AC) + P(BC)] + [P(A) + P(B) + P(C)]
```

Measured on overall GQA accuracy AND broken down by question type. The prediction: I₃ is largest on questions requiring all three capabilities.

### 6.2 Secondary Metrics

**Pairwise interaction terms:**
```
I₂(A,B) = P(AB) − P(A) − P(B)
I₂(A,C) = P(AC) − P(A) − P(C)
I₂(B,C) = P(BC) − P(B) − P(C)
```

These should be positive (pairwise composition helps) but the three-way interaction should be ABOVE what pairwise interactions predict.

**Composition vs. scale:**
```
Δ_scale = P(ABC_composed) − P(single_large_model)
```

If positive, composition beats scale at the same parameter budget.

**Composition vs. ensemble:**
```
Δ_ensemble = P(ABC_composed) − P(ABC_ensemble)
```

If positive, learned composition beats output-level combination. This isolates the "A's part in B" effect from the "different models are good at different things" effect.

**Connector contribution:**
```
Δ_connector = P(ABC_with_connectors) − P(ABC_concatenation_only)
```

If positive, the learned connective tissue matters above and beyond just having access to multiple feature sources.

### 6.3 Per-Question-Type Analysis

Break down ALL metrics by GQA question type. The prediction structure:

| Question Type | Expected I₃ | Rationale |
|--------------|-------------|-----------|
| Pure spatial | ~0 | Only needs DINOv2, others are noise |
| Pure semantic | ~0 | Only needs SigLIP |
| Pure material | ~0 | Only needs MAE |
| Two-capability | Small positive | Pairwise composition suffices |
| **Three-capability** | **Large positive** | **This is where the Lewis effect lives** |

If you see large I₃ ONLY on three-capability questions and ~0 on single-capability questions, that's strong evidence for the claim. If I₃ is uniformly positive across all question types, the effect is real but unspecific — composition helps generally rather than through the specific mechanism proposed.

### 6.4 Representational Analysis

After training, analyze what the connectors learn:

**Attention pattern analysis:** In the cross-attention connectors, which patches in model A attend to which patches in model B? Do spatially corresponding patches attend to each other (aligned representations)? Or do attention patterns differ by question type (dynamic routing)?

**Ablation of specific pathways:** After training the full system, ablate each connector C_ij by zeroing it out at inference time (without retraining). The performance drop when ablating C_ij tells you how much "i's part in j" contributes.

**Feature similarity before/after connectors:** Measure CKA (Centered Kernel Alignment) between the models' representations before and after the connector processing. If the connectors are doing useful work, the effective representation should change — but not collapse to a shared representation (which would indicate the connectors are just projecting to a common space, not enabling rich interaction).

---

## 7. Training Protocol

### 7.1 Data

- GQA train split: ~1M questions on ~80K images
- Each condition trains on the same data with the same hyperparameters
- Evaluation on GQA val split (~130K questions)

### 7.2 Training

- Optimizer: AdamW, lr=1e-4, cosine schedule, weight decay=0.01
- Batch size: 128
- Training duration: 10 epochs (early stopping on val loss)
- All vision encoders: frozen throughout
- Only connectors + task head: trained

### 7.3 Computational Budget

Per condition:
- Forward pass through 3 frozen ViTs: ~50ms/image (batched on A100)
- Connector + task head training: ~5M trainable parameters, ~1M training samples
- Estimated training time per condition: 2-4 hours on 1× A100

Total: 7 main conditions × ~3 hours + 4 controls × ~3 hours = ~33 hours of A100 time.

### 7.4 Statistical Rigor

Run each condition with 3 random seeds. Report mean ± standard deviation. Test I₃ > 0 with a one-sided paired t-test across seeds and question subsets. Given the large number of test questions (~130K), even small effects should be detectable.

---

## 8. Possible Outcomes and Interpretations

### Outcome 1: Strong Superadditivity (I₃ >> 0, especially on three-capability questions)

**Interpretation:** The Lewis effect is real. Composition of diverse models through learned connections produces emergent capabilities. The connective tissue enables representations that no individual model or pair of models can produce.

**Implication:** This validates the Learned Connective Architecture thesis. The next step is scaling: more models, richer connections, cross-modal composition.

### Outcome 2: Weak Superadditivity (I₃ slightly > 0, uniformly across question types)

**Interpretation:** Composition helps, but not through the specific mechanism proposed. The effect is likely just improved robustness / ensembling through richer feature combination, not true emergent capabilities.

**Implication:** The architecture may still be useful for efficiency, but the "Lewis" framing overstates the effect. Thicker connections help, but more as better ensembling than as emergent interaction.

### Outcome 3: No Superadditivity (I₃ ≈ 0)

**Interpretation:** The pairwise interactions fully explain the full system's performance. There is no "A's part in B that only C can bring out." Independently trained models have compatible representations (as model stitching theory suggests), but composition doesn't create new capabilities — it just combines existing ones.

**Implication:** The standard ensemble / MoE paradigm is sufficient. Learned connective tissue is useful for routing but not for emergence.

### Outcome 4: Negative I₃

**Interpretation:** Adding the third model interferes with the pairwise interactions. The connective tissue creates cross-talk that degrades the system. This would be analogous to the Dense Connector finding that adding parameters to connectors doesn't always help.

**Implication:** Composition needs better regularization, sparser connections, or the hub-and-spoke architecture proposed in the main document.

---

## 9. Connection to Phase 2: Cross-Modal Prediction Pretraining

If Outcome 1 or 2 is observed, the natural follow-up tests Phase 2 of the Learned Connective Architecture:

**Experiment Extension:** Before the GQA fine-tuning, pretrain the connectors via cross-modal prediction:

- C_AB: DINOv2 features predict SigLIP features (and vice versa)
- C_AC: DINOv2 features predict MAE features (and vice versa)
- C_BC: SigLIP features predict MAE features (and vice versa)

Loss: MSE in representation space (with stop-gradient on the target, à la I-JEPA, to prevent collapse).

Then fine-tune on GQA. Compare:
- Random init connectors → GQA (baseline)
- Cross-modal prediction pretrained connectors → GQA (Phase 2 hypothesis)

If pretraining increases I₃ specifically (the three-way interaction gets stronger), that's evidence for the infant-development-inspired calibration idea: the connective tissue learns which cross-model predictions are reliable before encountering any task-specific signal.

---

## 10. Timeline and Resources

### Option A: Free Tier ($0)

All three models are small enough (22M-86M) to run frozen on a T4 (16GB VRAM). Only ~17M params are trained. This runs on free compute.

**Key simplification:** Use a 50K-question GQA subset for training, 10K for eval. The I₃ measurement works at any sample size — 10K eval questions across 7 types gives ~1400 per type, plenty for statistical significance. Each condition trains in 1-2 hours on T4 instead of 2-4 on A100.

**Free compute sources:**
- Kaggle: 30 hours/week of free T4/P100
- Google Colab free: T4, session-limited but sufficient
- Yale HPC: Check Grace/McCleary cluster access through physics dept (likely free for students)

| Phase | Duration | Compute | Cost |
|-------|----------|---------|------|
| Setup: data pipeline, model loading, connector architecture, training loop | 1-2 weeks | Local / CPU | $0 |
| Condition training: 7 main + 6 controls × 3 seeds = 39 runs × ~1.5 hrs | 2 weeks | Kaggle + Colab (~60 T4-hours) | $0 |
| Evaluation and analysis | 3-4 days | Minimal | $0 |
| Phase 2 extension (if warranted) | 1 week | Kaggle + Colab (~20 T4-hours) | $0 |
| **Total** | **~5-6 weeks** | **~80 T4-hours** | **$0** |

### Option B: Budget ($10-50)

Google Colab Pro ($9.99/month) gives priority access to T4/V100 with longer sessions. Reduces the session-juggling friction significantly.

| Phase | Duration | Cost |
|-------|----------|------|
| Setup | 1-2 weeks | $0 |
| Training + eval | 2-3 weeks | $10-20 (1-2 months Colab Pro) |
| Phase 2 extension | 1 week | included |
| **Total** | **~4-5 weeks** | **$10-20** |

### Option C: Full Budget (~$200)

A100 cloud via Lambda Labs or RunPod. Faster but unnecessary for the core experiment.

| Phase | Duration | Cost |
|-------|----------|------|
| Setup | 1-2 weeks | $0 |
| Training + eval | 4-5 days | ~$150 |
| Phase 2 extension | 2 days | ~$50 |
| **Total** | **~3 weeks** | **~$200** |

### Recommendation

Start with Option A. The pipeline setup (data loading, model integration, training loop) is the hard part and doesn't need a GPU at all. Build and debug everything locally with tiny toy data. Only once the pipeline is solid, use Kaggle/Colab for the actual training runs. If Yale HPC is available, that's the best option — likely faster than T4 and free.

---

## 11. What a Positive Result Would Mean

A positive I₃ on three-capability questions, combined with the composition-beats-scale result (Δ_scale > 0), would establish:

1. **A new experimental paradigm** for studying model composition, distinct from MoE (which trains experts together) and ensembles (which combine outputs).

2. **Empirical evidence for the infant development analogy.** If cross-modal prediction pretraining of the connective tissue improves superadditive performance, the parallel to forward-model construction through sensorimotor babbling becomes concrete.

3. **A foundation for the Learned Connective Architecture.** The multi-timescale, precision-weighted, bidirectional connective tissue proposed in the main document becomes motivated by experimental evidence rather than just theoretical analogy.

4. **A connection between the agent harness optimization problem (Kairn) and the sub-symbolic composition problem.** If the interaction terms are the primary source of leverage in model composition, then the same optimization techniques (population-based search, Thompson sampling over connection topologies) that Kairn applies to agent harnesses could be applied to neural connective tissue.

The Lewis quote provides the intuition. The experiment provides the evidence. The architecture provides the engineering.
