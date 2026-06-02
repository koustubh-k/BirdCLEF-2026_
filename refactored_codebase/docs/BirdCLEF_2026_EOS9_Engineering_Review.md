# BirdCLEF 2026 — birdclef-2026-eos-9.ipynb Engineering Review (Comprehensive)

> Target: Improve from LB **0.950 → 0.960+** while staying fully **Kaggle-compatible**.

---

## Scope & what was analyzed

- Main active notebook: **`birdclef-2026-eos-9.ipynb`**.
- Active ensemble configuration (as documented in the notebook):
  - **Model_22 + Model_51 + Model_74**
  - `_runSED_once = True`
  - SED run reused across model components.
- Legacy notebooks/scripts were treated as **historical modules** (v1/v6/v7).

---

## 0. Executive summary (engineering takeaways)

This solution is a **multi-stage weakly-supervised / weakly-calibrated ensemble** built around a dominant **Perch → ProtoSSM/ResidualSSM** pipeline, with optional **SED ONNX** contributing temporal/rank evidence and a taxonomy-aware smoothing/postprocessing stack.

The dominant driver is the **engineered inference logic** (priors + rank/gate scaling + SED-rank blending), not heavy training.

The biggest bottleneck to leaderboard gains is not model capacity—it's **OOF/inference mismatch risk**:

- If validation/OOF artifacts weren’t generated with the exact inference-time transform stack, then the downstream threshold/probe/prior tuning is partially optimistic.
- Calibration is mostly indirect (rank power + temperature), which can leave systematic per-class errors uncorrected.

A second bottleneck is **probability geometry distortion**: multiple nonlinear steps (logit/prob transforms, rank power, continuity gates, thresholds sharpening) interact nonlinearly. Without a fold-safe calibration stage, this can hurt rare taxa.

---

## 1. Module-by-module engineering review

### 1.1 `KAGGLE_INPUTS.md`

**What it does**

- Documents Kaggle inputs required by the active configuration.
- Specifies expected mounts for:
  - BirdCLEF competition data
  - Perch ONNX runtime wheels
  - Perch TF fallback
  - ProtoSSM and ResidualSSM pretrained weights
  - Distilled SED ONNX folds
  - Perch meta cache artifacts

**Purpose of modeling decisions**

- Fail-fast reproducibility and submit safety.
- Avoid silent “default/empty” fallback behavior in inference.

**Inherited ideas**

- Kaggle master practice: explicit input manifests + early validation.

**Potential bugs**

- Provenance drift: paths exist but artifacts mismatch expected versions.
- “found=True” does not guarantee checksum correctness.

**Data leakage risks**

- Minimal (documentation only).

**Validation weaknesses**

- None.

**Calibration weaknesses**

- None.

**Ensemble weaknesses**

- None.

**Inference bottlenecks**

- None.

**LB impact**

- Indirect reliability only.

---

### 1.2 `solutions` cell + top-level blend

**What it does**

- Defines final ensemble weights:
  - Model_22 weight ~**0.020**
  - Model_51p weight ~**0.013**
  - Model_74 weight ~**0.967**
- Defines `_runSED_once = True`, allowing reuse of a SED inference artifact.

**Purpose of modeling decisions**

- Keep a dominant backbone for score stability.
- Preserve a small amount of diversity via other pipelines.

**Inherited ideas**

- Winner-style “dominant single-model + tiny diversity blends” approach.

**Potential bugs**

- Arithmetic blend assumes:
  - identical row ordering
  - identical probability semantics/scaling
- If any component outputs are already rank/temperature transformed, blending can become biased.

**Data leakage risks**

- Medium if SED reuse or priors/caches were built with any fold contamination.

**Validation weaknesses**

- If blend weights tuned on public LB, private gain may be smaller.

**Calibration weaknesses**

- No explicit OOF-calibration after blending.

**Ensemble weaknesses**

- Very low diversity ceiling.

**Inference bottlenecks**

- Dominated by Model_74 compute.

**Estimated leaderboard impact of each component**

- Model_74: **major** (anchors ~0.949–0.950)
- Model_22: **small diversity** (+0.000 to +0.0007)
- Model_51p: **small diversity** (+0.000 to +0.0004)

---

### 1.3 Utility/input cell (path resolution, manifest checks, CSV writer)

**What it does**

- Resolves Kaggle mount paths.
- Prints a manifest with “required for active run” and presence checks.
- Writes final `submission.csv` matching `sample_submission.csv` row_id order and column set.

**Purpose of modeling decisions**

- Prevent invalid submission.
- Ensure deterministic column order.

**Potential bugs**

- Edge-case `row_id` handling: mixing string/int conversion can produce mismatches.
- Filtering `Unnamed` columns may hide unintended schema problems.

**Data leakage risks**

- None.

**Validation/calibration/ensemble**

- None.

**Inference bottlenecks**

- Negligible.

---

### 1.4 Model_1 (inactive; efficient CNN-based SED + Perch distillation)

**What it does (high level)**

- Trains an EfficientNet-based SED on mel spectrograms.
- Uses:
  - GeMFreq pooling (learnable p)
  - attention pooling to aggregate frame logits to clip logits
  - Perch distillation head (GAP + Linear → 1536D)
- Exports ONNX and uses inference-time frame_max + clip blending.

**Purpose of modeling decisions**

- Distillation provides a representation alignment to Perch’s embedding geometry.
- GeM pooling + attention improve event localization.

**Inherited ideas**

- 1st-place-inspired SED inference design patterns.

**Potential bugs / risks**

- Stop-gradient distillation design: backbone may learn only from SED task, not distill (by design). If unintended, it can underperform.
- Fold correctness for any distill target generation must be audited.

**Data leakage risks**

- Potential if Perch distillation targets were precomputed with fold contamination.

**Validation weaknesses**

- If OOF doesn’t mirror inference-time blend/logit smoothing, gains may not carry over.

**Calibration weaknesses**

- Fixed inference blend; no per-class calibration.

**Ensemble weaknesses**

- If revived with tiny weight, may not help much unless outputs are meaningfully different.

**Inference bottlenecks**

- Mel computation + ONNX per fold.

**Estimated LB impact if revived**

- Typically **+0.001 to +0.004** by diversity.

---

### 1.5 Model_22 / Model_51 / Model_74 (Perch→ProtoSSM/ResidualSSM stack)

#### Common structure across these modules

**What they do**

- Run Perch to obtain per-window:
  - logits for mapped classes
  - embeddings (1536D)
- For unmapped/weak classes:
  - apply genus proxies based on Perch model label mapping
- Build **site/hour priors** (and sometimes site-hour/genus variants).
- Fuse Perch scores with priors using:
  - lambda weights
  - rank-aware scaling
  - smoothing over windows
- Use ProtoSSM and optionally ResidualSSM for temporal and second-pass corrections.
- Use MLP probes trained on embedding-based engineered features.

**Purpose of modeling decisions**

- ProtoSSM handles minute-level temporal coherence and context.
- ResidualSSM corrects systematic first-pass errors.
- Priors enforce plausible taxa distribution across time/site.
- Rank-aware transforms reduce false positives in low-confidence files.

**Inherited ideas from winners**

- Perch embeddings + ProtoSSM fusion approach taken from top BirdCLEF 2024/2025 Perch→SSM solutions.
- Taxonomy smoothing and rank/gate heuristics from later EOS improvements.

#### Key risks (applies strongest to Model_74)

**Potential bugs**

- Any OOF probe/stacker computations must have strict alignment with inference-time row order and transform order.
- If thresholds are tuned on train-time distributions but applied to submit-time distributions after additional nonlinear transforms, calibration breaks.

**Data leakage risks**

- High if prior tables and/or stacker features were computed using the same windows later used as OOF targets.
- Prior tables must be fold-safe: “exclude validation files from prior tables” must truly hold for every tuned component.

**Validation weaknesses**

- Often: validation harness isn’t a faithful replica of inference-time postprocessing.

**Calibration weaknesses**

- Temperature scaling + rank-power scaling ≠ calibrated probabilities.
- Threshold sharpening is not calibration; it changes decision geometry.

**Ensemble weaknesses**

- Dominance of a single module reduces diversity ceiling.

**Inference bottlenecks**

- Perch inference (mitigated by cache)
- ProtoSSM forward + TTA
- ResidualSSM forward if enabled

---

## 2. Architecture diagram (requested)

### 2.1 Data flow

```text
BirdCLEF Data + Taxonomy
  |
  |-- train_soundscapes_labels.csv --------------+
  |-- sample_submission.csv -------------------+ |
  |-- taxonomy.csv --------------------------+ | |
  +------------------------------------------+ | |
                                             v v
                               window metadata (row_id, site, hour, taxonomy groups)
```

### 2.2 Feature flow

```text
Audio (5s windows / 60s files)
  |
  |--> Perch v2 (ONNX/TF, frozen)
  |        - logits per window (mapped to 234 classes)
  |        - embeddings per window (1536D)
  |
  |--> SED ONNX (distilled, folds)
           - clip+frame logits per window
           - window smoothing / rank-based SED blend (optional)
```

### 2.3 Model flow

```text
Perch embeddings + Perch/proxy logits + metadata
  |
  |--> Priors (site/hour/genus)
  |--> ProtoSSM (prototype similarity + fusion gate)
  |--> MLP probes (class-specific stacking features)
  |--> ResidualSSM (optional corrections)
  |
  +--> Rank-aware scaling + continuity gates
  +--> (optional) SED rank blend
```

### 2.4 Ensemble flow

```text
subm_22  (w=0.020)
subm_51p  (w=0.013)
subm_74  (w=0.967)

   weighted sum in probability space
             |
             v
      final postprocessing + taxonomy smoothing
             |
             v
        submission.csv writer
```

### 2.5 Postprocessing flow

```text
raw scores
  |
  v
per-taxon temperature scaling
  |
  v
file-level confidence scaling (top-k)
  |
  v
rank-aware power scaling
  |
  v
delta / continuity smoothing across windows
  |
  v
per-class threshold sharpening
  |
  v
(optional) taxonomy smoothing postproc
  |
  v
clip to [0,1] + CSV write
```

---

## 3. Improvement table (requested)

| Component            | Current Implementation                                                            | Weakness                             | Suggested Improvement                                                              | Difficulty | Estimated LB Gain | Kaggle Feasibility | Runtime Cost |
| -------------------- | --------------------------------------------------------------------------------- | ------------------------------------ | ---------------------------------------------------------------------------------- | ---------- | ----------------: | ------------------ | ------------ |
| Validation harness   | OOF logic + many tuned nonlinear transforms                                       | Likely inference/OOF mismatch        | Canonical OOF pipeline that exactly reproduces inference-time postprocessing order | Med        |  +0.002 to +0.004 | Yes                | Low-Med      |
| Calibration          | Temperature scaling + rank power, no OOF-calibrated per-class probability mapping | Systematic class-wise miscalibration | OOF-trained per-class Platt/isotonic after all nonlinearities                      | Med        |  +0.001 to +0.003 | Yes                | Low          |
| Priors λ             | lambda differs train/test; fixed shrinkage                                        | Domain shift; over-regularization    | OOF-search λ per taxon with support-based shrink                                   | Low-Med    |  +0.001 to +0.003 | Yes                | Low          |
| Rank/gate tuning     | Hand-tuned power and continuity gates                                             | Distorts probability geometry        | Tune rank power + gate thresholds by grouped OOF (site/hour/taxon)                 | Med        |  +0.001 to +0.002 | Yes                | Low          |
| Ensemble diversity   | Dominant Model_74 (~97%)                                                          | Diversity ceiling too low            | Add 1 independent distilled teacher head (SED/HTS-AT/BEATs) with small weight      | Med        |  +0.002 to +0.006 | Yes (distill+ONNX) | Med-High     |
| SED reuse efficiency | “run SED once” but still may have overhead in IO/session init                     | Waste compute                        | Cache SED predictions to a single shared artifact                                  | Low        |                 0 | Yes                | Lower        |
| Probe stacking       | Probes trained with limited stacker-harness validation                            | Potential overfitting/OOF optimism   | Train probes using fold-safe OOF features and freeze stacker params                | Med        |            +0.001 | Yes                | Med          |
| Taxonomy smoothing   | Optional smoothing                                                                | Can smear rare positives             | Make taxonomy smoothing uncertainty-gated                                          | Med        | -0.0005 to +0.001 | Yes                | Low          |

---

## 4. Advanced improvements evaluation (requested)

| Idea                               | Verdict                            |           Expected gain | Key risk                                         |
| ---------------------------------- | ---------------------------------- | ----------------------: | ------------------------------------------------ |
| BEATs                              | Good if distilled                  |        +0.003 to +0.006 | Heavy training; distill complexity               |
| AST                                | Good diversity                     |        +0.002 to +0.005 | CPU inference cost unless distilled              |
| HTS-AT                             | Best transformer teacher candidate |        +0.003 to +0.010 | Requires distillation pipeline                   |
| Mamba / SSM                        | Likely incremental                 |        +0.001 to +0.004 | Needs strong validation first                    |
| State Space Models                 | Already present                    |           Keep refining | Hard to exceed current gains without calibration |
| Retrieval-Augmented Classification | Promising for rare taxa            |        +0.001 to +0.003 | Prototype/RAG correctness and caching            |
| Prototype Networks                 | Already part of ProtoSSM           | Extend multi-prototypes | Moderate                                         |
| Graph Neural Taxonomy              | Overkill                           |       +0.0005 to +0.002 | Risk of complexity                               |
| Adaptive Taxonomy Smoothing        | High ROI                           |       +0.0008 to +0.002 | Needs OOF gating                                 |
| Pseudo Labeling                    | Highest ceiling                    |        +0.004 to +0.010 | Leakage + rare species collapse                  |
| Semi-supervised learning           | Similar to pseudo labels           |        +0.003 to +0.008 | Requires careful confidence filtering            |
| Contrastive learning               | Offline embedding quality          |        +0.001 to +0.003 | Compute                                          |
| Habitat priors                     | Feasible and robust                |        +0.001 to +0.003 | Needs careful shrinkage                          |
| Species priors                     | Add co-occurrence Bayesian priors  |        +0.001 to +0.003 | Risk of wrong prior                              |

---

## 5. Competition strategy (requested)

Rank improvements by expected Public LB gain, Private LB gain, risk, compute cost, Kaggle compatibility.

1. **Canonical OOF + inference transform parity**
   - Public: +0.002 to +0.004 | Private: +0.003 to +0.005
   - Risk: Low
   - Cost: Medium
2. **OOF per-class calibration after nonlinear postproc**
   - Public: +0.001 to +0.002 | Private: +0.001 to +0.003
   - Risk: Low
   - Cost: Low
3. **OOF-grouped tuning of rank power + continuity gates**
   - Public: +0.001 to +0.002 | Private: +0.002 to +0.004
   - Risk: Medium
   - Cost: Low-Med
4. **Add independent distilled transformer/teacher head**
   - Public: +0.002 to +0.006 | Private: +0.003 to +0.010
   - Risk: Medium-High
   - Cost: Med-High
5. **Pseudo labeling/noisy student**
   - Public: +0.004 to +0.008 | Private: +0.006 to +0.012
   - Risk: High
   - Cost: High

---

## 6. Refactoring suggestions (requested)

### Modularization

Split notebook into modules:

- `io/paths.py` (resolve Kaggle mount variants)
- `io/submission_writer.py`
- `features/perch.py` (load onnx/TF, mapping, caching)
- `features/sed.py` (fold session + caching)
- `models/proto_ssm.py`, `models/residual_ssm.py`
- `postprocess/rank_ops.py`, `postprocess/tax_smoothing.py`
- `ensemble/blend.py`

### Inference optimization

- Cache Perch test outputs once per config hash.
- Cache SED predictions once if multiple models use SED.
- Remove repeated session init / repeated normalization loops.

### Memory optimization

- Store large arrays float32 (or float16 for rank-only steps).
- Avoid holding multiple redundant copies of `scores_full_raw`.

### ONNX optimization

- Force `onnxruntime` session option `ORT_ENABLE_ALL`.
- Reduce per-call overhead by batching windows.

### Mixed precision improvements

- Only where numerical stability allows (matmul-heavy blocks).
- Keep postprocessing in float32.

### Dataloader improvements (if training resumed)

- Precompute mels for SED training windows or implement batched mel.
- Remove Python loops over batch elements.

---

## 7. Final roadmap (requested)

### 7.1 1-month roadmap (LB 0.950 → 0.955+)

- Week 1: Canonical OOF pipeline + remove OOF bugs/mismatch.
- Week 2: OOF-trained per-class calibration artifacts.
- Week 3: OOF-grouped tuning for priors λ and rank power/gates.
- Week 4: Re-run final blend with calibration + tuned transforms.

### 7.2 3-month roadmap (LB 0.960 target)

- Add one independent distilled transformer-lite head.
- Add uncertainty-gated taxonomy smoothing.
- Implement safe pseudo labeling only if OOF correctness is verified.

### 7.3 Competition-winning roadmap (highest ceiling)

- Multi-teacher distillation into one Kaggle-fast model.
- Multi-stage pseudo labeling with confidence gating.
- Final calibrated ensemble: dominant ProtoSSM/Model_74 + small diversity head.

---

## Deliverable produced

- This file contains the complete review in **Markdown** as requested.
