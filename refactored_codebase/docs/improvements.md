# BirdCLEF 2026 — Phoenix Pipeline Future Improvements Report

This document details identified weaknesses in the current refactored Phoenix solution along with structural suggestions, estimation of implementation efforts, and projected Leaderboard (LB) improvements.

---

## 1. System Improvement Matrix

| Module | Component | Current Implementation | Core Weakness | Proposed Improvement | Est. LB Gain | Implementation Effort |
|---|---|---|---|---|---|---|
| **Feature Extraction** | Backbone Fuser | Frozen Perch v2 | Perch is optimized for focal recordings; soundscapes have significant background noise and vocabulary gaps. | **BEATs/HTS-AT Multi-Teacher Distillation**: Train a secondary distilled audio transformer on soundscape datasets to produce 768D embeddings, fusing them with Perch to yield a 2304D joint vector. | **+0.005 to +0.010** | Medium (Requires 1-2 GPU days offline) |
| **Sequence Modeling** | ProtoSSM | 4-layer Selective SSM with 2 prototypes per class | A static count of 2 prototypes fails to capture large intra-class variation (e.g. dawn song, contact notes, alert calls). | **Enhanced ProtoSSM v6**: Expand to 6 layers with 4 learnable prototypes per class. Add a contrastive prototype loss to pull matching window embeddings closer and push non-matching ones away. | **+0.005 to +0.012** | Medium (3 days) |
| **Sequence Modeling** | ResidualSSM | 2-layer sequence correction | Corrections are fitted globally; systematic errors vary heavily depending on site geography and recorder types. | **Site-Conditioned Residual Correction**: Condition ResidualSSM gating weights using site embeddings or local ambient noise levels. | **+0.003 to +0.006** | Low-Medium (2 days) |
| **Ensembling** | Submission Blend | Static division weights by species index (split 117/117) | Splitting class index groupings statically has no ecological or mathematical justification. | **Species-Specific OOF-Weighted Blending**: Solve for optimal blending weights for each target class independently on OOF validation folds. | **+0.003 to +0.005** | Low (1 day) |
| **Post-Processing** | Taxonomy Smoothing | Mean smoothing over genus & family | Simple mathematical averaging over taxonomical groups can smear signal spikes from rare species. | **Graph Neural Network (GNN) Taxonomy Propagation**: Propagate confidence scores along a phylogenetic tree graph where edge weights reflect acoustic similarity. | **+0.002 to +0.004** | High (5 days) |
| **Validation** | CV Strategy | GroupKFold by recording site | Folds do not consider biome, elevation, or seasonal shifts, leading to optimistic local scores. | **Stratified Habitat GroupKFold**: Stratify validation folds based on habitat classification (forest, marsh, savannah) and monthly recording distributions. | **+0.002 (Better CV-LB alignment)** | Low-Medium (2 days) |
| **Post-Processing** | Calibration | Isotonic regression with shared taxon fallbacks | Does not capture time-of-day variations in target calling rates. | **Hour-Calibrated Platt Scaling**: Integrate cyclical hour priors directly into the sigmoid calibration mapping function. | **+0.001 to +0.003** | Low (1 day) |

---

## 2. Recommended Roadmap

To maximize performance gains within a limited runtime/development budget, we recommend executing improvements in the following order:

```mermaid
graph LR
    Step1["1. Species-Specific Blend<br/>(LB +0.003)"] --> Step2["2. Stratified CV Strategy<br/>(Parity Parity)"]
    Step2 --> Step3["3. Enhanced ProtoSSM v6<br/>(LB +0.007)"]
    Step3 --> Step4["4. Multi-Teacher Distillation<br/>(LB +0.008)"]
```

1. **Species-Specific Blending (Ensemble)**: Quickest win. Avoids probability distortion without retraining models.
2. **Stratified Habitat CV (Validation)**: Crucial step before launching heavy models to prevent optimization of parameters on location leakages.
3. **Enhanced ProtoSSM v6 (Sequence Model)**: Significant sequence representation upgrade.
4. **Multi-Teacher Distillation (Feature Extraction)**: Strongest diversity gain by injecting transformer geometry into state-space models.
