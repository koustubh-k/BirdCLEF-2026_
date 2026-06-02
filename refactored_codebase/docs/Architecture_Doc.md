# BirdCLEF 2026 — Phoenix Architecture Documentation

This document describes the architectural design, mathematical foundations, ecological relevance, and expected leaderboard (LB) contribution for each major component of the BirdCLEF refactored research codebase.

---

## 1. Frozen Backbone Feature Extractor (Perch v2)

### Mathematical Formulation
The Google Bird Vocalization Classifier v2 (Perch) extracts features from 5-second mono audio windows at 32kHz:
- **Embeddings ($E \in \mathbb{R}^{B \times 1536}$)**: Extracted from the global average pooling layer.
- **Vocabulary Logits ($L_{raw} \in \mathbb{R}^{B \times 11000}$)**: Output of the final projection layer over the pre-training species vocabulary.
- **Genus Proxy Mapping**: For the $N_{unmapped} \approx 50$ species not directly present in Perch's vocabulary, we resolve a congeneric proxy using the maximum logit of all congeneric species present in Perch:
  $$S_{proxy} = \max_{j \in \text{Genus}(c)} L_{raw}[:, j]$$
  where $\text{Genus}(c)$ is the set of all species indexes in the same genus as unmapped species $c$.

### Ecological Relevance
- **Zero-Shot Transfer**: Leverages pre-trained representations learned from millions of bioacoustic recordings (iNaturalist + FSD50K) to represent rare species.
- **Manifold Alignment**: Embeddings act as a generalized bioacoustic manifold, providing a robust input geometry for downstream sequence modeling.

### Expected Leaderboard Contribution
- **Solo LB**: ~0.85–0.90 (raw mapped logits)
- **With Genus Proxies**: ~0.92–0.93

---

## 2. Sound Event Detection (Distilled SED)

### Mathematical Formulation
EfficientNet-B0 backbone operating on Mel spectrograms (256 bins, hop=512) with two main branches:
1. **GeMFreq Pooling**: A generalized mean pool over the frequency axis with learnable parameter $p$ (initialized at 3.0):
   $$\text{GeM}(X) = \left( \frac{1}{F} \sum_{f=1}^F X_f^p \right)^{1/p}$$
2. **Temporal Attention Head**: Combines attention weights and frame-level logits:
   $$A_t = \text{Softmax}(\tanh(W_a X_t + b_a))$$
   $$C_t = \sigma(W_c X_t + b_c)$$
   $$P_{clip} = \sum_{t=1}^T A_t C_t$$
3. **Perch Distillation Head**: Aligns the SED feature map to Perch's 1536D space via Mean Squared Error (MSE) loss during offline training, utilizing a gradient stop on the classification path.

### Ecological Relevance
- **Temporal Localization**: Unlike Perch's fixed 5-second windows, SED identifies sub-second vocalization frames, allowing fine-grained segmentation.
- **Noise Suppression**: Ignores continuous backgrounds (wind, rain, insects) via localized frequency pooling.

### Expected Leaderboard Contribution
- **Solo LB**: ~0.88–0.91
- **Ensemble Gain**: +0.003 to +0.007 when combined with ProtoSSM outputs.

---

## 3. Prototype-Augmented State Space Model (ProtoSSM)

### Mathematical Formulation
Operates on sequence matrices of 12 windows per file.
1. **Selective SSM (S4 Block)**: Implements input-dependent discretization:
   $$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$$
   $$y_t = C_t h_t + D_t x_t$$
   where $\bar{A}_t = \exp(\Delta_t A)$ and $\bar{B}_t = \Delta_t B_t$ are discretized matrices, and $\Delta_t = \text{Softplus}(\text{Linear}(x_t))$ is the step size.
2. **Prototype Similarity Attention**: A cross-attention layer matches hidden states $H \in \mathbb{R}^{12 \times d_{model}}$ to learnable class prototype vectors $P \in \mathbb{R}^{234 \times d_{model}}$:
   $$\text{Similarity}_c(t) = \tau \cdot \cos(H_t, P_c) + b_c$$
   where $\tau$ is a learnable temperature scale and $b_c$ is a class-specific bias.

### Ecological Relevance
- **Temporal Context**: Enforces acoustic consistency. E.g., a dawn-chorus bird calling in window $t$ increases the probability of it being detected in window $t+1$.
- **Prior Embeddings**: Site and hour embeddings represent local ecological niches, steering predictions based on typical site/hour distributions.

### Expected Leaderboard Contribution
- **Marginal Gain**: +0.03 to +0.06 over raw Perch logits.

---

## 4. Error-Correcting Second Pass (ResidualSSM)

### Mathematical Formulation
A smaller sequence model trained on validation residuals to correct systematic prediction errors.
1. **Residual Targets**:
   $$\Delta_{target} = Y - \sigma(L_{first\_pass})$$
2. **Residual Forward Pass**: Takes concatenated embeddings and first-pass predictions to predict correction logits:
   $$C_{residual} = \text{ResidualSSM}([\text{Emb}, L_{first\_pass}])$$
3. **Correction Blending**:
   $$L_{final} = L_{first\_pass} + w_{correction} \times C_{residual}$$
   where $w_{correction} \approx 0.35$ is tuned via grid search in the post-processed validation space.

### Ecological Relevance
- **Systematic Bias Correction**: Automatically suppresses classes that the first pass consistently over-predicts (e.g., confusing similar-sounding sympatric species at specific hours).

### Expected Leaderboard Contribution
- **Marginal Gain**: +0.005 to +0.015.

---

## 5. Parallelized MLP Probes (Stacking Probes)

### Mathematical Formulation
Per-class sklearn `MLPClassifiers` trained on standardized, PCA-reduced (128D) Perch embeddings concatenated with temporal sequence statistics:
$$X_{probe} = [\text{PCA}(E), L_{raw}, \text{Prev}(L_{raw}), \text{Next}(L_{raw}), \text{Mean}(L_{raw}), \text{Max}(L_{raw}), \text{Std}(L_{raw})]$$
During training, positive classes are upsampled dynamically based on class-frequency weights:
$$W_c = \text{Clip}\left(\sqrt{\frac{N_{total}}{N_{pos, c}}}, 1.0, 10.0\right)$$
Frequent classes ($\ge 50$ positives) use a wider architecture $(256, 128)$ while rare classes keep $(128, 64)$ to avoid overfitting.

### Ecological Relevance
- **Unmapped Species Recovery**: Resolves class probabilities for unmapped species by learning non-linear mappings directly from raw embedding manifolds.

### Expected Leaderboard Contribution
- **Marginal Gain**: +0.005 to +0.010 for unmapped species.

---

## 6. Bayesian Prior Fusion

### Mathematical Formulation
Computes empirical occurrence rates across training records, smoothed temporally using a circular Gaussian filter (sigma=1.5h) over the 24-hour cycle:
$$P_{hour, smoothed}(h) = \text{GaussianFilter}(P_{hour}, \sigma=1.5h)$$
At test time, logits are adjusted dynamically using bayesian log-prior addition:
$$L_{adjusted} = L_{raw} + \lambda_{prior} \left( \log(P_{prior}) - \log(1 - P_{prior}) \right)$$
where $P_{prior}$ is computed using shrinkage estimators based on site, hour, and joint site-hour frequencies.

### Ecological Relevance
- **Ecological Distribution Constraints**: Suppresses false positives for species that are biologically absent or inactive at certain times or locations (e.g., diurnal species vocalizing at midnight).

### Expected Leaderboard Contribution
- **Marginal Gain**: +0.01 to +0.03.

---

## 7. Post-Processing and Smoothing

### Mathematical Formulation
- **File Confidence Scaling**: Scales predictions by the average of the top-K window predictions:
  $$P_{scaled}(t) = P(t) \times \left( \frac{1}{K} \sum_{i \in \text{TopK}} P(i) \right)^{p_{fc}}$$
- **Rank-Aware Scaling**: Multiplies scores by file max probability to reject low-confidence files:
  $$P_{rank}(t) = P(t) \times (\max_j P(j))^{p_{ra}}$$
- **Confidence-Gated Smoothing**: Smoothes predictions with adjacent windows using an alpha parameter gated by predictions confidence:
  $$\alpha_t = \alpha_{base} \times (1 - \max_c P(t, c))$$
  $$P_{smooth}(t) = (1 - \alpha_t) P(t) + \alpha_t \left( \frac{P(t-1) + P(t+1)}{2} \right)$$

### Ecological Relevance
- **Noisy File Rejection**: Dampens brief, random, low-confidence noise triggers while keeping continuous, high-confidence vocal sequences intact.

### Expected Leaderboard Contribution
- **Marginal Gain**: +0.005 to +0.012.

---

## 8. Adaptive Taxonomy Smoothing

### Mathematical Formulation
Uses predictive uncertainty (derived from class entropy) to smooth individual species scores towards genus and family averages:
$$U_t = \text{Clip}\left(\frac{0.92 - \max_c P(t, c)}{0.57}, 0.0, 1.0\right)$$
$$P_{smooth}(t, c) = (1 - \alpha_{genus} U_t) P(t, c) + \alpha_{genus} U_t \text{Mean}_{j \in \text{Genus}(c)} P(t, j)$$

### Ecological Relevance
- **Taxonomic Coherence**: Species of the same genus share morphological and vocal traits. Moving uncertain classifications toward genus-level centers prevents high-variance prediction errors.

### Expected Leaderboard Contribution
- **Marginal Gain**: +0.001 to +0.003.
