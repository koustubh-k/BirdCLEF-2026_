# 📖 BirdCLEF Phoenix Pipeline — Line-by-Line Code Guide

This document provides a detailed breakdown of the core classes, functions, and execution blocks used in the BirdCLEF 2026 inference pipeline. Since the codebase contains highly redundant blocks copy-pasted across multiple model execution cells, this guide focuses on the active logic of the pipeline's core components:

---

## 1. Environment & Path Resolution Utilities

These functions run at the beginning of the notebook cells to resolve data folders and set up submit-safe execution paths.

### `find_competition_dir()`
```python
def find_competition_dir():
    candidates = [
        Path("/kaggle/input/competitions/birdclef-2026"),
        Path("/kaggle/input/birdclef-2026"),
    ]
    for p in candidates:
        if (p / "sample_submission.csv").exists() and (p / "taxonomy.csv").exists():
            return p
    for p in Path("/kaggle/input").rglob("sample_submission.csv"):
        parent = p.parent
        if (parent / "taxonomy.csv").exists() and (parent / "train_soundscapes_labels.csv").exists():
            return parent
    raise FileNotFoundError("BirdCLEF competition data directory not found.")
```
*   **Lines 2–5:** Defines candidate directories representing default Kaggle inputs.
*   **Lines 6–8:** Probes each candidate to check if they contain the key data files (`sample_submission.csv` and `taxonomy.csv`). If found, returns the directory.
*   **Lines 9–13:** If standard candidate paths fail, searches recursively across all Kaggle input mounts (`/kaggle/input`) for any folder containing a `sample_submission.csv` and checks if it also contains the taxonomy and training labels.
*   **Line 14:** Throws a `FileNotFoundError` if all search methods fail, raising a loud failure in the submission runtime.

---

## 2. Feature Extraction Backbone (Perch / Distilled SED)

This section covers the initialization and inference steps for ONNX backbones.

### `PerchBackbone.__init__()`
```python
def __init__(self, onnx_path: Optional[Path] = None, tf_model_path: Optional[Path] = None):
    self.use_onnx = _ONNX_AVAILABLE and onnx_path is not None and onnx_path.exists()
    self.onnx_session = None
    self.tf_infer_fn = None
    self.emb_dim = 1536
    if self.use_onnx:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.onnx_session = ort.InferenceSession(str(onnx_path), providers=providers)
```
*   **Line 2:** Detects whether to use the ONNX Runtime based on the availability of the `onnxruntime` import and the existence of the target ONNX file.
*   **Lines 7–10:** Configures session options (capping thread count to 4 to prevent CPU exhaustion on Kaggle instances) and instantiates the `InferenceSession` with GPU support fallback (`CUDAExecutionProvider` followed by `CPUExecutionProvider`).

### `PerchBackbone.predict()`
```python
def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if self.use_onnx:
        outs = self.onnx_session.run(None, {self.onnx_input_name: x})
        logits = outs[self.onnx_out_map["label"]].astype(np.float32)
        emb = outs[self.onnx_out_map["embedding"]].astype(np.float32)
    return logits, emb
```
*   **Line 3:** Runs ONNX inference on input waveform matrix `x` (shape `(B, 160000)`).
*   **Lines 4–5:** Extracts the classification logits (shape `(B, 3183)`) and the final pooling layer embeddings (shape `(B, 1536)`).

### `MultiBackboneExtractor.predict()`
```python
def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p_logits, p_emb = self.perch.predict(x)
    if self.teacher_session is not None:
        outs = self.teacher_session.run(None, {self.teacher_input_name: x})
        t_emb = outs[self.teacher_embed_idx].astype(np.float32)
        emb = np.concatenate([p_emb, t_emb], axis=-1)
    else:
        emb = p_emb
    return p_logits, emb
```
*   **Line 2:** Extracts Perch backbone features.
*   **Lines 3–5:** If a distilled Teacher model is active, runs inference on the same audio matrix `x` to extract a secondary 768-dimensional embedding `t_emb` and concatenates both embeddings to form a 2304-dimensional feature vector (`emb`).

---

## 3. Sequence Modeling (ProtoSSM & ResidualSSM)

These modules model sequence dependencies across contiguous 5-second chunks (total sequence length of 12 windows for a 60s file).

### `LightProtoSSM.forward()`
```python
def forward(self, x: torch.Tensor, site_ids: torch.Tensor, hour_ids: torch.Tensor) -> torch.Tensor:
    B, T, D = x.shape
    x = self.input_proj(x)
    
    # Metadata embedding injection
    meta_emb = self.site_emb(site_ids) + self.hour_emb(hour_ids)
    x = x + meta_emb.unsqueeze(1)
    
    # Mamba SSM sequence layers
    for layer in self.ssm_layers:
        x = x + layer(x)
        
    # Prototype projection
    x = F.normalize(x, p=2, dim=-1)
    prototypes = F.normalize(self.prototypes, p=2, dim=-1)
    logits = torch.matmul(x, prototypes.t()) / self.temperature
    return logits
```
*   **Line 3:** Projects input dimensions (`2304` or `1536`) to Mamba internal `d_model` size.
*   **Lines 6–7:** Embeds spatial site metadata and temporal hour metadata, summing them and broadcasting across the sequence length (`T`).
*   **Lines 10–11:** Passes sequence vectors through successive Selective State Space (SSM) layers with residual skips.
*   **Lines 14–16:** Performs cosine-similarity matching of sequence features against class prototypes, outputting classification logits.

### `ResidualSSM.forward()`
```python
def forward(self, x: torch.Tensor, base_scores: torch.Tensor, site_ids: torch.Tensor, hour_ids: torch.Tensor) -> torch.Tensor:
    B, T, D = x.shape
    x = self.input_proj(x)
    meta = self.site_emb(site_ids) + self.hour_emb(hour_ids)
    x = x + meta.unsqueeze(1)
    
    x = self.ssm_fwd(x)
    res_logits = self.output_head(x)
    return base_scores + res_logits
```
*   **Lines 3–5:** Performs projection and metadata embedding injection identical to ProtoSSM.
*   **Line 7:** Runs a forward Selective SSM pass to learn sequential residual connections.
*   **Lines 8–9:** Predicts a residual correction score `res_logits` and sums it directly with the original `base_scores` (classification logits from the Perch backbone).

---

## 4. Calibration & Post-Processing Pipeline

These functions calibrate and refine logit predictions into final submission probabilities.

### `file_confidence_scale()`
```python
def file_confidence_scale(probs: np.ndarray, n_windows: int = 12, top_k: int = 2, power: float = 0.4) -> np.ndarray:
    N, C = probs.shape
    padded, orig_n, _ = _pad_windows_matrix(probs, n_windows)
    view = padded.reshape(-1, n_windows, C)
    sorted_v = np.sort(view, axis=1)
    top_k_mean = sorted_v[:, -top_k:, :].mean(axis=1, keepdims=True)
    out = (view * np.power(top_k_mean, power)).reshape(-1, C)
    return out[:orig_n]
```
*   **Line 3:** Pads the row probability matrix to match file boundaries.
*   **Line 5:** Sorts window predictions per file.
*   **Line 6:** Computes the mean probability of the top-k highest scoring windows for each species.
*   **Line 7:** Scales the probabilities of all windows in that file by the top-k mean raised to a power (hyperparameter). This scales up predictions in files where a species is confidently detected multiple times.

### `apply_threshold_sharpening()`
```python
def apply_threshold_sharpening(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    C = probs.shape[1]
    scaled = np.copy(probs)
    for c in range(C):
        t = thresholds[c]
        above = probs[:, c] > t
        scaled[above, c] = 0.5 + 0.5 * (probs[above, c] - t) / (1 - t + 1e-8)
        scaled[~above, c] = 0.5 * probs[~above, c] / (t + 1e-8)
    return np.clip(scaled, 0.0, 1.0)
```
*   **Line 5:** Iterates over each class threshold $t$.
*   **Lines 6–8:** Maps values above threshold to $[0.5, 1.0]$ and values below to $[0.0, 0.5]$. This sharpens decision boundaries around class-specific optimal thresholds.

### `_adaptive_taxonomy_smoothing()`
```python
def _adaptive_taxonomy_smoothing(probs, cols, genus_alpha=0.12, class_alpha=0.025):
    tax = _load_taxonomy_for_postproc()
    tax_by_label = tax.set_index('primary_label')
    # ... Genus/Class grouping logic ...
    uncertainty = _row_uncertainty(probs)[:, None]
    
    for members in multi_genus.values():
        idx = [col_to_i[m] for m in members if m in col_to_i]
        group_mean = out[:, idx].mean(axis=1, keepdims=True)
        alpha = genus_alpha * uncertainty
        out[:, idx] = (1.0 - alpha) * out[:, idx] + alpha * group_mean
```
*   **Line 4:** Calculates prediction uncertainty for each window (higher uncertainty = higher alpha smoothing).
*   **Lines 7–10:** Computes the average prediction within taxonomic groups (genus/class) and performs an linear interpolation weighted by uncertainty-scaled alphas. This spreads probability mass to taxonomically related species only when the model is uncertain.

---

## 5. Ensembling & Blending

This section covers the final stage of ensembling prediction tables.

### `direct_addsafe()`
```python
def direct_addsafe():
    weight_sum = float(sum(_weights))
    norm_weights = [float(w) / weight_sum for w in _weights]
    dfs = [_read_submission_checked(path) for path in _files_subm]
    # ... Row/Column checks ...
    out = sum(w * df.loc[base_idx, base_cols] for w, df in zip(norm_weights, dfs))
    return out.astype(np.float32)
```
*   **Line 2:** Normalizes blending weights so that they sum to $1.0$.
*   **Line 3:** Reads each model's saved CSV file and runs schemas tests (asserts no NaN, shape mismatch, or duplicate indexes).
*   **Line 5:** Computes a weighted sum of the prediction probabilities across all aligned model DataFrames.
