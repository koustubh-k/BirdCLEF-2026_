# 📖 BirdCLEF Phoenix Pipeline (v14) — Line-by-Line Code Guide

This document provides a detailed breakdown of the core classes, functions, and execution blocks used in the `birdclef-2026-v14.ipynb` pipeline. It also includes instructions for setting up training and submission runs on Kaggle.

---

## 1. Environment & Path Resolution Utilities

### `_find_sample_submission_path()`
```python
def _find_sample_submission_path():
    base_obj = globals().get("BASE", None)
    if base_obj is not None:
        p = Path(base_obj) / "sample_submission.csv"
        if p.exists():
            return p
    for p in [
        Path("/kaggle/input/competitions/birdclef-2026/sample_submission.csv"),
        Path("/kaggle/input/birdclef-2026/sample_submission.csv"),
    ]:
        if p.exists():
            return p
    root = Path("/kaggle/input")
    if root.exists():
        hits = sorted(root.rglob("sample_submission.csv"))
        for p in hits:
            if (p.parent / "taxonomy.csv").exists():
                return p
    return None
```
*   **Lines 2–5:** Probes the global variable `BASE` to see if a base directory is defined and checks if it contains `sample_submission.csv`.
*   **Lines 6–11:** Checks default Kaggle competition mount folder paths.
*   **Lines 12–17:** Performs a recursive search (`rglob`) across `/kaggle/input` for `sample_submission.csv` and returns the path if its parent folder also contains `taxonomy.csv` (verifying it is the correct competition folder).

---

## 2. Feature Extraction Backbone (Perch / Distilled SED)

### `PerchBackbone.predict()`
```python
def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if self.use_onnx:
        outs = self.onnx_session.run(None, {self.onnx_input_name: x})
        logits = outs[self.onnx_out_map["label"]].astype(np.float32)
        emb = outs[self.onnx_out_map["embedding"]].astype(np.float32)
    else:
        out = self.tf_infer_fn(inputs=tf.convert_to_tensor(x))
        logits = out["label"].numpy().astype(np.float32)
        emb = out["embedding"].numpy().astype(np.float32)
    return logits, emb
```
*   **Lines 2–5:** If ONNX runtime is active, executes the session to run inference on input waveform matrix `x` (shape `(B, 160000)`), extracting class logits (shape `(B, 3183)`) and embeddings (shape `(B, 1536)`).
*   **Lines 6–9:** Fallback to standard TensorFlow signature runner if ONNX is disabled or unavailable.

---

## 3. Sequence Modeling (ProtoSSM & ResidualSSM)

### `LightProtoSSM`
```python
class LightProtoSSM(nn.Module):
    def __init__(self, d_input, d_model, d_state, n_classes, n_sites, meta_dim, n_windows=12, use_cross_attn=False):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model)
        )
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        # SelectiveSSM sequence layers
        self.ssm_layers = nn.ModuleList([
            SelectiveSSM(d_model=d_model, d_state=d_state, meta_dim=meta_dim * 2, dropout=0.1)
            for _ in range(2)
        ])
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model))
```
*   **Lines 3–6:** Defines input feature projection layer translating raw embedding dims to Mamba's internal `d_model` size.
*   **Lines 7–8:** Initializes embedding matrices for spatial metadata (`site_emb`) and temporal metadata (`hour_emb`).
*   **Lines 10–13:** Instantiates sequence modeling blocks (`SelectiveSSM` layers) with state space representations and residual dropout layers.
*   **Line 14:** Initializes the class prototype representation matrix used for cosine similarity classification.

---

## 4. Calibration & Post-Processing Pipeline

### `apply_calibration()`
```python
def apply_calibration(probs: np.ndarray, calibrators: Dict[int, IsotonicRegression], taxon_calibrators: Optional[Dict[str, IsotonicRegression]] = None, class_taxons: Optional[List[str]] = None) -> np.ndarray:
    result = probs.copy()
    n_classes = probs.shape[1]
    for c in range(n_classes):
        if c in calibrators:
            result[:, c] = calibrators[c].transform(probs[:, c])
        elif taxon_calibrators is not None and class_taxons is not None:
            taxon = class_taxons[c]
            if taxon in taxon_calibrators:
                result[:, c] = taxon_calibrators[taxon].transform(probs[:, c])
    return np.clip(result, 0.0, 1.0)
```
*   **Line 4:** Iterates over each classification column.
*   **Lines 5–6:** Applies class-specific isotonic regression models fit during cross-validation OOF tuning.
*   **Lines 7–10:** If a class had insufficient positive training samples for a custom model, falls back to an aggregated taxon-level calibrator (e.g. shared calibration model for all related birds in the same family).

### `f_TAX_SMOOTHING_POSTPROC()` (v14 Version)
```python
def f_TAX_SMOOTHING_POSTPROC(func_add=direct, genus_α=0.15, class_α=0.05):
    submission = func_add()
    submission = _to_indexed_prob_frame(submission)
    # ... Genus/Class taxonomy mapping loading ...
    probs = submission.to_numpy(dtype=np.float32, copy=True)
    cols = list(submission.columns)
    
    # Adaptive taxonomy smoothing loops
    for members in multi_genus.values():
        idx = [col_to_i[m] for m in members if m in col_to_i]
        group_mean = probs[:, idx].mean(axis=1, keepdims=True)
        probs[:, idx] = (1.0 - genus_α) * probs[:, idx] + genus_α * group_mean
        
    for members in multi_class.values():
        idx = [col_to_i[m] for m in members if m in col_to_i]
        group_mean = probs[:, idx].mean(axis=1, keepdims=True)
        probs[:, idx] = (1.0 - class_α) * probs[:, idx] + class_α * group_mean
        
    return pd.DataFrame(np.clip(probs, 0.0, 1.0), index=submission.index, columns=cols)
```
*   **Line 2:** Computes the ensembled raw predictions.
*   **Lines 9–12:** Blends predictions of species belonging to the same Genus taxonomy grouping using the global smoothing weight `genus_α`.
*   **Lines 14–17:** Blends predictions of species in the same broad taxonomic Class using `class_α`.

---

## 5. Kaggle Execution Workflow: How to Train and Submit

The notebook `birdclef-2026-v14.ipynb` supports both cross-validation training and final submit modes.

### A. How to Train New Models
1.  **Mount Data Sources**:
    *   Upload the target competition dataset (`birdclef-2026`).
    *   Mount the cache datasets containing pre-extracted features (such as `jaejohn/perch-meta`).
2.  **Configure Target Model Cell**:
    *   Go to the target execution block (for example, Cell 23 / `Model_74`).
    *   Set **`MODE = "train"`** (located at the top of the cell block):
        ```python
        MODE = "train"
        ```
3.  **Execute Cells**:
    *   Run the configuration and parameter setup cells (Cells 4–5).
    *   Run the target model cell.
    *   The model will split the cached training data (`meta_tr`, `emb_tr`, `sc_tr`) into folds using `GroupKFold` (based on audio filenames).
    *   It will fit:
        *   `train_light_proto_ssm`: Training Mamba sequence state-space model weights.
        *   `train_mlp_probes`: Fitting multi-layer perceptron dense classifiers.
        *   `train_residual_ssm`: Fitting residual correction sequence networks.
    *   It will output validation out-of-fold metrics (Macro AUC) and save the trained weights (e.g. `proto_ssm_best.pt` and `residual_ssm_best.pt`) to `/kaggle/working/`.

### B. How to Submit to Kaggle
1.  **Configure Target Model Cell**:
    *   Set **`MODE = "submit"`** inside the model cell.
    *   Verify that `solutions['Models']` contains the correct weights file paths pointing to your trained checkpoints (which can be uploaded as a private Kaggle dataset).
2.  **Save Notebook (Commit Run)**:
    *   Click **"Save Version"** in Kaggle.
    *   The notebook will run on the public test set, which has 0 files. The script will detect this dry-run environment and automatically fall back to the first 5 ogg files in `train_soundscapes` to generate a dummy `submission.csv` matching the `sample_submission.csv` row and column layout.
    *   This ensures the save completes successfully.
3.  **Submit to Leaderboard**:
    *   Once the save run completes, go to the notebook viewer and click **"Submit to Competition"**.
    *   Kaggle will swap in the hidden test set, run the code with `MODE = "submit"`, perform actual sequence predictions on the test soundscapes, and output the final prediction table.
