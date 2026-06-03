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

## 4. Calibration, Blending & Post-Processing Pipeline

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

### Tuned Ensembling & Post-Processing Optimizations

To maximize leaderboard performance, the following high-yield adjustments have been implemented directly in the pipeline code:

1.  **xSED Ensemble Weight Shift (Tactic 4):**
    The blending weights inside `solutions['Models']` for `Model_74` are shifted from `[0.60, 0.40]` to `[0.65, 0.35]`. This places higher confidence (65%) on the sequence-aware ProtoSSM predictions than the frame-wise SED predictions (`35%`), enhancing generalization on the hidden test soundscapes.
2.  **Increased Rank-Aware Scaling Power (Tactic 3):**
    The power scalar used in `rank_aware_scaling(probs, n_windows=12, power=power)` has been pushed from `0.60` (or `0.65`) to **`0.68`** across active models (`Model_51` and `Model_74`). This aggressive scaling dampens predictions in files with low overall confidence.
3.  **Tuned Japanese Amendment Gates (Tactic 2):**
    The logic gates controlling the blend between the SED and ProtoSSM models have been optimized.
    *   **Original:** `proto_cont = (xctx > 0.88) & (rank_proto > 0.77) & (p_sed < 0.14) & (~fake_only)`
    *   **Optimized:** `proto_cont = (xctx > 0.90) & (rank_proto > 0.77) & (p_sed < 0.15) & (~fake_only)`
    *   *Rationale:* Relaxing the SED constraint slightly (`p_sed < 0.15`) captures faint calls, while tightening the temporal context threshold (`xctx > 0.90`) guards against isolated false positives.

---

## 5. Kaggle Execution Workflow: How to Train and Submit

The notebook `birdclef-2026-v14.ipynb` features a centralized **GLOBAL PIPELINE CONTROLLER** (Cell 1, located at the very top of the notebook). You can manage training, submission, weight bypasses, and threshold overrides globally from this single block.

### A. How to Train New Models
1.  **Mount Data Sources**:
    *   Upload the target competition dataset (`birdclef-2026`).
    *   Mount the cache datasets containing pre-extracted features (such as `jaejohn/perch-meta`).
2.  **Configure Global Controller Cell (Cell 1)**:
    *   Set `GLOBAL_MODE = "train"`
    *   Set `FORCE_TRAIN_FROM_SCRATCH = True` (to bypass loading pre-trained weights)
3.  **Execute Cells**:
    *   Run the notebook cells sequentially.
    *   The active model blocks (`Model_22`, `Model_51`, `Model_74`) will read these global variables, bypass weights loading, train Mamba/ResidualSSM/MLP classifiers on the fly, output validation metrics, calibrate thresholds (generating the 234-element array in the logs), and save files to `/kaggle/working/`.

### B. How to Submit to Kaggle (with Threshold Injection)
1.  **Configure Global Controller Cell (Cell 1)**:
    *   Set `GLOBAL_MODE = "submit"`
    *   Set `FORCE_TRAIN_FROM_SCRATCH = False` (to load pre-trained weights)
    *   Copy the 234-element threshold array from your training log, and paste it to define `GLOBAL_INJECTED_THRESHOLDS`:
        ```python
        import numpy as np
        GLOBAL_INJECTED_THRESHOLDS = np.array([0.312, 0.450, 0.225, ..., 0.500]) # 234 elements
        ```
2.  **Save Notebook (Commit Run)**:
    *   Click **"Save Version"** in Kaggle.
    *   The notebook will run on the public test set, detect the dry-run, and output a dummy `submission.csv` matching `sample_submission.csv`.
3.  **Submit to Leaderboard**:
    *   Once the save run completes, click **"Submit to Competition"** in the notebook viewer.
    *   Kaggle will swap in the hidden test set, run sequence predictions, apply the globally injected thresholds, and generate the final output.


