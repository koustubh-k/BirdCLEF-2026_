# BirdCLEF 2026 — Phoenix Pipeline Dependency Graph

This document details the compile-time and execution-time dependencies across modules in the Phoenix architecture.

---

## 1. High-Level Flow Chart

The diagram below maps the execution sequence and dependencies from raw input audio through feature extraction, models, calibration, ensembling, and post-processing.

```mermaid
graph TD
    %% Audio & Config Inputs
    A["Audio Inputs (.ogg)"] -->|Audio Waveform| C["dataset.py<br/>(Audio & Meta Loader)"]
    B["config.py<br/>(Execution Constants)"] -->|Configuration CFG| C
    B -->|Configuration CFG| D["features.py<br/>(Backbone Extractors)"]
    B -->|Configuration CFG| E["models.py<br/>(PyTorch Architectures)"]
    B -->|Configuration CFG| F["training.py<br/>(Optimization Loops)"]
    B -->|Configuration CFG| G["validation.py<br/>(CV Fold Evaluators)"]
    B -->|Configuration CFG| H["postprocessing.py<br/>(Calibration & Smoothing)"]

    %% Feature Processing
    C -->|Waveform Batch| D
    D -->|Perch & Teacher Embeddings| E
    D -->|Genus-Proxy Logits| E

    %% Sequence Models
    E -->|LightProtoSSM| I["training.py / validation.py<br/>(Model Validation & Tuning)"]
    E -->|ResidualSSM| I
    E -->|EmbeddingRetrievalHead| I
    E -->|VectorizedMLPProbes| I

    %% Inference Pipeline
    I -->|Trained Models & Weights| J["inference.py<br/>(TTA Inference Coordinator)"]
    J -->|Raw Out-of-Fold Predictions| K["postprocessing.py<br/>(Taxon Tuning & Calibration)"]

    %% Ensembling and Post-processing
    K -->|Calibrated Probabilities| L["ensemble.py<br/>(Splitblend coordinator)"]
    L -->|Merged Probabilities| M["postprocessing.py<br/>(Taxonomy & Temporal Smoothing)"]
    M -->|Final Probabilities| N["submission.csv<br/>(Submission File)"]
```

---

## 2. Module Dependency Matrix

| Target Module | Depends On | Purpose of Dependency |
|---|---|---|
| [config.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/src/config.py) | *None* | Serves as the central repository for execution constants, settings, and model configurations. |
| [dataset.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/src/dataset.py) | [config.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/src/config.py) | Audio sampling rates, windows constants, class sizes, and batch settings. |
| [augmentation.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/src/augmentation.py) | *None* | Implements sequence rolls and bidirectional horizontal flip augmentations on raw arrays. |
| [features.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/src/features.py) | [config.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/src/config.py), [dataset.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/src/dataset.py) | Uses dataset.py utilities to batch-load raw 60-second audio files for ONNX extraction. |
| [models.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/src/models.py) | [config.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/src/config.py) | Configures sequence dimensions and class prototype counts dynamically. |
| [training.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/src/training.py) | [config.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/src/config.py), [models.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/src/models.py) | Accesses model parameters, optimizer architectures, learning rates, loss functions, and weight averaging parameters. |
| [validation.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/src/validation.py) | [config.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/src/config.py) | Cross-validation fold splitting strategies (GroupKFold by recording site). |
| [inference.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/src/inference.py) | [config.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/src/config.py), [models.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/src/models.py), [augmentation.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/src/augmentation.py) | Applies test-time sequence shift augmentations, executes forward inference sessions, and runs error correction. |
| [ensemble.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/src/ensemble.py) | [config.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/src/config.py) | Splits ensembling weights across taxons / scientific class indices. |
| [postprocessing.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/src/postprocessing.py) | [config.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/src/config.py) | Scales and smooths raw output logits dynamically using site/hour prior lookups and isotonic regression. |
| [main.py](file:///c:/Users/koust/Documents/Kaggle/Bird_AudioPred/refactored_codebase/main.py) | *All source modules* | Coordinates the entire execution, training, OOF tuning, ensembling, postprocessing, and evaluation pipeline. |
