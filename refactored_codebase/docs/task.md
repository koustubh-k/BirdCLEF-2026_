# BirdCLEF 2026 — Refactoring Tasks

## Phase 1: Modular Refactoring (B & C)
- [x] Create `refactored_codebase/src/config.py` with model-specific execution parameters
- [x] Create `refactored_codebase/src/dataset.py` with audio loading and metadata parsing
- [x] Create `refactored_codebase/src/augmentation.py` with TTA circular/flip shifts
- [x] Create `refactored_codebase/src/features.py` with ONNX session runners and embedding loading
- [x] Create `refactored_codebase/src/models.py` with SelectiveSSM, LightProtoSSM, and ResidualSSM
- [x] Create `refactored_codebase/src/training.py` with training loops and SWA
- [x] Create `refactored_codebase/src/validation.py` with evaluation and AUC calculations
- [x] Create `refactored_codebase/src/inference.py` with test pipeline coordinator
- [x] Create `refactored_codebase/src/ensemble.py` with blend algorithms
- [x] Create `refactored_codebase/src/postprocessing.py` with calibration and taxonomy smoothing

## Phase 2: Documentation & Architecture Reports (A, C, D, F)
- [x] Generate `refactored_codebase/docs/Architecture_Doc.md`
- [x] Generate `refactored_codebase/Refactoring_Report.md` (graphs, diagrams, execution flow)

## Phase 3: Notebook Restructuring (E)
- [x] Generate restructured Jupyter Notebook `refactored_codebase/birdclef_refactored.ipynb`
- [x] Add explainers before every cell

## Phase 4: Validation & Cleanup
- [x] Run `test_postprocessing.py` modified to test modular imports
- [x] Verify functionality remains identical

## Phase 5: Advanced Model Improvements (1 → 2 → 3 → 6 → 5 → 4)
- [x] Extend `config.py` with hyperparameter toggles and constants (k-NN, Teacher model paths, Calibration toggles)
- [x] Implement `EmbeddingRetrievalHead` class in `models.py` for rare/unmapped classes (Improvement 5)
- [x] Implement `MultiBackboneExtractor` in `features.py` to optionally load and fuse distilled teacher features (Improvement 4)
- [x] Implement uncertainty-gated taxonomy smoothing using predictive entropy and TTA variance in `postprocessing.py` (Improvement 6)
- [x] Implement shared taxon-level isotonic calibration for rare classes in `postprocessing.py` (Improvement 2)
- [x] Implement OOF grid search optimization for `rank_aware_power` and `lambda_prior` per taxon (Improvement 3)
- [x] Ensure full validation transform parity (Improvement 1) and update `inference.py` to support new components

Phase 5 completion note (2026-05-31):
- Retrieval blend now targets both unmapped and rare classes (`<10` positives), aligned with the implementation plan.
- Teacher-mode dimension handling validated (1536D and 2304D paths).
- Postprocessing parity test and compile checks passed after integration.

## Phase 6: Local Entry Runner & Kaggle Parity
- [x] Write dynamic Kaggle input path resolver in `config.py` / `main.py` referencing `KAGGLE_INPUTS.md`
- [x] Implement `main.py` runner to execute the full local training, OOF validation, tuning, and inference pipeline
- [x] Verify `main.py` and `test_postprocessing.py` execute and exit successfully
- [x] Fix Kaggle submission failure: implement actual test set feature extraction and resolve calibration shape mismatch on hidden test set

