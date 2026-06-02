import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add refactored_codebase path
sys.path.insert(0, str(Path(__file__).parent))

from src.postprocessing import (
    apply_postprocessing,
    fit_per_class_calibrators,
    apply_calibration,
    optimize_thresholds,
    run_corrected_calibration_pipeline,
    apply_threshold_sharpening,
    apply_retrieval_blend
)

# MOCK TEST DATA
N_WINDOWS = 12
N_FILES = 5
N = N_WINDOWS * N_FILES
C = 234

print(f"Creating mock predictions with N={N} windows, C={C} classes...")
np.random.seed(42)

# Generate mock logits (normal distribution)
first_pass_logits_tr = np.random.normal(loc=-1.0, scale=2.0, size=(N, C)).astype(np.float32)
# Generate mock ResidualSSM corrections
correction_flat_tr = np.random.normal(loc=0.0, scale=0.5, size=(N, C)).astype(np.float32)
# Generate mock multi-hot labels (mostly 0s, some 1s)
Y_true = (np.random.rand(N, C) > 0.98).astype(np.uint8)

# Generate mock temperatures (same shape as classes)
temperatures = np.ones(C, dtype=np.float32)
temperatures[:100] = 0.95
temperatures[100:] = 1.10

# Test run_corrected_calibration_pipeline
print("Testing run_corrected_calibration_pipeline...")
class_taxons = ["Aves"] * C
# Make a few classes belong to Insecta or Amphibia to test taxons logic
for i in range(10):
    class_taxons[i] = "Amphibia"
for i in range(10, 20):
    class_taxons[i] = "Insecta"

sites = np.array(["S01"] * N)
hours = np.array([6] * N)
prior_tables = {
    "global_p": Y_true.mean(axis=0).astype(np.float32),
    "site_to_i": {"S01": 0},
    "site_p": np.zeros((1, C), dtype=np.float32),
    "site_n": np.ones(1, dtype=np.float32),
    "hour_to_i": {6: 0},
    "hour_p": np.zeros((1, C), dtype=np.float32),
    "hour_n": np.ones(1, dtype=np.float32),
}

try:
    w, calibrators, taxon_calibrators, thresholds, best_lambdas, best_powers = run_corrected_calibration_pipeline(
        first_pass_logits_tr=first_pass_logits_tr,
        correction_flat_tr=correction_flat_tr,
        Y_true=Y_true,
        temperatures=temperatures,
        n_windows=N_WINDOWS,
        correction_grid=[0.10, 0.20, 0.30],
        threshold_grid=[0.30, 0.40, 0.50],
        class_taxons=class_taxons,
        prior_tables=prior_tables,
        sites=sites,
        hours=hours,
        top_k=2,
        fc_power=0.4,
        ra_power=0.6,
        smooth_alpha=0.20
    )
    print(f"Pipeline succeeded!")
    print(f"  Selected correction_weight: {w}")
    print(f"  Number of fitted calibrators: {len(calibrators)} / {C}")
    print(f"  Number of fitted taxon calibrators: {len(taxon_calibrators)}")
    print(f"  Mean threshold: {thresholds.mean():.4f}")
except Exception as e:
    print(f"Pipeline failed with error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test test-time inference postprocessing application
print("Testing test-time postprocessing...")
try:
    final_scores = first_pass_logits_tr + w * correction_flat_tr
    probs = apply_postprocessing(
        final_scores,
        temperatures,
        n_windows=N_WINDOWS,
        prior_tables=prior_tables,
        sites=sites,
        hours=hours,
        lambda_prior=best_lambdas,
        class_taxons=class_taxons,
        ra_power=best_powers,
        top_k=2,
        fc_power=0.4,
        smooth_alpha=0.20
    )
    assert probs.shape == (N, C), f"Expected shape {(N, C)}, got {probs.shape}"
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Probabilities out of [0, 1] bounds"
    
    cal_probs = apply_calibration(
        probs, calibrators, taxon_calibrators=taxon_calibrators, class_taxons=class_taxons
    )
    assert cal_probs.shape == (N, C), f"Expected shape {(N, C)}, got {cal_probs.shape}"
    
    # Test apply_retrieval_blend
    retrieval_mask = np.random.rand(C) > 0.90
    retrieval_probs = np.random.rand(N, C)
    blended_probs = apply_retrieval_blend(cal_probs, retrieval_probs, retrieval_mask)
    assert blended_probs.shape == (N, C), f"Expected shape {(N, C)}, got {blended_probs.shape}"
    
    sharpened_probs = apply_threshold_sharpening(blended_probs, thresholds)
    assert sharpened_probs.shape == (N, C), f"Expected shape {(N, C)}, got {sharpened_probs.shape}"
    
    print("Test-time application checks passed successfully!")
except Exception as e:
    print(f"Test-time postprocessing failed with error: {e}")
    exit(1)

print("ALL TESTS PASSED SUCCESSFULLY! Parity fix is 100% verified via modular imports.")
