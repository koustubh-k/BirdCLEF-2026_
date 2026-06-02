"""inference.py — Inference coordination and test-time augmentation (TTA) for BirdCLEF models.

This module coordinates test-time predictions, executing temporal shift/flip TTA sequences
for ProtoSSM and applying residual SSM corrections.
"""

from typing import List, Tuple, Dict, Any, Union, Optional
import numpy as np
import torch
import torch.nn as nn

from .models import LightProtoSSM, ResidualSSM


def run_tta_proto(
    proto_model: nn.Module,
    emb_files: np.ndarray,
    sc_files: np.ndarray,
    site_t: torch.Tensor,
    hour_t: torch.Tensor,
    shifts: List[int] = [0, 1, -1, 2, -2]
) -> Tuple[np.ndarray, np.ndarray]:
    """Executes sequence-level temporal shift and flip TTA on LightProtoSSM.

    Summary:
        Augments the sequence embeddings by shifting and flipping along the temporal dimension,
        computes predictions, aligns them back, and averages to reduce prediction variance.
        Also calculates the variance across TTA passes in probability space.

    Inputs:
        proto_model (nn.Module): Trained LightProtoSSM model.
        emb_files (np.ndarray): Grouped Perch embeddings per file.
        sc_files (np.ndarray): Grouped Perch score logits per file.
        site_t (torch.Tensor): Integer site identifiers.
        hour_t (torch.Tensor): Integer hour identifiers.
        shifts (List[int]): Shifting window offsets. Default is [0, 1, -1, 2, -2].

    Outputs:
        Tuple[np.ndarray, np.ndarray]:
            - mean_logits: TTA-averaged prediction logits array.
            - variance_probs: Variance across TTA passes in probability space.

    Shapes:
        - Input `emb_files`: `(N_files, 12, 1536)` or `(N_files, 12, 2304)`
        - Input `sc_files`: `(N_files, 12, N_classes)`
        - Output mean_logits: `(N_files, 12, N_classes)`
        - Output variance_probs: `(N_files, 12, N_classes)`
    """
    proto_model.eval()
    all_preds = []

    emb_t = torch.tensor(emb_files, dtype=torch.float32)
    sc_t = torch.tensor(sc_files, dtype=torch.float32)

    for shift in shifts:
        e = torch.roll(emb_t, shift, dims=1) if shift else emb_t
        s = torch.roll(sc_t, shift, dims=1) if shift else sc_t
        with torch.no_grad():
            out = proto_model(e, s, site_ids=site_t, hours=hour_t).cpu().numpy()
        if shift:
            out = np.roll(out, -shift, axis=1)
        all_preds.append(out)

    # Temporal flip as an extra TTA pass
    with torch.no_grad():
        out_flip = proto_model(
            emb_t.flip(1), sc_t.flip(1), site_ids=site_t, hours=hour_t
        ).cpu().numpy()
    all_preds.append(out_flip[:, ::-1, :].copy())

    mean_logits = np.mean(all_preds, axis=0)

    # Compute variance in probability space for gated taxonomy smoothing
    probs_passes = [1.0 / (1.0 + np.exp(-np.clip(p, -30, 30))) for p in all_preds]
    variance_probs = np.var(probs_passes, axis=0)

    return mean_logits, variance_probs


def run_sequence_inference(
    proto_model: nn.Module,
    res_model: nn.Module,
    emb_te: np.ndarray,
    sc_te: np.ndarray,
    sc_te_adjusted: np.ndarray,
    test_site_ids: np.ndarray,
    test_hour_ids: np.ndarray,
    mapped_mask: np.ndarray,
    correction_weight: float,
    retrieval_head: Optional[nn.Module] = None,
    shifts: List[int] = [0, 1, -1, 2, -2],
    n_windows: int = 12,
    n_classes: int = 234
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Runs the sequential inference pipeline combining ProtoSSM, ResidualSSM, and RetrievalHead.

    Summary:
        Sequentially runs TTA sequence classification, blends first-pass predictions,
        corrects predictions with ResidualSSM, and computes retrieval-augmented predictions.

    Inputs:
        proto_model (nn.Module): Trained LightProtoSSM model.
        res_model (nn.Module): Trained ResidualSSM model.
        emb_te (np.ndarray): Flat test embeddings.
        sc_te (np.ndarray): Flat raw test logits from Perch.
        sc_te_adjusted (np.ndarray): Prior-adjusted and MLP-stacked test scores.
        test_site_ids (np.ndarray): File site IDs.
        test_hour_ids (np.ndarray): File hour IDs.
        mapped_mask (np.ndarray): Boolean mask indicating if class maps to Perch.
        correction_weight (float): Blending factor for ResidualSSM outputs.
        retrieval_head (Optional[nn.Module]): Trained EmbeddingRetrievalHead module. Default is None.
        shifts (List[int]): Temporal window shifts for ProtoSSM TTA. Default is [0, 1, -1, 2, -2].
        n_windows (int): Length of sequence in windows. Default is 12.
        n_classes (int): Number of target classes. Default is 234.

    Outputs:
        Tuple: Contains:
            - final_scores (np.ndarray): Aligned final logit predictions.
            - first_pass_flat (np.ndarray): Intermediate first-pass blended predictions.
            - tta_variance (np.ndarray): Flattened TTA variance in probability space.
            - retrieval_probs (Optional[np.ndarray]): Retrieval probability votes.

    Shapes:
        - Input `emb_te`: `(N_samples, D_emb)` where `N_samples` is divisible by `n_windows`.
        - Input `sc_te`: `(N_samples, n_classes)`
        - Output `final_scores`: `(N_samples, n_classes)`
        - Output `tta_variance`: `(N_samples, n_classes)`
    """
    n_test_files = len(sc_te) // n_windows
    emb_te_f = emb_te.reshape(n_test_files, n_windows, -1)
    sc_te_f = sc_te.reshape(n_test_files, n_windows, -1)

    site_t = torch.tensor(test_site_ids, dtype=torch.long)
    hour_t = torch.tensor(test_hour_ids, dtype=torch.long)

    # Step 1: Run ProtoSSM TTA inference and get variance
    proto_out, variance_out = run_tta_proto(
        proto_model,
        emb_te_f,
        sc_te_f,
        site_t=site_t,
        hour_t=hour_t,
        shifts=shifts
    )
    proto_scores_flat = proto_out.reshape(-1, n_classes).astype(np.float32)
    tta_variance = variance_out.reshape(-1, n_classes).astype(np.float32)

    # Step 2: Blend first-pass predictions
    ensemble_w_per_class = np.where(mapped_mask, 0.60, 0.35).astype(np.float32)
    first_pass_flat = (
        ensemble_w_per_class[None, :] * proto_scores_flat
        + (1.0 - ensemble_w_per_class)[None, :] * sc_te_adjusted
    )

    # Step 3: Run ResidualSSM correction pass
    first_pass_te_f = first_pass_flat.reshape(n_test_files, n_windows, -1)
    res_model.eval()
    with torch.no_grad():
        test_correction = res_model(
            torch.tensor(emb_te_f, dtype=torch.float32),
            torch.tensor(first_pass_te_f, dtype=torch.float32),
            site_ids=site_t,
            hours=hour_t,
        ).cpu().numpy()

    correction_flat = test_correction.reshape(-1, n_classes).astype(np.float32)
    final_scores = first_pass_flat + correction_weight * correction_flat

    # Step 4: Run retrieval head (if available)
    retrieval_probs = None
    if retrieval_head is not None:
        retrieval_head.eval()
        with torch.no_grad():
            retrieval_probs = retrieval_head(torch.tensor(emb_te, dtype=torch.float32)).cpu().numpy()

    return final_scores, first_pass_flat, tta_variance, retrieval_probs
