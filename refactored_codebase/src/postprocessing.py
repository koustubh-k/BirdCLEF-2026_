"""postprocessing.py — Post-processing and calibration pipeline for BirdCLEF models.

This module houses post-inference adjustment algorithms, including prior mapping,
isotonic regression calibration, F1-based threshold sharpening, and adaptive
uncertainty-gated taxonomy/temporal smoothing.
"""

from typing import Dict, List, Tuple, Union, Optional, Any
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function.

    Summary:
        Computes the logistic sigmoid function, clamping inputs to avoid overflow/underflow.

    Inputs:
        x (np.ndarray): Input values.

    Outputs:
        np.ndarray: Bounded sigmoid values in [0, 1].

    Shapes:
        - Input `x`: Arbitrary shape.
        - Output: Same shape as `x`.

    Side effects:
        None.

    Usage example:
        >>> sigmoid(np.array([0.0, 2.0, -100.0]))
        array([5.0000000e-01, 8.8079708e-01, 3.7200760e-44], dtype=float32)
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def file_confidence_scale(probs: np.ndarray, n_windows: int = 12, top_k: int = 2, power: float = 0.4) -> np.ndarray:
    """Scales predictions in each window by the top-K mean confidence of the file.

    Summary:
        Applies a confidence multiplier to suppress frames in files where the class is not confident.

    Inputs:
        probs (np.ndarray): Probability scores.
        n_windows (int): Length of sequence in windows. Default is 12.
        top_k (int): Number of top windows to average. Default is 2.
        power (float): Scaling exponent. Default is 0.4.

    Outputs:
        np.ndarray: Bounded scaled probabilities.

    Shapes:
        - Input `probs`: `(N_samples, N_classes)` where N_samples is divisible by `n_windows`.
        - Output: `(N_samples, N_classes)`

    Side effects:
        None.

    Usage example:
        >>> p = np.random.rand(24, 2)
        >>> scaled = file_confidence_scale(p, n_windows=12, top_k=2)
    """
    N, C = probs.shape
    view = probs.reshape(-1, n_windows, C)
    sorted_v = np.sort(view, axis=1)
    top_k_mean = sorted_v[:, -top_k:, :].mean(axis=1, keepdims=True)
    return (view * np.power(top_k_mean, power)).reshape(N, C)


def rank_aware_scaling(probs: np.ndarray, n_windows: int = 12, power: float = 0.6) -> np.ndarray:
    """Scales window-level scores by (file_max)^power to push confident files higher.

    Summary:
        Suppresses false positives in noisy or unconfident recordings by scaling with peak file probabilities.

    Inputs:
        probs (np.ndarray): Bounded probability scores.
        n_windows (int): Windows sequence length. Default is 12.
        power (float): Exponent scaling parameter. Default is 0.6.

    Outputs:
        np.ndarray: Rank-aware scaled probabilities.

    Shapes:
        - Input `probs`: `(N_samples, N_classes)`
        - Output: `(N_samples, N_classes)`

    Side effects:
        None.

    Usage example:
        >>> p = np.random.rand(12, 2)
        >>> scaled = rank_aware_scaling(p, power=0.5)
    """
    N, C = probs.shape
    view = probs.reshape(-1, n_windows, C)
    file_max = view.max(axis=1, keepdims=True)
    return (view * np.power(file_max, power)).reshape(N, C)


def adaptive_delta_smooth(probs: np.ndarray, n_windows: int = 12, base_alpha: float = 0.20) -> np.ndarray:
    """Confidence-gated temporal smoothing across adjacent windows.

    Summary:
        Smoothes predictions over adjacent windows, scaling the smoothing strength inversely
        proportional to prediction confidence to protect sharp confident transients.

    Inputs:
        probs (np.ndarray): Bounded prediction probabilities.
        n_windows (int): Windows sequence length. Default is 12.
        base_alpha (float): Peak smoothing factor. Default is 0.20.

    Outputs:
        np.ndarray: Smoothed prediction probabilities.

    Shapes:
        - Input `probs`: `(N_samples, N_classes)`
        - Output: `(N_samples, N_classes)`

    Side effects:
        None.

    Usage example:
        >>> p = np.random.rand(24, 2)
        >>> smoothed = adaptive_delta_smooth(p, base_alpha=0.15)
    """
    N, C = probs.shape
    result = probs.copy()
    view = probs.reshape(-1, n_windows, C)
    out = result.reshape(-1, n_windows, C)
    for t in range(n_windows):
        conf = view[:, t, :].max(axis=-1, keepdims=True)
        alpha = base_alpha * (1.0 - conf)
        if t == 0:
            neighbor_avg = (view[:, t, :] + view[:, t + 1, :]) / 2.0
        elif t == n_windows - 1:
            neighbor_avg = (view[:, t - 1, :] + view[:, t, :]) / 2.0
        else:
            neighbor_avg = (view[:, t - 1, :] + view[:, t + 1, :]) / 2.0
        out[:, t, :] = (1.0 - alpha) * view[:, t, :] + alpha * neighbor_avg
    return result


def apply_postprocessing(
    logits: np.ndarray,
    temperatures: np.ndarray,
    n_windows: int = 12,
    top_k: int = 2,
    fc_power: float = 0.4,
    ra_power: Union[float, Dict[str, float]] = 0.6,
    smooth_alpha: float = 0.20,
    prior_tables: Optional[Dict[str, Any]] = None,
    sites: Optional[np.ndarray] = None,
    hours: Optional[np.ndarray] = None,
    lambda_prior: Union[float, Dict[str, float]] = 0.0,
    class_taxons: Optional[List[str]] = None
) -> np.ndarray:
    """Executes the standard post-inference scaling and temporal smoothing pipeline.

    Summary:
        Sequentially runs prior adjustments, temperature scaling, sigmoid activations,
        file confidence scaling, rank-aware scaling, and adaptive temporal smoothing,
        supporting taxon-specific prior weights and scaling exponents.

    Inputs:
        logits (np.ndarray): Unscaled model prediction scores.
        temperatures (np.ndarray): Class-specific temperature scale factors.
        n_windows (int): Windows count per file. Default is 12.
        top_k (int): Top windows count for file confidence scaling. Default is 2.
        fc_power (float): Exponent for file confidence scaling. Default is 0.4.
        ra_power (Union[float, Dict[str, float]]): Exponent for rank-aware scaling. Default is 0.6.
        smooth_alpha (float): Peak temporal smoothing factor. Default is 0.20.
        prior_tables (Optional[Dict[str, Any]]): Reference prior tables dictionary. Default is None.
        sites (Optional[np.ndarray]): Array of site labels. Default is None.
        hours (Optional[np.ndarray]): Array of recording hours. Default is None.
        lambda_prior (Union[float, Dict[str, float]]): Weight factor for prior adjustments. Default is 0.0.
        class_taxons (Optional[List[str]]): List of taxon strings matching class indices. Default is None.

    Outputs:
        np.ndarray: Bounded prediction probabilities.

    Shapes:
        - Input `logits`: `(N_samples, N_classes)`
        - Input `temperatures`: `(N_classes,)`
        - Output: `(N_samples, N_classes)`
    """
    n_samples, n_classes = logits.shape

    # Step 1: Prior logit adjustment
    scores = logits.copy()
    if prior_tables is not None and sites is not None and hours is not None:
        if isinstance(lambda_prior, dict) and class_taxons is not None:
            for c in range(n_classes):
                taxon = class_taxons[c]
                lam = lambda_prior.get(taxon, 0.4)
                if lam > 0:
                    col_prior = apply_prior(scores[:, c:c+1], sites, hours, prior_tables, lambda_prior=lam, class_idx=c)
                    scores[:, c] = col_prior[:, 0]
        elif isinstance(lambda_prior, (int, float)) and lambda_prior > 0:
            scores = apply_prior(scores, sites, hours, prior_tables, lambda_prior=lambda_prior)

    # Step 2: Temperature scaling
    scores = scores / temperatures[None, :]

    # Step 3: Convert to probabilities
    probs = sigmoid(scores)

    # Step 4: File-level confidence scaling
    probs = file_confidence_scale(probs, n_windows=n_windows, top_k=top_k, power=fc_power)

    # Step 5: Rank-aware scaling (with optional taxon-specific exponents)
    if isinstance(ra_power, dict) and class_taxons is not None:
        view = probs.reshape(-1, n_windows, n_classes)
        file_max = view.max(axis=1, keepdims=True)
        class_powers = np.array([ra_power.get(class_taxons[c], 0.6) for c in range(n_classes)], dtype=np.float32)
        scaled = view * np.power(file_max, class_powers[None, None, :])
        probs = scaled.reshape(n_samples, n_classes)
    else:
        power_val = float(ra_power) if not isinstance(ra_power, dict) else 0.6
        probs = rank_aware_scaling(probs, n_windows=n_windows, power=power_val)

    # Step 6: Adaptive temporal smoothing
    probs = adaptive_delta_smooth(probs, n_windows=n_windows, base_alpha=smooth_alpha)

    return np.clip(probs, 0.0, 1.0)


def fit_per_class_calibrators(
    oof_probs: np.ndarray,
    Y_true: np.ndarray,
    class_taxons: Optional[List[str]] = None,
    n_windows: int = 12,
    min_pos: int = 5
) -> Tuple[Dict[int, IsotonicRegression], Dict[str, IsotonicRegression]]:
    """Fits isotonic regression calibrators on out-of-fold validation scores.

    Summary:
        Trains individual class calibrators for frequent classes, and fits shared taxon-level
        calibrators for rare classes to prevent validation overfitting.

    Inputs:
        oof_probs (np.ndarray): Post-processed OOF prediction probabilities.
        Y_true (np.ndarray): Target multi-hot labels.
        class_taxons (Optional[List[str]]): List of taxon strings matching class indices. Default is None.
        n_windows (int): Window sequence length. Default is 12.
        min_pos (int): Minimum positive labels count required to calibrate individually. Default is 5.

    Outputs:
        Tuple: Contains:
            - calibrators (Dict[int, IsotonicRegression]): Individual class calibrators.
            - taxon_calibrators (Dict[str, IsotonicRegression]): Shared taxon calibrators.
    """
    n_classes = oof_probs.shape[1]
    calibrators = {}
    taxon_calibrators = {}

    # 1. Fit individual class calibrators
    for c in range(n_classes):
        y_true_c = Y_true[:, c]
        y_pred_c = oof_probs[:, c]

        if y_true_c.sum() < min_pos:
            continue

        try:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(y_pred_c, y_true_c)
            calibrators[c] = ir
        except Exception:
            continue

    # 2. Fit shared taxon-level calibrators for classes with insufficient positives
    if class_taxons is not None:
        taxon_to_rare_classes = {}
        for c in range(n_classes):
            if c not in calibrators:
                taxon = class_taxons[c]
                taxon_to_rare_classes.setdefault(taxon, []).append(c)

        for taxon, class_indices in taxon_to_rare_classes.items():
            y_true_list = []
            y_pred_list = []
            for c in class_indices:
                y_true_list.append(Y_true[:, c])
                y_pred_list.append(oof_probs[:, c])

            if len(y_true_list) > 0:
                y_true_rare = np.concatenate(y_true_list)
                y_pred_rare = np.concatenate(y_pred_list)

                if y_true_rare.sum() >= 2:
                    try:
                        ir = IsotonicRegression(out_of_bounds="clip")
                        ir.fit(y_pred_rare, y_true_rare)
                        taxon_calibrators[taxon] = ir
                    except Exception:
                        pass

    print(
        f"[calibration] Fit {len(calibrators)} per-class isotonic calibrators "
        f"and {len(taxon_calibrators)} shared taxon-level calibrators."
    )
    return calibrators, taxon_calibrators


def apply_calibration(
    probs: np.ndarray,
    calibrators: Dict[int, IsotonicRegression],
    taxon_calibrators: Optional[Dict[str, IsotonicRegression]] = None,
    class_taxons: Optional[List[str]] = None
) -> np.ndarray:
    """Applies per-class and taxon-level isotonic calibration to probabilities.

    Summary:
        Runs individual class calibrators first, then applies taxon-level fallbacks
        for uncalibrated classes.
    """
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


def optimize_thresholds(
    calibrated_probs: np.ndarray,
    Y_true: np.ndarray,
    n_windows: int = 12,
    threshold_grid: Optional[List[float]] = None
) -> np.ndarray:
    """Finds optimal F1-based decision thresholds per class on file-level predictions.

    Summary:
        Performs a grid search over threshold values to maximize the class F1 score
        calculated on file-level peak probabilities.

    Inputs:
        calibrated_probs (np.ndarray): Calibrated prediction probabilities.
        Y_true (np.ndarray): Target multi-hot labels.
        n_windows (int): Window sequence length. Default is 12.
        threshold_grid (Optional[List[float]]): Array of thresholds to sweep. Default is None.

    Outputs:
        np.ndarray: Optimized class thresholds.

    Shapes:
        - Input `calibrated_probs`: `(N_samples, N_classes)`
        - Input `Y_true`: `(N_samples, N_classes)`
        - Output thresholds: `(N_classes,)`

    Side effects:
        Prints threshold summary statistics to stdout.

    Usage example:
        >>> thrs = optimize_thresholds(cal_probs, Y_tr)
    """
    if threshold_grid is None:
        threshold_grid = [round(t, 3) for t in np.arange(0.20, 0.75, 0.025)]

    n_samples, n_classes = calibrated_probs.shape
    n_files = n_samples // n_windows
    thresholds = np.full(n_classes, 0.5, dtype=np.float32)

    file_probs = calibrated_probs.reshape(n_files, n_windows, n_classes).max(axis=1)
    file_y = Y_true.reshape(n_files, n_windows, n_classes).max(axis=1)

    n_optimized = 0
    for c in range(n_classes):
        y_c = file_y[:, c]
        p_c = file_probs[:, c]

        if y_c.sum() < 3:
            continue

        best_f1, best_t = 0.0, 0.5
        for t in threshold_grid:
            pred = (p_c >= t).astype(int)
            tp = ((pred == 1) & (y_c == 1)).sum()
            fp = ((pred == 1) & (y_c == 0)).sum()
            fn = ((pred == 0) & (y_c == 1)).sum()
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        thresholds[c] = best_t
        n_optimized += 1

    print(
        f"[thresholds] Optimized {n_optimized} classes | "
        f"mean={thresholds.mean():.3f} range=[{thresholds.min():.2f}, {thresholds.max():.2f}]"
    )
    return thresholds


def apply_threshold_sharpening(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Applies per-class threshold sharpening to prediction probabilities.

    Summary:
        Scales prediction probabilities such that scores above their class threshold are pushed
        towards 1.0, and scores below their class threshold are scaled towards 0.0.

    Inputs:
        probs (np.ndarray): Probability prediction scores.
        thresholds (np.ndarray): Target class thresholds array.

    Outputs:
        np.ndarray: Sharpened prediction probabilities.

    Shapes:
        - Input `probs`: `(N_samples, N_classes)`
        - Input `thresholds`: `(N_classes,)`
        - Output: `(N_samples, N_classes)`

    Side effects:
        None.

    Usage example:
        >>> sharp_probs = apply_threshold_sharpening(cal_probs, thrs)
    """
    C = probs.shape[1]
    scaled = np.copy(probs)
    for c in range(C):
        t = thresholds[c]
        above = probs[:, c] > t
        scaled[above, c] = 0.5 + 0.5 * (probs[above, c] - t) / (1 - t + 1e-8)
        scaled[~above, c] = 0.5 * probs[~above, c] / (t + 1e-8)
    return np.clip(scaled, 0.0, 1.0)


def grid_search_correction_weight(
    first_pass_logits: np.ndarray,
    correction_flat: np.ndarray,
    Y_true: np.ndarray,
    temperatures: np.ndarray,
    grid: Optional[List[float]] = None,
    n_windows: int = 12,
    **postproc_kwargs: Any,
) -> float:
    """Performs grid search to find the optimal residual correction weight.

    Summary:
        Sweeps through scale factors for the error-correction logits, evaluates the macro-AUC
        on post-processed outputs, and returns the optimal blending parameter.

    Inputs:
        first_pass_logits (np.ndarray): Uncorrected first-pass prediction logits.
        correction_flat (np.ndarray): Raw prediction residuals correction logits.
        Y_true (np.ndarray): Ground truth bird target labels.
        temperatures (np.ndarray): Taxon temperature scaling settings.
        grid (Optional[List[float]]): Scale factor values to sweep. Default is None.
        n_windows (int): Sequence length in windows. Default is 12.
        postproc_kwargs (Any): Arguments passed forward to `apply_postprocessing`.

    Outputs:
        float: Optimal correction blending weight.

    Shapes:
        - Input `first_pass_logits`: `(N_samples, N_classes)`
        - Input `correction_flat`: `(N_samples, N_classes)`
        - Input `Y_true`: `(N_samples, N_classes)`

    Side effects:
        Prints intermediate search metrics to stdout.

    Usage example:
        >>> best_w = grid_search_correction_weight(fp_log, corr, Y_tr, temps)
    """
    if grid is None:
        grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    best_auc, best_w = -1.0, 0.30
    for w in grid:
        trial_logits = first_pass_logits + w * correction_flat
        trial_probs = apply_postprocessing(
            trial_logits, temperatures, n_windows=n_windows, **postproc_kwargs
        )
        # Macro AUC calculation helper
        keep = Y_true.sum(axis=0) > 0
        if keep.sum() == 0:
            auc = 0.0
        else:
            auc = float(roc_auc_score(Y_true[:, keep], trial_probs[:, keep], average="macro"))

        print(f"  correction_weight={w:.2f}  post-transform macro-AUC={auc:.5f}")
        if auc > best_auc:
            best_auc, best_w = auc, w

    print(f"[grid] Best correction_weight={best_w:.2f} (AUC={best_auc:.5f})")
    return best_w


def build_prior_tables(sc_df: pd.DataFrame, Y_labels: np.ndarray) -> Dict[str, Any]:
    """Calculates empirical probability prior distributions from training targets.

    Summary:
        Computes class occurrence likelihoods grouped globally, site-wise, hour-wise, and
        jointly, applying wrap-around Gaussian smoothing to temporal hour distributions.

    Inputs:
        sc_df (pd.DataFrame): Training soundscapes metadata dataframe containing ('site', 'hour_utc').
        Y_labels (np.ndarray): Multi-hot label targets matrix.

    Outputs:
        Dict[str, Any]: Prior tables mapping dictionaries and average class occurrences.

    Shapes:
        - Input `Y_labels`: `(N_samples, N_classes)`
        - Output prior mappings: Contains arrays for global (N_classes,), site (N_sites, N_classes),
          hour (N_hours, N_classes), and joint (N_joint, N_classes) frequencies.

    Side effects:
        Executes Gaussian filtering.

    Usage example:
        >>> tables = build_prior_tables(sc_tr, Y_tr)
    """
    sc_df = sc_df.reset_index(drop=True)
    global_p = Y_labels.mean(axis=0).astype(np.float32)

    site_keys = sorted(sc_df["site"].dropna().astype(str).unique())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_p = np.zeros((len(site_keys), Y_labels.shape[1]), dtype=np.float32)
    site_n = np.zeros(len(site_keys), dtype=np.float32)
    for s in site_keys:
        i = site_to_i[s]
        mask = sc_df["site"].astype(str).values == s
        site_n[i] = mask.sum()
        site_p[i] = Y_labels[mask].mean(axis=0)

    hour_keys = sorted(sc_df["hour_utc"].dropna().astype(int).unique())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_p = np.zeros((len(hour_keys), Y_labels.shape[1]), dtype=np.float32)
    hour_n = np.zeros(len(hour_keys), dtype=np.float32)
    for h in hour_keys:
        i = hour_to_i[h]
        mask = sc_df["hour_utc"].astype(int).values == h
        hour_n[i] = mask.sum()
        hour_p[i] = Y_labels[mask].mean(axis=0)

    # Joint site-hour bucket (prior table shrink factor = 4)
    sh_keys = sorted(
        {
            (str(s), int(h))
            for s, h in zip(sc_df["site"].dropna(), sc_df["hour_utc"].dropna())
            if not pd.isna(s) and not pd.isna(h)
        }
    )
    sh_to_i = {k: i for i, k in enumerate(sh_keys)}
    sh_p = np.zeros((len(sh_keys), Y_labels.shape[1]), dtype=np.float32)
    sh_n = np.zeros(len(sh_keys), dtype=np.float32)
    for (s, h) in sh_keys:
        i = sh_to_i[(s, h)]
        mask = (sc_df["site"].astype(str).values == s) & (
            sc_df["hour_utc"].astype(int).values == h
        )
        sh_n[i] = mask.sum()
        sh_p[i] = Y_labels[mask].mean(axis=0)

    # Tweak D: Circular Gaussian smoothing on hour priors (sigma=1.5h)
    if len(hour_keys) >= 3:
        full_hour_p = np.zeros((24, hour_p.shape[1]), dtype=np.float32)
        for h, idx in hour_to_i.items():
            full_hour_p[int(h)] = hour_p[idx]
        tiled = np.tile(full_hour_p, (3, 1))
        tiled_smooth = gaussian_filter1d(tiled, sigma=1.5, axis=0, mode="wrap")
        full_smooth = tiled_smooth[24:48]
        for h, idx in hour_to_i.items():
            hour_p[idx] = full_smooth[int(h)]
        hour_p = np.clip(hour_p, 0.0, 1.0)

    return {
        "global_p": global_p,
        "site_to_i": site_to_i,
        "site_p": site_p,
        "site_n": site_n,
        "hour_to_i": hour_to_i,
        "hour_p": hour_p,
        "hour_n": hour_n,
        "sh_to_i": sh_to_i,
        "sh_p": sh_p,
        "sh_n": sh_n,
    }


def apply_prior(
    scores: np.ndarray,
    sites: np.ndarray,
    hours: np.ndarray,
    tables: Dict[str, Any],
    lambda_prior: float = 0.4,
    class_idx: Optional[Union[int, List[int], np.ndarray]] = None
) -> np.ndarray:
    """Applies bayesian prior logit adjustments to predictions.

    Summary:
        Computes conditional occurrence probability combining global, site, hour,
        and joint site-hour frequencies with empirical sample size shrinkage,
        then offsets inputs in logit space. Supports subsetting via class_idx.

    Inputs:
        scores (np.ndarray): Raw prediction logits or scores.
        sites (np.ndarray): Array of site labels.
        hours (np.ndarray): Array of recording hours.
        tables (Dict[str, Any]): Reference prior tables dictionary.
        lambda_prior (float): Scaling factor weight for the prior adjustments. Default is 0.4.
        class_idx (Optional[Union[int, List[int], np.ndarray]]): Index or list of indices
            of classes to extract priors for. Default is None (extract all).

    Outputs:
        np.ndarray: Prior-adjusted logit scores.

    Shapes:
        - Input `scores`: `(N_samples, N_classes)`
        - Output: `(N_samples, N_classes)`

    Side effects:
        None.

    Usage example:
        >>> adj_scores = apply_prior(sc_te, sites, hours, prior_tables)
    """
    eps = 1e-4
    n = len(scores)
    out = scores.copy()

    global_p = tables["global_p"]
    hour_p = tables["hour_p"]
    site_p = tables["site_p"]
    sh_p = tables.get("sh_p", None)

    if class_idx is not None:
        if isinstance(class_idx, (int, np.integer)):
            idx_list = [class_idx]
        else:
            idx_list = class_idx
        global_p = global_p[idx_list]
        hour_p = hour_p[:, idx_list]
        site_p = site_p[:, idx_list]
        if sh_p is not None:
            sh_p = sh_p[:, idx_list]

    p = np.tile(global_p, (n, 1))

    for i, h in enumerate(hours):
        h = int(h)
        if h in tables["hour_to_i"]:
            j = tables["hour_to_i"][h]
            nh = tables["hour_n"][j]
            w = nh / (nh + 8.0)
            p[i] = w * hour_p[j] + (1 - w) * global_p

    for i, s in enumerate(sites):
        s = str(s)
        if s in tables["site_to_i"]:
            j = tables["site_to_i"][s]
            ns = tables["site_n"][j]
            w = ns / (ns + 8.0)
            p[i] = w * site_p[j] + (1 - w) * p[i]

    if "sh_to_i" in tables and sh_p is not None:
        for i, (s, h) in enumerate(zip(sites, hours)):
            key = (str(s), int(h))
            if key in tables["sh_to_i"]:
                j = tables["sh_to_i"][key]
                nsh = tables["sh_n"][j]
                w = nsh / (nsh + 4.0)
                p[i] = w * sh_p[j] + (1 - w) * p[i]

    p = np.clip(p, eps, 1.0 - eps)
    out += lambda_prior * (np.log(p) - np.log1p(-p))
    return out.astype(np.float32)


def build_fold_safe_prior_tables(
    sc_df: pd.DataFrame, Y_labels: np.ndarray, fold_column: str = "fold", exclude_fold: Optional[int] = None
) -> Dict[str, Any]:
    """Generates prior tables with fold isolation to prevent data leakage.

    Summary:
        Masks samples belonging to a specific evaluation fold and generates priors on remaining entries.

    Inputs:
        sc_df (pd.DataFrame): SOUNDSCAPE metadata.
        Y_labels (np.ndarray): Class occurrence matrices.
        fold_column (str): Folder classification column name. Default is "fold".
        exclude_fold (Optional[int]): Folds index value to isolate. Default is None.

    Outputs:
        Dict[str, Any]: Prior tables mapping dictionaries.

    Shapes:
        - Input `Y_labels`: `(N_samples, N_classes)`
        - Output prior mappings: Contains arrays for global, site, and hour frequencies.

    Side effects:
        None.

    Usage example:
        >>> tables = build_fold_safe_prior_tables(sc_tr, Y_tr, exclude_fold=0)
    """
    if exclude_fold is not None:
        mask = sc_df[fold_column].values != exclude_fold
        sc_df = sc_df[mask].reset_index(drop=True)
        Y_labels = Y_labels[mask]

    return build_prior_tables(sc_df, Y_labels)


def run_corrected_calibration_pipeline(
    first_pass_logits_tr: np.ndarray,
    correction_flat_tr: np.ndarray,
    Y_true: np.ndarray,
    temperatures: np.ndarray,
    n_windows: int = 12,
    correction_grid: Optional[List[float]] = None,
    threshold_grid: Optional[List[float]] = None,
    class_taxons: Optional[List[str]] = None,
    prior_tables: Optional[Dict[str, Any]] = None,
    sites: Optional[np.ndarray] = None,
    hours: Optional[np.ndarray] = None,
    **postproc_kwargs: Any,
) -> Tuple[float, Dict[int, IsotonicRegression], Dict[str, IsotonicRegression], np.ndarray, Dict[str, float], Dict[str, float]]:
    """Runs the unified post-transform calibration and optimization sequence.

    Summary:
        Tunes taxon hyperparameters, sweeps ResidualSSM weights, fits calibrators,
        and optimizes class decision thresholds.
    """
    # Step 1: Tune prior and rank aware hyperparameters per taxon
    best_lambdas = {}
    best_powers = {}
    if class_taxons is not None and prior_tables is not None and sites is not None and hours is not None:
        best_lambdas, best_powers = tune_hyperparameters_per_taxon(
            first_pass_logits_tr, Y_true, sites, hours, prior_tables,
            class_taxons, temperatures, n_windows=n_windows
        )
        postproc_kwargs["lambda_prior"] = best_lambdas
        postproc_kwargs["ra_power"] = best_powers
        postproc_kwargs["prior_tables"] = prior_tables
        postproc_kwargs["sites"] = sites
        postproc_kwargs["hours"] = hours
        postproc_kwargs["class_taxons"] = class_taxons

    # Step 2: Find best correction weight
    correction_weight = grid_search_correction_weight(
        first_pass_logits_tr,
        correction_flat_tr,
        Y_true,
        temperatures,
        grid=correction_grid,
        n_windows=n_windows,
        **postproc_kwargs,
    )

    # Step 3: Apply best correction and postprocess
    corrected_logits = first_pass_logits_tr + correction_weight * correction_flat_tr
    oof_probs = apply_postprocessing(
        corrected_logits, temperatures, n_windows=n_windows, **postproc_kwargs
    )

    # Step 4: Fit per-class and taxon calibrators
    calibrators, taxon_calibrators = fit_per_class_calibrators(
        oof_probs, Y_true, class_taxons=class_taxons, n_windows=n_windows
    )

    # Step 5: Apply calibration
    calibrated_probs = apply_calibration(
        oof_probs, calibrators, taxon_calibrators=taxon_calibrators, class_taxons=class_taxons
    )

    # Step 6: Optimize thresholds
    thresholds = optimize_thresholds(
        calibrated_probs,
        Y_true,
        n_windows=n_windows,
        threshold_grid=threshold_grid,
    )

    return correction_weight, calibrators, taxon_calibrators, thresholds, best_lambdas, best_powers


def _to_indexed_prob_frame(pred: pd.DataFrame) -> pd.DataFrame:
    """Verifies format and index compatibility of predictions dataframes.

    Summary:
        Coerces input predictions to a standard format where `row_id` acts as the index
        and values are checked for valid bounds.
    """
    if not isinstance(pred, pd.DataFrame):
        raise AssertionError("prediction must be a pandas DataFrame")
    df = pred.copy()
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    if "row_id" in df.columns:
        df["row_id"] = df["row_id"].astype(str)
        df = df.set_index("row_id")
    elif df.index.name == "row_id":
        df.index = df.index.astype(str)
    else:
        raise AssertionError("prediction must have row_id as column or index")
    assert df.index.is_unique, "duplicate row_id values before postprocessing"
    values = df.to_numpy(dtype=np.float32)
    assert np.isfinite(values).all(), "NaN/inf before postprocessing"
    assert (
        values.min() >= 0.0 and values.max() <= 1.0
    ), "probabilities outside [0, 1] before postprocessing"
    return df.astype(np.float32)


def _row_uncertainty(probs: np.ndarray) -> np.ndarray:
    """Calculates row prediction uncertainty based on the maximum predicted probability.

    Summary:
        Computes a soft uncertainty scaling ratio from peak window predictions.
    """
    row_max = probs.max(axis=1)
    return np.clip((0.92 - row_max) / 0.57, 0.0, 1.0).astype(np.float32)


def _load_taxonomy_for_postproc(base_path: Optional[Union[str, Path]] = None) -> Optional[pd.DataFrame]:
    """Loads taxonomy file to provide taxonomic grouping contexts."""
    paths = []
    if base_path is not None:
        paths.append(Path(base_path) / "taxonomy.csv")
    paths.extend([
        Path("/kaggle/input/competitions/birdclef-2026/taxonomy.csv"),
        Path("/kaggle/input/birdclef-2026/taxonomy.csv"),
        Path("./taxonomy.csv")
    ])
    for p in paths:
        if p.exists():
            return pd.read_csv(p)
    print("[postproc] taxonomy.csv not found; skipping taxonomy smoothing")
    return None


def _adaptive_taxonomy_smoothing(
    probs: np.ndarray,
    cols: List[str],
    genus_alpha: float = 0.12,
    class_alpha: float = 0.025,
    base_path: Optional[Union[str, Path]] = None,
    tta_variance: Optional[np.ndarray] = None
) -> np.ndarray:
    """Smooths target predictions towards genus and class averages based on uncertainty."""
    tax = _load_taxonomy_for_postproc(base_path)
    if tax is None:
        return probs
    tax = tax.copy()
    tax["primary_label"] = tax["primary_label"].astype(str)
    tax_by_label = tax.set_index("primary_label")

    genus_groups: Dict[str, List[str]] = {}
    class_groups: Dict[str, List[str]] = {}
    for c in cols:
        if c not in tax_by_label.index:
            continue
        row = tax_by_label.loc[c]
        sci = str(row.get("scientific_name", "")).strip()
        cls = str(row.get("class_name", "")).strip()
        genus = sci.split(" ")[0] if " " in sci else sci
        genus_bad = genus == "" or genus.lower() == "nan" or genus == c
        if (not genus_bad) and not ("son" in c and genus_bad):
            genus_groups.setdefault(genus, []).append(c)
        if cls and cls.lower() != "nan":
            class_groups.setdefault(cls, []).append(c)

    multi_genus = {g: m for g, m in genus_groups.items() if len(m) > 1}
    multi_class = {g: m for g, m in class_groups.items() if 2 <= len(m) <= 80}
    col_to_i = {c: i for i, c in enumerate(cols)}

    # Apply new uncertainty-gated alpha computation
    genus_alphas = compute_gated_smoothing_alpha(probs, tta_variance, base_alpha=genus_alpha)
    class_alphas = compute_gated_smoothing_alpha(probs, tta_variance, base_alpha=class_alpha)

    print(
        f"[postproc] adaptive taxonomy groups: genus={len(multi_genus)}, class={len(multi_class)}"
    )
    out = probs.copy()
    for members in multi_genus.values():
        idx = [col_to_i[m] for m in members if m in col_to_i]
        if len(idx) < 2:
            continue
        group_mean = out[:, idx].mean(axis=1, keepdims=True)
        out[:, idx] = (1.0 - genus_alphas) * out[:, idx] + genus_alphas * group_mean

    for members in multi_class.values():
        idx = [col_to_i[m] for m in members if m in col_to_i]
        if len(idx) < 2:
            continue
        group_mean = out[:, idx].mean(axis=1, keepdims=True)
        out[:, idx] = (1.0 - class_alphas) * out[:, idx] + class_alphas * group_mean
    return out


def _split_row_id(row_id: str) -> Tuple[str, int]:
    """Splits a window identifier row_id into recording stem and window offset."""
    text = str(row_id)
    stem, sep, end = text.rpartition("_")
    if not sep:
        return text, -1
    try:
        return stem, int(end)
    except Exception:
        return stem, -1


def _conservative_temporal_consistency(
    probs: np.ndarray, row_ids: List[str], temporal_alpha: float = 0.08
) -> np.ndarray:
    """Applies temporal context constraints to smooth predictions conservatively."""
    if temporal_alpha <= 0 or len(row_ids) < 3:
        return probs
    out = probs.copy()
    parsed = [_split_row_id(r) for r in row_ids]
    file_to_pos: Dict[str, List[Tuple[int, int]]] = {}
    for i, (stem, end) in enumerate(parsed):
        file_to_pos.setdefault(stem, []).append((end, i))

    adjusted = 0
    uncertainty = _row_uncertainty(probs)
    for items in file_to_pos.values():
        items = sorted(items, key=lambda x: x[0])
        pos = [i for _, i in items]
        if len(pos) < 3:
            continue
        for j in range(1, len(pos) - 1):
            i = pos[j]
            a = temporal_alpha * float(uncertainty[i])
            if a <= 0:
                continue
            prev_i, next_i = pos[j - 1], pos[j + 1]
            neighbor = 0.5 * (probs[prev_i] + probs[next_i])
            current = probs[i]
            # Keep isolated high-confidence spikes; smooth only uncertain texture.
            spike = (current > 0.65) & (current > neighbor * 1.25)
            candidate = (1.0 - a) * current + a * neighbor
            out[i] = np.where(spike, current, candidate)
            adjusted += 1
    print(f"[postproc] temporal consistency adjusted middle windows: {adjusted}")
    return out


def _apply_optional_extra_artifact_blend(
    probs: np.ndarray, row_ids: List[str], cols: List[str], extra_csvs: List[Union[str, Path]] = [], weight: float = 0.0
) -> np.ndarray:
    """Blends current predictions with external predictions if configured."""
    if weight <= 0 or not extra_csvs:
        return probs
    weight = min(max(weight, 0.0), 0.20)
    base = pd.DataFrame(probs, index=pd.Index(row_ids, name="row_id"), columns=cols)
    blended = base.copy()
    used = 0
    for path in extra_csvs:
        p = Path(path)
        if not p.exists():
            continue
        extra = pd.read_csv(p)
        if "row_id" not in extra.columns:
            continue
        extra["row_id"] = extra["row_id"].astype(str)
        if set(extra["row_id"]) != set(row_ids) or any(c not in extra.columns for c in cols):
            continue
        extra = extra.set_index("row_id").loc[row_ids, cols].astype(np.float32)
        vals = extra.to_numpy(dtype=np.float32)
        if not np.isfinite(vals).all() or vals.min() < 0 or vals.max() > 1:
            continue
        blended = (1.0 - weight) * blended + weight * extra
        used += 1
    return blended.to_numpy(dtype=np.float32) if used else probs


def f_TAX_SMOOTHING_POSTPROC(
    pred_df: pd.DataFrame,
    genus_alpha: float = 0.12,
    class_alpha: float = 0.025,
    temporal_alpha: float = 0.08,
    base_path: Optional[Union[str, Path]] = None,
    extra_csvs: List[Union[str, Path]] = [],
    extra_blend_weight: float = 0.0,
    tta_variance: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Applies adaptive taxonomy smoothing and temporal consistency checking on prediction frames.

    Summary:
        Formats input predictions and executes external blending, uncertainty-gated congeneric/class-level
        smoothing, and adjacent window temporal smoothing.

    Inputs:
        pred_df (pd.DataFrame): Blended predictions dataframe.
        genus_alpha (float): Scaling factor weight for genus-level smoothing. Default is 0.12.
        class_alpha (float): Scaling factor weight for class-level smoothing. Default is 0.025.
        temporal_alpha (float): Scaling factor weight for adjacent temporal smoothing. Default is 0.08.
        base_path (Optional[Union[str, Path]]): Base path directory containing competition metadata. Default is None.
        extra_csvs (List[Union[str, Path]]): external blend file paths. Default is [].
        extra_blend_weight (float): blend weight fraction for external predictions. Default is 0.0.
        tta_variance (Optional[np.ndarray]): Variance matrix from TTA passes. Default is None.

    Outputs:
        pd.DataFrame: Post-processed and smoothed probabilities dataframe.

    Shapes:
        - Input `pred_df`: `(N_samples, N_classes)`
        - Output: Same shape.

    Side effects:
        Prints diagnostic metrics to stdout. Loads taxonomy data.

    Usage example:
        >>> final_sub = f_TAX_SMOOTHING_POSTPROC(sub_df, base_path="/kaggle/input/birdclef-2026")
    """
    submission = _to_indexed_prob_frame(pred_df)
    print(
        f"[postproc] pre: shape={submission.shape}, "
        f"mean={submission.values.mean():.6f}, max={submission.values.max():.6f}"
    )

    sample_path = None
    if base_path is not None:
        sample_path = Path(base_path) / "sample_submission.csv"
    if sample_path is not None and sample_path.exists():
        sample = pd.read_csv(sample_path)
        sample["row_id"] = sample["row_id"].astype(str)
        expected_cols = sample.columns[1:].tolist()
        missing = [c for c in expected_cols if c not in submission.columns]
        assert not missing, f"postproc missing sample columns: {missing[:8]}"
        submission = submission.loc[sample["row_id"], expected_cols]

    probs = submission.to_numpy(dtype=np.float32, copy=True)
    cols = list(submission.columns)
    row_ids = submission.index.astype(str).tolist()

    # Apply external artifacts, taxonomy smoothing, and temporal consistency
    probs = _apply_optional_extra_artifact_blend(probs, row_ids, cols, extra_csvs, extra_blend_weight)
    probs = _adaptive_taxonomy_smoothing(
        probs, cols, genus_alpha=genus_alpha, class_alpha=class_alpha, base_path=base_path, tta_variance=tta_variance
    )
    probs = _conservative_temporal_consistency(probs, row_ids, temporal_alpha=temporal_alpha)

    probs = np.clip(probs, 0.0, 1.0).astype(np.float32)
    out = pd.DataFrame(probs, index=submission.index, columns=cols)
    out.index.name = "row_id"

    print(
        f"[postproc] post: shape={out.shape}, "
        f"mean={out.values.mean():.6f}, max={out.values.max():.6f}"
    )
    return out


def compute_gated_smoothing_alpha(
    probs: np.ndarray,
    tta_variance: Optional[np.ndarray] = None,
    base_alpha: float = 0.12
) -> np.ndarray:
    """Computes an uncertainty-gated smoothing alpha per window.

    Summary:
        Melds predictive entropy (from probs) and optional epistemic uncertainty
        (from TTA variance) to scale taxonomy smoothing.
    """
    eps = 1e-8
    entropy_matrix = -probs * np.log(probs + eps) - (1.0 - probs) * np.log(1.0 - probs + eps)
    mean_entropy = np.mean(entropy_matrix, axis=1)
    norm_entropy = np.clip(mean_entropy / 0.693147, 0.0, 1.0)

    if tta_variance is not None:
        mean_var = np.mean(tta_variance, axis=1)
        norm_var = np.clip(mean_var * 4.0, 0.0, 1.0)
        uncertainty = 0.5 * norm_entropy + 0.5 * norm_var
    else:
        row_max = probs.max(axis=1)
        max_unc = np.clip((0.92 - row_max) / 0.57, 0.0, 1.0)
        uncertainty = 0.5 * norm_entropy + 0.5 * max_unc

    return base_alpha * uncertainty[:, None]


def tune_hyperparameters_per_taxon(
    oof_logits: np.ndarray,
    Y_true: np.ndarray,
    sites: np.ndarray,
    hours: np.ndarray,
    prior_tables: Dict[str, Any],
    class_taxons: List[str],
    temperatures: np.ndarray,
    n_windows: int = 12,
    lambda_grid: Optional[List[float]] = None,
    power_grid: Optional[List[float]] = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Tunes lambda_prior and rank_aware_power per taxon to maximize macro-AUC on OOF.

    Summary:
        Sweeps combination grids of prior weights and scaling exponents for each taxon group,
        optimizing parameters independently.
    """
    if lambda_grid is None:
        lambda_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    if power_grid is None:
        power_grid = [0.0, 0.2, 0.4, 0.6, 0.8]

    taxons = sorted(list(set(class_taxons)))
    best_lambdas = {}
    best_powers = {}

    # Define a simple internal macro-auc function for tuning
    def _local_macro_auc(y_true, y_score):
        keep = y_true.sum(axis=0) > 0
        if keep.sum() == 0:
            return 0.0
        return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")

    for taxon in taxons:
        taxon_indices = [c for c, t in enumerate(class_taxons) if t == taxon]
        if len(taxon_indices) == 0:
            continue

        y_true_taxon = Y_true[:, taxon_indices]
        if y_true_taxon.sum() == 0:
            best_lambdas[taxon] = 0.4
            best_powers[taxon] = 0.6
            continue

        logits_taxon = oof_logits[:, taxon_indices]
        temps_taxon = temperatures[taxon_indices]

        best_auc = -1.0
        opt_lambda = 0.4
        opt_power = 0.6

        for lam in lambda_grid:
            adjusted_logits = apply_prior(oof_logits, sites, hours, prior_tables, lambda_prior=lam)[:, taxon_indices]
            probs = sigmoid(adjusted_logits / temps_taxon[None, :])
            probs = file_confidence_scale(probs, n_windows=n_windows, top_k=2, power=0.4)

            for pwr in power_grid:
                scaled_probs = rank_aware_scaling(probs, n_windows=n_windows, power=pwr)
                smoothed_probs = adaptive_delta_smooth(scaled_probs, n_windows=n_windows, base_alpha=0.20)

                auc = _local_macro_auc(y_true_taxon, smoothed_probs)
                if auc > best_auc:
                    best_auc = auc
                    opt_lambda = lam
                    opt_power = pwr

        best_lambdas[taxon] = opt_lambda
        best_powers[taxon] = opt_power
        print(f"[tuning] Taxon '{taxon}' optimized: lambda_prior={opt_lambda:.2f}, rank_aware_power={opt_power:.2f} (AUC={best_auc:.5f})")

    return best_lambdas, best_powers


def apply_retrieval_blend(
    probs: np.ndarray,
    retrieval_probs: np.ndarray,
    retrieval_mask: np.ndarray,
    retrieval_weight: float = 0.2
) -> np.ndarray:
    """Blends retrieval predictions with model predictions for rare/unmapped classes.

    Summary:
        Combines probabilities in [0, 1] using a weighted average for specified classes.
    """
    out = probs.copy()
    for c in range(probs.shape[1]):
        if retrieval_mask[c]:
            out[:, c] = (1.0 - retrieval_weight) * probs[:, c] + retrieval_weight * retrieval_probs[:, c]
    return np.clip(out, 0.0, 1.0)
