"""validation.py — Validation helpers and metrics for BirdCLEF models.

This module contains calculation methods for macro-averaged ROC-AUC,
out-of-fold prediction smoothing, and cross-validation splitting utility methods.
"""

from typing import List, Tuple, Dict, Any, Union, Generator
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes the macro-averaged Area Under the ROC Curve (ROC-AUC).

    Summary:
        Calculates the average ROC-AUC score over classes that have at least
        one positive sample.

    Inputs:
        y_true (np.ndarray): Ground truth multi-hot labels.
        y_score (np.ndarray): Predicted probability or logit scores.

    Outputs:
        float: Macro-averaged ROC-AUC score.

    Shapes:
        - Input `y_true`: `(N_samples, N_classes)`
        - Input `y_score`: `(N_samples, N_classes)`

    Side effects:
        None.

    Usage example:
        >>> y_true = np.array([[1, 0], [0, 1]])
        >>> y_score = np.array([[0.9, 0.1], [0.2, 0.8]])
        >>> score = macro_auc(y_true, y_score)
        >>> print(f"{score:.2f}")
        1.00
    """
    keep = y_true.sum(axis=0) > 0
    if keep.sum() == 0:
        return 0.0
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


def smooth_predictions(probs: np.ndarray, n_windows: int = 12, alpha: float = 0.3) -> np.ndarray:
    """Applies simple bidirectional window smoothing on sequence predictions.

    Summary:
        Applies a symmetric temporal smoothing window across adjacent prediction frames
        within each 60-second file sequence.

    Inputs:
        probs (np.ndarray): Prediction probabilities matrix.
        n_windows (int): Number of windows per file sequence. Default is 12.
        alpha (float): Smoothing factor weight assigned to neighbors. Default is 0.3.

    Outputs:
        np.ndarray: Smoothed prediction probabilities.

    Shapes:
        - Input `probs`: `(N_samples, N_classes)` where N_samples must be divisible by `n_windows`.
        - Output: `(N_samples, N_classes)`

    Side effects:
        Asserts that the sample size is divisible by `n_windows`.

    Usage example:
        >>> probs = np.random.rand(24, 2)
        >>> smoothed = smooth_predictions(probs, n_windows=12, alpha=0.2)
        >>> smoothed.shape
        (24, 2)
    """
    N, C = probs.shape
    assert N % n_windows == 0, f"Total samples {N} must be divisible by window size {n_windows}"
    view = probs.reshape(-1, n_windows, C).copy()
    prev_w = np.concatenate([view[:, :1, :], view[:, :-1, :]], axis=1)
    next_w = np.concatenate([view[:, 1:, :], view[:, -1:, :]], axis=1)
    return ((1 - alpha) * view + 0.5 * alpha * (prev_w + next_w)).reshape(N, C)


def get_group_kfold_splits(
    meta_df: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = "filename"
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Generates out-of-fold cross validation split indices using GroupKFold.

    Summary:
        Yields train and validation indices partitioned by a grouping column (e.g. filename or site)
        to prevent target leakage across split boundaries.

    Inputs:
        meta_df (pd.DataFrame): Dataframe containing metadata information.
        n_splits (int): Folds split count. Default is 5.
        group_col (str): Metadata column used as the grouping identifier. Default is "filename".

    Outputs:
        Generator[Tuple[np.ndarray, np.ndarray], None, None]: Yields train index array and validation index array.

    Shapes:
        - Output: Tuple of integer arrays `(train_idx, val_idx)`.

    Side effects:
        None.

    Usage example:
        >>> splits = get_group_kfold_splits(meta_tr, n_splits=5)
        >>> for tr_idx, val_idx in splits:
        ...     print(len(tr_idx), len(val_idx))
    """
    groups = meta_df[group_col].values
    gkf = GroupKFold(n_splits=n_splits)
    # Target values can be dummy zeros for GroupKFold
    dummy_y = np.zeros(len(meta_df))
    for train_idx, val_idx in gkf.split(meta_df, dummy_y, groups=groups):
        yield train_idx, val_idx
