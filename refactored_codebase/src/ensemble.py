"""ensemble.py — Ensembling and prediction blending algorithms for BirdCLEF.

This module implements submission blending helpers including direct weighted blending,
split-weight division of attention blending, and verification safety checks.
"""

from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd
from pathlib import Path


def _read_submission_checked(path: Union[str, Path]) -> pd.DataFrame:
    """Reads a submission CSV file and runs safety and value range diagnostics.

    Summary:
        Loads a submission dataframe from disk and verifies schema validity,
        checking for NaNs, unique row IDs, and bounded probability values in [0, 1].

    Inputs:
        path (Union[str, Path]): Path to the target submission CSV file.

    Outputs:
        pd.DataFrame: Aligned checked dataframe indexed by row_id.

    Shapes:
        - Output dataframe: `(N_samples, N_classes)` indexed by `row_id`.

    Side effects:
        Asserts validity conditions and throws AssertionError on check failures.

    Usage example:
        >>> df = _read_submission_checked("submission.csv")
    """
    df = pd.read_csv(path)
    assert "row_id" in df.columns, f"row_id column missing in {path}"
    assert not any(
        str(c).startswith("Unnamed") for c in df.columns
    ), f"unexpected unnamed column in {path}: {df.columns.tolist()[:5]}"
    assert df["row_id"].is_unique, f"duplicate row_id values in {path}"

    prob_cols = [c for c in df.columns if c != "row_id"]
    assert prob_cols, f"no probability columns in {path}"

    values = df[prob_cols].to_numpy(dtype=np.float32)
    assert np.isfinite(values).all(), f"NaN/inf values in {path}"
    assert (
        values.min() >= 0.0 and values.max() <= 1.0
    ), f"probabilities outside [0, 1] in {path}"

    out = df.set_index("row_id")
    out.index = out.index.astype(str)
    out.index.name = "row_id"
    return out


def direct_blend(files: List[Union[str, Path]], weights: List[float]) -> pd.DataFrame:
    """Computes a direct normalized weighted average across multiple submission files.

    Summary:
        Aligns rows and columns across several submission CSVs and averages their
        probabilities using normalized weights.

    Inputs:
        files (List[Union[str, Path]]): Submission file paths.
        weights (List[float]): Weights vector for each submission.

    Outputs:
        pd.DataFrame: Blended submission dataframe.

    Shapes:
        - Output: `(N_samples, N_classes)` indexed by `row_id`.

    Side effects:
        Reads files from disk and asserts shape matching.

    Usage example:
        >>> files = ["subm_22.csv", "subm_51.csv", "subm_74.csv"]
        >>> weights = [0.014, 0.021, 0.965]
        >>> blend_df = direct_blend(files, weights)
    """
    assert len(files) == len(weights), "submission file / weight length mismatch"
    weight_sum = float(sum(weights))
    assert weight_sum > 0, "ensemble weights must sum to a positive value"

    norm_weights = [float(w) / weight_sum for w in weights]
    dfs = [_read_submission_checked(path) for path in files]

    base_idx = dfs[0].index
    base_cols = dfs[0].columns

    for path, df in zip(files, dfs):
        assert df.columns.equals(base_cols), f"Column mismatch in {path}"
        missing = base_idx.difference(df.index)
        extra = df.index.difference(base_idx)
        assert len(missing) == 0 and len(extra) == 0, (
            f"row_id mismatch in {path}: missing={len(missing)}, extra={len(extra)}"
        )

    out = sum(w * df.loc[base_idx, base_cols] for w, df in zip(norm_weights, dfs))
    out.index.name = "row_id"

    values = out.to_numpy(dtype=np.float32)
    assert np.isfinite(values).all(), "NaN/inf in final blend"
    assert values.min() >= 0.0 and values.max() <= 1.0, "final probabilities outside [0, 1]"

    return out.reset_index()


def division_attention_blend(
    files: List[Union[str, Path]],
    sub_w1: List[float] = [0.014, 0.021, 0.965],
    sub_w2: List[float] = [0.0137, 0.0213, 0.965]
) -> pd.DataFrame:
    """Performs split-weight ensembling by partition indices across species.

    Summary:
        Blends three submissions using two distinct weight vectors: one for the first
        half of classes and another for the remaining half of classes.

    Inputs:
        files (List[Union[str, Path]]): Paths to the three submission CSVs.
        sub_w1 (List[float]): Weights for the first half of species. Default is [0.014, 0.021, 0.965].
        sub_w2 (List[float]): Weights for the second half of species. Default is [0.0137, 0.0213, 0.965].

    Outputs:
        pd.DataFrame: A blended submission dataframe.

    Shapes:
        - Input `files`: List of length 3.
        - Output: `(N_samples, N_classes)` with a `row_id` column.

    Side effects:
        Reads files from disk and asserts class partitions.

    Usage example:
        >>> files = ["subm_22.csv", "subm_51.csv", "subm_74.csv"]
        >>> blend_df = division_attention_blend(files)
    """
    assert len(files) == 3, "Division of attention blend is designed specifically for 3 models."

    subm_1 = _read_submission_checked(files[0])
    subm_2 = _read_submission_checked(files[1])
    subm_3 = _read_submission_checked(files[2])

    list_species = subm_1.columns.tolist()
    n_half = len(list_species) // 2

    out = pd.DataFrame(index=subm_1.index)

    # First half of columns (0 to n_half - 1)
    for col in list_species[:n_half]:
        out[col] = sub_w1[0] * subm_1[col] + sub_w1[1] * subm_2[col] + sub_w1[2] * subm_3[col]

    # Second half of columns (n_half to N)
    for col in list_species[n_half:]:
        out[col] = sub_w2[0] * subm_1[col] + sub_w2[1] * subm_2[col] + sub_w2[2] * subm_3[col]

    out.index.name = "row_id"
    return out.reset_index()
