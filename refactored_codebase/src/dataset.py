"""dataset.py — Audio reading, metadata loading, and label alignment utilities.

This module handles competition manifest files, parses recording metadata,
aligns multi-hot labels, and windows audio recordings.
"""

import re
from pathlib import Path
from typing import Dict, Tuple, List, Set, Union
import numpy as np
import pandas as pd
import soundfile as sf

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


def parse_fname(name: str) -> Dict[str, Union[str, int]]:
    """Parses a soundscape filename to extract recording site and hour in UTC.

    Summary:
        Extracts ecological metadata encoded in standard BirdCLEF filenames.

    Inputs:
        name (str): The filename string (e.g., 'BC2026_Train_12345_S01_20260404_060000.ogg').

    Outputs:
        Dict[str, Union[str, int]]: Dictionary with keys 'site' (str) and 'hour_utc' (int).

    Shapes:
        None.

    Side effects:
        None.

    Usage example:
        >>> parse_fname("BC2026_Train_12345_S01_20260404_060000.ogg")
        {"site": "S01", "hour_utc": 6}
    """
    m = FNAME_RE.match(name)
    if not m:
        return {"site": "unknown", "hour_utc": -1}
    _, site, _, hms = m.groups()
    return {"site": site, "hour_utc": int(hms[:2])}


def union_labels(series: pd.Series) -> List[str]:
    """Combines semicolon-separated bird labels into a sorted list of unique labels.

    Summary:
        Merges multiple overlapping labels detected within a soundscape window.

    Inputs:
        series (pd.Series): A pandas series of semicolon-separated label strings.

    Outputs:
        List[str]: A sorted list of unique bird labels.

    Shapes:
        None.

    Side effects:
        None.

    Usage example:
        >>> union_labels(pd.Series(["mallar3;canvas1", "mallar3"]))
        ['canvas1', 'mallar3']
    """
    out: Set[str] = set()
    for x in series:
        if pd.notna(x):
            for t in str(x).split(";"):
                t = t.strip()
                if t:
                    out.add(t)
    return sorted(out)


def read_60s(path: Union[str, Path], file_samples: int = 1_920_000) -> np.ndarray:
    """Reads a 60-second audio file, averages channels to mono, and pads/crops.

    Summary:
        Loads and standardizes audio signal length to precisely 60 seconds at 32kHz.

    Inputs:
        path (Union[str, Path]): Path to the target audio file.
        file_samples (int): Required size of the output array. Default is 1,920,000 (60s * 32kHz).

    Outputs:
        np.ndarray: The mono audio signal.

    Shapes:
        Output shape: (file_samples,)

    Side effects:
        Reads from disk.

    Usage example:
        >>> audio = read_60s("test.ogg")
        >>> audio.shape
        (1920000,)
    """
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) < file_samples:
        y = np.pad(y, (0, file_samples - len(y)))
    else:
        y = y[:file_samples]
    return y.astype(np.float32)


def load_competition_metadata(
    base_path: Union[str, Path]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], Dict[str, int]]:
    """Loads taxonomy, sample submission, and soundscape labels.

    Summary:
        Reads files from the competition directory and builds name-to-index mappings.

    Inputs:
        base_path (Union[str, Path]): Path to the competition data root directory.

    Outputs:
        Tuple: Contains:
            - taxonomy (pd.DataFrame)
            - sample_sub (pd.DataFrame)
            - soundscape_labels (pd.DataFrame)
            - primary_labels (List[str]): List of bird labels in sample submission.
            - label_to_idx (Dict[str, int]): Map from label string to integer index.

    Shapes:
        None.

    Side effects:
        Reads multiple CSV files from disk.

    Usage example:
        >>> tax, sub, labels, classes, mapping = load_competition_metadata("/kaggle/input/birdclef-2026")
    """
    base = Path(base_path)
    taxonomy = pd.read_csv(base / "taxonomy.csv")
    sample_sub = pd.read_csv(base / "sample_submission.csv")
    soundscape_labels = pd.read_csv(base / "train_soundscapes_labels.csv")

    primary_labels = sample_sub.columns[1:].tolist()
    label_to_idx = {c: i for i, c in enumerate(primary_labels)}

    return taxonomy, sample_sub, soundscape_labels, primary_labels, label_to_idx


def prepare_labeled_matrix(
    soundscape_labels: pd.DataFrame,
    primary_labels: List[str],
    label_to_idx: Dict[str, int],
    n_windows: int = 12
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Builds the multi-hot target label matrices for soundscape windows.

    Summary:
        Groups soundscape annotations and creates aligned targets for windows and full files.

    Inputs:
        soundscape_labels (pd.DataFrame): Dataframe containing raw annotations.
        primary_labels (List[str]): List of bird class labels.
        label_to_idx (Dict[str, int]): Target label string-to-index dictionary.
        n_windows (int): Number of windows per file. Default is 12.

    Outputs:
        Tuple: Contains:
            - sc (pd.DataFrame): Window-level metadata.
            - Y_SC (np.ndarray): Multi-hot label matrix for all windows.
            - full_rows (pd.DataFrame): Window metadata filtered for fully-labeled files.
            - Y_FULL (np.ndarray): Multi-hot label matrix aligned with full files.

    Shapes:
        - Y_SC: (N_windows_total, N_classes)
        - Y_FULL: (N_windows_fully_labeled, N_classes)

    Side effects:
        None.

    Usage example:
        >>> sc, Y_SC, full_rows, Y_FULL = prepare_labeled_matrix(labels, classes, mapping)
    """
    sc = (
        soundscape_labels.groupby(["filename", "start", "end"])["primary_label"]
        .apply(union_labels)
        .reset_index(name="label_list")
    )

    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
    sc["row_id"] = (
        sc["filename"].str.replace(".ogg", "", regex=False)
        + "_"
        + sc["end_sec"].astype(str)
    )

    # Parse and append site and hour metadata
    meta = sc["filename"].apply(parse_fname).apply(pd.Series)
    sc = pd.concat([sc, meta], axis=1)

    n_classes = len(primary_labels)
    Y_SC = np.zeros((len(sc), n_classes), dtype=np.uint8)
    for i, lbls in enumerate(sc["label_list"]):
        for lbl in lbls:
            if lbl in label_to_idx:
                Y_SC[i, label_to_idx[lbl]] = 1

    windows_per_file = sc.groupby("filename").size()
    full_files = sorted(
        windows_per_file[windows_per_file == n_windows].index.tolist()
    )
    sc["fully_labeled"] = sc["filename"].isin(full_files)

    full_rows = (
        sc[sc["fully_labeled"]]
        .sort_values(["filename", "end_sec"])
        .reset_index(drop=False)
    )
    Y_FULL = Y_SC[full_rows["index"].to_numpy()]

    return sc, Y_SC, full_rows, Y_FULL
