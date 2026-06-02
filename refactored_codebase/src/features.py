"""features.py — Feature extraction and mapping for BirdCLEF models.

This module encapsulates loading the Perch v2 model (via ONNX Runtime or TF),
mapping outputs from Perch label space to BirdCLEF label space, resolving genus-level
proxies for unmapped species, and loading precomputed arrays from cache files.
"""

import os
import gc
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False

try:
    import tensorflow as tf
except ImportError:
    tf = None


class TaxonomyMapper:
    """Handles mapping from Perch output classes to BirdCLEF target classes.

    Summary:
        Maps Perch scientific names to the target 234 species and resolves genus
        proxies for unmapped species.

    Attributes:
        primary_labels (List[str]): Bird target species labels.
        n_classes (int): Total number of bird classes (234).
        mapped_mask (np.ndarray): Boolean mask indicating if target has Perch representation.
        mapped_bc_idx (np.ndarray): Array of Perch indices for mapped targets.
        mapped_pos (np.ndarray): Indices of mapped classes in BirdCLEF space.
        unmapped_pos (np.ndarray): Indices of unmapped classes.
        proxy_map (Dict[int, List[int]]): Maps unmapped class index to congeneric Perch indices.
        temperatures (np.ndarray): Temperature scaling factors per class.
    """

    def __init__(self, taxonomy: pd.DataFrame, primary_labels: List[str], perch_labels_path: Path):
        """Initializes taxonomy mappings and proxies.

        Summary:
            Builds index arrays and proxy relationships based on taxonomy CSV.

        Inputs:
            taxonomy (pd.DataFrame): Dataframe of bird taxonomy.
            primary_labels (List[str]): List of bird target species.
            perch_labels_path (Path): Path to labels.csv asset from Google Perch.

        Outputs:
            TaxonomyMapper: A initialized instance.

        Shapes:
            None.

        Side effects:
            Reads labels.csv from disk.

        Usage example:
            >>> mapper = TaxonomyMapper(tax, classes, Path("labels.csv"))
        """
        self.primary_labels = primary_labels
        self.n_classes = len(primary_labels)
        label_to_idx = {c: i for i, c in enumerate(primary_labels)}

        # Load Perch label vocabulary
        bc_labels = pd.read_csv(perch_labels_path).reset_index().rename(
            columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"}
        )
        if "scientific_name" not in bc_labels.columns:
            for c in ["label", "labels", "name"]:
                if c in bc_labels.columns:
                    bc_labels = bc_labels.rename(columns={c: "scientific_name"})
                    break
        assert "scientific_name" in bc_labels.columns, "No scientific name in labels file."

        no_label = len(bc_labels)

        # Merge taxonomy with Perch labels
        mapping = taxonomy.merge(
            bc_labels.rename(columns={"scientific_name": "scientific_name"}),
            on="scientific_name",
            how="left"
        )
        mapping["bc_index"] = mapping["bc_index"].fillna(no_label).astype(int)
        lbl2bc = mapping.set_index("primary_label")["bc_index"]

        self.bc_indices = np.array([int(lbl2bc.loc[c]) for c in primary_labels], dtype=np.int32)
        self.mapped_mask = self.bc_indices != no_label
        self.mapped_pos = np.where(self.mapped_mask)[0].astype(np.int32)
        self.mapped_bc_idx = self.bc_indices[self.mapped_mask].astype(np.int32)

        # Build proxy mappings for unmapped species
        self.unmapped_pos = np.where(~self.mapped_mask)[0].astype(np.int32)
        class_name_map = taxonomy.set_index("primary_label")["class_name"].to_dict()
        texture_taxa = {"Amphibia", "Insecta"}

        self.proxy_map: Dict[int, List[int]] = {}
        unmapped_df = taxonomy[taxonomy["primary_label"].isin([primary_labels[i] for i in self.unmapped_pos])].copy()

        for _, row in unmapped_df.iterrows():
            target = row["primary_label"]
            sci = str(row["scientific_name"])
            genus = sci.split()[0]
            hits = bc_labels[
                bc_labels["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\s", na=False)
            ]
            if len(hits) > 0:
                self.proxy_map[label_to_idx[target]] = hits["bc_index"].astype(int).tolist()

        proxy_taxa = {"Amphibia", "Insecta", "Aves"}
        self.proxy_map = {
            idx: bc_idxs
            for idx, bc_idxs in self.proxy_map.items()
            if class_name_map.get(primary_labels[idx]) in proxy_taxa
        }

        # Temperature scaling parameters per taxon
        self.temperatures = np.ones(self.n_classes, dtype=np.float32)
        for ci, label in enumerate(primary_labels):
            cls = class_name_map.get(label, "Aves")
            self.temperatures[ci] = 0.95 if cls in texture_taxa else 1.10


class PerchBackbone:
    """Manages the Perch model session and performs audio inference.

    Summary:
        Runs Perch model inference on mono waveforms using ONNX Runtime or TensorFlow.

    Attributes:
        onnx_path (Optional[Path]): Path to Perch ONNX weights.
        tf_model_path (Optional[Path]): Path to TF Perch SavedModel directory.
        use_onnx (bool): Whether using ONNX backend.
    """

    def __init__(self, onnx_path: Optional[Path] = None, tf_model_path: Optional[Path] = None):
        """Initializes model sessions.

        Summary:
            Sets up ONNX Session or TF infer function based on availability.

        Inputs:
            onnx_path (Optional[Path]): ONNX model path.
            tf_model_path (Optional[Path]): TensorFlow model path.

        Outputs:
            PerchBackbone: An instance of PerchBackbone.

        Shapes:
            None.

        Side effects:
            Loads weights and allocates GPU/CPU sessions.
        """
        self.use_onnx = _ONNX_AVAILABLE and onnx_path is not None and onnx_path.exists()
        self.onnx_session = None
        self.tf_infer_fn = None
        self.emb_dim = 1536

        if self.use_onnx:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 4
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.onnx_session = ort.InferenceSession(
                str(onnx_path), sess_options=so, providers=providers
            )
            self.onnx_input_name = self.onnx_session.get_inputs()[0].name
            self.onnx_out_map = {o.name: i for i, o in enumerate(self.onnx_session.get_outputs())}
            print(f"[features] Using ONNX Perch: {onnx_path.name}")
        elif tf_model_path is not None and tf_model_path.exists() and tf is not None:
            tf.config.set_visible_devices([], "GPU")
            birdclassifier = tf.saved_model.load(str(tf_model_path))
            self.tf_infer_fn = birdclassifier.signatures["serving_default"]
            print("[features] Using TF SavedModel Perch")
        else:
            raise FileNotFoundError("No usable Perch model backend found.")

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Runs model inference on raw audio batches.

        Summary:
            Calculates raw logits and embeddings for audio signals.

        Inputs:
            x (np.ndarray): Audio batch of shape (Batch * N_windows, window_samples).

        Outputs:
            Tuple[np.ndarray, np.ndarray]: Logits (B, 11000) and Embeddings (B, 1536).

        Shapes:
            - Input `x`: `(B * 12, 160000)`
            - Output Logits: `(B * 12, N_vocab)`
            - Output Embeddings: `(B * 12, 1536)`

        Side effects:
            Runs GPU/CPU calculations.
        """
        if self.use_onnx:
            outs = self.onnx_session.run(None, {self.onnx_input_name: x})
            logits = outs[self.onnx_out_map["label"]].astype(np.float32)
            emb = outs[self.onnx_out_map["embedding"]].astype(np.float32)
        else:
            out = self.tf_infer_fn(inputs=tf.convert_to_tensor(x))
            logits = out["label"].numpy().astype(np.float32)
            emb = out["embedding"].numpy().astype(np.float32)
        return logits, emb


class MultiBackboneExtractor:
    """Manages Perch and Teacher backbone models to extract concatenated features.

    Summary:
        Runs Perch along with an optional distilled teacher ONNX model, concatenating
        their embedding outputs to enrich feature geometry.

    Attributes:
        perch (PerchBackbone): Core Perch backbone module.
        teacher_session (ort.InferenceSession): Distilled teacher ONNX session (if available).
        emb_dim (int): Dimensionality of the joint concatenated embedding.
    """

    def __init__(self, perch_path: Optional[Path] = None, tf_model_path: Optional[Path] = None, teacher_path: Optional[Path] = None):
        """Initializes the multi-backbone sessions.

        Summary:
            Instantiates the core Perch session and queries the optional teacher ONNX.

        Inputs:
            perch_path (Optional[Path]): Path to Perch ONNX weights.
            tf_model_path (Optional[Path]): Path to TF Perch SavedModel directory.
            teacher_path (Optional[Path]): Path to distilled teacher ONNX weights.

        Outputs:
            MultiBackboneExtractor: An instance of the extractor.

        Shapes:
            None.

        Side effects:
            Loads weights and allocates memory sessions.
        """
        self.perch = PerchBackbone(onnx_path=perch_path, tf_model_path=tf_model_path)
        self.teacher_session = None
        self.teacher_emb_dim = 768
        self.use_teacher = teacher_path is not None
        self.emb_dim = 1536 + self.teacher_emb_dim if self.use_teacher else 1536

        if teacher_path is not None and teacher_path.exists():
            try:
                import onnxruntime as ort
                so = ort.SessionOptions()
                so.intra_op_num_threads = 4
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self.teacher_session = ort.InferenceSession(
                    str(teacher_path), sess_options=so, providers=providers
                )
                self.teacher_input_name = self.teacher_session.get_inputs()[0].name
                self.teacher_out_map = {o.name: i for i, o in enumerate(self.teacher_session.get_outputs())}
                # Find output corresponding to embedding (e.g. 768D or 1536D)
                self.teacher_embed_idx = 1
                for idx, o in enumerate(self.teacher_session.get_outputs()):
                    if o.shape and o.shape[-1] == 768:
                        self.teacher_embed_idx = idx
                        break
                self.emb_dim = 1536 + self.teacher_emb_dim
                print(f"[features] Loaded distilled teacher ONNX from: {teacher_path.name} | emb_dim={self.emb_dim}")
            except Exception as e:
                print(f"[features] Warning: Failed to load teacher ONNX: {e}. Using zero-padded teacher fallback.")
                self.teacher_session = None
        else:
            if teacher_path is not None:
                print(f"[features] Warning: Teacher ONNX path {teacher_path} not found. Using zero-padded teacher fallback.")

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Runs joint inference to extract Perch logits and concatenated embeddings.

        Summary:
            Aggregates output tensors from both active ONNX models.

        Inputs:
            x (np.ndarray): Audio batch of shape (Batch * N_windows, window_samples).

        Outputs:
            Tuple[np.ndarray, np.ndarray]: Perch Logits (B, 11000) and joint concatenated Embeddings (B, emb_dim).

        Shapes:
            - Input `x`: `(B * 12, 160000)`
            - Output Logits: `(B * 12, N_vocab)`
            - Output Embeddings: `(B * 12, emb_dim)`

        Side effects:
            Runs CPU/GPU session calculations.
        """
        p_logits, p_emb = self.perch.predict(x)

        if self.teacher_session is not None:
            try:
                # Distilled teacher inference
                # We assume the teacher model accepts raw waveform same shape as Perch
                outs = self.teacher_session.run(None, {self.teacher_input_name: x})
                t_emb = outs[self.teacher_embed_idx].astype(np.float32)
                emb = np.concatenate([p_emb, t_emb], axis=-1)
            except Exception as e:
                # Graceful fallback: pad with zeros if teacher predict fails.
                print(f"[features] Warning: Teacher predict failed: {e}. Using zero-padded teacher fallback.")
                padding = np.zeros((p_emb.shape[0], self.teacher_emb_dim), dtype=np.float32)
                emb = np.concatenate([p_emb, padding], axis=-1)
        else:
            if self.use_teacher:
                # Keep feature shape stable when teacher is requested but unavailable.
                padding = np.zeros((p_emb.shape[0], self.teacher_emb_dim), dtype=np.float32)
                emb = np.concatenate([p_emb, padding], axis=-1)
            else:
                emb = p_emb

        return p_logits, emb


def extract_features(
    paths: List[Path],
    backbone: PerchBackbone,
    mapper: TaxonomyMapper,
    batch_files: int = 16,
    n_windows: int = 12,
    window_samples: int = 160_000,
    file_samples: int = 1_920_000,
    verbose: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Runs end-to-end Perch feature extraction on a list of audio files.

    Summary:
        Loads waveforms, runs inference, maps target classes, and aggregates features.

    Inputs:
        paths (List[Path]): Waveform file paths.
        backbone (PerchBackbone): Loaded Perch engine.
        mapper (TaxonomyMapper): Label mapper.
        batch_files (int): Batch size in files.
        n_windows (int): Windows count (12).
        window_samples (int): Samples per window (160000).
        file_samples (int): Total samples per file (1920000).
        verbose (bool): Whether to show progress bar.

    Outputs:
        Tuple: Contains:
            - meta_df (pd.DataFrame): Recording metadata.
            - scores (np.ndarray): Multi-class logit scores.
            - embs (np.ndarray): Latent feature embeddings.

    Shapes:
        - scores: `(N_files * 12, N_classes)`
        - embs: `(N_files * 12, 1536)`

    Side effects:
        Frees memory via garbage collection.
    """
    from .dataset import read_60s, parse_fname
    import concurrent.futures
    from tqdm.auto import tqdm

    n_rows = len(paths) * n_windows
    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.zeros(n_rows, dtype=np.int16)
    scores = np.zeros((n_rows, mapper.n_classes), dtype=np.float32)
    emb_dim = getattr(backbone, "emb_dim", 1536)
    embs = np.zeros((n_rows, emb_dim), dtype=np.float32)

    wr = 0
    itr = range(0, len(paths), batch_files)
    if verbose:
        itr = tqdm(itr, desc="Perch Features")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as io_executor:
        next_paths = paths[0:batch_files]
        future_audio = [io_executor.submit(read_60s, p, file_samples) for p in next_paths]
        for start in itr:
            batch_paths = next_paths
            batch_n = len(batch_paths)
            batch_audio = [f.result() for f in future_audio]
            del future_audio

            next_start = start + batch_files
            if next_start < len(paths):
                next_paths = paths[next_start:next_start + batch_files]
                future_audio = [io_executor.submit(read_60s, p, file_samples) for p in next_paths]
            else:
                future_audio = []

            x = np.empty((batch_n * n_windows, window_samples), dtype=np.float32)
            br = wr
            for bi, path in enumerate(batch_paths):
                y = batch_audio[bi]
                meta = parse_fname(path.name)
                stem = path.stem
                x[bi * n_windows:(bi + 1) * n_windows] = y.reshape(n_windows, window_samples)
                row_ids[wr:wr + n_windows] = [f"{stem}_{t}" for t in range(5, 65, 5)]
                filenames[wr:wr + n_windows] = path.name
                sites[wr:wr + n_windows] = meta["site"]
                hours[wr:wr + n_windows] = meta["hour_utc"]
                wr += n_windows

            logits, emb = backbone.predict(x)
            scores[br:wr, mapper.mapped_pos] = logits[:, mapper.mapped_bc_idx]
            embs[br:wr] = emb

            # Apply congeneric genus-level proxy scores
            for pos_idx, bc_idxs in mapper.proxy_map.items():
                bc_arr = np.array(bc_idxs, dtype=np.int32)
                scores[br:wr, pos_idx] = logits[:, bc_arr].max(axis=1)

            del x, logits, emb, batch_audio
            gc.collect()

    meta_df = pd.DataFrame({"row_id": row_ids, "filename": filenames, "site": sites, "hour_utc": hours})
    return meta_df, scores, embs


def load_cached_features(
    cache_meta_path: Path, cache_npz_path: Path, primary_labels: List[str], target_emb_dim: int = 1536
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Loads precomputed Perch feature arrays and metadata from files.

    Summary:
        Loads Parquet meta and NPZ embeddings, ensuring column schemas align.

    Inputs:
        cache_meta_path (Path): Path to Parquet metadata file.
        cache_npz_path (Path): Path to cached arrays (.npz).
        primary_labels (List[str]): List of target species classes.

    Outputs:
        Tuple: Contains:
            - meta_tr (pd.DataFrame)
            - sc_tr (np.ndarray): Training class logits.
            - emb_tr (np.ndarray): Training embeddings.

    Shapes:
        - sc_tr: `(N_train_windows, 234)`
        - emb_tr: `(N_train_windows, 1536)`

    Side effects:
        Loads large array objects into RAM.
    """
    meta_tr = pd.read_parquet(cache_meta_path)
    arr = np.load(cache_npz_path)

    # Pick arrays based on dimension matches
    sc_tr = None
    emb_tr = None
    for k in ["scores", "sc", "logits", "arr_0", "scores_full_raw"]:
        if k in arr.files:
            v = arr[k]
            if v.ndim == 2 and v.shape[1] == len(primary_labels):
                sc_tr = v.astype(np.float32)
                break
    for k in ["embs", "emb", "embeddings", "arr_1", "emb_full"]:
        if k in arr.files:
            v = arr[k]
            if v.ndim == 2 and (v.shape[1] == 1536 or v.shape[1] == target_emb_dim):
                emb_tr = v.astype(np.float32)
                break

    if sc_tr is None or emb_tr is None:
        raise KeyError("Could not resolve valid scores or embeddings from cache npz.")

    if emb_tr.shape[1] < target_emb_dim:
        print(f"[features] Warning: Cached embeddings are {emb_tr.shape[1]}D, padding to {target_emb_dim}D.")
        padding = np.zeros((emb_tr.shape[0], target_emb_dim - emb_tr.shape[1]), dtype=np.float32)
        emb_tr = np.concatenate([emb_tr, padding], axis=1)

    # Reorder metadata and array rows sequentially
    if "end_sec" not in meta_tr.columns:
        meta_tr["end_sec"] = meta_tr["row_id"].str.rsplit("_", n=1).str[-1].astype(int)
    meta_tr = meta_tr.copy()
    meta_tr["_cache_pos"] = np.arange(len(meta_tr))
    order = meta_tr.sort_values(["filename", "end_sec"])["_cache_pos"].to_numpy()

    meta_tr = meta_tr.iloc[order].drop(columns=["_cache_pos"]).reset_index(drop=True)
    sc_tr = sc_tr[order]
    emb_tr = emb_tr[order]

    return meta_tr, sc_tr, emb_tr
