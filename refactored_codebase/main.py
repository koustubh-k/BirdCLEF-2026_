"""main.py — Local entry runner and Kaggle-compatible execution coordinator.

This script runs the entire BirdCLEF Phoenix audio prediction pipeline locally
or in a Kaggle notebook. It resolves input paths dynamically from KAGGLE_INPUTS.md,
falls back to mock metadata if raw competition files are not found, trains models,
performs OOF hyperparameter tuning, fits calibrators, and formats submissions.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add modular code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.dataset import load_competition_metadata, prepare_labeled_matrix, read_60s
from src.features import load_cached_features, extract_features, TaxonomyMapper, MultiBackboneExtractor
from src.models import LightProtoSSM, ResidualSSM, VectorizedMLPProbes, EmbeddingRetrievalHead
from src.training import train_light_proto_ssm, train_residual_ssm, train_mlp_probes
from src.validation import get_group_kfold_splits, macro_auc
from src.inference import run_sequence_inference
from src.postprocessing import (
    apply_postprocessing,
    apply_calibration,
    apply_threshold_sharpening,
    apply_retrieval_blend,
    f_TAX_SMOOTHING_POSTPROC,
    run_corrected_calibration_pipeline,
    build_fold_safe_prior_tables,
    apply_prior
)
from src.ensemble import direct_blend, division_attention_blend


def load_or_mock_metadata(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], Dict[str, int]]:
    """Loads competition CSV metadata, falling back to mock tables if not available.

    Summary:
        Attempts loading CSV resources from competition mount. If missing, resolves classes
        from cached submission files and synthesizes consistent mock DataFrames.
    """
    try:
        taxonomy, sample_sub, soundscape_labels, primary_labels, label_to_idx = load_competition_metadata(config.COMP_DIR)
        print(f"[main] Successfully loaded competition metadata from {config.COMP_DIR}")
        return taxonomy, sample_sub, soundscape_labels, primary_labels, label_to_idx
    except Exception as e:
        print(f"[main] Could not load competition metadata from {config.COMP_DIR} ({e}). Creating mock metadata...")

        # Load the 234 classes from the cached submission.csv if possible, otherwise use a default mock
        try:
            cached_sub = pd.read_csv(Path(os.path.expanduser("~/.cache/kagglehub/datasets/hideyukizushi/sgkfk-202604041716/versions/1/submission.csv")))
            primary_labels = cached_sub.columns[1:].tolist()
            assert len(primary_labels) == 234
        except Exception:
            # Fallback mock labels
            primary_labels = [f"label_{i}" for i in range(234)]

        label_to_idx = {lbl: i for i, lbl in enumerate(primary_labels)}

        # Create mock taxonomy
        taxonomy_rows = []
        for lbl in primary_labels:
            # Let's mock a few non-Aves to test taxon calibration / smoothing
            taxon = "Aves"
            if "0" in lbl or "1" in lbl:
                taxon = "Amphibia"
            elif "2" in lbl or "3" in lbl:
                taxon = "Insecta"
            taxonomy_rows.append({
                "primary_label": lbl,
                "scientific_name": f"Scientific {lbl}",
                "class_name": taxon
            })
        taxonomy = pd.DataFrame(taxonomy_rows)

        # Create mock sample_submission
        try:
            full_meta = pd.read_parquet(Path(os.path.expanduser("~/.cache/kagglehub/datasets/jaejohn/perch-meta/versions/1/full_perch_meta.parquet")))
            row_ids = full_meta["row_id"].tolist()
        except Exception:
            row_ids = [f"BC2026_Train_0001_S08_20250606_030007_{t}" for t in range(5, 65, 5)]

        sample_sub_data = {"row_id": row_ids}
        for lbl in primary_labels:
            sample_sub_data[lbl] = 0.5
        sample_sub = pd.DataFrame(sample_sub_data)

        # Create mock soundscape labels (aligned with full_perch_meta)
        sc_labels_rows = []
        try:
            full_meta = pd.read_parquet(Path(os.path.expanduser("~/.cache/kagglehub/datasets/jaejohn/perch-meta/versions/1/full_perch_meta.parquet")))
            np.random.seed(42)
            for _, row in full_meta.iterrows():
                # Randomly assign 1-2 bird labels
                lbls = np.random.choice(primary_labels, size=np.random.randint(1, 3), replace=False)
                for lbl in lbls:
                    end_sec = int(row["row_id"].split("_")[-1])
                    start_sec = end_sec - 5
                    h_start, m_start = divmod(start_sec, 3600)
                    m_start, s_start = divmod(m_start, 60)
                    h_end, m_end = divmod(end_sec, 3600)
                    m_end, s_end = divmod(m_end, 60)
                    sc_labels_rows.append({
                        "filename": row["filename"],
                        "start": f"{h_start:02d}:{m_start:02d}:{s_start:02d}",
                        "end": f"{h_end:02d}:{m_end:02d}:{s_end:02d}",
                        "primary_label": lbl
                    })
        except Exception:
            for rid in row_ids:
                end_sec = int(rid.split("_")[-1])
                start_sec = end_sec - 5
                h_start, m_start = divmod(start_sec, 3600)
                m_start, s_start = divmod(m_start, 60)
                h_end, m_end = divmod(end_sec, 3600)
                m_end, s_end = divmod(m_end, 60)
                sc_labels_rows.append({
                    "filename": "BC2026_Train_0001_S08_20250606_030007.ogg",
                    "start": f"{h_start:02d}:{m_start:02d}:{s_start:02d}",
                    "end": f"{h_end:02d}:{m_end:02d}:{s_end:02d}",
                    "primary_label": primary_labels[0]
                })
        soundscape_labels = pd.DataFrame(sc_labels_rows)

        print(f"[main] Mock metadata generated successfully: {len(primary_labels)} classes, {len(row_ids)} windows.")
        return taxonomy, sample_sub, soundscape_labels, primary_labels, label_to_idx


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 Local Pipeline Entry Runner")
    parser.add_argument("--mode", type=str, default="submit", choices=["train", "submit"],
                        help="Execution mode: train (quick run CV) or submit (mimic inference submit)")
    parser.add_argument("--use-teacher", action="store_true", help="Enable multi-backbone teacher distillation head")
    args = parser.parse_args()

    config = Config(mode=args.mode)
    if args.use_teacher:
        config.USE_TEACHER = True

    print(f"\n=======================================================")
    print(f"Starting BirdCLEF Phoenix Entry Runner in {config.MODE.upper()} mode")
    print(f"  USE_TEACHER: {config.USE_TEACHER}")
    print(f"=======================================================\n")

    # 1. Load/mock metadata
    taxonomy, sample_sub, soundscape_labels, primary_labels, label_to_idx = load_or_mock_metadata(config)

    # 2. Instantiate taxonomy mapper
    perch_labels_path = config.PERCH_ONNX_PATH.parent / "labels.csv"
    if not perch_labels_path.exists():
        perch_labels_path = Path(os.path.expanduser("~/.cache/kagglehub/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/versions/2/labels.csv"))
    mapper = TaxonomyMapper(taxonomy, primary_labels, perch_labels_path)

    # 3. Load cached Perch features
    cache_meta_path = config.PERCH_CACHE_DIR / "full_perch_meta.parquet"
    cache_npz_path = config.PERCH_CACHE_DIR / "full_perch_arrays.npz"
    if not (cache_meta_path.exists() and cache_npz_path.exists()):
        cache_meta_path = config.resolve_path(
            "/kaggle/input/datasets/jaejohn/perch-meta/full_perch_meta.parquet",
            "./full_perch_meta.parquet",
            "jaejohn/perch-meta"
        )
        cache_npz_path = config.resolve_path(
            "/kaggle/input/datasets/jaejohn/perch-meta/full_perch_arrays.npz",
            "./full_perch_arrays.npz",
            "jaejohn/perch-meta"
        )

    target_emb_dim = 2304 if config.USE_TEACHER else 1536
    print(f"[main] Loading features from {cache_meta_path.parent.name}... (Target dimension: {target_emb_dim}D)")
    meta_tr, sc_tr, emb_tr = load_cached_features(
        cache_meta_path, cache_npz_path, primary_labels, target_emb_dim=target_emb_dim
    )
    print(f"[main] Loaded training features: logits={sc_tr.shape}, embeddings={emb_tr.shape}")

    # 4. Prepare label matrices
    sc_df, Y_SC, full_rows, Y_FULL = prepare_labeled_matrix(soundscape_labels, primary_labels, label_to_idx)
    print(f"[main] Prepared labeled matrices: Y_SC={Y_SC.shape}, Y_FULL={Y_FULL.shape}")

    # Align features with soundscape labels metadata
    meta_tr = meta_tr.set_index("row_id")
    valid_ids = [rid for rid in full_rows["row_id"] if rid in meta_tr.index]
    full_rows = full_rows[full_rows["row_id"].isin(valid_ids)].reset_index(drop=True)

    emb_tr = emb_tr[[meta_tr.index.get_loc(rid) for rid in valid_ids]]
    sc_tr = sc_tr[[meta_tr.index.get_loc(rid) for rid in valid_ids]]
    meta_tr = meta_tr.loc[valid_ids].reset_index()
    Y_FULL = Y_SC[[sc_df.set_index("row_id").index.get_loc(rid) for rid in valid_ids]]

    print(f"[main] Aligned features size: {emb_tr.shape}, labels: {Y_FULL.shape}")

    class_name_map = taxonomy.set_index("primary_label")["class_name"].to_dict()
    class_taxons = [class_name_map.get(lbl, "Aves") for lbl in primary_labels]

    if config.MODE == "submit":
        print("\n--- INFERENCE / SUBMIT RUN ---")

        # --- Auto-infer model architecture from checkpoint ---
        def _infer_proto_config(state_dict, default_d_input):
            """Infer LightProtoSSM hyperparameters from checkpoint keys."""
            d_model = state_dict.get("input_proj.0.bias", torch.zeros(128)).shape[0]
            d_input = state_dict.get("input_proj.0.weight", torch.zeros(128, default_d_input)).shape[1]
            d_state = state_dict.get("ssm_fwd.0.A_log", torch.zeros(128, 16)).shape[1]
            n_sites = state_dict.get("site_emb.weight", torch.zeros(40, 16)).shape[0]
            meta_dim = state_dict.get("site_emb.weight", torch.zeros(40, 16)).shape[1]
            n_classes = state_dict.get("prototypes", torch.zeros(234, 128)).shape[0]
            # Count SSM layers by checking ssm_fwd.{i}.A_log keys
            n_ssm_layers = sum(1 for k in state_dict if k.startswith("ssm_fwd.") and k.endswith(".A_log"))
            n_ssm_layers = max(n_ssm_layers, 2)
            return dict(d_input=d_input, d_model=d_model, d_state=d_state, n_sites=n_sites,
                        meta_dim=meta_dim, n_classes=n_classes, n_ssm_layers=n_ssm_layers)

        def _infer_res_config(state_dict, default_d_input, default_d_scores):
            """Infer ResidualSSM hyperparameters from checkpoint keys."""
            d_model = state_dict.get("output_head.weight", torch.zeros(234, 64)).shape[1]
            total_input = state_dict.get("input_proj.0.weight", torch.zeros(64, default_d_input + default_d_scores)).shape[1]
            d_state = state_dict.get("ssm_fwd.A_log", torch.zeros(64, 8)).shape[1]
            n_sites = state_dict.get("site_emb.weight", torch.zeros(40, 8)).shape[0]
            meta_dim = state_dict.get("site_emb.weight", torch.zeros(40, 8)).shape[1]
            n_classes = state_dict.get("output_head.weight", torch.zeros(234, 64)).shape[0]
            return dict(d_model=d_model, d_state=d_state, n_sites=n_sites,
                        meta_dim=meta_dim, n_classes=n_classes)

        # Load checkpoints and infer architecture
        proto_ckpt, res_ckpt = None, None
        proto_cfg = dict(d_input=target_emb_dim, d_model=128, d_state=16, n_classes=config.N_CLASSES,
                         n_sites=40, meta_dim=16, n_ssm_layers=2)
        res_cfg = dict(d_model=64, d_state=8, n_classes=config.N_CLASSES, n_sites=40, meta_dim=8)

        if config.PROTO_SSM_PATH.exists():
            proto_ckpt = torch.load(config.PROTO_SSM_PATH, map_location="cpu")
            proto_cfg = _infer_proto_config(proto_ckpt, target_emb_dim)
            print(f"[main] ProtoSSM checkpoint inferred: d_model={proto_cfg['d_model']}, d_state={proto_cfg['d_state']}, "
                  f"n_ssm_layers={proto_cfg['n_ssm_layers']}, n_sites={proto_cfg['n_sites']}, meta_dim={proto_cfg['meta_dim']}")

        if config.RESIDUAL_SSM_PATH.exists():
            res_ckpt = torch.load(config.RESIDUAL_SSM_PATH, map_location="cpu")
            res_cfg = _infer_res_config(res_ckpt, target_emb_dim, config.N_CLASSES)
            print(f"[main] ResidualSSM checkpoint inferred: d_model={res_cfg['d_model']}, d_state={res_cfg['d_state']}, "
                  f"n_sites={res_cfg['n_sites']}, meta_dim={res_cfg['meta_dim']}")

        # Instantiate models with checkpoint-matched dimensions
        proto_model = LightProtoSSM(
            d_input=proto_cfg['d_input'],
            d_model=proto_cfg['d_model'],
            d_state=proto_cfg['d_state'],
            n_classes=config.N_CLASSES,
            n_sites=proto_cfg['n_sites'],
            meta_dim=proto_cfg['meta_dim'],
            n_ssm_layers=proto_cfg['n_ssm_layers'],
            n_windows=config.N_WINDOWS,
            use_cross_attn=False,  # Structural cross_attn layout varies across checkpoints
        )
        res_model = ResidualSSM(
            d_input=target_emb_dim,
            d_scores=config.N_CLASSES,
            d_model=res_cfg['d_model'],
            d_state=res_cfg['d_state'],
            n_classes=config.N_CLASSES,
            n_sites=res_cfg['n_sites'],
            meta_dim=res_cfg['meta_dim'],
            n_windows=config.N_WINDOWS
        )

        # Load weights with strict=False to handle structural differences
        if proto_ckpt is not None:
            try:
                if config.USE_TEACHER:
                    proto_ckpt = {k: v for k, v in proto_ckpt.items() if "input_proj" not in k}
                missing, unexpected = proto_model.load_state_dict(proto_ckpt, strict=False)
                print(f"[main] Loaded ProtoSSM weights (strict=False): {len(missing)} missing, {len(unexpected)} unexpected keys")
            except Exception as e:
                print(f"[main] Failed to load ProtoSSM weights: {e}")

        if res_ckpt is not None:
            try:
                if config.USE_TEACHER:
                    res_ckpt = {k: v for k, v in res_ckpt.items() if "input_proj" not in k}
                missing, unexpected = res_model.load_state_dict(res_ckpt, strict=False)
                print(f"[main] Loaded ResidualSSM weights (strict=False): {len(missing)} missing, {len(unexpected)} unexpected keys")
            except Exception as e:
                print(f"[main] Failed to load ResidualSSM weights: {e}")

        # Fit retrieval head
        retrieval_head = None
        if config.USE_RETRIEVAL_HEAD:
            retrieval_head = EmbeddingRetrievalHead(k=config.RETRIEVAL_K, retrieval_weight=config.RETRIEVAL_WEIGHT)
            retrieval_head.fit(torch.tensor(emb_tr), torch.tensor(Y_FULL))
            print("[main] Fit EmbeddingRetrievalHead successfully.")

        # Build site/hour prior tables
        prior_tables = build_fold_safe_prior_tables(meta_tr, Y_FULL)
        print("[main] Built prior tables.")

        # Locate test audio files
        test_audio_dir = config.COMP_DIR / "test_soundscapes"
        test_files = list(test_audio_dir.glob("*.ogg")) if test_audio_dir.exists() else []
        
        if len(test_files) > 0:
            # LIVE INFERENCE: Extract features from the hidden test set
            is_real_test = True
            print(f"[main] Found {len(test_files)} test files. Running feature extraction...")
            # Ensure we read the actual sample submission for row IDs
            sample_sub_path = config.COMP_DIR / "sample_submission.csv"
            sample_sub = pd.read_csv(sample_sub_path)
            
            extractor = MultiBackboneExtractor(perch_path=config.PERCH_ONNX_PATH, teacher_path=config.TEACHER_ONNX_PATH if config.USE_TEACHER else None)
            test_meta, test_sc, test_emb = extract_features(test_files, extractor, mapper, batch_files=config.BATCH_FILES, n_windows=config.N_WINDOWS)
        else:
            # PUBLIC RUN FALLBACK: If no test files (e.g., standard public commit), mock it to avoid crashing
            is_real_test = False
            print("[main] No test files found. Using cached dummy data for pipeline validation.")
            test_emb = emb_tr[:120]
            test_sc = sc_tr[:120]
            test_meta = meta_tr.iloc[:120]

        sites_u = sorted(meta_tr["site"].dropna().unique())
        site2i = {s: i + 1 for i, s in enumerate(sites_u)}
        test_site_ids = np.array([site2i.get(s, 0) for s in test_meta["site"].values[::config.N_WINDOWS]])
        test_hour_ids = test_meta["hour_utc"].values[::config.N_WINDOWS]
        test_sc_adjusted = apply_prior(
            test_sc, test_meta["site"].values, test_meta["hour_utc"].values, prior_tables, lambda_prior=0.4
        )

        # Run calibration fitting on training data
        print("[main] Running calibration fitting on training data...")
        tr_site_ids = np.array([site2i.get(s, 0) for s in meta_tr["site"].values[::config.N_WINDOWS]])
        tr_hour_ids = meta_tr["hour_utc"].values[::config.N_WINDOWS]
        tr_sc_adjusted = apply_prior(sc_tr, meta_tr["site"].values, meta_tr["hour_utc"].values, prior_tables, lambda_prior=0.4)
        
        final_scores_tr, first_pass_tr, _, _ = run_sequence_inference(
            proto_model=proto_model,
            res_model=res_model,
            emb_te=emb_tr,
            sc_te=sc_tr,
            sc_te_adjusted=tr_sc_adjusted,
            test_site_ids=tr_site_ids,
            test_hour_ids=tr_hour_ids,
            mapped_mask=mapper.mapped_mask,
            correction_weight=0.35,
            retrieval_head=retrieval_head,
            n_windows=config.N_WINDOWS,
            n_classes=config.N_CLASSES
        )

        print("[main] Optimizing calibration and thresholds on full training set...")
        w, calibrators, taxon_calibrators, thresholds, best_lambdas, best_powers = run_corrected_calibration_pipeline(
            first_pass_logits_tr=first_pass_tr,
            correction_flat_tr=final_scores_tr - first_pass_tr,
            Y_true=Y_FULL,
            temperatures=mapper.temperatures,
            n_windows=config.N_WINDOWS,
            correction_grid=config.CORRECTION_WEIGHT_GRID,
            class_taxons=class_taxons,
            prior_tables=prior_tables,
            sites=meta_tr["site"].values,
            hours=meta_tr["hour_utc"].values
        )
        print(f"[main] Calibration optimized on training set. Selected weight w={w:.2f}")

        # Run test set sequence inference
        print("[main] Running sequence inference on test set...")
        final_scores, first_pass, tta_variance, retrieval_probs = run_sequence_inference(
            proto_model=proto_model,
            res_model=res_model,
            emb_te=test_emb,
            sc_te=test_sc,
            sc_te_adjusted=test_sc_adjusted,
            test_site_ids=test_site_ids,
            test_hour_ids=test_hour_ids,
            mapped_mask=mapper.mapped_mask,
            correction_weight=w,
            retrieval_head=retrieval_head,
            n_windows=config.N_WINDOWS,
            n_classes=config.N_CLASSES
        )

        # Postprocessing, calibration, and taxonomy smoothing
        try:
            probs = apply_postprocessing(
                final_scores, mapper.temperatures, n_windows=config.N_WINDOWS,
                prior_tables=prior_tables, sites=test_meta["site"].values, hours=test_meta["hour_utc"].values,
                lambda_prior=best_lambdas, class_taxons=class_taxons, ra_power=best_powers
            )
            
            # Apply retrieval blend for low-sample/unmapped species
            class_pos_counts = Y_FULL.sum(axis=0)
            rare_mask = class_pos_counts < config.RETRIEVAL_MIN_POS
            retrieval_mask = np.logical_or(~mapper.mapped_mask, rare_mask)
            if retrieval_probs is not None and config.USE_RETRIEVAL_HEAD:
                probs = apply_retrieval_blend(probs, retrieval_probs, retrieval_mask, retrieval_weight=config.RETRIEVAL_WEIGHT)
                n_retrieval_classes = int(retrieval_mask.sum())
                print(f"[main] Applied retrieval head blend on {n_retrieval_classes} rare/unmapped classes.")

            # Apply calibration
            calibrated_probs = apply_calibration(
                probs, calibrators, taxon_calibrators=taxon_calibrators, class_taxons=class_taxons
            )

            # Apply threshold sharpening
            sharpened = apply_threshold_sharpening(calibrated_probs, thresholds)

            # Convert to DataFrame
            sub_df = pd.DataFrame(sharpened, columns=primary_labels)
            sub_df.insert(0, "row_id", test_meta["row_id"].values)

            # Apply taxonomy smoothing and temporal consistency
            # Only align against sample_submission when real test files exist
            post_smoothed = f_TAX_SMOOTHING_POSTPROC(
                sub_df, base_path=config.COMP_DIR if is_real_test else None, tta_variance=tta_variance
            )
            
            # Write output
            out_csv = Path("./submission.csv")
            post_smoothed.to_csv(out_csv, index=True)
            print(f"[main] Final submission successfully written to {out_csv.resolve()}")
            print(f"  Shape: {post_smoothed.shape}")
        except Exception as e:
            print(f"[main] Postprocessing or writing submission failed: {e}")
            import traceback
            traceback.print_exc()
            raise e
    else:
        print("\n--- CV TRAINING RUN ---")
        splits = list(get_group_kfold_splits(meta_tr, n_splits=config.OOF_N_SPLITS))
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"\n--- Fold {fold} Training Run ---")
            
            # Train LightProtoSSM
            print("[main] Training LightProtoSSM (quick 2 epochs)...")
            proto_model, site2i = train_light_proto_ssm(
                emb_full=emb_tr[train_idx],
                scores_full=sc_tr[train_idx],
                Y_full=Y_FULL[train_idx],
                meta_full=meta_tr.iloc[train_idx],
                n_epochs=2,
                patience=1,
                lr=8e-4,
                n_windows=config.N_WINDOWS,
                n_classes=config.N_CLASSES,
                verbose=True
            )

            # Train ResidualSSM
            print("[main] Training ResidualSSM (quick 2 epochs)...")
            res_model, corr_w = train_residual_ssm(
                emb_full=emb_tr[train_idx],
                first_pass_flat=sc_tr[train_idx],
                Y_full=Y_FULL[train_idx],
                site_ids=np.zeros(len(train_idx) // config.N_WINDOWS, dtype=np.int64),
                hour_ids=np.zeros(len(train_idx) // config.N_WINDOWS, dtype=np.int64),
                n_epochs=2,
                patience=1,
                lr=8e-4,
                n_windows=config.N_WINDOWS,
                n_classes=config.N_CLASSES,
                verbose=True
            )

            print(f"[main] Fold {fold} training done. Aborting further fold training for verification speed.")
            break

    print("\n[main] BirdCLEF Phoenix Entry Runner completed successfully.\n")


if __name__ == "__main__":
    main()
