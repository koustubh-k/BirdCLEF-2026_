"""config.py — Configuration module for BirdCLEF 2026 Phoenix codebase.

This module stores global execution settings, audio signal parameters,
and model-specific hyperparameters (e.g. Model_22, Model_51, Model_74).
"""

from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Global configuration settings for the BirdCLEF codebase.

    Attributes:
        MODE (str): Execution mode ('train' or 'submit').
        SR (int): Audio sampling rate in Hz.
        WINDOW_SEC (int): Window duration in seconds.
        WINDOW_SAMPLES (int): Number of audio samples per window.
        FILE_SAMPLES (int): Total samples in a 60-second audio file.
        N_WINDOWS (int): Total 5-second windows in a 60-second file (12).
        N_CLASSES (int): Number of target bird classes (234).
        BATCH_FILES (int): Files batch size for feature extraction.
        OOF_N_SPLITS (int): Number of splits for cross-validation.
        DRYRUN_N_FILES (int): Number of files to process in dry-run mode.
        RUN_OOF (bool): Whether to evaluate out-of-fold validation.
        VERBOSE (bool): Verbosity level during execution.
    """

    def __init__(self, mode: str = "submit"):
        """Initializes configuration settings based on the run mode.

        Summary:
            Constructs configuration parameters tailored for either submission
            or local model training.

        Inputs:
            mode (str): Run mode, must be either "train" or "submit". Default is "submit".

        Outputs:
            Config: An instance of the Config class.

        Shapes:
            None.

        Side effects:
            None.

        Usage example:
            >>> config = Config(mode="submit")
            >>> print(config.MODE)
            "submit"
        """
        assert mode in {"train", "submit"}, "Mode must be 'train' or 'submit'."
        self.MODE = mode
        
        # Audio preprocessing constants
        self.SR = 32_000
        self.WINDOW_SEC = 5
        self.WINDOW_SAMPLES = self.SR * self.WINDOW_SEC
        self.FILE_SAMPLES = 60 * self.SR
        self.N_WINDOWS = 12
        self.N_CLASSES = 234

        # Run settings
        self.BATCH_FILES = 16
        self.OOF_N_SPLITS = 5 if self.MODE == "train" else 3
        self.DRYRUN_N_FILES = 20 if self.MODE == "train" else 0
        self.RUN_OOF = self.MODE == "train"
        self.VERBOSE = self.MODE == "train"

        # Advanced Phoenix improvements config
        self.USE_TEACHER = False  # Dynamic toggle
        self.USE_RETRIEVAL_HEAD = True
        self.USE_TAXON_SHARED_CALIBRATION = True
        self.USE_UNCERTAINTY_GATED_SMOOTHING = True
        self.USE_TAXON_OOF_TUNING = True
        self.RETRIEVAL_K = 5
        self.RETRIEVAL_WEIGHT = 0.2
        self.RETRIEVAL_MIN_POS = 10
        self.CALIBRATION_METHOD = "isotonic"
        self.MIN_POS_CALIBRATION = 5
        self.RANK_AWARE_POWER_DEFAULT = 0.6
        self.LAMBDA_PRIOR_DEFAULT = 0.4
        self.CORRECTION_WEIGHT_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        self.LAMBDA_PRIOR_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        self.RANK_AWARE_POWER_GRID = [0.0, 0.2, 0.4, 0.6, 0.8]

        # Path attributes (resolved dynamically)
        self.COMP_DIR = self.resolve_path(
            "/kaggle/input/competitions/birdclef-2026",
            "./birdclef-2026",
            "birdclef-2026"
        )
        self.PERCH_ONNX_PATH = self.resolve_path(
            "/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/perch_v2.onnx",
            "./perch_v2.onnx",
            "rishikeshjani/perch-onnx-for-birdclef-2026"
        )
        self.TEACHER_ONNX_PATH = self.resolve_path(
            "/kaggle/input/datasets/tuckerarrants/bc2026-distilled-sed-public/sed_fold0.onnx",
            "./sed_fold0.onnx",
            "tuckerarrants/bc2026-distilled-sed-public"
        )
        self.PROTO_SSM_PATH = self.resolve_path(
            "/kaggle/input/datasets/hideyukizushi/sgkfk-202604041716/train_proto_ssm_single/models/proto_ssm_best.pt",
            "./proto_ssm_best.pt",
            "hideyukizushi/sgkfk-202604041716"
        )
        self.RESIDUAL_SSM_PATH = self.resolve_path(
            "/kaggle/input/datasets/hideyukizushi/sgkfk-202604041716/ResidualSSM/models/residual_ssm_best.pt",
            "./residual_ssm_best.pt",
            "hideyukizushi/sgkfk-202604041716"
        )
        self.PERCH_CACHE_DIR = self.resolve_path(
            "/kaggle/input/datasets/jaejohn/perch-meta",
            "./perch-meta",
            "jaejohn/perch-meta"
        )

    @staticmethod
    def resolve_path(kaggle_path: str, local_fallback: str, dataset_slug: Optional[str] = None) -> Path:
        """Resolves paths by checking Kaggle environment, local fallback, or kagglehub cache.

        Summary:
            Guarantees execution parity by searching standard Kaggle notebook directories
            first, then local fallbacks, and finally local kagglehub directories.
        """
        p = Path(kaggle_path)
        if p.exists():
            return p

        p_local = Path(local_fallback)
        if p_local.exists():
            return p_local

        # Try to resolve via kagglehub cache
        if dataset_slug:
            import os
            import glob
            cache_base = os.path.expanduser("~/.cache/kagglehub/datasets")
            slug_dir = os.path.join(cache_base, dataset_slug.replace("/", os.sep))
            if os.path.exists(slug_dir):
                leaf = Path(kaggle_path).name
                matches = glob.glob(os.path.join(slug_dir, "**", leaf), recursive=True)
                if matches:
                    return Path(matches[0])
                # If searching for directory (no leaf extension)
                if not Path(kaggle_path).suffix:
                    return Path(slug_dir)

        return p_local

    def get_model_hyperparameters(self, model_name: str) -> Dict[str, Any]:
        """Returns hyperparameters for a specific model setup.

        Summary:
            Retrieves optimizer learning rates, epochs, regularization, and SSM/MLP
            parameters based on the model's design version.

        Inputs:
            model_name (str): Name of the model configuration to fetch
                              (e.g., 'Model_74', 'Model_51', 'Model_22').

        Outputs:
            Dict[str, Any]: Model training and layer size hyperparameters.

        Shapes:
            None.

        Side effects:
            None.

        Usage example:
            >>> config = Config(mode="submit")
            >>> params = config.get_model_hyperparameters("Model_74")
            >>> print(params["proto_ssm_train"]["n_epochs"])
            40
        """
        is_train = self.MODE == "train"
        
        # Base templates for ProtoSSM, ResidualSSM, and MLP probes
        proto_ssm = {
            "n_epochs": 80 if is_train else 40,
            "lr": 8e-4,
            "weight_decay": 1e-3,
            "val_ratio": 0.15,
            "patience": 20 if is_train else 8,
            "pos_weight_cap": 25.0,
            "distill_weight": 0.15,
            "proto_margin": 0.15,
            "label_smoothing": 0.03,
            "mixup_alpha": 0.4,
            "focal_gamma": 2.5,
            "swa_start_frac": 0.65,
            "swa_lr": 4e-4,
            "use_cosine_restart": True,
            "restart_period": 20,
        }

        residual_ssm = {
            "d_model": 128,
            "d_state": 16,
            "n_ssm_layers": 2,
            "dropout": 0.1,
            "correction_weight": 0.35,
            "n_epochs": 40 if is_train else 20,
            "lr": 8e-4,
            "patience": 12 if is_train else 6,
        }

        mlp_params = {
            "hidden_layer_sizes": (256, 128),
            "activation": "relu",
            "max_iter": 500 if is_train else 200,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 20 if is_train else 10,
            "random_state": 42,
            "learning_rate_init": 5e-4,
            "alpha": 0.005,
        }

        # Apply model-specific overrides if applicable
        if model_name in {"Model_21", "Model_22"}:
            proto_ssm["lr"] = 1e-3
            proto_ssm["n_epochs"] = 60 if is_train else 30
            residual_ssm["correction_weight"] = 0.30
        elif model_name in {"Model_51", "Model_52"}:
            proto_ssm["lr"] = 9e-4
            residual_ssm["correction_weight"] = 0.33

        return {
            "proto_ssm_train": proto_ssm,
            "residual_ssm": residual_ssm,
            "mlp_params": mlp_params,
        }
