"""training.py — Model training pipelines for BirdCLEF sequence and stacking models.

This module implements the training logic for LightProtoSSM, ResidualSSM,
and MLP probes, including support for SWA, learning rate scheduling,
class frequency weighting, and sequence feature generation.
"""

from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from tqdm.auto import tqdm

from .models import LightProtoSSM, ResidualSSM


def build_class_freq_weights(Y: np.ndarray, cap: float = 10.0) -> np.ndarray:
    """Computes class frequency weights for loss balancing based on positive ratios.

    Summary:
        Calculates normalized class weights inversely proportional to the square root of
        class frequencies to handle severe class imbalance.

    Inputs:
        Y (np.ndarray): Multi-hot label targets matrix.
        cap (float): Maximum scaling factor cap. Default is 10.0.

    Outputs:
        np.ndarray: Aligned class weights.

    Shapes:
        - Input `Y`: `(N, N_classes)`
        - Output weights: `(N_classes,)`

    Side effects:
        None.

    Usage example:
        >>> Y = np.array([[1, 0], [0, 1], [0, 0]])
        >>> weights = build_class_freq_weights(Y)
        >>> weights.shape
        (2,)
    """
    pos_count = Y.sum(axis=0).astype(np.float32) + 1.0
    freq = pos_count / Y.shape[0]
    weights = np.clip(1.0 / (freq ** 0.5), 1.0, cap)
    return (weights / weights.mean()).astype(np.float32)


def build_sequential_features(scores_col: np.ndarray, n_windows: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates temporal sequential context features for a class predictions column.

    Summary:
        Computes previous, next, mean, max, and std features within a 60-second window sequence.

    Inputs:
        scores_col (np.ndarray): Single-class scores column vector.
        n_windows (int): Length of sequence in windows. Default is 12.

    Outputs:
        Tuple: Contains:
            - prev (np.ndarray): Shifted previous predictions.
            - next_ (np.ndarray): Shifted next predictions.
            - mean (np.ndarray): File-level mean prediction.
            - max_ (np.ndarray): File-level max prediction.
            - std (np.ndarray): File-level standard deviation.

    Shapes:
        - Input `scores_col`: `(N_samples,)` where N_samples is divisible by `n_windows`.
        - Output elements: Each has shape `(N_samples,)`.

    Side effects:
        None.

    Usage example:
        >>> scores = np.random.rand(24)
        >>> prev, nxt, mean, mx, std = build_sequential_features(scores, n_windows=12)
        >>> prev.shape
        (24,)
    """
    x = scores_col.reshape(-1, n_windows)
    prev = np.concatenate([x[:, :1], x[:, :-1]], axis=1)
    next_ = np.concatenate([x[:, 1:], x[:, -1:]], axis=1)
    mean = np.repeat(x.mean(axis=1), n_windows)
    max_ = np.repeat(x.max(axis=1), n_windows)
    std = np.repeat(x.std(axis=1), n_windows)
    return prev.reshape(-1), next_.reshape(-1), mean, max_, std


def train_mlp_probes(
    emb: np.ndarray,
    scores_raw: np.ndarray,
    Y: np.ndarray,
    min_pos: int = 5,
    pca_dim: int = 64,
    alpha_blend: float = 0.4,
    verbose: bool = True
) -> Tuple[Dict[int, MLPClassifier], StandardScaler, PCA, float]:
    """Trains per-class MLPClassifiers (stacking probes) on Perch embeddings and context.

    Summary:
        Fits individual MLP models for active species with balanced sampling and dynamic architectures.

    Inputs:
        emb (np.ndarray): Latent embeddings matrix.
        scores_raw (np.ndarray): Raw class prediction logits.
        Y (np.ndarray): Target multi-hot labels.
        min_pos (int): Minimum positive samples required to train a probe. Default is 5.
        pca_dim (int): Components count for PCA reduction. Default is 64.
        alpha_blend (float): Interpolation ratio for MLP predictions. Default is 0.4.
        verbose (bool): Whether to show progress bars and logging. Default is True.

    Outputs:
        Tuple: Contains:
            - probe_models (Dict[int, MLPClassifier]): Maps class index to fitted classifier.
            - scaler (StandardScaler): Fitted standard scaler for embeddings.
            - pca (PCA): Fitted PCA transformer.
            - alpha_blend (float): Copied blend ratio.

    Shapes:
        - Input `emb`: `(N_samples, D_emb)`
        - Input `scores_raw`: `(N_samples, N_classes)`
        - Input `Y`: `(N_samples, N_classes)`

    Side effects:
        Prints variance information to stdout.

    Usage example:
        >>> probes, scaler, pca, blend = train_mlp_probes(emb_tr, sc_tr, Y_tr)
    """
    scaler = StandardScaler()
    emb_s = scaler.fit_transform(emb)

    pca = PCA(n_components=min(pca_dim, emb_s.shape[1] - 1))
    Z = pca.fit_transform(emb_s).astype(np.float32)

    if verbose:
        print(
            f"[training] MLP embedding: {emb.shape} → PCA: {Z.shape} "
            f"(variance retained: {pca.explained_variance_ratio_.sum():.2%})"
        )

    class_weights = build_class_freq_weights(Y, cap=10.0)
    probe_models = {}
    active = np.where(Y.sum(axis=0) >= min_pos)[0]
    MAX_ROWS = 3000

    iterator = tqdm(active, desc="MLP probes") if verbose else active

    for ci in iterator:
        y = Y[:, ci]
        if y.sum() == 0 or y.sum() == len(y):
            continue

        prev, next_, mean, max_, std = build_sequential_features(scores_raw[:, ci])
        X = np.hstack(
            [
                Z,
                scores_raw[:, ci : ci + 1],
                prev[:, None],
                next_[:, None],
                mean[:, None],
                max_[:, None],
                std[:, None],
            ]
        )

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_idx = np.where(y == 1)[0]
        w = float(class_weights[ci])
        repeat = max(1, min(int(round(w * n_neg / max(n_pos, 1))), 8))
        if n_pos * repeat + len(y) > MAX_ROWS:
            repeat = max(1, (MAX_ROWS - len(y)) // max(n_pos, 1))

        X_bal = np.vstack([X, np.tile(X[pos_idx], (repeat, 1))])
        y_bal = np.concatenate([y, np.ones(n_pos * repeat, dtype=y.dtype)])

        # Tweak E: Wider MLP for frequent classes (>= 50 positives)
        hidden_shape = (256, 128) if n_pos >= 50 else (128, 64)
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_shape,
            activation="relu",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=42,
            learning_rate_init=5e-4,
            alpha=0.005,
        )
        clf.fit(X_bal, y_bal)
        probe_models[ci] = clf

    if verbose:
        print(f"[training] Trained {len(probe_models)} MLP probes")

    return probe_models, scaler, pca, alpha_blend


def train_light_proto_ssm(
    emb_full: np.ndarray,
    scores_full: np.ndarray,
    Y_full: np.ndarray,
    meta_full: pd.DataFrame,
    n_epochs: int = 40,
    patience: int = 8,
    lr: float = 1e-3,
    n_sites: int = 20,
    n_windows: int = 12,
    n_classes: int = 234,
    verbose: bool = False
) -> Tuple[nn.Module, Dict[str, int]]:
    """Trains the prototype-augmented State Space Model (LightProtoSSM).

    Summary:
        Runs gradient descent optimization on sequences of window embeddings, incorporating
        OneCycle learning rates, SWA updates, and congeneric proxy initializations.

    Inputs:
        emb_full (np.ndarray): Flat Perch embeddings array.
        scores_full (np.ndarray): Flat raw Perch prediction logits.
        Y_full (np.ndarray): Flat multi-hot bird target labels.
        meta_full (pd.DataFrame): Dataframe containing metadata ('filename', 'site', 'hour_utc').
        n_epochs (int): Total epochs to train. Default is 40.
        patience (int): Early stopping patience. Default is 8.
        lr (float): Peak learning rate. Default is 1e-3.
        n_sites (int): Total sites in vocab. Default is 20.
        n_windows (int): Length of sequence in windows. Default is 12.
        n_classes (int): Total number of bird classes. Default is 234.
        verbose (bool): Whether to output training progress. Default is False.

    Outputs:
        Tuple: Contains:
            - model (nn.Module): The optimized LightProtoSSM model (potentially SWA averaged).
            - site2i (Dict[str, int]): Map from site string to embedding index.

    Shapes:
        - Input `emb_full`: `(N_samples, D_emb)` where N_samples is divisible by `n_windows`.
        - Input `scores_full`: `(N_samples, n_classes)`
        - Input `Y_full`: `(N_samples, n_classes)`

    Side effects:
        Instantiates PyTorch optimizer/schedulers and allocates memory on CPU/GPU.

    Usage example:
        >>> model, site2i = train_light_proto_ssm(emb_tr, sc_tr, Y_tr, meta_tr)
    """
    n_files = len(emb_full) // n_windows
    emb_f = emb_full.reshape(n_files, n_windows, -1)
    log_f = scores_full.reshape(n_files, n_windows, -1)
    lab_f = Y_full.reshape(n_files, n_windows, -1).astype(np.float32)

    fnames = meta_full["filename"].unique()
    sites_u = sorted(meta_full["site"].dropna().unique())
    site2i = {s: i + 1 for i, s in enumerate(sites_u)}

    site_ids = np.array([
        min(site2i.get(meta_full.loc[meta_full["filename"] == fn, "site"].iloc[0], 0), n_sites - 1)
        for fn in fnames
    ], dtype=np.int64)

    hour_ids = np.array([
        int(meta_full.loc[meta_full["filename"] == fn, "hour_utc"].iloc[0]) % 24
        for fn in fnames
    ], dtype=np.int64)

    model = LightProtoSSM(
        d_input=int(emb_full.shape[1]),
        n_classes=n_classes,
        n_sites=n_sites,
        n_windows=n_windows,
        use_cross_attn=True,
        cross_attn_heads=2
    )
    model.init_prototypes(
        torch.tensor(emb_full, dtype=torch.float32),
        torch.tensor(Y_full, dtype=torch.float32)
    )

    emb_t = torch.tensor(emb_f, dtype=torch.float32)
    log_t = torch.tensor(log_f, dtype=torch.float32)
    lab_t = torch.tensor(lab_f, dtype=torch.float32)
    site_t = torch.tensor(site_ids, dtype=torch.long)
    hour_t = torch.tensor(hour_ids, dtype=torch.long)

    pos_cnt = lab_t.sum(dim=(0, 1))
    total = lab_t.shape[0] * lab_t.shape[1]
    pos_weight = ((total - pos_cnt) / (pos_cnt + 1)).clamp(max=25.0)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=n_epochs, steps_per_epoch=1, pct_start=0.1, anneal_strategy="cos"
    )

    best_loss = float("inf")
    best_state = None
    wait = 0

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = int(n_epochs * 0.65)
    swa_sched = torch.optim.swa_utils.SWALR(opt, swa_lr=4e-4)

    for ep in range(n_epochs):
        model.train()
        out = model(emb_t, log_t, site_ids=site_t, hours=hour_t)
        loss = F.binary_cross_entropy_with_logits(
            out, lab_t, pos_weight=pos_weight[None, None, :]
        ) + 0.15 * F.mse_loss(out, log_t)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ep >= swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()
        else:
            sched.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"[training] Early stopping triggered at epoch {ep}")
                break

    if ep >= swa_start:
        torch.optim.swa_utils.update_bn(emb_t.unsqueeze(0), swa_model)
        model = swa_model
    else:
        model.load_state_dict(best_state)

    model.eval()
    return model, site2i


def train_residual_ssm(
    emb_full: np.ndarray,
    first_pass_flat: np.ndarray,
    Y_full: np.ndarray,
    site_ids: np.ndarray,
    hour_ids: np.ndarray,
    n_epochs: int = 30,
    patience: int = 8,
    lr: float = 1e-3,
    correction_weight: float = 0.30,
    n_windows: int = 12,
    n_classes: int = 234,
    verbose: bool = False
) -> Tuple[nn.Module, float]:
    """Trains the second-pass error-correction ResidualSSM on validation targets.

    Summary:
        Trains a state space model on first-pass prediction residuals using validation split
        early stopping control.

    Inputs:
        emb_full (np.ndarray): Flat latent audio embeddings array.
        first_pass_flat (np.ndarray): Flat first-pass model prediction logits.
        Y_full (np.ndarray): Flat target multi-hot labels.
        site_ids (np.ndarray): Integer array representing file site indices.
        hour_ids (np.ndarray): Integer array representing file hour indices.
        n_epochs (int): Epochs limit. Default is 30.
        patience (int): Validation loss patience steps. Default is 8.
        lr (float): Peak learning rate. Default is 1e-3.
        correction_weight (float): Blending factor for residual correction. Default is 0.30.
        n_windows (int): Sequence length in windows. Default is 12.
        n_classes (int): Total number of bird classes. Default is 234.
        verbose (bool): Whether to log validation loss updates. Default is False.

    Outputs:
        Tuple: Contains:
            - model (nn.Module): The fitted ResidualSSM model.
            - correction_weight (float): The input blending weight coefficient.

    Shapes:
        - Input `emb_full`: `(N_samples, D_emb)`
        - Input `first_pass_flat`: `(N_samples, n_classes)`
        - Input `Y_full`: `(N_samples, n_classes)`
        - Input `site_ids`: `(N_files,)` where `N_files = N_samples / n_windows`.
        - Input `hour_ids`: `(N_files,)`

    Side effects:
        Allocates PyTorch model sessions on CPU/GPU.

    Usage example:
        >>> res_model, c_weight = train_residual_ssm(emb_tr, fp_tr, Y_tr, sites, hours)
    """
    n_files = len(emb_full) // n_windows
    emb_f = emb_full.reshape(n_files, n_windows, -1)
    fp_f = first_pass_flat.reshape(n_files, n_windows, -1)
    lab_f = Y_full.reshape(n_files, n_windows, -1).astype(np.float32)

    # Compute prediction residuals
    fp_prob = 1.0 / (1.0 + np.exp(-np.clip(fp_f, -30, 30)))
    residuals = lab_f - fp_prob

    # Holdout validation split for ResidualSSM early stopping
    n_val = max(1, int(n_files * 0.15))
    rng = torch.Generator()
    rng.manual_seed(42)
    perm = torch.randperm(n_files, generator=rng).numpy()
    val_i = perm[:n_val]
    train_i = perm[n_val:]

    emb_t = torch.tensor(emb_f, dtype=torch.float32)
    fp_t = torch.tensor(fp_f, dtype=torch.float32)
    res_t = torch.tensor(residuals, dtype=torch.float32)
    site_t = torch.tensor(site_ids, dtype=torch.long)
    hour_t = torch.tensor(hour_ids, dtype=torch.long)

    n_sites_model = max(20, int(site_ids.max()) + 1) if len(site_ids) > 0 else 20
    model = ResidualSSM(
        d_input=int(emb_full.shape[1]),
        d_scores=int(first_pass_flat.shape[1]),
        n_classes=n_classes,
        n_windows=n_windows,
        n_sites=n_sites_model,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=n_epochs, steps_per_epoch=1, pct_start=0.1, anneal_strategy="cos"
    )

    best_loss = float("inf")
    best_state = None
    wait = 0

    for ep in range(n_epochs):
        model.train()
        corr = model(emb_t[train_i], fp_t[train_i], site_ids=site_t[train_i], hours=hour_t[train_i])
        loss = F.mse_loss(corr, res_t[train_i])

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        # Validation pass
        model.eval()
        with torch.no_grad():
            val_corr = model(emb_t[val_i], fp_t[val_i], site_ids=site_t[val_i], hours=hour_t[val_i])
            val_loss = F.mse_loss(val_corr, res_t[val_i])

        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"[training] Residual SSM Early stopping at epoch {ep}")
                break

    model.load_state_dict(best_state)
    return model, correction_weight
