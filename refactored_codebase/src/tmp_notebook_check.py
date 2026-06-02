# --- Cell 1 ---
solutions = {
 'type_add' :'single',
 'Models'   : [
 {'Model':'Model_7','subm':'submission.csv','weight':1, 'xSED':[],'LB':'0.948'}
 ]
}

# --- Cell 2 ---
_ensemble_models = [model['Model' ] for model in solutions['Models']]
_files_subm      = [model['subm'  ] for model in solutions['Models']]
_weights         = [model['weight'] for model in solutions['Models']]
_xsed            = [model['xSED'  ] for model in solutions['Models']]
_lbs             = [model['LB'    ] for model in solutions['Models']]

_single_solution = True if solutions['type_add']=='single' else False

# --- Cell 5 ---
if 'Model_2' in _ensemble_models or \
   'task' in solutions and 'run Model_2_SED once' in solutions['task']:

    _file_name_submission = "subm_2.csv"

    # Install from bundled wheel (no network needed at scoring time)
    !pip install -q /kaggle/input/datasets/tuckerarrants/perch-v2-no-dft-onnx/onnxruntime-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
    
    # =================================================================
    # S1 -- IMPORTS + CONFIG
    # =================================================================
    import os, sys, time, json, pickle, gc, random, math
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from collections import defaultdict
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, ConcatDataset
    from torch.cuda.amp import GradScaler, autocast
    import torchaudio
    import timm
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from scipy.special import expit as sigmoid_np
    import warnings
    warnings.filterwarnings("ignore")
    
    SEED = 42
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name()}")
    
    # ------------------------------------------------------------------
    # Notebook Mode
    # -----------------------------------------------------------------
    MODE = "infer"  # "train" or "infer"
    
    # =================================================================
    # CONFIGURATION
    # =================================================================
    COMP_DIR = Path("/kaggle/input/competitions/birdclef-2026")
    WAVEFORM_CACHE_DIR = Path("/kaggle/input/datasets/tuckerarrants/birdclef-2026-waveform-cache/waveform_cache")
    PERCH_ONNX_PATH = Path("/kaggle/input/datasets/tuckerarrants/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx")
    
    LABELS_PATH     = COMP_DIR / "train_soundscapes_labels.csv"
    TAXONOMY_PATH   = COMP_DIR / "taxonomy.csv"
    SAMPLE_SUB_PATH = COMP_DIR / "sample_submission.csv"
    TEST_DIR        = COMP_DIR / "test_soundscapes"
    
    OUT_DIR = Path("/kaggle/working")
    
    NUM_CLASSES = 234
    SR = 32000
    
    # --- Duration ---
    TRAIN_DURATION = 5    # seconds
    VAL_DURATION   = 5    # always 5s for competition eval
    TRAIN_SAMPLES  = SR * TRAIN_DURATION
    VAL_SAMPLES    = SR * VAL_DURATION
    
    N_FOLDS = 5
    
    # --- Mel spectrogram ---
    N_FFT      = 2048
    HOP_LENGTH = 512
    N_MELS     = 256
    FMIN       = 20
    FMAX       = 16000
    
    # --- Model ---
    BACKBONE_NAME = "segformer_b0"  # Upgraded from EfficientNet-B0 to SegFormer-B0
    
    # --- Perch distillation ---
    USE_PERCH_DISTILL = True
    PERCH_EMBED_DIM   = 1536
    ALPHA_DISTILL     = 1.0   # MSE loss weight
    
    # --- Training ---
    FOLDS  = [0, 1, 2, 3, 4]
    EPOCHS = 25
    BATCH  = 16 if MODE == "train" else 64  # Larger batch for inference
    LR     = 5e-4
    MIN_LR = 1e-5
    WD     = 1e-4
    WARMUP_EPOCHS = 2
    
    # --- Upsampling ---
    MIN_SAMPLE = 20
    
    # --- Augmentation ---
    AUG_PROB = 0.5
    AUG_GAIN_DB_RANGE      = (-6.0, 6.0)
    AUG_NOISE_SNR_DB_RANGE = (10.0, 30.0)
    
    # --- MixUp ---
    USE_FOCAL_MIXUP    = True
    MIXUP_PROB         = 0.5
    MIXUP_ALPHA        = 0.4
    MIXUP_HARD         = True    # union labels (hard) vs weighted blend (soft)
    
    USE_FOCAL_SC_MIXUP     = True
    FOCAL_SC_MIXUP_PROB    = 0.5
    FOCAL_SC_MIXUP_ALPHA   = 0.4
    
    # --- FreqMixStyle (disabled by default) ---
    FREQ_MIXSTYLE_PROB  = 0.0
    FREQ_MIXSTYLE_ALPHA = 0.1
    
    # --- SpecAugment ---
    FREQ_MASK_PARAM = 10
    TIME_MASK_PARAM = 10
    NUM_FREQ_MASKS  = 1
    NUM_TIME_MASKS  = 2
    
    # --- Source weights ---
    USE_FOCAL           = True
    USE_FOCAL_SECONDARY = True
    USE_LABELED_SC      = True
    
    ACTIVE_SOURCES = ["focal", "sc"]
    SHARES = {"focal": 0.9, "sc": 0.1}
    SOURCE_WEIGHTS = {
        "focal":         1.0,
        "focal_missing": 0.0,
        "sc":            1.0,
    }
    
    print(f"Backbone: {BACKBONE_NAME}")
    print(f"Train duration: {TRAIN_DURATION}s | Mel: {N_MELS} mels, n_fft={N_FFT}, hop={HOP_LENGTH}")
    print(f"Distillation: {'ON' if USE_PERCH_DISTILL else 'OFF'} (alpha={ALPHA_DISTILL})")
    print(f"Batch: {BATCH} | Epochs: {EPOCHS} | Folds: {FOLDS}")
    
    
    if MODE == "train":
        !pip install -q timm torchaudio onnxscript onnx
    
    # =================================================================
    # S2 -- LOAD DATA
    # =================================================================
    
    # --- Label ordering from sample_submission (defines column order) ---
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    PRIMARY_LABELS = sample_sub.columns[1:].tolist()
    LABEL2IDX = {label: idx for idx, label in enumerate(PRIMARY_LABELS)}
    taxonomy = pd.read_csv(TAXONOMY_PATH)
    label_to_taxon = dict(zip(taxonomy["primary_label"].astype(str),
                              taxonomy["class_name"].astype(str)))
    TAXON_MASKS = {t: np.array([i for i, l in enumerate(PRIMARY_LABELS)
                                if label_to_taxon.get(l, "") == t])
                   for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]}
    
    # --- Focal recording metadata ---
    audio_cache_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "audio_cache_meta.csv")
    train_df = pd.read_csv(COMP_DIR / "train.csv")
    audio_cache_meta = audio_cache_meta.merge(
        train_df[["filename", "secondary_labels"]], on="filename", how="left"
    )
    audio_cache_meta = audio_cache_meta[
        audio_cache_meta["primary_label"].isin(LABEL2IDX)
    ].reset_index(drop=True)
    print(f"Focal audio cache: {len(audio_cache_meta)} entries")
    
    # --- Soundscape window metadata ---
    sc_cache_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "soundscape_cache_meta.csv")
    sc_cache_meta["label_list"] = sc_cache_meta["label_list"].apply(
        lambda x: x.split(";") if isinstance(x, str) else []
    )
    print(f"Soundscape cache: {len(sc_cache_meta)} windows")
    
    # --- Build soundscape label matrix from ground truth ---
    sc_labels_raw = pd.read_csv(LABELS_PATH).drop_duplicates()
    sc_labels_raw["start_sec"] = pd.to_timedelta(sc_labels_raw["start"]).dt.total_seconds().astype(int)
    
    Y_SC = np.zeros((len(sc_cache_meta), NUM_CLASSES), dtype=np.float32)
    for i, row in sc_cache_meta.iterrows():
        matches = sc_labels_raw[
            (sc_labels_raw["filename"] == row["filename"]) &
            (sc_labels_raw["start_sec"] == row["start_sec"])
        ]
        for _, m in matches.iterrows():
            for lbl in str(m["primary_label"]).split(";"):
                lbl = lbl.strip()
                if lbl in LABEL2IDX:
                    Y_SC[i, LABEL2IDX[lbl]] = 1.0
    
    labeled_sc_mask = Y_SC.sum(axis=1) > 0
    print(f"Soundscape labels: {labeled_sc_mask.sum()}/{len(Y_SC)} windows labeled, "
          f"{int(Y_SC.sum())} positives, {int((Y_SC.sum(axis=0) > 0).sum())} species")
    
    # =================================================================
    # FOLD ASSIGNMENT
    # =================================================================
    
    # --- Focal: StratifiedKFold by species ---
    audio_for_split = audio_cache_meta.drop_duplicates("original_idx").reset_index(drop=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    audio_for_split["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(audio_for_split, audio_for_split["primary_label"])):
        audio_for_split.loc[val_idx, "fold"] = fold
    audio_cache_meta = audio_cache_meta.merge(
        audio_for_split[["original_idx", "fold"]], on="original_idx", how="left"
    )
    print(f"\nFocal fold distribution:\n{audio_cache_meta['fold'].value_counts().sort_index()}")
    
    # --- Soundscape: file-level folds, all 66 files distributed ---
    # All files including S22 participate in CV for maximum species coverage
    # (46 multi-fold species vs 32 with S22 holdout). The non_s22_mask_sc
    # filter in evaluation still excludes S22 windows from the primary metric.
    from sklearn.model_selection import GroupKFold
    
    sc_files = sc_cache_meta[["filename", "site"]].drop_duplicates().reset_index(drop=True)
    gkf = GroupKFold(n_splits=N_FOLDS)
    sc_files["fold"] = -1
    for fold, (_, val_idx) in enumerate(gkf.split(sc_files, groups=sc_files["filename"])):
        sc_files.loc[sc_files.index[val_idx], "fold"] = fold
    
    file_to_fold = dict(zip(sc_files["filename"], sc_files["fold"]))
    sc_cache_meta["fold"] = sc_cache_meta["filename"].map(file_to_fold).fillna(-1).astype(int)
    print(f"\nSoundscape fold distribution:")
    print(sc_cache_meta["fold"].value_counts().sort_index())
    # =================================================================
    # UPSAMPLE RARE SPECIES
    # =================================================================
    counts = audio_cache_meta["primary_label"].value_counts()
    rare_species = counts[counts < MIN_SAMPLE].index
    extra_rows = []
    for sp in rare_species:
        sp_rows = audio_cache_meta[audio_cache_meta["primary_label"] == sp]
        n_copies = int(np.ceil(MIN_SAMPLE / len(sp_rows))) - 1
        for _ in range(n_copies):
            extra_rows.append(sp_rows)
    
    n_before = len(audio_cache_meta)
    if extra_rows:
        audio_cache_meta = pd.concat([audio_cache_meta] + extra_rows, ignore_index=True)
    print(f"\nUpsampled {len(rare_species)} rare species (min={MIN_SAMPLE}): "
          f"{n_before} -> {len(audio_cache_meta)} samples")
    
    # Non-S22 mask for evaluation (S22 is a site with known label noise)
    sc_sites = sc_cache_meta["site"].values
    non_s22_mask_sc = sc_sites != "S22"
    print(f"S22: {(~non_s22_mask_sc).sum()}, non-S22: {non_s22_mask_sc.sum()}")
    print("OK Data loaded")
    
    
    if DEBUG:
        EPOCHS = 1
        FOLDS = [0]
        audio_cache_meta = audio_cache_meta.groupby("primary_label").head(3).reset_index(drop=True)
        sc_cache_meta = sc_cache_meta.head(50)
        Y_SC = Y_SC[:50]
        non_s22_mask_sc = non_s22_mask_sc[:50]
        print(f"DEBUG MODE: {len(audio_cache_meta)} focal, {len(sc_cache_meta)} sc, "
              f"{EPOCHS} epoch, folds={FOLDS}")
    
    # =================================================================
    # S3 -- EVAL UTILITIES + MEL TRANSFORM + SED MODELS
    # =================================================================
    
    def compute_macro_auc(y_true, y_pred, mask=None, class_mask=None):
        """Macro-averaged AUC across evaluable species."""
        if mask is not None:
            y_true, y_pred = y_true[mask], y_pred[mask]
        if class_mask is not None:
            y_true, y_pred = y_true[:, class_mask], y_pred[:, class_mask]
        aucs = []
        for c in range(y_true.shape[1]):
            col = y_true[:, c]
            if col.sum() == 0 or col.sum() == len(col):
                continue
            try:
                aucs.append(roc_auc_score(col, y_pred[:, c]))
            except ValueError:
                continue
        return (np.mean(aucs) if aucs else float("nan")), len(aucs)
    
    def full_eval(y_true, y_pred, ns22, tm):
        r = {}
        a, n = compute_macro_auc(y_true, y_pred)
        r["macro_auc_all"], r["n_all"] = round(a, 4), n
        a, n = compute_macro_auc(y_true, y_pred, mask=ns22)
        r["non_s22_macro"], r["n_ns22"] = round(a, 4), n
        for t, cm in tm.items():
            a, n = compute_macro_auc(y_true, y_pred, mask=ns22, class_mask=cm)
            r[f"non_s22_{t}"] = round(a, 4)
        return r
    
    # ------------------------------------------------------------------
    # GPU Mel Spectrogram
    # ------------------------------------------------------------------
    class MelSpecTransform(nn.Module):
        def __init__(self):
            super().__init__()
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
            )
            self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
    
        def forward(self, waveform):
            return self.db_transform(self.mel_spec(waveform))
    
    # ------------------------------------------------------------------
    # GPU SpecAugment
    # ------------------------------------------------------------------
    class SpecAugment(nn.Module):
        def __init__(self):
            super().__init__()
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=FREQ_MASK_PARAM)
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=TIME_MASK_PARAM)
    
        def forward(self, mel):
            for _ in range(NUM_FREQ_MASKS):
                mel = self.freq_mask(mel)
            for _ in range(NUM_TIME_MASKS):
                mel = self.time_mask(mel)
            return mel
    
    # ------------------------------------------------------------------
    # Frozen Perch teacher -- ONNX inference, no gradients
    # ------------------------------------------------------------------
    import onnxruntime as ort
    
    class PerchTeacher:
        """Frozen Perch v2 via ONNX. Takes 5s waveforms, returns 1536-d embeddings.
        The teacher is never updated -- it provides a stable distillation target."""
    
        def __init__(self, onnx_path, device_str="cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
                if device_str == "cuda" else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(onnx_path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self._out_names = [o.name for o in self.session.get_outputs()]
            self._embed_idx = None
            for i, o in enumerate(self.session.get_outputs()):
                if o.shape and o.shape[-1] == PERCH_EMBED_DIM:
                    self._embed_idx = i
                    break
            if self._embed_idx is None:
                self._embed_idx = 1
            print(f"Perch ONNX loaded: embed_idx={self._embed_idx}")
    
        @torch.no_grad()
        def embed(self, waveforms_5s):
            """waveforms_5s: (B, 160000) float32, returns (B, 1536) embeddings."""
            wav_np = waveforms_5s.cpu().numpy()
            results = self.session.run(None, {self.input_name: wav_np})
            return torch.from_numpy(results[self._embed_idx]).float()
    
    # ------------------------------------------------------------------
    # Distillation head: GAP + Linear to Perch embedding space
    # ------------------------------------------------------------------
    class DistillHead(nn.Module):
        """Projects backbone features to Perch's 1536-d space via GAP + Linear."""
        def __init__(self, backbone_dim, embed_dim=1536):
            super().__init__()
            self.proj = nn.Linear(backbone_dim, embed_dim)
    
        def forward(self, feature_map):
            gap = feature_map.mean(dim=[2, 3])   # (B, C, F, T) -> (B, C)
            return self.proj(gap)                 # (B, embed_dim)
    
    # ------------------------------------------------------------------
    # SED Model V2: SegFormer + GeMFreq + bottleneck + AttBlock
    # ------------------------------------------------------------------
    class GeMFreqPool(nn.Module):
        """Generalized Mean pooling over frequency. Learnable p starts at 3.0
        (sharper than mean, softer than max). Lets the model emphasize
        frequency bands where species vocalize."""
        def __init__(self, p_init=3.0, eps=1e-6):
            super().__init__()
            self.p = nn.Parameter(torch.tensor(float(p_init)))
            self.eps = eps
    
        def forward(self, x):
            p = self.p.clamp(min=1.0)
            x = x.clamp(min=self.eps).pow(p)
            x = x.mean(dim=2)
            return x.pow(1.0 / p)
    
    class BirdSEDModel(nn.Module):
        """SED model with SegFormer backbone and 1st-place-inspired design.
        - SegFormer-B0: Lightweight transformer with efficient attention.
        - GeMFreq pooling (learnable, sharper than mean)
        - 512-dim bottleneck with dropout
        - Attention-weighted clip logits from frame logits
        - Distillation: GAP+Linear branch for MSE to Perch
        - Stop gradient: backbone trains from distillation only
        """
        def __init__(self, backbone_name=BACKBONE_NAME, num_classes=NUM_CLASSES,
                     drop_path_rate=0.1, hidden_dim=512):
            super().__init__()
            self.backbone = timm.create_model(
                backbone_name, pretrained=True, in_chans=1,
                num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
            )
            with torch.no_grad():
                n_tf = TRAIN_SAMPLES // HOP_LENGTH + 1
                dummy = torch.randn(1, 1, N_MELS, n_tf)
                feat = self.backbone(dummy)
                self.backbone_dim = feat.shape[1]
                print(f"V2 backbone: {tuple(feat.shape)}  (C={self.backbone_dim})")
    
            self.gem_freq = GeMFreqPool(p_init=3.0)
            self.dense = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(self.backbone_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
            )
            self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
            self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
            nn.init.xavier_uniform_(self.att.weight)
            nn.init.xavier_uniform_(self.cla.weight)
            self.att.bias.data.fill_(0.)
            self.cla.bias.data.fill_(0.)
            if USE_PERCH_DISTILL:
                self.distill_head = DistillHead(self.backbone_dim, PERCH_EMBED_DIM)
    
        def forward(self, x, return_framewise=False, return_distill=False):
            h = self.backbone(x)
            distill_emb = None
            if return_distill and hasattr(self, 'distill_head'):
                distill_emb = self.distill_head(h)
    
            # Stop gradient: SED head doesn't update the backbone
            h_cls = h.detach() if USE_PERCH_DISTILL else h
    
            h_cls = self.gem_freq(h_cls)            # (B, C, T)
            h_cls = h_cls.permute(0, 2, 1)          # (B, T, C)
            h_cls = self.dense(h_cls)               # (B, T, 512)
            h_cls = h_cls.permute(0, 2, 1)          # (B, 512, T)
    
            norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
            framewise_logits = self.cla(h_cls)
            clip_logits = torch.sum(norm_att * framewise_logits, dim=2)
    
            fw = framewise_logits.permute(0, 2, 1) if return_framewise else None
            if return_framewise and return_distill: return clip_logits, fw, distill_emb
            elif return_framewise: return clip_logits, fw
            elif return_distill: return clip_logits, distill_emb
            return clip_logits
    
    def make_model():
            return BirdSEDModel(BACKBONE_NAME).to(device)
    
    print("OK Model definitions ready")
    
    
    # =================================================================
    # S4 -- DATA PIPELINE
    # =================================================================
    
    def load_int16(path):
        """Load int16 waveform tensor to float32 in [-1, 1]."""
        waveform_int16 = torch.load(path, map_location="cpu")
        return waveform_int16.float() / 32767.0
    
    _FC = {}
    def load_focal(p):
        """Load focal waveform with simple LRU cache."""
        if p in _FC: return _FC[p]
        pp = WAVEFORM_CACHE_DIR / p
        if not pp.exists(): return None
        a = load_int16(pp).numpy()
        if len(_FC) >= 2000:
            _FC.pop(next(iter(_FC)))
        _FC[p] = a
        return a
    
    _SC_CACHE = {}
    def load_sc_waveform_from(cache_dir, cache_file):
        """Load a soundscape waveform with LRU cache."""
        key = str(cache_dir / cache_file)
        if key in _SC_CACHE: return _SC_CACHE[key]
        pp = cache_dir / cache_file
        if not pp.exists(): return None
        a = load_int16(pp).numpy()
        if len(_SC_CACHE) >= 200:
            _SC_CACHE.pop(next(iter(_SC_CACHE)))
        _SC_CACHE[key] = a
        return a
    
    def extract_chunk_np(waveform, start_sample, n_samples):
        """Extract a chunk, left-padding if the recording is too short."""
        total = len(waveform)
        if total <= n_samples:
            return np.pad(waveform, (n_samples - total, 0))
        end = start_sample + n_samples
        if end > total:
            start_sample = max(0, total - n_samples)
        return waveform[start_sample:start_sample + n_samples]
    
    def apply_aug(w):
        """Simple waveform augmentation: gain jitter + noise + shift."""
        if np.random.random() < AUG_PROB:
            w = w * (10 ** (np.random.uniform(*AUG_GAIN_DB_RANGE) / 20))
        if np.random.random() < AUG_PROB:
            sp = (w ** 2).mean()
            if sp > 1e-10:
                w = w + np.random.randn(*w.shape).astype(w.dtype) * np.sqrt(
                    sp / (10 ** (np.random.uniform(*AUG_NOISE_SNR_DB_RANGE) / 10)))
        return w
    
    # ------------------------------------------------------------------
    # Build soundscape MixUp pool (labeled windows only)
    # ------------------------------------------------------------------
    sc_mixup_sources = []
    _sc_file_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "soundscape_file_meta.csv")
    _sc_file_dict = dict(zip(_sc_file_meta["filename"], _sc_file_meta["cache_file"]))
    _labeled_rows = []
    for i in range(len(sc_cache_meta)):
        row = sc_cache_meta.iloc[i]
        if Y_SC[i].sum() > 0:
            cf = _sc_file_dict.get(row["filename"])
            if cf is not None:
                _labeled_rows.append({
                    "filename": row["filename"], "start_sec": int(row["start_sec"]),
                    "cache_file": cf, "label_idx": i, "fold": int(row.get("fold", -1)),
                })
    if _labeled_rows:
        _labeled_meta = pd.DataFrame(_labeled_rows)
        sc_mixup_sources.append((WAVEFORM_CACHE_DIR, _labeled_meta, Y_SC))
        print(f"SC MixUp pool: {len(_labeled_meta)} labeled windows")
    
    # ------------------------------------------------------------------
    # FocalDS -- with Focal-Focal AND Focal-Soundscape MixUp
    # ------------------------------------------------------------------
    class FocalDS(Dataset):
        """Focal recording dataset. Returns (waveform, label, weight, mask, source_tag)."""
        def __init__(self, df, l2i, secondary_lookup=None,
                     sc_mixup_sources=None, fold_k=None, aug=False):
            self.df, self.l2i, self.aug = df.reset_index(drop=True), l2i, aug
            self.secondary_lookup = secondary_lookup
            self.sc_mixup_sources = sc_mixup_sources
            self.fold_k = fold_k
    
        def __len__(self): return len(self.df)
    
        def _load_chunk(self, r):
            w = load_focal(r["cache_file"])
            if w is None: return None, None
            if self.aug:
                start = np.random.randint(0, max(1, len(w) - TRAIN_SAMPLES + 1)) if len(w) > TRAIN_SAMPLES else 0
            else:
                start = int(r.get("start_sec", 0)) * SR
            ch = extract_chunk_np(w, start, TRAIN_SAMPLES)
            lb = np.zeros(NUM_CLASSES, dtype=np.float32)
            if str(r["primary_label"]) in self.l2i:
                lb[self.l2i[str(r["primary_label"])]] = 1.0
            if self.secondary_lookup is not None and "original_idx" in self.df.columns:
                for s in self.secondary_lookup.get(int(r["original_idx"]), []):
                    if s in self.l2i: lb[self.l2i[s]] = 1.0
            return ch, lb
    
        def __getitem__(self, i):
            r1 = self.df.iloc[i]
            ch1, lb1 = self._load_chunk(r1)
            if ch1 is None:
                return (torch.zeros(1, TRAIN_SAMPLES), torch.zeros(NUM_CLASSES),
                        torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal_missing")
    
            # Focal-Focal MixUp
            if USE_FOCAL_MIXUP and self.aug and np.random.random() < MIXUP_PROB:
                ch2 = None
                for _ in range(3):
                    j = np.random.randint(len(self.df))
                    ch2, lb2 = self._load_chunk(self.df.iloc[j])
                    if ch2 is not None: break
                if ch2 is not None:
                    lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                    ch_mix = (lam * ch1 + (1 - lam) * ch2).astype(np.float32)
                    if self.aug: ch_mix = apply_aug(ch_mix)
                    lb = np.maximum(lb1, lb2) if MIXUP_HARD else (lam * lb1 + (1 - lam) * lb2)
                    return (torch.from_numpy(ch_mix).unsqueeze(0), torch.from_numpy(lb),
                            torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal")
    
            # Focal-Soundscape MixUp
            if (USE_FOCAL_SC_MIXUP and self.aug and self.sc_mixup_sources
                    and np.random.random() < FOCAL_SC_MIXUP_PROB):
                src_idx = np.random.randint(len(self.sc_mixup_sources))
                cache_dir, meta_df_sc, labels = self.sc_mixup_sources[src_idx]
                eligible = meta_df_sc[meta_df_sc["fold"] != self.fold_k] if self.fold_k is not None else meta_df_sc
                if len(eligible) > 0:
                    sc_row = eligible.iloc[np.random.randint(len(eligible))]
                    sc_wav = load_sc_waveform_from(cache_dir, sc_row["cache_file"])
                    if sc_wav is not None and len(sc_wav) >= TRAIN_SAMPLES:
                        sc_chunk = extract_chunk_np(sc_wav, int(sc_row["start_sec"]) * SR, TRAIN_SAMPLES)
                        lam = np.random.beta(FOCAL_SC_MIXUP_ALPHA, FOCAL_SC_MIXUP_ALPHA)
                        ch_mix = (lam * ch1 + (1 - lam) * sc_chunk).astype(np.float32)
                        if self.aug: ch_mix = apply_aug(ch_mix)
                        lb_sc = labels[int(sc_row["label_idx"])].astype(np.float32)
                        lb = np.maximum(lb1, lb_sc) if MIXUP_HARD else lam * lb1 + (1 - lam) * lb_sc
                        return (torch.from_numpy(ch_mix).unsqueeze(0), torch.from_numpy(lb),
                                torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal")
    
            # No MixUp
            if self.aug: ch1 = apply_aug(ch1)
            return (torch.from_numpy(ch1.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(lb1),
                    torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal")
    
    # ------------------------------------------------------------------
    # ScDS -- Labeled soundscape windows
    # ------------------------------------------------------------------
    class ScDS(Dataset):
        def __init__(self, Y, sc_df, aug=False):
            self.Y, self.df, self.aug = Y, sc_df.reset_index(drop=True), aug
        def __len__(self): return len(self.Y)
        def __getitem__(self, i):
            row = self.df.iloc[i]
            wav_full = load_sc_waveform_from(WAVEFORM_CACHE_DIR, row.get("cache_file")) \
                       if row.get("cache_file") else None
            if wav_full is None:
                wav_t = torch.zeros(1, TRAIN_SAMPLES)
            else:
                chunk = extract_chunk_np(wav_full, int(row["start_sec"]) * SR, TRAIN_SAMPLES)
                if self.aug: chunk = apply_aug(chunk)
                wav_t = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0)
            return (wav_t, torch.from_numpy(self.Y[i].astype(np.float32)),
                    torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "sc")
    
    # ------------------------------------------------------------------
    # Load focal secondary labels
    # ------------------------------------------------------------------
    focal_secondary_labels = None
    if USE_FOCAL_SECONDARY:
        focal_secondary_labels = {}
        for idx, row in train_df.iterrows():
            sec = row.get("secondary_labels", "")
            if pd.isna(sec) or sec in ("", "[]"): continue
            try:
                sec_list = eval(sec) if isinstance(sec, str) else []
                valid = [s for s in sec_list if s in LABEL2IDX]
                if valid: focal_secondary_labels[idx] = valid
            except: continue
        print(f"Focal secondary labels: {len(focal_secondary_labels)} files")
    
    # =================================================================
    # S5 -- TRAINING
    # =================================================================
    
    def _load_val_waveforms(val_sc_df):
        """Load validation waveforms (always 5s)."""
        sc_file_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "soundscape_file_meta.csv")
        sc_file_dict = dict(zip(sc_file_meta["filename"], sc_file_meta["cache_file"]))
        wavs = []
        for _, row in val_sc_df.iterrows():
            cf = sc_file_dict.get(row["filename"])
            if cf is not None:
                w = load_sc_waveform_from(WAVEFORM_CACHE_DIR, cf)
                if w is not None:
                    chunk = extract_chunk_np(w, int(row["start_sec"]) * SR, VAL_SAMPLES)
                    wavs.append(torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0))
                else: wavs.append(torch.zeros(1, VAL_SAMPLES))
            else: wavs.append(torch.zeros(1, VAL_SAMPLES))
        return wavs
    
    def _predict_from_waveforms(model, mel_transform, wav_list, batch_size=64):
        """Inference: mel -> model -> sigmoid. Distillation head is NOT used."""
        model.eval()
        preds_clip, preds_fmax, preds_blend = [], [], []
        with torch.no_grad():
            for s in range(0, len(wav_list), batch_size):
                batch = torch.stack(wav_list[s:s+batch_size]).to(device)
                mel = mel_transform(batch)
                B = mel.size(0)
                for i in range(B):
                    mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
                with autocast():
                    clip_logits, framewise = model(mel, return_framewise=True)
                    frame_max = framewise.max(dim=1).values
                    p_clip = torch.sigmoid(clip_logits).cpu().numpy()
                    p_fmax = torch.sigmoid(frame_max).cpu().numpy()
                    p_blend = 0.5 * p_clip + 0.5 * p_fmax
                preds_clip.append(p_clip); preds_fmax.append(p_fmax); preds_blend.append(p_blend)
        return {"clip": np.concatenate(preds_clip), "fmax": np.concatenate(preds_fmax),
                "blend": np.concatenate(preds_blend)}
    
    def build_active_datasets(fold_k):
        items = []
        if USE_FOCAL:
            fds = FocalDS(audio_cache_meta[audio_cache_meta["fold"] != fold_k],
                          LABEL2IDX, secondary_lookup=focal_secondary_labels,
                          sc_mixup_sources=sc_mixup_sources if USE_FOCAL_SC_MIXUP else None,
                          fold_k=fold_k, aug=True)
            items.append(("focal", fds, len(fds)))
        if USE_LABELED_SC:
            vm = sc_cache_meta["fold"].values == fold_k
            Y_tr = Y_SC[~vm]
            sc_train_df = sc_cache_meta[~vm].reset_index(drop=True)
            sds = ScDS(Y_tr, sc_train_df, aug=True)
            items.append(("sc", sds, len(sds)))
        return items
    
    def train_fold(fold_k):
        vm = sc_cache_meta["fold"].values == fold_k
        Y_val = Y_SC[vm]
        ns22_val = non_s22_mask_sc[vm]
        val_sc_df = sc_cache_meta[vm].reset_index(drop=True)
    
        active = build_active_datasets(fold_k)
        names, datasets, sizes = zip(*active)
        mds = ConcatDataset(list(datasets))
        nst = max(100, int(sum(sizes) / BATCH))
    
        print(f"  Streams: {dict(zip(names, sizes))}  steps/ep: {nst}")
    
        m = make_model()
        mel_transform = MelSpecTransform().to(device)
        spec_augment = SpecAugment().to(device)
        perch_teacher = PerchTeacher(PERCH_ONNX_PATH,
                                      "cuda" if torch.cuda.is_available() else "cpu") \
                        if USE_PERCH_DISTILL else None
    
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
        scaler = GradScaler()
        warmup_steps = nst * WARMUP_EPOCHS
        total_steps  = nst * EPOCHS
        warmup_sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1/25, end_factor=1.0,
                                                          total_iters=warmup_steps)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - warmup_steps,
                                                                   eta_min=1e-6)
        sch = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup_sched, cosine_sched],
                                                     milestones=[warmup_steps])
    
        history = {"ep": [], "train_loss": [], "cls_loss": [], "dist_loss": [],
                   "macro": [], "ns22_macro": [],
                   "ns22_Aves": [], "ns22_Amphibia": [], "ns22_Insecta": [], "ns22_Mammalia": [],
                   "val_preds": []}
        best_ns22, best_state_ns22 = -1.0, None
        best_macro, best_state_macro = -1.0, None
        val_wavs = _load_val_waveforms(val_sc_df_k)
    
        for ep in range(EPOCHS):
            m.train()
            smp = MixSamp(list(sizes), list(names), SHARES, BATCH, nst, seed=42 + ep)
            tl = DataLoader(mds, batch_sampler=smp, collate_fn=collate_m,
                            num_workers=0, pin_memory=True)
            el, el_cls, el_dist, nb_count = 0.0, 0.0, 0.0, 0
            t0 = time.time()
    
            for wav, lb, wt, mk, sr in tl:
                wav, lb, wt, mk = wav.to(device), lb.to(device), wt.to(device), mk.to(device)
                sw = mk_sw(sr).to(device)
    
                with torch.no_grad():
                    mel = mel_transform(wav)
                    B = mel.size(0)
                    for i in range(B):
                        mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
                    mel = spec_augment(mel)
    
                with autocast():
                    if USE_PERCH_DISTILL:
                        clip_logits, framewise, distill_emb = m(mel, return_framewise=True,
                                                                return_distill=True)
                    else:
                        clip_logits, framewise = m(mel, return_framewise=True)
    
                    frame_max_logits = framewise.max(dim=1).values
    
                    # Classification loss
                    bce_clip = F.binary_cross_entropy_with_logits(clip_logits, lb, reduction="none")
                    bce_frame = F.binary_cross_entropy_with_logits(frame_max_logits, lb, reduction="none")
                    bce = 0.5 * bce_clip + 0.5 * bce_frame
                    ps = (bce * wt * mk).sum(1) / (mk.sum(1) + 1e-8)
                    cls_loss = (ps * sw).mean()
    
                    # Distillation loss
                    if USE_PERCH_DISTILL and perch_teacher is not None:
                        with torch.no_grad():
                            wav_5s = wav.squeeze(1)
                            N = wav_5s.shape[1]
                            if N > 160000:
                                start = (N - 160000) // 2
                                wav_5s = wav_5s[:, start:start + 160000]
                            elif N < 160000:
                                wav_5s = F.pad(wav_5s, (0, 160000 - N))
                            perch_emb = perch_teacher.embed(wav_5s).to(device)
                        distill_loss = F.mse_loss(distill_emb, perch_emb)
                        loss = cls_loss + ALPHA_DISTILL * distill_loss
                    else:
                        distill_loss = torch.tensor(0.0)
                        loss = cls_loss
    
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sch.step()
                el += loss.item(); el_cls += cls_loss.item()
                el_dist += distill_loss.item(); nb_count += 1
    
            # Validation
            val_preds_dict = _predict_from_waveforms(m, mel_transform, val_wavs)
            val_preds = val_preds_dict["blend"]
            r = full_eval(Y_val, val_preds, ns22_val, TAXON_MASKS)
            for mode in ["clip", "fmax", "blend"]:
                r_mode = full_eval(Y_val, val_preds_dict[mode], ns22_val, TAXON_MASKS)
                r[f"ns22_{mode}"] = r_mode["non_s22_macro"]
    
            history["ep"].append(ep)
            history["train_loss"].append(round(el / nb_count, 5))
            history["cls_loss"].append(round(el_cls / nb_count, 5))
            history["dist_loss"].append(round(el_dist / nb_count, 5))
            history["macro"].append(r["macro_auc_all"])
            history["ns22_macro"].append(r["non_s22_macro"])
            for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]:
                history[f"ns22_{t}"].append(r[f"non_s22_{t}"])
            history["val_preds"].append(val_preds.astype(np.float32))
    
            tag_ns22 = ""; tag_macro = ""
            if r["non_s22_macro"] > best_ns22:
                best_ns22 = r["non_s22_macro"]
                best_state_ns22 = {k: v.cpu().clone() for k, v in m.state_dict().items()}
                tag_ns22 = " *ns22"
            if r["macro_auc_all"] > best_macro:
                best_macro = r["macro_auc_all"]
                best_state_macro = {k: v.cpu().clone() for k, v in m.state_dict().items()}
                tag_macro = " *macro"
    
            dist_str = f" dist={el_dist/nb_count:.4f}" if USE_PERCH_DISTILL else ""
            print(f"    Ep{ep:02d}: loss={el/nb_count:.4f} cls={el_cls/nb_count:.4f}{dist_str} "
                  f"lr={opt.param_groups[0]['lr']:.1e} | "
                  f"ns22: {r['ns22_blend']:.4f} | "
                  f"Av={r['non_s22_Aves']:.4f} Am={r['non_s22_Amphibia']:.4f} "
                  f"In={r['non_s22_Insecta']:.4f} Ma={r['non_s22_Mammalia']:.4f} "
                  f"[{time.time()-t0:.0f}s]{tag_ns22}{tag_macro}")
    
        del perch_teacher, m, mel_transform, spec_augment
        torch.cuda.empty_cache(); gc.collect()
        return best_state_ns22, best_state_macro, history
    
    print("OK Training function ready")
    
    
    # =================================================================
    # S6 -- FOLD LOOP + ONNX EXPORT
    # =================================================================
    
    if MODE != "train":
        print("Skipping training (MODE='infer')")
        oof_ns22 = None
        all_hist = {}
    else:
    
        oof_ns22 = np.full((len(sc_cache_meta), NUM_CLASSES), np.nan, dtype=np.float32)
        all_hist = {}
        for fold_k in FOLDS:
            print(f"\n{'='*60}")
            print(f"FOLD {fold_k}")
            print(f"{'='*60}")
            vm = sc_cache_meta["fold"].values == fold_k
            val_sc_df_k = sc_cache_meta[vm].reset_index(drop=True)
        
            best_ns22_state, best_macro_state, hist = train_fold(fold_k)
            all_hist[fold_k] = hist
        
            mel_tf = MelSpecTransform().to(device)
            val_wavs_k = _load_val_waveforms(val_sc_df_k)
        
            if best_macro_state is not None:
                # Save PyTorch checkpoint
                torch.save(best_macro_state, OUT_DIR / f"fold{fold_k}_best_ns22.pt")
                m = make_model()
                m.load_state_dict(best_macro_state, strict=False)
                oof_ns22[vm] = _predict_from_waveforms(m, mel_tf, val_wavs_k)["blend"]
        
                # --- ONNX Export (Conv1d remap for stable tracing) ---
                m.eval()
                INF_N_MELS = 128
                INF_N_FRAMES = VAL_SAMPLES // HOP_LENGTH + 1
        
                class SEDExportWrapper(nn.Module):
                    def __init__(self, backbone_name, num_classes, backbone_dim, hidden_dim=512):
                        super().__init__()
                        self.backbone = timm.create_model(
                            backbone_name, pretrained=False, in_chans=1,
                            num_classes=0, global_pool="", drop_path_rate=0.1,
                        )
                        self.gem_freq = GeMFreqPool(p_init=3.0)
                        self.dense_drop1 = nn.Dropout(0.25)
                        self.dense_conv = nn.Conv1d(backbone_dim, hidden_dim, 1)
                        self.dense_relu = nn.ReLU(inplace=True)
                        self.dense_drop2 = nn.Dropout(0.5)
                        self.att = nn.Conv1d(hidden_dim, num_classes, 1)
                        self.cla = nn.Conv1d(hidden_dim, num_classes, 1)
        
                    def forward(self, mel):
                        h = self.backbone(mel)
                        h = self.gem_freq(h)
                        h = self.dense_drop1(h)
                        h = self.dense_conv(h)
                        h = self.dense_relu(h)
                        h = self.dense_drop2(h)
                        norm_att = torch.softmax(torch.tanh(self.att(h)), dim=-1)
                        framewise = self.cla(h)
                        clip = torch.sum(norm_att * framewise, dim=2)
                        return clip, framewise.permute(0, 2, 1)
        
                def load_and_remap_state(export_model, trained_state):
                    remap = {}
                    for k, v in trained_state.items():
                        if k.startswith("distill_head."):
                            continue
                        if k == "dense.1.weight":
                            remap["dense_conv.weight"] = v.unsqueeze(-1)
                        elif k == "dense.1.bias":
                            remap["dense_conv.bias"] = v
                        else:
                            remap[k] = v
                    export_model.load_state_dict(remap, strict=False)
        
                export_model = SEDExportWrapper(
                    BACKBONE_NAME, NUM_CLASSES, m.backbone_dim
                ).to(device)
                load_and_remap_state(export_model, best_macro_state)
                export_model.eval()
        
                dummy_mel = torch.randn(1, 1, INF_N_MELS, INF_N_FRAMES).to(device)
                onnx_path = OUT_DIR / f"sed_distill_fold{fold_k}.onnx"
                torch.onnx.export(
                    export_model, dummy_mel, str(onnx_path),
                    input_names=["mel"],
                    output_names=["clip_logits", "framewise_logits"],
                    dynamic_axes={"mel": {0: "batch"},
                                  "clip_logits": {0: "batch"},
                                  "framewise_logits": {0: "batch"}},
                    opset_version=17,
                )
        
                # Verify
                _sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
                _onnx_out = _sess.run(None, {'mel': dummy_mel.cpu().numpy()})
                with torch.no_grad():
                    _ref_clip, _ref_frame = export_model(dummy_mel)
                _diff = np.abs(_ref_clip.cpu().numpy() - _onnx_out[0]).max()
                print(f"  ONNX verify: max|diff|={_diff:.3e}")
                assert _diff < 1e-3, f"ONNX export diverged: {_diff}"
                del _sess
        
                size_mb = onnx_path.stat().st_size / 1e6
                print(f"  Exported {onnx_path.name} ({size_mb:.1f} MB)")
                del m, export_model
    
    # =================================================================
    # S7 -- EVALUATION SUMMARY
    # =================================================================
    
    if MODE != "train":
        print("Skipping evaluation (MODE='infer')")
    else:
    
        has = ~np.isnan(oof_ns22[:, 0])
        if has.sum() > 0:
            r_all = full_eval(Y_SC[has], oof_ns22[has], non_s22_mask_sc[has], TAXON_MASKS)
            print("=" * 60)
            print("OOF RESULTS (best-ns22 checkpoints)")
            print("=" * 60)
            print(f"  macro AUC (all):        {r_all['macro_auc_all']:.4f}")
            print(f"  macro AUC (non-S22):    {r_all['non_s22_macro']:.4f}")
            for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]:
                print(f"    {t:<12}: {r_all.get(f'non_s22_{t}', float('nan')):.4f}")
        
        # Per-epoch progression
        print("\nPer-epoch pooled non-S22 AUC:")
        fold_true, fold_ns22_m = {}, {}
        for fk in range(N_FOLDS):
            vm = sc_cache_meta["fold"].values == fk
            fold_true[fk] = Y_SC[vm]
            fold_ns22_m[fk] = non_s22_mask_sc[vm]
        
        n_eps = [len(all_hist[k]["val_preds"]) for k in range(N_FOLDS) if k in all_hist]
        max_ep = min(n_eps) if n_eps else 0
        for ep in range(max_ep):
            pp = np.concatenate([all_hist[k]["val_preds"][ep] for k in range(N_FOLDS) if k in all_hist])
            pt = np.concatenate([fold_true[k] for k in range(N_FOLDS) if k in all_hist])
            pm = np.concatenate([fold_ns22_m[k] for k in range(N_FOLDS) if k in all_hist])
            ns, _ = compute_macro_auc(pt, pp, mask=pm)
            print(f"  Ep{ep:02d}: {ns:.4f}")
    
    
    # =================================================================
    # INFERENCE -- Mel spectrogram (librosa)
    # =================================================================
    if MODE != "infer":
        print("Skipping inference setup (MODE='train')")
    else:
        import librosa
    
        INF_N_MELS   = 256
        INF_N_FFT    = 2048
        INF_HOP      = 512
        INF_FMIN     = 20
        INF_FMAX     = 16000
        INF_TOP_DB   = 80
        INF_SR       = 32000
        INF_CHUNK_S  = 5
        INF_CHUNK_N  = INF_SR * INF_CHUNK_S   # 160,000
        INF_N_FRAMES = INF_CHUNK_N // INF_HOP + 1  # 313
        
        def audio_to_mel(chunks):
            """Raw audio chunks (N, 160000) -> normalized mel dB (N, 1, 128, 313)."""
            mels = []
            for i in range(chunks.shape[0]):
                S = librosa.feature.melspectrogram(
                    y=chunks[i], sr=INF_SR, n_fft=INF_N_FFT, hop_length=INF_HOP,
                    n_mels=INF_N_MELS, fmin=INF_FMIN, fmax=INF_FMAX, power=2.0,
                )
                S_dB = librosa.power_to_db(S, top_db=INF_TOP_DB)
                S_dB = (S_dB - S_dB.mean()) / (S_dB.std() + 1e-6)
                mels.append(S_dB)
            return np.stack(mels)[:, np.newaxis, :, :].astype(np.float32)
        
        print(f"Inference mel: {INF_N_MELS} mels, {INF_N_FRAMES} frames/chunk")
    
    
    # =================================================================
    # INFERENCE -- Load ONNX sessions
    # =================================================================
    if MODE != "infer":
        print("Skipping ONNX loading (MODE='train')")
    else:
        import re, glob
    
        def discover_folds(sed_dir):
            pat = re.compile(r'sed_fold(\d+)\.onnx$')
            folds = []
            for fname in os.listdir(sed_dir):
                m = pat.match(fname)
                if m: folds.append(int(m.group(1)))
            return sorted(folds)
        
        def make_session(onnx_path):
            so = ort.SessionOptions()
            so.intra_op_num_threads = 4
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        
        SED_DIR_CANDIDATES = [
            str(OUT_DIR),
            '/kaggle/input/datasets/tuckerarrants/bc2026-distilled-sed-public',
        ]
        SED_DIR = next((p for p in SED_DIR_CANDIDATES if os.path.isdir(p) and
                        any(f.endswith('.onnx') and 'sed' in f for f in os.listdir(p))), None)
        assert SED_DIR, f'No SED ONNX files found in {SED_DIR_CANDIDATES}'
        
        INF_FOLDS = discover_folds(SED_DIR)
        assert INF_FOLDS, f'No sed_distill_fold*.onnx in {SED_DIR}'
        print(f'Found {len(INF_FOLDS)} fold(s) in {SED_DIR}: {INF_FOLDS}')
        
        fold_sessions = []
        for fold in INF_FOLDS:
            p = f'{SED_DIR}/sed_fold{fold}.onnx'
            sess = make_session(p)
            fold_sessions.append(sess)
            size_mb = os.path.getsize(p) / 1e6
            print(f'  fold {fold}: {size_mb:5.1f}MB  providers={sess.get_providers()}')
    
    
    # =================================================================
    # INFERENCE -- Audio loading + main loop
    # =================================================================
    from scipy.ndimage import convolve1d
    
    GAUSSIAN_KERNEL = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    N_WINDOWS = 12  # 60s / 5s
    
    if MODE != "infer":
        print("Skipping inference (MODE='train')")
    else:
        try:
            import soundfile as sf
            DECODER = 'soundfile'
        except ImportError:
            DECODER = 'librosa'
        print(f'Audio decoder: {DECODER}')
    
        def load_audio_32k_mono(path):
            if DECODER == 'soundfile':
                wav, sr = sf.read(path, dtype='float32', always_2d=False)
                if wav.ndim > 1: wav = wav.mean(axis=1)
                if sr != INF_SR:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=INF_SR)
                return wav.astype(np.float32)
            else:
                wav, _ = librosa.load(path, sr=INF_SR, mono=True)
                return wav.astype(np.float32)
    
        def file_to_chunks(path):
            wav = load_audio_32k_mono(path)
            target_len = 60 * INF_SR
            if len(wav) < target_len:
                wav = np.pad(wav, (0, target_len - len(wav)))
            elif len(wav) > target_len:
                wav = wav[:target_len]
            n_chunks = target_len // INF_CHUNK_N
            chunks = wav[:n_chunks * INF_CHUNK_N].reshape(n_chunks, INF_CHUNK_N)
            end_times = np.arange(1, n_chunks + 1) * INF_CHUNK_S
            return chunks.astype(np.float32), end_times
    
        def sigmoid_inf(x):
            return np.where(
                x >= 0,
                1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))),
                np.exp(np.clip(x, -50, 50)) / (1.0 + np.exp(np.clip(x, -50, 50))),
            ).astype(np.float32)
    
        def gauss_smooth_final(scores, weights=GAUSSIAN_KERNEL):
            """Gaussian smooth predictions across the 12 windows within each file.
            Operates on the leading axis of the per-file slab."""
            smoothed = scores.reshape(-1, N_WINDOWS, scores.shape[1]).copy()
            for i in range(smoothed.shape[0]):
                smoothed[i] = convolve1d(smoothed[i], weights, axis=0, mode='nearest')
            return smoothed.reshape(-1, scores.shape[1])
    
        # Discover test files (with train_soundscapes fallback for debugging)
        test_files = sorted(glob.glob(f'{TEST_DIR}/*.ogg')) if TEST_DIR.is_dir() else []
        if len(test_files) == 0:
            fallback = COMP_DIR / "train_soundscapes"
            if fallback.is_dir():
                test_files = sorted(glob.glob(f'{fallback}/*.ogg'))[:5]
                print(f'No test_soundscapes -- using {len(test_files)} train files for debug')
        print(f'Test files: {len(test_files)}')
    
        # --- Main inference loop ---
        t0 = time.time()
        all_rows, all_preds = [], []
    
        for file_idx, file_path in enumerate(test_files):
            basename = os.path.basename(file_path).replace('.ogg', '')
            chunks, end_times = file_to_chunks(file_path)
            mel = audio_to_mel(chunks)
    
            # Accumulate logits in logit space across folds (matches the
            # smoothing-in-logit-space convention from the other inference notebook).
            # Per fold: blend clip + frame_max in LOGIT space.
            logits_sum = np.zeros((chunks.shape[0], NUM_CLASSES), dtype=np.float32)
            for sess in fold_sessions:
                outs = sess.run(None, {'mel': mel})
                clip_logits = outs[0]
                frame_max = outs[1].max(axis=1)
                logits_sum += 0.5 * clip_logits + 0.5 * frame_max
            logits_mean = logits_sum / len(INF_FOLDS)
    
            # Smooth across windows in logit space, then sigmoid once.
            logits_smoothed = gauss_smooth_final(logits_mean)
            probs = sigmoid_inf(logits_smoothed)
    
            all_rows.extend([f'{basename}_{int(t)}' for t in end_times])
            all_preds.append(probs)
    
            if (file_idx + 1) % 50 == 0 or file_idx == 0 or file_idx == len(test_files) - 1:
                elapsed = time.time() - t0
                rate = (file_idx + 1) / elapsed
                print(f'  [{file_idx+1:4d}/{len(test_files)}] {elapsed:.1f}s  {rate:.2f} files/s')
    
        all_preds_arr = np.concatenate(all_preds) if all_preds else np.zeros((0, NUM_CLASSES), np.float32)
        print(f'\nInference: {len(all_rows)} rows, {time.time()-t0:.1f}s total')
    
    # =================================================================
    # WRITE SUBMISSION
    # =================================================================
    if MODE != "infer":
        print("Skipping submission (MODE='train')")
    else:
        submission = pd.DataFrame(all_preds_arr, columns=PRIMARY_LABELS)
        submission.insert(0, 'row_id', all_rows)
        
        assert submission.shape[1] == NUM_CLASSES + 1
        assert submission['row_id'].is_unique
        assert not submission.iloc[:, 1:].isna().any().any()
        submission.iloc[:, 1:] = submission.iloc[:, 1:].clip(0.0, 1.0)
        
        submission.to_csv(_file_name_submission, index=False)
        print(f'Wrote submission.csv: {len(submission)} rows x {submission.shape[1]} cols')
        print(submission.head(3).iloc[:, :6])
    
    # =================================================================
    # PIPELINE 2: PROTOSSM V5 WITH META-LEARNER AND HARD NEGATIVE SUPPRESSION
    # =================================================================
    if 'Model_3' in _ensemble_models:
        
        _file_name_submission = "subm_3.csv"
        
        # Install required packages
        !pip install -q timm torchaudio onnxscript onnx lightgbm scikit-learn
        
        import os, sys, json, pickle, gc, random, math, re, glob
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from collections import defaultdict
        
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader, ConcatDataset
        from torch.cuda.amp import GradScaler, autocast
        import torchaudio
        import timm
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from scipy.special import expit as sigmoid_np
        import warnings
        warnings.filterwarnings("ignore")
        
        SEED = 42
        random.seed(SEED)
        os.environ["PYTHONHASHSEED"] = str(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name()}")
        
        MODE = "infer"  # "train" or "infer"
        
        # =================================================================
        # CONFIGURATION
        # =================================================================
        COMP_DIR = Path("/kaggle/input/competitions/birdclef-2026")
        WAVEFORM_CACHE_DIR = Path("/kaggle/input/datasets/tuckerarrants/birdclef-2026-waveform-cache/waveform_cache")
        PERCH_ONNX_PATH = Path("/kaggle/input/datasets/tuckerarrants/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx")
        
        LABELS_PATH     = COMP_DIR / "train_soundscapes_labels.csv"
        TAXONOMY_PATH   = COMP_DIR / "taxonomy.csv"
        SAMPLE_SUB_PATH = COMP_DIR / "sample_submission.csv"
        TEST_DIR        = COMP_DIR / "test_soundscapes"
        
        OUT_DIR = Path("/kaggle/working")
        
        NUM_CLASSES = 234
        SR = 32000
        
        # --- Duration ---
        TRAIN_DURATION = 5    # seconds
        VAL_DURATION   = 5    # always 5s for competition eval
        TRAIN_SAMPLES  = SR * TRAIN_DURATION
        VAL_SAMPLES    = SR * VAL_DURATION
        
        N_FOLDS = 5
        
        # --- Mel spectrogram ---
        N_FFT      = 2048
        HOP_LENGTH = 512
        N_MELS     = 256
        FMIN       = 20
        FMAX       = 16000
        
        # --- Model ---
        BACKBONE_NAME = "tf_efficientnet_b0.ns_jft_in1k"  # Original SED backbone
        
        # --- Perch distillation ---
        USE_PERCH_DISTILL = True
        PERCH_EMBED_DIM   = 1536
        ALPHA_DISTILL     = 1.0   # MSE loss weight
        
        # --- Training ---
        FOLDS  = [0, 1, 2, 3, 4]
        EPOCHS = 25
        BATCH  = 16 if MODE == "train" else 64
        LR     = 5e-4
        MIN_LR = 1e-5
        WD     = 1e-4
        WARMUP_EPOCHS = 2
        
        # --- Upsampling ---
        MIN_SAMPLE = 20
        
        # --- Augmentation ---
        AUG_PROB = 0.5
        AUG_GAIN_DB_RANGE      = (-6.0, 6.0)
        AUG_NOISE_SNR_DB_RANGE = (10.0, 30.0)
        
        # --- MixUp ---
        USE_FOCAL_MIXUP    = True
        MIXUP_PROB         = 0.5
        MIXUP_ALPHA        = 0.4
        MIXUP_HARD         = True
        
        USE_FOCAL_SC_MIXUP     = True
        FOCAL_SC_MIXUP_PROB    = 0.5
        FOCAL_SC_MIXUP_ALPHA   = 0.4
        
        # --- FreqMixStyle (disabled by default) ---
        FREQ_MIXSTYLE_PROB  = 0.0
        FREQ_MIXSTYLE_ALPHA = 0.1
        
        # --- SpecAugment ---
        FREQ_MASK_PARAM = 10
        TIME_MASK_PARAM = 10
        NUM_FREQ_MASKS  = 1
        NUM_TIME_MASKS  = 2
        
        # --- Source weights ---
        USE_FOCAL           = True
        USE_FOCAL_SECONDARY = True
        USE_LABELED_SC      = True
        
        ACTIVE_SOURCES = ["focal", "sc"]
        SHARES = {"focal": 0.9, "sc": 0.1}
        SOURCE_WEIGHTS = {
            "focal":         1.0,
            "focal_missing": 0.0,
            "sc":            1.0,
        }
        
        print(f"Backbone: {BACKBONE_NAME}")
        print(f"Train duration: {TRAIN_DURATION}s | Mel: {N_MELS} mels, n_fft={N_FFT}, hop={HOP_LENGTH}")
        print(f"Distillation: {'ON' if USE_PERCH_DISTILL else 'OFF'} (alpha={ALPHA_DISTILL})")
        print(f"Batch: {BATCH} | Epochs: {EPOCHS} | Folds: {FOLDS}")
    
    # =================================================================
    # S2 -- LOAD DATA
    # =================================================================
    
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    PRIMARY_LABELS = sample_sub.columns[1:].tolist()
    LABEL2IDX = {label: idx for idx, label in enumerate(PRIMARY_LABELS)}
    taxonomy = pd.read_csv(TAXONOMY_PATH)
    label_to_taxon = dict(zip(taxonomy["primary_label"].astype(str),
                              taxonomy["class_name"].astype(str)))
    TAXON_MASKS = {t: np.array([i for i, l in enumerate(PRIMARY_LABELS)
                                if label_to_taxon.get(l, "") == t])
                   for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]}
    
    audio_cache_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "audio_cache_meta.csv")
    train_df = pd.read_csv(COMP_DIR / "train.csv")
    audio_cache_meta = audio_cache_meta.merge(
        train_df[["filename", "secondary_labels"]], on="filename", how="left"
    )
    audio_cache_meta = audio_cache_meta[
        audio_cache_meta["primary_label"].isin(LABEL2IDX)
    ].reset_index(drop=True)
    print(f"Focal audio cache: {len(audio_cache_meta)} entries")
    
    sc_cache_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "soundscape_cache_meta.csv")
    sc_cache_meta["label_list"] = sc_cache_meta["label_list"].apply(
        lambda x: x.split(";") if isinstance(x, str) else []
    )
    print(f"Soundscape cache: {len(sc_cache_meta)} windows")
    
    sc_labels_raw = pd.read_csv(LABELS_PATH).drop_duplicates()
    sc_labels_raw["start_sec"] = pd.to_timedelta(sc_labels_raw["start"]).dt.total_seconds().astype(int)
    
    Y_SC = np.zeros((len(sc_cache_meta), NUM_CLASSES), dtype=np.float32)
    for i, row in sc_cache_meta.iterrows():
        matches = sc_labels_raw[
            (sc_labels_raw["filename"] == row["filename"]) &
            (sc_labels_raw["start_sec"] == row["start_sec"])
        ]
        for _, m in matches.iterrows():
            for lbl in str(m["primary_label"]).split(";"):
                lbl = lbl.strip()
                if lbl in LABEL2IDX:
                    Y_SC[i, LABEL2IDX[lbl]] = 1.0
    
    labeled_sc_mask = Y_SC.sum(axis=1) > 0
    print(f"Soundscape labels: {labeled_sc_mask.sum()}/{len(Y_SC)} windows labeled, "
          f"{int(Y_SC.sum())} positives, {int((Y_SC.sum(axis=0) > 0).sum())} species")
    
    audio_for_split = audio_cache_meta.drop_duplicates("original_idx").reset_index(drop=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    audio_for_split["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(audio_for_split, audio_for_split["primary_label"])):
        audio_for_split.loc[val_idx, "fold"] = fold
    audio_cache_meta = audio_cache_meta.merge(
        audio_for_split[["original_idx", "fold"]], on="original_idx", how="left"
    )
    print(f"\nFocal fold distribution:\n{audio_cache_meta['fold'].value_counts().sort_index()}")
    
    sc_files = sc_cache_meta[["filename", "site"]].drop_duplicates().reset_index(drop=True)
    gkf = GroupKFold(n_splits=N_FOLDS)
    sc_files["fold"] = -1
    for fold, (_, val_idx) in enumerate(gkf.split(sc_files, groups=sc_files["filename"])):
        sc_files.loc[sc_files.index[val_idx], "fold"] = fold
    
    file_to_fold = dict(zip(sc_files["filename"], sc_files["fold"]))
    sc_cache_meta["fold"] = sc_cache_meta["filename"].map(file_to_fold).fillna(-1).astype(int)
    print(f"\nSoundscape fold distribution:")
    print(sc_cache_meta["fold"].value_counts().sort_index())
    
    counts = audio_cache_meta["primary_label"].value_counts()
    rare_species = counts[counts < MIN_SAMPLE].index
    extra_rows = []
    for sp in rare_species:
        sp_rows = audio_cache_meta[audio_cache_meta["primary_label"] == sp]
        n_copies = int(np.ceil(MIN_SAMPLE / len(sp_rows))) - 1
        for _ in range(n_copies):
            extra_rows.append(sp_rows)
    
    n_before = len(audio_cache_meta)
    if extra_rows:
        audio_cache_meta = pd.concat([audio_cache_meta] + extra_rows, ignore_index=True)
    print(f"\nUpsampled {len(rare_species)} rare species (min={MIN_SAMPLE}): "
          f"{n_before} -> {len(audio_cache_meta)} samples")
    
    sc_sites = sc_cache_meta["site"].values
    non_s22_mask_sc = sc_sites != "S22"
    print(f"S22: {(~non_s22_mask_sc).sum()}, non-S22: {non_s22_mask_sc.sum()}")
    print("OK Data loaded")
    
    # =================================================================
    # S3 -- EVAL UTILITIES + MEL TRANSFORM + PROTOSSM MODEL
    # =================================================================
    
    def compute_macro_auc(y_true, y_pred, mask=None, class_mask=None):
        if mask is not None:
            y_true, y_pred = y_true[mask], y_pred[mask]
        if class_mask is not None:
            y_true, y_pred = y_true[:, class_mask], y_pred[:, class_mask]
        aucs = []
        for c in range(y_true.shape[1]):
            col = y_true[:, c]
            if col.sum() == 0 or col.sum() == len(col):
                continue
            try:
                aucs.append(roc_auc_score(col, y_pred[:, c]))
            except ValueError:
                continue
        return (np.mean(aucs) if aucs else float("nan")), len(aucs)
    
    def full_eval(y_true, y_pred, ns22, tm):
        r = {}
        a, n = compute_macro_auc(y_true, y_pred)
        r["macro_auc_all"], r["n_all"] = round(a, 4), n
        a, n = compute_macro_auc(y_true, y_pred, mask=ns22)
        r["non_s22_macro"], r["n_ns22"] = round(a, 4), n
        for t, cm in tm.items():
            a, n = compute_macro_auc(y_true, y_pred, mask=ns22, class_mask=cm)
            r[f"non_s22_{t}"] = round(a, 4)
        return r
    
    # ------------------------------------------------------------------
    # GPU Mel Spectrogram
    # ------------------------------------------------------------------
    class MelSpecTransform(nn.Module):
        def __init__(self):
            super().__init__()
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
            )
            self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
    
        def forward(self, waveform):
            return self.db_transform(self.mel_spec(waveform))
    
    # ------------------------------------------------------------------
    # GPU SpecAugment
    # ------------------------------------------------------------------
    class SpecAugment(nn.Module):
        def __init__(self):
            super().__init__()
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=FREQ_MASK_PARAM)
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=TIME_MASK_PARAM)
    
        def forward(self, mel):
            for _ in range(NUM_FREQ_MASKS):
                mel = self.freq_mask(mel)
            for _ in range(NUM_TIME_MASKS):
                mel = self.time_mask(mel)
            return mel
    
    # ------------------------------------------------------------------
    # Frozen Perch teacher -- ONNX inference, no gradients
    # ------------------------------------------------------------------
    import onnxruntime as ort
    
    class PerchTeacher:
        """Frozen Perch v2 via ONNX. Takes 5s waveforms, returns 1536-d embeddings.
        The teacher is never updated -- it provides a stable distillation target."""
    
        def __init__(self, onnx_path, device_str="cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
                if device_str == "cuda" else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(onnx_path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self._out_names = [o.name for o in self.session.get_outputs()]
            self._embed_idx = None
            for i, o in enumerate(self.session.get_outputs()):
                if o.shape and o.shape[-1] == PERCH_EMBED_DIM:
                    self._embed_idx = i
                    break
            if self._embed_idx is None:
                self._embed_idx = 1
            print(f"Perch ONNX loaded: embed_idx={self._embed_idx}")
    
        @torch.no_grad()
        def embed(self, waveforms_5s):
            """waveforms_5s: (B, 160000) float32, returns (B, 1536) embeddings."""
            wav_np = waveforms_5s.cpu().numpy()
            results = self.session.run(None, {self.input_name: wav_np})
            return torch.from_numpy(results[self._embed_idx]).float()
    
    # ------------------------------------------------------------------
    # Distillation head: GAP + Linear to Perch embedding space
    # ------------------------------------------------------------------
    class DistillHead(nn.Module):
        """Projects backbone features to Perch's 1536-d space via GAP + Linear."""
        def __init__(self, backbone_dim, embed_dim=1536):
            super().__init__()
            self.proj = nn.Linear(backbone_dim, embed_dim)
    
        def forward(self, feature_map):
            gap = feature_map.mean(dim=[2, 3])   # (B, C, F, T) -> (B, C)
            return self.proj(gap)                 # (B, embed_dim)
    
    # ------------------------------------------------------------------
    # ProtoSSMv5 Model
    # ------------------------------------------------------------------
    class ProtoSSMv5(nn.Module):
        """State Space Model with temporal cross-attention and prototypical classification.
        - 4-layer bidirectional Mamba-style SSM
        - 8-head temporal cross-attention
        - Perch embedding distillation
        - Residual correction network
        """
        def __init__(self, num_classes=NUM_CLASSES, perch_dim=PERCH_EMBED_DIM, device='cpu'):
            super().__init__()
            self.device = device
            self.perch_dim = perch_dim
            
            # Learnable class prototypes (234, perch_dim)
            self.prototypes = nn.Parameter(torch.randn(num_classes, perch_dim).to(device) / 10.0)
            
            # Metadata projection (site_id + hour_utc -> 24-dim)
            self.meta_proj = nn.Sequential(
                nn.Linear(6 + 24, 64),  # 6 for site_id one-hot, 24 for hour embedding
                nn.ReLU(),
                nn.Linear(64, 24)
            )
            
            # Selective State Space Model (Mamba-style)
            self.ssm = nn.ModuleList([
                nn.TransformerXLEncoderLayer(d_model=320, nhead=4, dim_feedforward=512, dropout=0.1, activation='relu'),
                nn.TransformerXLEncoderLayer(d_model=320, nhead=4, dim_feedforward=512, dropout=0.1, activation='relu'),
                nn.TransformerXLEncoderLayer(d_model=320, nhead=4, dim_feedforward=512, dropout=0.1, activation='relu'),
                nn.TransformerXLEncoderLayer(d_model=320, nhead=4, dim_feedforward=512, dropout=0.1, activation='relu'),
            ])
            
            # Temporal Cross-Attention
            self.tca = nn.MultiheadAttention(embed_dim=320, num_heads=8, batch_first=True)
            
            # Residual Correction Network
            self.residual_corr = nn.Sequential(
                nn.Linear(320, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 320)
            )
            
        def forward(self, perch_embeddings, site_ids, hours):
            """Perch embeddings: (B, 12, 1536), site_ids: (B, 6), hours: (B, 24) -> (B, 12, 320)"""
            B, N_W, D = perch_embeddings.shape
            
            # Project metadata and add to embeddings
            meta_emb = self.meta_proj(torch.cat([site_ids, hours], dim=1)).unsqueeze(1).repeat(1, N_W, 1)  # (B, 1, 24) -> (B, N_W, 24)
            x = torch.cat([perch_embeddings, meta_emb], dim=-1)  # (B, N_W, 1560)
            
            # Selective SSM
            for layer in self.ssm:
                x = layer(x)
            
            # Temporal Cross-Attention
            x, _ = self.tca(x)
            
            # Sequence pooling
            seq_emb = x.mean(dim=1)  # (B, 320)
            
            # Prototypical classification
            logits = F.cosine_similarity(seq_emb.unsqueeze(1), self.prototypes.unsqueeze(0), dim=-1).squeeze(-1)
            
            # Residual correction (optional)
            correction = self.residual_corr(seq_emb)
            logits = logits + correction @ self.prototypes.T  # Apply correction in logit space
            
            return logits, x
    
    # ------------------------------------------------------------------
    # MLP Probes (Ridge Regression) for all 234 classes
    # ------------------------------------------------------------------
    class RidgeProbes:
        """Ridge regression probes on Perch embeddings for all 234 classes.
        Uses closed-form solution for fast training and inference.
        """
        def __init__(self, perch_dim=1536):
            self.perch_dim = perch_dim
            self.weights = []  # List of 234 weight vectors (1D)
            self.biases = []
            
        def train(self, X, y):
            """X: (N, perch_dim), y: (N,) binary labels for one class."""
            X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias term
            XtX_inv = np.linalg.pinv(X.T @ X)
            w = XtX_inv @ X.T @ y
            return w[:-1], w[-1]  # weights, bias
            
        def train_all(self, X, Y):
            """X: (N, perch_dim), Y: (N, C) binary matrix."""
            N, C = Y.shape
            X_with_bias = np.hstack([X, np.ones((N, 1))])
            XtX_inv = np.linalg.pinv(X.T @ X)
            for c in range(C):
                w = XtX_inv @ X.T @ Y[:, c]
                self.weights.append(w[:-1])
                self.biases.append(w[-1])
            
        def predict(self, X):
            X = np.hstack([X, np.ones((X.shape[0], 1))])
            preds = np.zeros((X.shape[0], len(self.weights)))
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                preds[:, i] = X @ w + b
            return preds
    
    # ------------------------------------------------------------------
    # Hard Negative Suppression using Metadata
    # ------------------------------------------------------------------
    class HardNegativeMask:
        """Apply biological constraints to suppress impossible predictions.
        Uses site-specific and time-specific species occurrence data.
        """
        def __init__(self, taxonomy_path):
            self.taxa = pd.read_csv(taxonomy_path)
            self.diurnal_species = self._get_diurnal_species()
            
        def _get_diurnal_species(self):
            """Get species that are strictly diurnal from taxonomy."""
            # For simplicity, we'll use a heuristic: birds active during day (6:00-18:00)
            # In practice, this would come from detailed ornithology data
            diurnal = set()
            # Example: diurnal birds of prey, songbirds
            # This is a placeholder - in production, load actual species-specific activity patterns
            return diurnal
        
        def apply_mask(self, probs, recording_time, site_id):
            """Set probabilities to 0 for species that cannot be present at this time.
            probs: (N, 234), recording_time: datetime, site_id: string
            """
            hours = pd.to_datetime(recording_time).hour
            mask = np.ones(probs.shape[0], dtype=bool)
            
            # Apply diurnal constraints
            for idx, label in enumerate(PRIMARY_LABELS):
                if label in self.diurnal_species:
                    start, end = 6, 18  # 6 AM - 6 PM
                    mask[:, idx] = (hours >= start) & (hours < end)
            
            # Apply site-specific constraints (e.g., species not found in certain regions)
            # This would use range maps or habitat preferences
            
            probs = np.where(mask, probs, 0.0)
            return probs
    
    # =================================================================
    # S2 -- LOAD DATA
    # =================================================================
    
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    PRIMARY_LABELS = sample_sub.columns[1:].tolist()
    LABEL2IDX = {label: idx for idx, label in enumerate(PRIMARY_LABELS)}
    taxonomy = pd.read_csv(TAXONOMY_PATH)
    label_to_taxon = dict(zip(taxonomy["primary_label"].astype(str),
                              taxonomy["class_name"].astype(str)))
    TAXON_MASKS = {t: np.array([i for i, l in enumerate(PRIMARY_LABELS)
                                if label_to_taxon.get(l, "") == t])
                   for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]}
    
    audio_cache_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "audio_cache_meta.csv")
    train_df = pd.read_csv(COMP_DIR / "train.csv")
    audio_cache_meta = audio_cache_meta.merge(
        train_df[["filename", "secondary_labels"]], on="filename", how="left"
    )
    audio_cache_meta = audio_cache_meta[
        audio_cache_meta["primary_label"].isin(LABEL2IDX)
    ].reset_index(drop=True)
    print(f"Focal audio cache: {len(audio_cache_meta)} entries")
    
    sc_cache_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "soundscape_cache_meta.csv")
    sc_cache_meta["label_list"] = sc_cache_meta["label_list"].apply(
        lambda x: x.split(";") if isinstance(x, str) else []
    )
    print(f"Soundscape cache: {len(sc_cache_meta)} windows")
    
    sc_labels_raw = pd.read_csv(LABELS_PATH).drop_duplicates()
    sc_labels_raw["start_sec"] = pd.to_timedelta(sc_labels_raw["start"]).dt.total_seconds().astype(int)
    
    Y_SC = np.zeros((len(sc_cache_meta), NUM_CLASSES), dtype=np.float32)
    for i, row in sc_cache_meta.iterrows():
        matches = sc_labels_raw[
            (sc_labels_raw["filename"] == row["filename"]) &
            (sc_labels_raw["start_sec"] == row["start_sec"])
        ]
        for _, m in matches.iterrows():
            for lbl in str(m["primary_label"]).split(";"):
                lbl = lbl.strip()
                if lbl in LABEL2IDX:
                    Y_SC[i, LABEL2IDX[lbl]] = 1.0
    
    labeled_sc_mask = Y_SC.sum(axis=1) > 0
    print(f"Soundscape labels: {labeled_sc_mask.sum()}/{len(Y_SC)} windows labeled, "
          f"{int(Y_SC.sum())} positives, {int((Y_SC.sum(axis=0) > 0).sum())} species")
    
    audio_for_split = audio_cache_meta.drop_duplicates("original_idx").reset_index(drop=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    audio_for_split["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(audio_for_split, audio_for_split["primary_label"])):
        audio_for_split.loc[val_idx, "fold"] = fold
    audio_cache_meta = audio_cache_meta.merge(
        audio_for_split[["original_idx", "fold"]], on="original_idx", how="left"
    )
    print(f"\nFocal fold distribution:\n{audio_cache_meta['fold'].value_counts().sort_index()}")
    
    sc_files = sc_cache_meta[["filename", "site"]].drop_duplicates().reset_index(drop=True)
    gkf = GroupKFold(n_splits=N_FOLDS)
    sc_files["fold"] = -1
    for fold, (_, val_idx) in enumerate(gkf.split(sc_files, groups=sc_files["filename"])):
        sc_files.loc[sc_files.index[val_idx], "fold"] = fold
    
    file_to_fold = dict(zip(sc_files["filename"], sc_files["fold"]))
    sc_cache_meta["fold"] = sc_cache_meta["filename"].map(file_to_fold).fillna(-1).astype(int)
    print(f"\nSoundscape fold distribution:")
    print(sc_cache_meta["fold"].value_counts().sort_index())
    
    counts = audio_cache_meta["primary_label"].value_counts()
    rare_species = counts[counts < MIN_SAMPLE].index
    extra_rows = []
    for sp in rare_species:
        sp_rows = audio_cache_meta[audio_cache_meta["primary_label"] == sp]
        n_copies = int(np.ceil(MIN_SAMPLE / len(sp_rows))) - 1
        for _ in range(n_copies):
            extra_rows.append(sp_rows)
    
    n_before = len(audio_cache_meta)
    if extra_rows:
        audio_cache_meta = pd.concat([audio_cache_meta] + extra_rows, ignore_index=True)
    print(f"\nUpsampled {len(rare_species)} rare species (min={MIN_SAMPLE}): "
          f"{n_before} -> {len(audio_cache_meta)} samples")
    
    sc_sites = sc_cache_meta["site"].values
    non_s22_mask_sc = sc_sites != "S22"
    print(f"S22: {(~non_s22_mask_sc).sum()}, non-S22: {non_s22_mask_sc.sum()}")
    print("OK Data loaded")
    
    # =================================================================
    # S3 -- EVAL UTILITIES + MEL TRANSFORM + PROTOSSM MODEL
    # =================================================================
    
    def compute_macro_auc(y_true, y_pred, mask=None, class_mask=None):
        if mask is not None:
            y_true, y_pred = y_true[mask], y_pred[mask]
        if class_mask is not None:
            y_true, y_pred = y_true[:, class_mask], y_pred[:, class_mask]
        aucs = []
        for c in range(y_true.shape[1]):
            col = y_true[:, c]
            if col.sum() == 0 or col.sum() == len(col):
                continue
            try:
                aucs.append(roc_auc_score(col, y_pred[:, c]))
            except ValueError:
                continue
        return (np.mean(aucs) if aucs else float("nan")), len(aucs)
    
    def full_eval(y_true, y_pred, ns22, tm):
        r = {}
        a, n = compute_macro_auc(y_true, y_pred)
        r["macro_auc_all"], r["n_all"] = round(a, 4), n
        a, n = compute_macro_auc(y_true, y_pred, mask=ns22)
        r["non_s22_macro"], r["n_ns22"] = round(a, 4), n
        for t, cm in tm.items():
            a, n = compute_macro_auc(y_true, y_pred, mask=ns22, class_mask=cm)
            r[f"non_s22_{t}"] = round(a, 4)
        return r
    
    # ------------------------------------------------------------------
    # GPU Mel Spectrogram
    # ------------------------------------------------------------------
    class MelSpecTransform(nn.Module):
        def __init__(self):
            super().__init__()
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
            )
            self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
    
        def forward(self, waveform):
            return self.db_transform(self.mel_spec(waveform))
    
    # ------------------------------------------------------------------
    # GPU SpecAugment
    # ------------------------------------------------------------------
    class SpecAugment(nn.Module):
        def __init__(self):
            super().__init__()
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=FREQ_MASK_PARAM)
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=TIME_MASK_PARAM)
    
        def forward(self, mel):
            for _ in range(NUM_FREQ_MASKS):
                mel = self.freq_mask(mel)
            for _ in range(NUM_TIME_MASKS):
                mel = self.time_mask(mel)
            return mel
    
    # ------------------------------------------------------------------
    # Frozen Perch teacher -- ONNX inference, no gradients
    # ------------------------------------------------------------------
    import onnxruntime as ort
    
    class PerchTeacher:
        """Frozen Perch v2 via ONNX. Takes 5s waveforms, returns 1536-d embeddings.
        The teacher is never updated -- it provides a stable distillation target."""
    
        def __init__(self, onnx_path, device_str="cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
                if device_str == "cuda" else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(onnx_path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self._out_names = [o.name for o in self.session.get_outputs()]
            self._embed_idx = None
            for i, o in enumerate(self.session.get_outputs()):
                if o.shape and o.shape[-1] == PERCH_EMBED_DIM:
                    self._embed_idx = i
                    break
            if self._embed_idx is None:
                self._embed_idx = 1
            print(f"Perch ONNX loaded: embed_idx={self._embed_idx}")
    
        @torch.no_grad()
        def embed(self, waveforms_5s):
            """waveforms_5s: (B, 160000) float32, returns (B, 1536) embeddings."""
            wav_np = waveforms_5s.cpu().numpy()
            results = self.session.run(None, {self.input_name: wav_np})
            return torch.from_numpy(results[self._embed_idx]).float()
    
    # =================================================================
    # S4 -- DATA PIPELINE
    # =================================================================
    
    def load_int16(path):
        """Load int16 waveform tensor to float32 in [-1, 1]."""
        waveform_int16 = torch.load(path, map_location="cpu")
        return waveform_int16.float() / 32767.0
    
    _FC = {}
    def load_focal(p):
        """Load focal waveform with simple LRU cache."""
        if p in _FC: return _FC[p]
        pp = WAVEFORM_CACHE_DIR / p
        if not pp.exists(): return None
        a = load_int16(pp).numpy()
        if len(_FC) >= 2000:
            _FC.pop(next(iter(_FC)))
        _FC[p] = a
        return a
    
    _SC_CACHE = {}
    def load_sc_waveform_from(cache_dir, cache_file):
        """Load a soundscape waveform with LRU cache."""
        key = str(cache_dir / cache_file)
        if key in _SC_CACHE: return _SC_CACHE[key]
        pp = cache_dir / cache_file
        if not pp.exists(): return None
        a = load_int16(pp).numpy()
        if len(_SC_CACHE) >= 200:
            _SC_CACHE.pop(next(iter(_SC_CACHE)))
        _SC_CACHE[key] = a
        return a
    
    def extract_chunk_np(waveform, start_sample, n_samples):
        """Extract a chunk, left-padding if the recording is too short."""
        total = len(waveform)
        if total <= n_samples:
            return np.pad(waveform, (n_samples - total, 0))
        end = start_sample + n_samples
        if end > total:
            start_sample = max(0, total - n_samples)
        return waveform[start_sample:start_sample + n_samples]
    
    def apply_aug(w):
        """Simple waveform augmentation: gain jitter + noise + shift."""
        if np.random.random() < AUG_PROB:
            w = w * (10 ** (np.random.uniform(*AUG_GAIN_DB_RANGE) / 20))
        if np.random.random() < AUG_PROB:
            sp = (w ** 2).mean()
            if sp > 1e-10:
                w = w + np.random.randn(*w.shape).astype(w.dtype) * np.sqrt(
                    sp / (10 ** (np.random.uniform(*AUG_NOISE_SNR_DB_RANGE) / 10)))
        return w
    
    # ------------------------------------------------------------------
    # Build soundscape MixUp pool (labeled windows only)
    # ------------------------------------------------------------------
    sc_mixup_sources = []
    _sc_file_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "soundscape_file_meta.csv")
    _sc_file_dict = dict(zip(_sc_file_meta["filename"], _sc_file_meta["cache_file"]))
    _labeled_rows = []
    for i in range(len(sc_cache_meta)):
        row = sc_cache_meta.iloc[i]
        if Y_SC[i].sum() > 0:
            cf = _sc_file_dict.get(row["filename"])
            if cf is not None:
                _labeled_rows.append({
                    "filename": row["filename"], "start_sec": int(row["start_sec"]),
                    "cache_file": cf, "label_idx": i, "fold": int(row.get("fold", -1)),
                })
    if _labeled_rows:
        _labeled_meta = pd.DataFrame(_labeled_rows)
        sc_mixup_sources.append((WAVEFORM_CACHE_DIR, _labeled_meta, Y_SC))
        print(f"SC MixUp pool: {len(_labeled_meta)} labeled windows")
    
    # ------------------------------------------------------------------
    # FocalDS -- with Focal-Focal AND Focal-Soundscape MixUp
    # ------------------------------------------------------------------
    class FocalDS(Dataset):
        """Focal recording dataset. Returns (waveform, label, weight, mask, source_tag)."""
        def __init__(self, df, l2i, secondary_lookup=None,
                     sc_mixup_sources=None, fold_k=None, aug=False):
            self.df, self.l2i, self.aug = df.reset_index(drop=True), l2i, aug
            self.secondary_lookup = secondary_lookup
            self.sc_mixup_sources = sc_mixup_sources
            self.fold_k = fold_k
    
        def __len__(self): return len(self.df)
    
        def _load_chunk(self, r):
            w = load_focal(r["cache_file"])
            if w is None: return None, None
            if self.aug:
                start = np.random.randint(0, max(1, len(w) - TRAIN_SAMPLES + 1)) if len(w) > TRAIN_SAMPLES else 0
            else:
                start = int(r.get("start_sec", 0)) * SR
            ch = extract_chunk_np(w, start, TRAIN_SAMPLES)
            lb = np.zeros(NUM_CLASSES, dtype=np.float32)
            if str(r["primary_label"]) in self.l2i:
                lb[self.l2i[str(r["primary_label"])]] = 1.0
            if self.secondary_lookup is not None and "original_idx" in self.df.columns:
                for s in self.secondary_lookup.get(int(r["original_idx"]), []):
                    if s in self.l2i: lb[self.l2i[s]] = 1.0
            return ch, lb
    
        def __getitem__(self, i):
            r1 = self.df.iloc[i]
            ch1, lb1 = self._load_chunk(r1)
            if ch1 is None:
                return (torch.zeros(1, TRAIN_SAMPLES), torch.zeros(NUM_CLASSES),
                        torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal_missing")
    
            # Focal-Focal MixUp
            if USE_FOCAL_MIXUP and self.aug and np.random.random() < MIXUP_PROB:
                ch2 = None
                for _ in range(3):
                    j = np.random.randint(len(self.df))
                    ch2, lb2 = self._load_chunk(self.df.iloc[j])
                    if ch2 is not None: break
                if ch2 is not None:
                    lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                    ch_mix = (lam * ch1 + (1 - lam) * ch2).astype(np.float32)
                    if self.aug: ch_mix = apply_aug(ch_mix)
                    lb = np.maximum(lb1, lb2) if MIXUP_HARD else (lam * lb1 + (1 - lam) * lb2)
                    return (torch.from_numpy(ch_mix).unsqueeze(0), torch.from_numpy(lb),
                            torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal")
    
            # Focal-Soundscape MixUp
            if (USE_FOCAL_SC_MIXUP and self.aug and self.sc_mixup_sources
                    and np.random.random() < FOCAL_SC_MIXUP_PROB):
                src_idx = np.random.randint(len(self.sc_mixup_sources))
                cache_dir, meta_df_sc, labels = self.sc_mixup_sources[src_idx]
                eligible = meta_df_sc[meta_df_sc["fold"] != self.fold_k] if self.fold_k is not None else meta_df_sc
                if len(eligible) > 0:
                    sc_row = eligible.iloc[np.random.randint(len(eligible))]
                    sc_wav = load_sc_waveform_from(cache_dir, sc_row["cache_file"])
                    if sc_wav is not None and len(sc_wav) >= TRAIN_SAMPLES:
                        sc_chunk = extract_chunk_np(sc_wav, int(sc_row["start_sec"]) * SR, TRAIN_SAMPLES)
                        lam = np.random.beta(FOCAL_SC_MIXUP_ALPHA, FOCAL_SC_MIXUP_ALPHA)
                        ch_mix = (lam * ch1 + (1 - lam) * sc_chunk).astype(np.float32)
                        if self.aug: ch_mix = apply_aug(ch_mix)
                        lb_sc = labels[int(sc_row["label_idx"])].astype(np.float32)
                        lb = np.maximum(lb1, lb_sc) if MIXUP_HARD else lam * lb1 + (1 - lam) * lb_sc
                        return (torch.from_numpy(ch_mix).unsqueeze(0), torch.from_numpy(lb),
                                torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal")
    
            # No MixUp
            if self.aug: ch1 = apply_aug(ch1)
            return (torch.from_numpy(ch1.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(lb1),
                    torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal")
    
    # ------------------------------------------------------------------
    # ScDS -- Labeled soundscape windows
    # ------------------------------------------------------------------
    class ScDS(Dataset):
        def __init__(self, Y, sc_df, aug=False):
            self.Y, self.df, self.aug = Y, sc_df.reset_index(drop=True), aug
        def __len__(self): return len(self.Y)
        def __getitem__(self, i):
            row = self.df.iloc[i]
            wav_full = load_sc_waveform_from(WAVEFORM_CACHE_DIR, row.get("cache_file")) \
                       if row.get("cache_file") else None
            if wav_full is None:
                wav_t = torch.zeros(1, TRAIN_SAMPLES)
            else:
                chunk = extract_chunk_np(wav_full, int(row["start_sec"]) * SR, TRAIN_SAMPLES)
                if self.aug: chunk = apply_aug(chunk)
                wav_t = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0)
            return (wav_t, torch.from_numpy(self.Y[i].astype(np.float32)),
                    torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "sc")
    
    # ------------------------------------------------------------------
    # Load focal secondary labels
    # ------------------------------------------------------------------
    focal_secondary_labels = None
    if USE_FOCAL_SECONDARY:
        focal_secondary_labels = {}
        for idx, row in train_df.iterrows():
            sec = row.get("secondary_labels", "")
            if pd.isna(sec) or sec in ("", "[]"): continue
            try:
                sec_list = eval(sec) if isinstance(sec, str) else []
                valid = [s for s in sec_list if s in LABEL2IDX]
                if valid: focal_secondary_labels[idx] = valid
            except: continue
        print(f"Focal secondary labels: {len(focal_secondary_labels)} files")
    
    # =================================================================
    # S5 -- TRAINING
    # =================================================================
    
    def _load_val_waveforms(val_sc_df):
        """Load validation waveforms (always 5s)."""
        sc_file_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "soundscape_file_meta.csv")
        sc_file_dict = dict(zip(sc_file_meta["filename"], sc_file_meta["cache_file"]))
        wavs = []
        for _, row in val_sc_df.iterrows():
            cf = sc_file_dict.get(row["filename"])
            if cf is not None:
                w = load_sc_waveform_from(WAVEFORM_CACHE_DIR, cf)
                if w is not None:
                    chunk = extract_chunk_np(w, int(row["start_sec"]) * SR, VAL_SAMPLES)
                    wavs.append(torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0))
                else: wavs.append(torch.zeros(1, VAL_SAMPLES))
            else: wavs.append(torch.zeros(1, VAL_SAMPLES))
        return wavs
    
    def _predict_from_waveforms(model, mel_transform, wav_list, batch_size=64):
        """Inference: mel -> model -> sigmoid. Distillation head is NOT used."""
        model.eval()
        preds_clip, preds_fmax, preds_blend = [], [], []
        with torch.no_grad():
            for s in range(0, len(wav_list), batch_size):
                batch = torch.stack(wav_list[s:s+batch_size]).to(device)
                mel = mel_transform(batch)
                B = mel.size(0)
                for i in range(B):
                    mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
                with autocast():
                    clip_logits, framewise = model(mel, return_framewise=True)
                    frame_max = framewise.max(dim=1).values
                    p_clip = torch.sigmoid(clip_logits).cpu().numpy()
                    p_fmax = torch.sigmoid(frame_max).cpu().numpy()
                    p_blend = 0.5 * p_clip + 0.5 * p_fmax
                preds_clip.append(p_clip); preds_fmax.append(p_fmax); preds_blend.append(p_blend)
        return {"clip": np.concatenate(preds_clip), "fmax": np.concatenate(preds_fmax),
                "blend": np.concatenate(preds_blend)}
    
    def build_active_datasets(fold_k):
        items = []
        if USE_FOCAL:
            fds = FocalDS(audio_cache_meta[audio_cache_meta["fold"] != fold_k],
                          LABEL2IDX, secondary_lookup=focal_secondary_labels,
                          sc_mixup_sources=sc_mixup_sources if USE_FOCAL_SC_MIXUP else None,
                          fold_k=fold_k, aug=True)
            items.append(("focal", fds, len(fds)))
        if USE_LABELED_SC:
            vm = sc_cache_meta["fold"].values == fold_k
            Y_tr = Y_SC[~vm]
            sc_train_df = sc_cache_meta[~vm].reset_index(drop=True)
            sds = ScDS(Y_tr, sc_train_df, aug=True)
            items.append(("sc", sds, len(sds)))
        return items
    
    def train_fold(fold_k):
        vm = sc_cache_meta["fold"].values == fold_k
        Y_val = Y_SC[vm]
        ns22_val = non_s22_mask_sc[vm]
        val_sc_df_k = sc_cache_meta[vm].reset_index(drop=True)
    
        active = build_active_datasets(fold_k)
        names, datasets, sizes = zip(*active)
        mds = ConcatDataset(list(datasets))
        nst = max(100, int(sum(sizes) / BATCH))
    
        print(f"  Streams: {dict(zip(names, sizes))}  steps/ep: {nst}")
    
        m = make_model()
        mel_transform = MelSpecTransform().to(device)
        spec_augment = SpecAugment().to(device)
        perch_teacher = PerchTeacher(PERCH_ONNX_PATH,
                                      "cuda" if torch.cuda.is_available() else "cpu") \
                        if USE_PERCH_DISTILL else None
    
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
        scaler = GradScaler()
        warmup_steps = nst * WARMUP_EPOCHS
        total_steps  = nst * EPOCHS
        warmup_sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1/25, end_factor=1.0,
                                                          total_iters=warmup_steps)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - warmup_steps,
                                                                   eta_min=1e-6)
        sch = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup_sched, cosine_sched],
                                                     milestones=[warmup_steps])
    
        history = {"ep": [], "train_loss": [], "cls_loss": [], "dist_loss": [],
                   "macro": [], "ns22_macro": [],
                   "ns22_Aves": [], "ns22_Amphibia": [], "ns22_Insecta": [], "ns22_Mammalia": [],
                   "val_preds": []}
        best_ns22, best_state_ns22 = -1.0, None
        best_macro, best_state_macro = -1.0, None
        val_wavs_k = _load_val_waveforms(val_sc_df_k)
    
        for ep in range(EPOCHS):
            m.train()
            smp = MixSamp(list(sizes), list(names), SHARES, BATCH, nst, seed=42 + ep)
            tl = DataLoader(mds, batch_sampler=smp, collate_fn=collate_m,
                            num_workers=0, pin_memory=True)
            el, el_cls, el_dist, nb_count = 0.0, 0.0, 0.0, 0
            t0 = time.time()
    
            for wav, lb, wt, mk, sr in tl:
                wav, lb, wt, mk = wav.to(device), lb.to(device), wt.to(device), mk.to(device)
                sw = mk_sw(sr).to(device)
    
                with torch.no_grad():
                    mel = mel_transform(wav)
                    B = mel.size(0)
                    for i in range(B):
                        mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
                    mel = spec_augment(mel)
    
                with autocast():
                    if USE_PERCH_DISTILL:
                        clip_logits, framewise, distill_emb = m(mel, return_framewise=True,
                                                                return_distill=True)
                    else:
                        clip_logits, framewise = m(mel, return_framewise=True)
    
                    frame_max_logits = framewise.max(dim=1).values
    
                    # Classification loss
                    bce_clip = F.binary_cross_entropy_with_logits(clip_logits, lb, reduction="none")
                    bce_frame = F.binary_cross_entropy_with_logits(frame_max_logits, lb, reduction="none")
                    bce = 0.5 * bce_clip + 0.5 * bce_frame
                    ps = (bce * wt * mk).sum(1) / (mk.sum(1) + 1e-8)
                    cls_loss = (ps * sw).mean()
    
                    # Distillation loss
                    if USE_PERCH_DISTILL and perch_teacher is not None:
                        with torch.no_grad():
                            wav_5s = wav.squeeze(1)
                            N = wav_5s.shape[1]
                            if N > 160000:
                                start = (N - 160000) // 2
                                wav_5s = wav_5s[:, start:start + 160000]
                            elif N < 160000:
                                wav_5s = F.pad(wav_5s, (0, 160000 - N))
                            perch_emb = perch_teacher.embed(wav_5s).to(device)
                        distill_loss = F.mse_loss(distill_emb, perch_emb)
                        loss = cls_loss + ALPHA_DISTILL * distill_loss
                    else:
                        distill_loss = torch.tensor(0.0)
                        loss = cls_loss
    
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sch.step()
                el += loss.item(); el_cls += cls_loss.item()
                el_dist += distill_loss.item(); nb_count += 1
    
            # Validation
            val_preds_dict = _predict_from_waveforms(m, mel_transform, val_wavs_k)
            val_preds = val_preds_dict["blend"]
            r = full_eval(Y_val, val_preds, ns22_val, TAXON_MASKS)
            for mode in ["clip", "fmax", "blend"]:
                r_mode = full_eval(Y_val, val_preds_dict[mode], ns22_val, TAXON_MASKS)
                r[f"ns22_{mode}"] = r_mode["non_s22_macro"]
    
            history["ep"].append(ep)
            history["train_loss"].append(round(el / nb_count, 5))
            history["cls_loss"].append(round(el_cls / nb_count, 5))
            history["dist_loss"].append(round(el_dist / nb_count, 5))
            history["macro"].append(r["macro_auc_all"])
            history["ns22_macro"].append(r["non_s22_macro"])
            for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]:
                history[f"ns22_{t}"].append(r[f"non_s22_{t}"])
            history["val_preds"].append(val_preds.astype(np.float32))
    
            tag_ns22 = ""; tag_macro = ""
            if r["non_s22_macro"] > best_ns22:
                best_ns22 = r["non_s22_macro"]
                best_state_ns22 = {k: v.cpu().clone() for k, v in m.state_dict().items()}
                tag_ns22 = " *ns22"
            if r["macro_auc_all"] > best_macro:
                best_macro = r["macro_auc_all"]
                best_state_macro = {k: v.cpu().clone() for k, v in m.state_dict().items()}
                tag_macro = " *macro"
    
            dist_str = f" dist={el_dist/nb_count:.4f}" if USE_PERCH_DISTILL else ""
            print(f"    Ep{ep:02d}: loss={el/nb_count:.4f} cls={el_cls/nb_count:.4f}{dist_str} "
                  f"lr={opt.param_groups[0]['lr']:.1e} | "
                  f"ns22: {r['ns22_blend']:.4f} | "
                  f"Av={r['non_s22_Aves']:.4f} Am={r['non_s22_Amphibia']:.4f} "
                  f"In={r['non_s22_Insecta']:.4f} Ma={r['non_s22_Mammalia']:.4f} "
                  f"[{time.time()-t0:.0f}s]{tag_ns22}{tag_macro}")
    
        del perch_teacher, m, mel_transform, spec_augment
        torch.cuda.empty_cache(); gc.collect()
        return best_state_ns22, best_state_macro, history
    
    print("OK Training function ready")
    
    # =================================================================
    # S6 -- FOLD LOOP + ONNX EXPORT
    # =================================================================
    
    if MODE != "train":
        print("Skipping training (MODE='infer')")
        oof_ns22 = None
        all_hist = {}
    else:
    
        oof_ns22 = np.full((len(sc_cache_meta), NUM_CLASSES), np.nan, dtype=np.float32)
        all_hist = {}
        for fold_k in FOLDS:
            print(f"\n{'='*60}")
            print(f"FOLD {fold_k}")
            print(f"{'='*60}")
            vm = sc_cache_meta["fold"].values == fold_k
            val_sc_df_k = sc_cache_meta[vm].reset_index(drop=True)
        
            best_ns22_state, best_macro_state, hist = train_fold(fold_k)
            all_hist[fold_k] = hist
        
            mel_tf = MelSpecTransform().to(device)
            val_wavs_k = _load_val_waveforms(val_sc_df_k)
        
            if best_macro_state is not None:
                torch.save(best_macro_state, OUT_DIR / f"fold{fold_k}_best_ns22.pt")
                m = make_model()
                m.load_state_dict(best_macro_state, strict=False)
                oof_ns22[vm] = _predict_from_waveforms(m, mel_tf, val_wavs_k)["blend"]
        
                # --- ONNX Export ---
                m.eval()
                INF_N_MELS = 128
                INF_N_FRAMES = VAL_SAMPLES // HOP_LENGTH + 1
        
                class SEDExportWrapper(nn.Module):
                    def __init__(self, backbone_name, num_classes, backbone_dim, hidden_dim=512):
                        super().__init__()
                        self.backbone = timm.create_model(
                            backbone_name, pretrained=False, in_chans=1,
                            num_classes=0, global_pool="", drop_path_rate=0.1,
                        )
                        self.gem_freq = GeMFreqPool(p_init=3.0)
                        self.dense_drop1 = nn.Dropout(0.25)
                        self.dense_conv = nn.Conv1d(backbone_dim, hidden_dim, 1)
                        self.dense_relu = nn.ReLU(inplace=True)
                        self.dense_drop2 = nn.Dropout(0.5)
                        self.att = nn.Conv1d(hidden_dim, num_classes, 1)
                        self.cla = nn.Conv1d(hidden_dim, num_classes, 1)
        
                    def forward(self, mel):
                        h = self.backbone(mel)
                        h = self.gem_freq(h)
                        h = self.dense_drop1(h)
                        h = self.dense_conv(h)
                        h = self.dense_relu(h)
                        h = self.dense_drop2(h)
                        norm_att = torch.softmax(torch.tanh(self.att(h)), dim=-1)
                        framewise = self.cla(h)
                        clip = torch.sum(norm_att * framewise, dim=2)
                        return clip, framewise.permute(0, 2, 1)
        
                def load_and_remap_state(export_model, trained_state):
                    remap = {}
                    for k, v in trained_state.items():
                        if k.startswith("distill_head."):
                            continue
                        if k == "dense.1.weight":
                            remap["dense_conv.weight"] = v.unsqueeze(-1)
                        elif k == "dense.1.bias":
                            remap["dense_conv.bias"] = v
                        else:
                            remap[k] = v
                    export_model.load_state_dict(remap, strict=False)
        
                export_model = SEDExportWrapper(
                    BACKBONE_NAME, NUM_CLASSES, m.backbone_dim
                ).to(device)
                load_and_remap_state(export_model, best_macro_state)
                export_model.eval()
        
                dummy_mel = torch.randn(1, 1, INF_N_MELS, INF_N_FRAMES).to(device)
                onnx_path = OUT_DIR / f"sed_distill_fold{fold_k}.onnx"
                torch.onnx.export(
                    export_model, dummy_mel, str(onnx_path),
                    input_names=["mel"],
                    output_names=["clip_logits", "framewise_logits"],
                    dynamic_axes={"mel": {0: "batch"},
                                  "clip_logits": {0: "batch"},
                                  "framewise_logits": {0: "batch"}},
                    opset_version=17,
                )
        
                # Verify
                _sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
                _onnx_out = _sess.run(None, {'mel': dummy_mel.cpu().numpy()})
                with torch.no_grad():
                    _ref_clip, _ref_frame = export_model(dummy_mel)
                _diff = np.abs(_ref_clip.cpu().numpy() - _onnx_out[0]).max()
                print(f"  ONNX verify: max|diff|={_diff:.3e}")
                assert _diff < 1e-3, f"ONNX export diverged: {_diff}"
                del _sess
        
                size_mb = onnx_path.stat().st_size / 1e6
                print(f"  Exported {onnx_path.name} ({size_mb:.1f} MB)")
                del m, export_model
    
    # =================================================================
    # S7 -- EVALUATION SUMMARY
    # =================================================================
    
    if MODE != "train":
        print("Skipping evaluation (MODE='infer')")
    else:
    
        has = ~np.isnan(oof_ns22[:, 0])
        if has.sum() > 0:
            r_all = full_eval(Y_SC[has], oof_ns22[has], non_s22_mask_sc[has], TAXON_MASKS)
            print("=" * 60)
            print("OOF RESULTS (best-ns22 checkpoints)")
            print("=" * 60)
            print(f"  macro AUC (all):        {r_all['macro_auc_all']:.4f}")
            print(f"  macro AUC (non-S22):    {r_all['non_s22_macro']:.4f}")
            for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]:
                print(f"    {t:<12}: {r_all.get(f'non_s22_{t}', float('nan')):.4f}")
        
        # Per-epoch progression
        print("\nPer-epoch pooled non-S22 AUC:")
        fold_true, fold_ns22_m = {}, {}
        for fk in range(N_FOLDS):
            vm = sc_cache_meta["fold"].values == fk
            fold_true[fk] = Y_SC[vm]
            fold_ns22_m[fk] = non_s22_mask_sc[vm]
        
        n_eps = [len(all_hist[k]["val_preds"]) for k in range(N_FOLDS) if k in all_hist]
        max_ep = min(n_eps) if n_eps else 0
        for ep in range(max_ep):
            pp = np.concatenate([all_hist[k]["val_preds"][ep] for k in range(N_FOLDS) if k in all_hist])
            pt = np.concatenate([fold_true[k] for k in range(N_FOLDS) if k in all_hist])
            pm = np.concatenate([fold_ns22_m[k] for k in range(N_FOLDS) if k in all_hist])
            ns, _ = compute_macro_auc(pt, pp, mask=pm)
            print(f"  Ep{ep:02d}: {ns:.4f}")
    
    # =================================================================
    # INFERENCE -- Mel spectrogram (librosa)
    # =================================================================
    if MODE != "infer":
        print("Skipping inference setup (MODE='train')")
    else:
        import librosa
    
        INF_N_MELS   = 256
        INF_N_FFT    = 2048
        INF_HOP      = 512
        INF_FMIN     = 20
        INF_FMAX     = 16000
        INF_TOP_DB   = 80
        INF_SR       = 32000
        INF_CHUNK_S  = 5
        INF_CHUNK_N  = INF_SR * INF_CHUNK_S   # 160,000
        INF_N_FRAMES = INF_CHUNK_N // INF_HOP + 1  # 313
        
        def audio_to_mel(chunks):
            """Raw audio chunks (N, 160000) -> normalized mel dB (N, 1, 128, 313)."""
            mels = []
            for i in range(chunks.shape[0]):
                S = librosa.feature.melspectrogram(
                    y=chunks[i], sr=INF_SR, n_fft=INF_N_FFT, hop_length=INF_HOP,
                    n_mels=INF_N_MELS, fmin=INF_FMIN, fmax=INF_FMAX, power=2.0,
                )
                S_dB = librosa.power_to_db(S, top_db=INF_TOP_DB)
                S_dB = (S_dB - S_dB.mean()) / (S_dB.std() + 1e-6)
                mels.append(S_dB)
            return np.stack(mels)[:, np.newaxis, :, :].astype(np.float32)
        
        print(f"Inference mel: {INF_N_MELS} mels, {INF_N_FRAMES} frames/chunk")
    
    # =================================================================
    # INFERENCE -- Load ONNX sessions
    # =================================================================
    if MODE != "infer":
        print("Skipping ONNX loading (MODE='train')")
    else:
        import re, glob
    
        def discover_folds(sed_dir):
            pat = re.compile(r'sed_fold(\d+)\.onnx$')
            folds = []
            for fname in os.listdir(sed_dir):
                m = pat.match(fname)
                if m: folds.append(int(m.group(1)))
            return sorted(folds)
        
        def make_session(onnx_path):
            so = ort.SessionOptions()
            so.intra_op_num_threads = 4
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        
        SED_DIR_CANDIDATES = [
            str(OUT_DIR),
            '/kaggle/input/datasets/tuckerarrants/bc2026-distilled-sed-public',
        ]
        SED_DIR = next((p for p in SED_DIR_CANDIDATES if os.path.isdir(p) and
                        any(f.endswith('.onnx') and 'sed' in f for f in os.listdir(p))), None)
        assert SED_DIR, f'No SED ONNX files found in {SED_DIR_CANDIDATES}'
        
        INF_FOLDS = discover_folds(SED_DIR)
        assert INF_FOLDS, f'No sed_distill_fold*.onnx in {SED_DIR}'
        print(f'Found {len(INF_FOLDS)} fold(s) in {SED_DIR}: {INF_FOLDS}')
        
        fold_sessions = []
        for fold in INF_FOLDS:
            p = f'{SED_DIR}/sed_fold{fold}.onnx'
            sess = make_session(p)
            fold_sessions.append(sess)
            size_mb = os.path.getsize(p) / 1e6
            print(f'  fold {fold}: {size_mb:5.1f}MB  providers={sess.get_providers()}')
    
    # =================================================================
    # INFERENCE -- Audio loading + main loop
    # =================================================================
    from scipy.ndimage import convolve1d
    
    GAUSSIAN_KERNEL = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    N_WINDOWS = 12  # 60s / 5s
    
    if MODE != "infer":
        print("Skipping inference (MODE='train')")
    else:
        try:
            import soundfile as sf
            DECODER = 'soundfile'
        except ImportError:
            DECODER = 'librosa'
        print(f'Audio decoder: {DECODER}')
    
        def load_audio_32k_mono(path):
            if DECODER == 'soundfile':
                wav, sr = sf.read(path, dtype='float32', always_2d=False)
                if wav.ndim > 1: wav = wav.mean(axis=1)
                if sr != INF_SR:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=INF_SR)
                return wav.astype(np.float32)
            else:
                wav, _ = librosa.load(path, sr=INF_SR, mono=True)
                return wav.astype(np.float32)
    
        def file_to_chunks(path):
            wav = load_audio_32k_mono(path)
            target_len = 60 * INF_SR
            if len(wav) < target_len:
                wav = np.pad(wav, (0, target_len - len(wav)))
            elif len(wav) > target_len:
                wav = wav[:target_len]
            n_chunks = target_len // INF_CHUNK_N
            chunks = wav[:n_chunks * INF_CHUNK_N].reshape(n_chunks, INF_CHUNK_N)
            end_times = np.arange(1, n_chunks + 1) * INF_CHUNK_S
            return chunks.astype(np.float32), end_times
    
        def sigmoid_inf(x):
            return np.where(
                x >= 0,
                1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))),
                np.exp(np.clip(x, -50, 50)) / (1.0 + np.exp(np.clip(x, -50, 50))),
            ).astype(np.float32)
    
        def gauss_smooth_final(scores, weights=GAUSSIAN_KERNEL):
            """Gaussian smooth predictions across the 12 windows within each file.
            Operates on the leading axis of the per-file slab."""
            smoothed = scores.reshape(-1, N_WINDOWS, scores.shape[1]).copy()
            for i in range(smoothed.shape[0]):
                smoothed[i] = convolve1d(smoothed[i], weights, axis=0, mode='nearest')
            return smoothed.reshape(-1, scores.shape[1])
    
        # Discover test files (with train_soundscapes fallback for debugging)
        test_files = sorted(glob.glob(f'{TEST_DIR}/*.ogg')) if TEST_DIR.is_dir() else []
        if len(test_files) == 0:
            fallback = COMP_DIR / "train_soundscapes"
            if fallback.is_dir():
                test_files = sorted(glob.glob(f'{fallback}/*.ogg'))[:5]
                print(f'No test_soundscapes -- using {len(test_files)} train files for debug')
        print(f'Test files: {len(test_files)}')
    
        # --- Main inference loop ---
        t0 = time.time()
        all_rows, all_preds = [], []
    
        for file_idx, file_path in enumerate(test_files):
            basename = os.path.basename(file_path).replace('.ogg', '')
            chunks, end_times = file_to_chunks(file_path)
            mel = audio_to_mel(chunks)
    
            # Accumulate logits in logit space across folds
            logits_sum = np.zeros((chunks.shape[0], NUM_CLASSES), dtype=np.float32)
            for sess in fold_sessions:
                outs = sess.run(None, {'mel': mel})
                clip_logits = outs[0]
                frame_max = outs[1].max(axis=1)
                logits_sum += 0.5 * clip_logits + 0.5 * frame_max
            logits_mean = logits_sum / len(INF_FOLDS)
    
            # Smooth across windows in logit space, then sigmoid once.
            logits_smoothed = gauss_smooth_final(logits_mean)
            probs = sigmoid_inf(logits_smoothed)
    
            all_rows.extend([f'{basename}_{int(t)}' for t in end_times])
            all_preds.append(probs)
    
            if (file_idx + 1) % 50 == 0 or file_idx == 0 or file_idx == len(test_files) - 1:
                elapsed = time.time() - t0
                rate = (file_idx + 1) / elapsed
                print(f'  [{file_idx+1:4d}/{len(test_files)}] {elapsed:.1f}s  {rate:.2f} files/s')
    
        all_preds_arr = np.concatenate(all_preds) if all_preds else np.zeros((0, NUM_CLASSES), np.float32)
        print(f'\nInference: {len(all_rows)} rows, {time.time()-t0:.1f}s total')
    
    # =================================================================
    # WRITE SUBMISSION
    # =================================================================
    if MODE != "infer":
        print("Skipping submission (MODE='train')")
    else:
        submission = pd.DataFrame(all_preds_arr, columns=PRIMARY_LABELS)
        submission.insert(0, 'row_id', all_rows)
        
        assert submission.shape[1] == NUM_CLASSES + 1
        assert submission['row_id'].is_unique
        assert not submission.iloc[:, 1:].isna().any().any()
        submission.iloc[:, 1:] = submission.iloc[:, 1:].clip(0.0, 1.0)
        
        submission.to_csv(_file_name_submission, index=False)
        print(f'Wrote submission.csv: {len(submission)} rows x {submission.shape[1]} cols')
        print(submission.head(3).iloc[:, :6])
    
    # =================================================================
    # PIPELINE 2: PROTOSSM V5 WITH META-LEARNER AND HARD NEGATIVE SUPPRESSION
    # =================================================================
    if 'Model_3' in _ensemble_models:
        
        _file_name_submission = "subm_3.csv"
        
        # Install required packages
        !pip install -q timm torchaudio onnxscript onnx lightgbm scikit-learn
        
        import os, sys, json, pickle, gc, random, math, re, glob
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from collections import defaultdict
        
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader, ConcatDataset
        from torch.cuda.amp import GradScaler, autocast
        import torchaudio
        import timm
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from scipy.special import expit as sigmoid_np
        import warnings
        warnings.filterwarnings("ignore")
        
        SEED = 42
        random.seed(SEED)
        os.environ["PYTHONHASHSEED"] = str(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name()}")
        
        MODE = "infer"  # "train" or "infer"
        
        COMP_DIR = Path("/kaggle/input/competitions/birdclef-2026")
        WAVEFORM_CACHE_DIR = Path("/kaggle/input/datasets/tuckerarrants/birdclef-2026-waveform-cache/waveform_cache")
        PERCH_ONNX_PATH = Path("/kaggle/input/datasets/tuckerarrants/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx")
        
        LABELS_PATH     = COMP_DIR / "train_soundscapes_labels.csv"
        TAXONOMY_PATH   = COMP_DIR / "taxonomy.csv"
        SAMPLE_SUB_PATH = COMP_DIR / "sample_submission.csv"
        TEST_DIR        = COMP_DIR / "test_soundscapes"
        
        OUT_DIR = Path("/kaggle/working")
        
        NUM_CLASSES = 234
        SR = 32000
        
        TRAIN_DURATION = 5
        VAL_DURATION   = 5
        TRAIN_SAMPLES  = SR * TRAIN_DURATION
        VAL_SAMPLES    = SR * VAL_DURATION
        
        N_FOLDS = 5
        
        N_FFT      = 2048
        HOP_LENGTH = 512
        N_MELS     = 256
        FMIN       = 20
        FMAX       = 16000
        
        BACKBONE_NAME = "tf_efficientnet_b0.ns_jft_in1k"
        
        USE_PERCH_DISTILL = True
        PERCH_EMBED_DIM   = 1536
        ALPHA_DISTILL     = 1.0
        
        FOLDS  = [0, 1, 2, 3, 4]
        EPOCHS = 25
        BATCH  = 16 if MODE == "train" else 64
        LR     = 5e-4
        MIN_LR = 1e-5
        WD     = 1e-4
        WARMUP_EPOCHS = 2
        
        MIN_SAMPLE = 20
        
        AUG_PROB = 0.5
        AUG_GAIN_DB_RANGE      = (-6.0, 6.0)
        AUG_NOISE_SNR_DB_RANGE = (10.0, 30.0)
        
        USE_FOCAL_MIXUP    = True
        MIXUP_PROB         = 0.5
        MIXUP_ALPHA        = 0.4
        MIXUP_HARD         = True
        
        USE_FOCAL_SC_MIXUP     = True
        FOCAL_SC_MIXUP_PROB    = 0.5
        FOCAL_SC_MIXUP_ALPHA   = 0.4
        
        FREQ_MIXSTYLE_PROB  = 0.0
        FREQ_MIXSTYLE_ALPHA = 0.1
        
        FREQ_MASK_PARAM = 10
        TIME_MASK_PARAM = 10
        NUM_FREQ_MASKS  = 1
        NUM_TIME_MASKS  = 2
        
        USE_FOCAL           = True
        USE_FOCAL_SECONDARY = True
        USE_LABELED_SC      = True
        
        ACTIVE_SOURCES = ["focal", "sc"]
        SHARES = {"focal": 0.9, "sc": 0.1}
        SOURCE_WEIGHTS = {
            "focal":         1.0,
            "focal_missing": 0.0,
            "sc":            1.0,
        }
        
        print(f"Backbone: {BACKBONE_NAME}")
        print(f"Train duration: {TRAIN_DURATION}s | Mel: {N_MELS} mels, n_fft={N_FFT}, hop={HOP_LENGTH}")
        print(f"Distillation: {'ON' if USE_PERCH_DISTILL else 'OFF'} (alpha={ALPHA_DISTILL})")
        print(f"Batch: {BATCH} | Epochs: {EPOCHS} | Folds: {FOLDS}")
    
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    PRIMARY_LABELS = sample_sub.columns[1:].tolist()
    LABEL2IDX = {label: idx for idx, label in enumerate(PRIMARY_LABELS)}
    taxonomy = pd.read_csv(TAXONOMY_PATH)
    label_to_taxon = dict(zip(taxonomy["primary_label"].astype(str),
                              taxonomy["class_name"].astype(str)))
    TAXON_MASKS = {t: np.array([i for i, l in enumerate(PRIMARY_LABELS)
                                if label_to_taxon.get(l, "") == t])
                   for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]}
    
    audio_cache_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "audio_cache_meta.csv")
    train_df = pd.read_csv(COMP_DIR / "train.csv")
    audio_cache_meta = audio_cache_meta.merge(
        train_df[["filename", "secondary_labels"]], on="filename", how="left"
    )
    audio_cache_meta = audio_cache_meta[
        audio_cache_meta["primary_label"].isin(LABEL2IDX)
    ].reset_index(drop=True)
    print(f"Focal audio cache: {len(audio_cache_meta)} entries")
    
    sc_cache_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "soundscape_cache_meta.csv")
    sc_cache_meta["label_list"] = sc_cache_meta["label_list"].apply(
        lambda x: x.split(";") if isinstance(x, str) else []
    )
    print(f"Soundscape cache: {len(sc_cache_meta)} windows")
    
    sc_labels_raw = pd.read_csv(LABELS_PATH).drop_duplicates()
    sc_labels_raw["start_sec"] = pd.to_timedelta(sc_labels_raw["start"]).dt.total_seconds().astype(int)
    
    Y_SC = np.zeros((len(sc_cache_meta), NUM_CLASSES), dtype=np.float32)
    for i, row in sc_cache_meta.iterrows():
        matches = sc_labels_raw[
            (sc_labels_raw["filename"] == row["filename"]) &
            (sc_labels_raw["start_sec"] == row["start_sec"])
        ]
        for _, m in matches.iterrows():
            for lbl in str(m["primary_label"]).split(";"):
                lbl = lbl.strip()
                if lbl in LABEL2IDX:
                    Y_SC[i, LABEL2IDX[lbl]] = 1.0
    
    labeled_sc_mask = Y_SC.sum(axis=1) > 0
    print(f"Soundscape labels: {labeled_sc_mask.sum()}/{len(Y_SC)} windows labeled, "
          f"{int(Y_SC.sum())} positives, {int((Y_SC.sum(axis=0) > 0).sum())} species")
    
    audio_for_split = audio_cache_meta.drop_duplicates("original_idx").reset_index(drop=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    audio_for_split["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(audio_for_split, audio_for_split["primary_label"])):
        audio_for_split.loc[val_idx, "fold"] = fold
    audio_cache_meta = audio_cache_meta.merge(
        audio_for_split[["original_idx", "fold"]], on="original_idx", how="left"
    )
    print(f"\nFocal fold distribution:\n{audio_cache_meta['fold'].value_counts().sort_index()}")
    
    sc_files = sc_cache_meta[["filename", "site"]].drop_duplicates().reset_index(drop=True)
    gkf = GroupKFold(n_splits=N_FOLDS)
    sc_files["fold"] = -1
    for fold, (_, val_idx) in enumerate(gkf.split(sc_files, groups=sc_files["filename"])):
        sc_files.loc[sc_files.index[val_idx], "fold"] = fold
    
    file_to_fold = dict(zip(sc_files["filename"], sc_files["fold"]))
    sc_cache_meta["fold"] = sc_cache_meta["filename"].map(file_to_fold).fillna(-1).astype(int)
    print(f"\nSoundscape fold distribution:")
    print(sc_cache_meta["fold"].value_counts().sort_index())
    
    counts = audio_cache_meta["primary_label"].value_counts()
    rare_species = counts[counts < MIN_SAMPLE].index
    extra_rows = []
    for sp in rare_species:
        sp_rows = audio_cache_meta[audio_cache_meta["primary_label"] == sp]
        n_copies = int(np.ceil(MIN_SAMPLE / len(sp_rows))) - 1
        for _ in range(n_copies):
            extra_rows.append(sp_rows)
    
    n_before = len(audio_cache_meta)
    if extra_rows:
        audio_cache_meta = pd.concat([audio_cache_meta] + extra_rows, ignore_index=True)
    print(f"\nUpsampled {len(rare_species)} rare species (min={MIN_SAMPLE}): "
          f"{n_before} -> {len(audio_cache_meta)} samples")
    
    sc_sites = sc_cache_meta["site"].values
    non_s22_mask_sc = sc_sites != "S22"
    print(f"S22: {(~non_s22_mask_sc).sum()}, non-S22: {non_s22_mask_sc.sum()}")
    print("OK Data loaded")
    
    # =================================================================
    # S3 -- EVAL UTILITIES + MEL TRANSFORM + PROTOSSM MODEL
    # =================================================================
    
    def compute_macro_auc(y_true, y_pred, mask=None, class_mask=None):
        if mask is not None:
            y_true, y_pred = y_true[mask], y_pred[mask]
        if class_mask is not None:
            y_true, y_pred = y_true[:, class_mask], y_pred[:, class_mask]
        aucs = []
        for c in range(y_true.shape[1]):
            col = y_true[:, c]
            if col.sum() == 0 or col.sum() == len(col):
                continue
            try:
                aucs.append(roc_auc_score(col, y_pred[:, c]))
            except ValueError:
                continue
        return (np.mean(aucs) if aucs else float("nan")), len(aucs)
    
    def full_eval(y_true, y_pred, ns22, tm):
        r = {}
        a, n = compute_macro_auc(y_true, y_pred)
        r["macro_auc_all"], r["n_all"] = round(a, 4), n
        a, n = compute_macro_auc(y_true, y_pred, mask=ns22)
        r["non_s22_macro"], r["n_ns22"] = round(a, 4), n
        for t, cm in tm.items():
            a, n = compute_macro_auc(y_true, y_pred, mask=ns22, class_mask=cm)
            r[f"non_s22_{t}"] = round(a, 4)
        return r
    
    # ------------------------------------------------------------------
    # GPU Mel Spectrogram
    # ------------------------------------------------------------------
    class MelSpecTransform(nn.Module):
        def __init__(self):
            super().__init__()
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
            )
            self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
    
        def forward(self, waveform):
            return self.db_transform(self.mel_spec(waveform))
    
    # ------------------------------------------------------------------
    # GPU SpecAugment
    # ------------------------------------------------------------------
    class SpecAugment(nn.Module):
        def __init__(self):
            super().__init__()
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=FREQ_MASK_PARAM)
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=TIME_MASK_PARAM)
    
        def forward(self, mel):
            for _ in range(NUM_FREQ_MASKS):
                mel = self.freq_mask(mel)
            for _ in range(NUM_TIME_MASKS):
                mel = self.time_mask(mel)
            return mel
    
    # ------------------------------------------------------------------
    # Frozen Perch teacher -- ONNX inference, no gradients
    # ------------------------------------------------------------------
    import onnxruntime as ort
    
    class PerchTeacher:
        """Frozen Perch v2 via ONNX. Takes 5s waveforms, returns 1536-d embeddings.
        The teacher is never updated -- it provides a stable distillation target."""
    
        def __init__(self, onnx_path, device_str="cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
                if device_str == "cuda" else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(onnx_path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self._out_names = [o.name for o in self.session.get_outputs()]
            self._embed_idx = None
            for i, o in enumerate(self.session.get_outputs()):
                if o.shape and o.shape[-1] == PERCH_EMBED_DIM:
                    self._embed_idx = i
                    break
            if self._embed_idx is None:
                self._embed_idx = 1
            print(f"Perch ONNX loaded: embed_idx={self._embed_idx}")
    
        @torch.no_grad()
        def embed(self, waveforms_5s):
            """waveforms_5s: (B, 160000) float32, returns (B, 1536) embeddings."""
            wav_np = waveforms_5s.cpu().numpy()
            results = self.session.run(None, {self.input_name: wav_np})
            return torch.from_numpy(results[self._embed_idx]).float()
    
    # ------------------------------------------------------------------
    # ProtoSSMv5 Model
    # ------------------------------------------------------------------
    class ProtoSSMv5(nn.Module):
        """State Space Model with temporal cross-attention and prototypical classification.
        - 4-layer bidirectional Mamba-style SSM
        - 8-head temporal cross-attention
        - Perch embedding distillation
        - Residual correction network
        """
        def __init__(self, num_classes=NUM_CLASSES, perch_dim=PERCH_EMBED_DIM, device='cpu'):
            super().__init__()
            self.device = device
            self.perch_dim = perch_dim
            
            # Learnable class prototypes (234, perch_dim)
            self.prototypes = nn.Parameter(torch.randn(num_classes, perch_dim).to(device) / 10.0)
            
            # Metadata projection (site_id + hour_utc -> 24-dim)
            self.meta_proj = nn.Sequential(
                nn.Linear(6 + 24, 64),  # 6 for site_id one-hot, 24 for hour embedding
                nn.ReLU(),
                nn.Linear(64, 24)
            )
            
            # Selective State Space Model (Mamba-style)
            self.ssm = nn.ModuleList([
                nn.TransformerXLEncoderLayer(d_model=320, nhead=4, dim_feedforward=512, dropout=0.1, activation='relu'),
                nn.TransformerXLEncoderLayer(d_model=320, nhead=4, dim_feedforward=512, dropout=0.1, activation='relu'),
                nn.TransformerXLEncoderLayer(d_model=320, nhead=4, dim_feedforward=512, dropout=0.1, activation='relu'),
                nn.TransformerXLEncoderLayer(d_model=320, nhead=4, dim_feedforward=512, dropout=0.1, activation='relu'),
            ])
            
            # Temporal Cross-Attention
            self.tca = nn.MultiheadAttention(embed_dim=320, num_heads=8, batch_first=True)
            
            # Residual Correction Network
            self.residual_corr = nn.Sequential(
                nn.Linear(320, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 320)
            )
            
        def forward(self, perch_embeddings, site_ids, hours):
            """Perch embeddings: (B, 12, 1536), site_ids: (B, 6), hours: (B, 24) -> (B, 12, 320)"""
            B, N_W, D = perch_embeddings.shape
            
            # Project metadata and add to embeddings
            meta_emb = self.meta_proj(torch.cat([site_ids, hours], dim=1)).unsqueeze(1).repeat(1, N_W, 1)  # (B, 1, 24) -> (B, N_W, 24)
            x = torch.cat([perch_embeddings, meta_emb], dim=-1)  # (B, N_W, 1560)
            
            # Selective SSM
            for layer in self.ssm:
                x = layer(x)
            
            # Temporal Cross-Attention
            x, _ = self.tca(x)
            
            # Sequence pooling
            seq_emb = x.mean(dim=1)  # (B, 320)
            
            # Prototypical classification
            logits = F.cosine_similarity(seq_emb.unsqueeze(1), self.prototypes.unsqueeze(0), dim=-1).squeeze(-1)
            
            # Residual correction
            correction = self.residual_corr(seq_emb)
            logits = logits + correction @ self.prototypes.T  # Apply correction in logit space
            
            return logits, x
    
    # ------------------------------------------------------------------
    # Ridge Regression Probes for all 234 classes
    # ------------------------------------------------------------------
    class RidgeProbes:
        """Ridge regression probes on Perch embeddings for all 234 classes.
        Uses closed-form solution for fast training and inference.
        """
        def __init__(self, perch_dim=1536):
            self.perch_dim = perch_dim
            self.weights = []  # List of 234 weight vectors (1D)
            self.biases = []
            
        def train(self, X, y):
            """X: (N, perch_dim), y: (N,) binary labels for one class."""
            X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias term
            XtX_inv = np.linalg.pinv(X.T @ X)
            w = XtX_inv @ X.T @ y
            return w[:-1], w[-1]  # weights, bias
            
        def train_all(self, X, Y):
            """X: (N, perch_dim), Y: (N, C) binary matrix."""
            N, C = Y.shape
            X_with_bias = np.hstack([X, np.ones((N, 1))])
            XtX_inv = np.linalg.pinv(X.T @ X)
            for c in range(C):
                w = XtX_inv @ X.T @ Y[:, c]
                self.weights.append(w[:-1])
                self.biases.append(w[-1])
            
        def predict(self, X):
            X = np.hstack([X, np.ones((X.shape[0], 1))])
            preds = np.zeros((X.shape[0], len(self.weights)))
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                preds[:, i] = X @ w + b
            return preds
    
    # ------------------------------------------------------------------
    # Hard Negative Suppression using Metadata
    # ------------------------------------------------------------------
    class HardNegativeMask:
        """Apply biological constraints to suppress impossible predictions.
        Uses site-specific and time-specific species occurrence data.
        """
        def __init__(self, taxonomy_path):
            self.taxa = pd.read_csv(taxonomy_path)
            self.diurnal_species = self._get_diurnal_species()
            
        def _get_diurnal_species(self):
            """Get species that are strictly diurnal from taxonomy."""
            diurnal = set()
            # Placeholder - in production, load actual species-specific activity patterns
            return diurnal
        
        def apply_mask(self, probs, recording_time, site_id):
            """Set probabilities to 0 for species that cannot be present at this time.
            probs: (N, 234), recording_time: datetime, site_id: string
            """
            hours = pd.to_datetime(recording_time).hour
            mask = np.ones(probs.shape[0], dtype=bool)
            
            for idx, label in enumerate(PRIMARY_LABELS):
                if label in self.diurnal_species:
                    start, end = 6, 18  # 6 AM - 6 PM
                    mask[:, idx] = (hours >= start) & (hours < end)
            
            probs = np.where(mask, probs, 0.0)
            return probs
    
    # =================================================================
    # S2 -- LOAD DATA
    # =================================================================
    
    # (Data loading code remains the same as Pipeline 1, but with different model)
    
    # =================================================================
    # S3 -- EVAL UTILITIES + MEL TRANSFORM + PROTOSSM MODEL
    # =================================================================
    
    # (Same utilities as Pipeline 1)
    
    # =================================================================
    # S4 -- DATA PIPELINE
    # =================================================================
    
    # (Same data pipeline as Pipeline 1)
    
    # =================================================================
    # S5 -- TRAINING
    # =================================================================
    
    def train_fold(fold_k):
        vm = sc_cache_meta["fold"].values == fold_k
        Y_val = Y_SC[vm]
        ns22_val = non_s22_mask_sc[vm]
        val_sc_df_k = sc_cache_meta[vm].reset_index(drop=True)
    
        active = build_active_datasets(fold_k)
        names, datasets, sizes = zip(*active)
        mds = ConcatDataset(list(datasets))
        nst = max(100, int(sum(sizes) / BATCH))
    
        print(f"  Streams: {dict(zip(names, sizes))}  steps/ep: {nst}")
    
        # Initialize both models
        m_sed = make_model()  # SegFormer-based SED model
        m_proto = ProtoSSMv5()  # ProtoSSM v5 model
        
        mel_transform = MelSpecTransform().to(device)
        spec_augment = SpecAugment().to(device)
        perch_teacher = PerchTeacher(PERCH_ONNX_PATH,
                                      "cuda" if torch.cuda.is_available() else "cpu") \
                        if USE_PERCH_DISTILL else None
    
        # Optimizers for both models
        opt_sed = torch.optim.AdamW(m_sed.parameters(), lr=LR, weight_decay=WD)
        opt_proto = torch.optim.AdamW(m_proto.parameters(), lr=LR, weight_decay=WD)
        
        scaler = GradScaler()
        warmup_steps = nst * WARMUP_EPOCHS
        total_steps  = nst * EPOCHS
        warmup_sched = torch.optim.lr_scheduler.LinearLR(opt_sed, start_factor=1/25, end_factor=1.0,
                                                          total_iters=warmup_steps)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_sed, T_max=total_steps - warmup_steps,
                                                                   eta_min=1e-6)
        sch = torch.optim.lr_scheduler.SequentialLR(opt_sed, schedulers=[warmup_sched, cosine_sched],
                                                     milestones=[warmup_steps])
    
        history = {"ep": [], "train_loss": [], "cls_loss": [], "dist_loss": [],
                   "macro": [], "ns22_macro": [],
                   "ns22_Aves": [], "ns22_Amphibia": [], "ns22_Insecta": [], "ns22_Mammalia": [],
                   "val_preds": []}
        best_ns22, best_state_ns22 = -1.0, None
        best_macro, best_state_macro = -1.0, None
        val_wavs_k = _load_val_waveforms(val_sc_df_k)
    
        for ep in range(EPOCHS):
            m_sed.train()
            m_proto.train()
            smp = MixSamp(list(sizes), list(names), SHARES, BATCH, nst, seed=42 + ep)
            tl = DataLoader(mds, batch_sampler=smp, collate_fn=collate_m,
                            num_workers=0, pin_memory=True)
            el, el_cls, el_dist, nb_count = 0.0, 0.0, 0.0, 0
            t0 = time.time()
    
            for wav, lb, wt, mk, sr in tl:
                wav, lb, wt, mk = wav.to(device), lb.to(device), wt.to(device), mk.to(device)
                sw = mk_sw(sr).to(device)
    
                with torch.no_grad():
                    mel = mel_transform(wav)
                    B = mel.size(0)
                    for i in range(B):
                        mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
                    mel = spec_augment(mel)
    
                with autocast():
                    # SED model forward pass
                    if USE_PERCH_DISTILL:
                        clip_logits, framewise, distill_emb = m_sed(mel, return_framewise=True, return_distill=True)
                    else:
                        clip_logits, framewise = m_sed(mel, return_framewise=True)
                    frame_max_logits = framewise.max(dim=1).values
                    
                    # Classification loss for SED
                    bce_clip = F.binary_cross_entropy_with_logits(clip_logits, lb, reduction="none")
                    bce_frame = F.binary_cross_entropy_with_logits(frame_max_logits, lb, reduction="none")
                    bce = 0.5 * bce_clip + 0.5 * bce_frame
                    ps = (bce * wt * mk).sum(1) / (mk.sum(1) + 1e-8)
                    cls_loss_sed = (ps * sw).mean()
                    
                    # Distillation loss for SED
                    if USE_PERCH_DISTILL and perch_teacher is not None:
                        with torch.no_grad():
                            wav_5s = wav.squeeze(1)
                            N = wav_5s.shape[1]
                            if N > 160000:
                                start = (N - 160000) // 2
                                wav_5s = wav_5s[:, start:start + 160000]
                            elif N < 160000:
                                wav_5s = F.pad(wav_5s, (0, 160000 - N))
                            perch_emb = perch_teacher.embed(wav_5s).to(device)
                        distill_loss_sed = F.mse_loss(distill_emb, perch_emb)
                        loss_sed = cls_loss_sed + ALPHA_DISTILL * distill_loss_sed
                    else:
                        distill_loss_sed = torch.tensor(0.0)
                        loss_sed = cls_loss_sed
    
                # Step SED model
                opt_sed.zero_grad()
                scaler.scale(loss_sed).backward()
                scaler.unscale_(opt_sed)
                torch.nn.utils.clip_grad_norm_(m_sed.parameters(), 1.0)
                scaler.step(opt_sed)
                
                # ProtoSSM forward pass (requires Perch embeddings from SED's distill_head)
                with torch.no_grad():
                    perch_emb = m_sed.distill_head(m_sed.backbone(mel))
                
                # For ProtoSSM, we need site_ids and hours - these are derived from metadata
                # In practice, we'd have these available from the dataset
                # For now, we'll use dummy values for training the ProtoSSM (will be replaced in full implementation)
                dummy_site_ids = torch.zeros(B, 6).to(device)  # one-hot for 6 sites
                dummy_hours = torch.randn(B, 24).to(device)  # hour embedding
                
                proto_logits = m_proto(perch_emb, dummy_site_ids, dummy_hours)
                
                # ProtoSSM classification loss (MSE to targets)
                proto_loss = F.mse_loss(proto_logits, clip_logits)  # Simple proxy
                
                # Step ProtoSSM model
                opt_proto.zero_grad()
                proto_loss.backward()
                torch.nn.utils.clip_grad_norm_(m_proto.parameters(), 1.0)
                opt_proto.Trinity-Large()
    
                el += loss_sed.item(); el_cls += cls_loss_sed.item()
                el_dist += distill_loss_sed.item(); nb_count += 1
    
            # Validation for SED model
            val_preds_dict = _predict_from_waveforms(m_sed, mel_tf, val_wavs_k)
            val_preds = val_preds_dict["blend"]
            r = full_eval(Y_val, val_preds, ns22_val, TAXON_MASKS)
            for mode in ["clip", "fmax", "blend"]:
                r_mode = full_eval(Y_val, val_preds_dict[mode], ns22_val, TAXON_MASKS)
                r[f"ns22_{mode}"] = r_mode["non_s22_macro"]
    
            history["ep"].append(ep)
            history["train_loss"].append(round(el / nb_count, 5))
            history["cls_loss"].append(round(el_cls / nb_count, 5))
            history["dist_loss"].append(round(el_dist / nb_count, 5))
            history["macro"].append(r["macro_auc_all"])
            history["ns22_macro"].append(r["non_s22_macro"])
            for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]:
                history[f"ns22_{t}"].append(r[f"non_s22_{t}"])
            history["val_preds"].append(val_preds.astype(np.float32))
    
            tag_ns22 = ""; tag_macro = ""
            if r["non_s22_macro"] > best_ns22:
                best_ns22 = r["non_s22_macro"]
                best_state_ns22 = {k: v.cpu().clone() for k, v in m_sed.state_dict().items()}
                tag_ns22 = " *ns22"
            if r["macro_auc_all"] > best_macro:
                best_macro = r["macro_auc_all"]
                best_state_macro = {k: v.cpu().clone() for k, v in m_sed.state_dict().items()}
                tag_macro = " *macro"
    
            dist_str = f" dist={el_dist/nb_count:.4f}" if USE_PERCH_DISTILL else ""
            print(f"    Ep{ep:02d}: loss={el/nb_count:.4f} cls={el_cls/nb_count:.4f}{dist_str} "
                  f"lr={opt_sed.param_groups[0]['lr']:.1e} | "
                  f"ns22: {r['ns22_blend']:.4f} | "
                  f"Av={r['non_s22_Aves']:.4f} Am={r['non_s22_Amphibia']:.4f} "
                  f"In={r['non_s22_Insecta']:.4f} Ma={r['non_s22_Mammalia']:.4f} "
                  f"[{time.time()-t0:.0f}s]{tag_ns22}{tag_macro}")
    
        del m_sed, m_proto, mel_transform, spec_augment, perch_teacher
        torch.cuda.empty_cache(); gc.collect()
        return best_state_ns22, best_state_macro, history
    
    print("OK Training function ready")
    
    # =================================================================
    # S6 -- OOF PREDICTION + META-LEARNER TRAINING
    # =================================================================
    
    if MODE != "train":
        print("Skipping OOF prediction (MODE='infer')")
        oof_ns22 = None
        all_hist = {}
    else:
        # Generate OOF predictions for both SED and ProtoSSM
        oof_ns22 = np.full((len(sc_cache_meta), NUM_CLASSES), np.nan, dtype=np.float32)
        oof_proto = np.full((len(sc_cache_meta), NUM_CLASSES), np.nan, dtype=np.float32)
        all_hist = {}
        for fold_k in FOLDS:
            print(f"\n{'='*60}")
            print(f"FOLD {fold_k}")
            print(f"{'='*60}")
            vm = sc_cache_meta["fold"].values == fold_k
            val_sc_df_k = sc_cache_meta[vm].reset_index(drop=True)
        
            # Train both models
            best_ns22_state, best_macro_state, hist = train_fold(fold_k)
            all_hist[fold_k] = hist
        
            mel_tf = MelSpecTransform().to(device)
            val_wavs_k = _load_val_waveforms(val_sc_df_k)
        
            if best_macro_state is not None:
                # Save checkpoints
                torch.save(best_macro_state, OUT_DIR / f"fold{fold_k}_best_ns22.pt")
                
                # Load models for OOF prediction
                m_sed = make_model()
                m_sed.load_state_dict(best_macro_state, strict=False)
                
                # Get SED predictions
                oof_ns22[vm] = _predict_from_waveforms(m_sed, mel_tf, val_wavs_k)["blend"]
        
        # Save OOF predictions to disk for meta-learner training
        np.save(OUT_DIR / "oof_sed.npy", oof_ns22)
        np.save(OUT_DIR / "oof_proto.npy", oof_proto)
        np.save(OUT_DIR / "oof_labels.npy", Y_SC)
    
    # =================================================================
    # S7 -- META-LEARNER TRAINING (Ridge Regression)
    # =================================================================
    if MODE != "train":
        print("Skipping meta-learner training (MODE='infer')")
    else:
        print("Training class-specific Ridge regression meta-learner...")
        
        # Load OOF predictions
        oof_sed = np.load(OUT_DIR / "oof_sed.npy")
        oof_proto = np.load(OUT_DIR / "oof_proto.npy")
        oof_labels = np.load(OUT_DIR / "oof_labels.npy")
        
        # Stack predictions as features for meta-learner
        X_meta = np.stack([oof_sed, oof_proto], axis=-1)  # (N, C, 2)
        X_meta = X_meta.reshape(-1, NUM_CLASSES * 2)  # (N, 2C)
        y_meta = oof_labels  # (N, C)
        
        # Train per-class Ridge regression
        meta_learner = RidgeProbes(perch_dim=2)  # 2 features per class
        meta_learner.train_all(X_meta, y_meta)
        
        # Save meta-learner
        with open(OUT_DIR / "meta_learner.pkl", "wb") as f:
            pickle.dump(meta_learner, f)
    
    # =================================================================
    # S8 -- INFERENCE WITH META-LEARNING AND HARD NEGATIVE SUPPRESSION
    # =================================================================
    if MODE != "infer":
        print("Skipping inference (MODE='train')")
    else:
        print("Loading models for inference...")
        
        # Load SED model (SegFormer-B0)
        m_sed = make_model()
        # Load best state dict
        checkpoint = torch.load(OUT_DIR / "fold0_best_ns22.pt")
        m_sed.load_state_dict(checkpoint, strict=False)
        m_sed.eval()
        
        # Export to ONNX and quantize
        INF_N_MELS = 128
        INF_N_FRAMES = VAL_SAMPLES // HOP_LENGTH + 1
        
        class SEDExportWrapper(nn.Module):
            def __init__(self, backbone_name, num_classes, backbone_dim, hidden_dim=512):
                super().__init__()
                self.backbone = timm.create_model(
                    backbone_name, pretrained=False, in_chans=1,
                    num_classes=0, global_pool="", drop_path_rate=0.1,
                )
                self.gem_freq = GeMFreqPool(p_init=3.0)
                self.dense_drop1 = nn.Dropout(0.25)
                self.dense_conv = nn.Conv1d(backbone_dim, hidden_dim, 1)
                self.dense_relu = nn.ReLU(inplace=True)
                self.dense_drop2 = nn.Dropout(0.5)
                self.att = nn.Conv1d(hidden_dim, num_classes, 1)
                self.cla = nn.Conv1d(hidden_dim, num_classes, 1)
    
            def forward(self, mel):
                h = self.backbone(mel)
                h = self.gem_freq(h)
                h = self.dense_drop1(h)
                h = self.dense_conv(h)
                h = self.dense_relu(h)
                h = self.dense_drop2(h)
                norm_att = torch.softmax(torch.tanh(self.att(h)), dim=-1)
                framewise = self.cla(h)
                clip = torch.sum(norm_att * framewise, dim=2)
                return clip, framewise.permute(0, 2, 1)
    
        def load_and_remap_state(export_model, trained_state):
            remap = {}
            for k, v in trained_state.items():
                if k.startswith("distill_head."):
                    continue
                if k == "dense.1.weight":
                    remap["dense_conv.weight"] = v.unsqueeze(-1)
                elif k == "dense.1.bias":
                    remap["dense_conv.bias"] = v
                else:
                    remap[k] = v
            export_model.load_state_dict(remap, strict=False)
    
        export_model = SEDExportWrapper(
            BACKBONE_NAME, NUM_CLASSES, m_sed.backbone_dim
        ).to(device)
        load_and_remap_state(export_model, checkpoint)
        export_model.eval()
        
        dummy_mel = torch.randn(1, 1, INF_N_MELS, INF_N_FRAMES).to(device)
        onnx_path = OUT_DIR / "sed_distill_fold0.onnx"
        torch.onnx.export(
            export_model, dummy_mel, str(onnx_path),
            input_names=["mel"],
            output_names=["clip_logits", "framewise_logits"],
            dynamic_axes={"mel": {0: "batch"},
                          "clip_logits": {0: "batch"},
                          "framewise_logits": {0: "batch"}},
            opset_version=17,
        )
        
        # Quantize to INT8
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quant_path = OUT_DIR / "sed_distill_fold0_int8.onnx"
        quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.INT8, input_type=QuantType.FP32)
        
        # Load ONNX sessions with OpenVINO
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
        sed_sess = ort.InferenceSession(str(quant_path), sess_options=so, providers=providers)
        
        # Load meta-learner
        with open(OUT_DIR / "meta_learner.pkl", "rb") as f:
            meta_learner = pickle.load(f)
        
        # Load hard negative mask
        hard_mask = HardNegativeMask(TAXONOMY_PATH)
        
        # Inference loop
        INF_N_MELS   = 256
        INF_N_FFT    = 2048
        INF_HOP      = 512
        INF_FMIN     = 20
        INF_FMAX     = 16000
        INF_TOP_DB   = 80
        INF_SR       = 32000
        INF_CHUNK_S  = 5
        INF_CHUNK_N  = INF_SR * INF_CHUNK_S   # 160,000
        INF_N_FRAMES = INF_CHUNK_N // INF_HOP + 1  # 313
        
        def audio_to_mel(chunks):
            mels = []
            for i in range(chunks.shape[0]):
                S = librosa.feature.melspectrogram(
                    y=chunks[i], sr=INF_SR, n_fft=INF_N_FFT, hop_length=INF_HOP,
                    n_mels=INF_N_MELS, fmin=INF_FMIN, fmax=INF_FMAX, power=2.0,
                )
                S_dB = librosa.power_to_db(S, top_db=INF_TOP_DB)
                S_dB = (S_dB - S_dB.mean()) / (S_dB.std() + 1e-6)
                mels.append(S_dB)
            return np.stack(mels)[:, np.newaxis, :, :].astype(np.float32)
        
        def file_to_chunks(path):
            wav = load_audio_32k_mono(path)
            target_len = 60 * INF_SR
            if len(wav) < target_len:
                wav = np.pad(wav, (0, target_len - len(wav)))
            elif len(wav) > target_len:
                wav = wav[:target_len]
            n_chunks = target_len // INF_CHUNK_N
            chunks = wav[:n_chunks * INF_CHUNK_N].reshape(n_chunks, INF_CHUNK_N)
            end_times = np.arange(1, n_chunks + 1) * INF_CHUNK_S
            return chunks.astype(np.float32), end_times
    
        def sigmoid_inf(x):
            return np.where(
                x >= 0,
                1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))),
                np.exp(np.clip(x, -50, 50)) / (1.0 + np.exp(np.clip(x, -50, 50))),
            ).astype(np.float32)
    
        def gauss_smooth_final(scores, weights=GAUSSIAN_KERNEL):
            smoothed = scores.reshape(-1, N_WINDOWS, scores.shape[1]).copy()
            for i in range(smoothed.shape[0]):
                smoothed[i] = convolve1d(smoothed[i], weights, axis=0, mode='nearest')
            return smoothed.reshape(-1, scores.shape[1])
    
        test_files = sorted(glob.glob(f'{TEST_DIR}/*.ogg')) if TEST_DIR.is_dir() else []
        if len(test_files) == 0:
            fallback = COMP_DIR / "train_soundscapes"
            if fallback.is_dir():
                test_files = sorted(glob.glob(f'{fallback}/*.ogg'))[:5]
                print(f'No test_soundscapes -- using {len(test_files)} train files for debug')
        print(f'Test files: {len(test_files)}')
    
        t0 = time.time()
        all_rows, all_preds = [], []
    
        for file_idx, file_path in enumerate(test_files):
            basename = os.path.basename(file_path).replace('.ogg', '')
            chunks, end_times = file_to_chunks(file_path)
            mel = audio_to_mel(chunks)
            
            # Get SED predictions from ONNX
            outs = sed_sess.run(None, {'mel': mel})
            clip_logits = outs[0]
            frame_max = outs[1].max(axis=1)
            logits_sed = 0.5 * clip_logits + 0.5 * frame_max
            
            # For now, we'll use a dummy ProtoSSM prediction (to be implemented with full training)
            # In practice, you'd run the ProtoSSM model here and get its logits
            logits_proto = np.random.randn(*logits_sed.shape).astype(np.float32) * 0.1  # Placeholder
            
            # Apply class-specific stacking via Ridge regression
            X_stacked = np.stack([logits_sed, logits_proto], axis=-1)  # (N, C, 2)
            X_stacked = X_stacked.reshape(-1, NUM_CLASSES * 2)  # (N, 2C)
            meta_preds = meta_learner.predict(X_stacked)  # (N, C)
            
            # Apply hard negative suppression
            # In practice, you'd have recording_time and site_id available
            # For now, we'll apply a simple diurnal mask to all predictions
            probs = sigmoid_inf(meta_preds)
            # hard_mask.apply_mask(probs, recording_time, site_id)  # Placeholder
            
            logits_smoothed = gauss_smooth_final(meta_preds)
            probs = sigmoid_inf(logits_smoothed)
    
            all_rows.extend([f'{basename}_{int(t)}' for t in end_times])
            all_preds.append(probs)
    
            if (file_idx + 1) % 50 == 0 or file_idx == 0 or file_idx == len(test_files) - 1:
                elapsed = time.time() - t0
                rate = (file_idx + 1) / elapsed
                print(f'  [{file_idx+1:4d}/{len(test_files)}] {elapsed:.1f}s  {rate:.2f} files/s')
    
        all_preds_arr = np.concatenate(all_preds) if all_preds else np.zeros((0, NUM_CLASSES), np.float32)
        print(f'\nInference: {len(all_rows)} rows, {time.time()-t0:.1f}s total')
    
    # =================================================================
    # WRITE SUBMISSION
    # =================================================================
    if MODE != "infer":
        print("Skipping submission (MODE='train')")
    else:
        submission = pd.DataFrame(all_preds_arr, columns=PRIMARY_LABELS)
        submission.insert(0, 'row_id', all_rows)
        
        assert submission.shape[1] == NUM_CLASSES + 1
        assert submission['row_id'].is_unique
        assert not submission.iloc[:, 1:].isna().any().any()
        submission.iloc[:, 1:] = submission.iloc[:, 1:].clip(0.0, 1.0)
        
        submission.to_csv(_file_name_submission, index=False)
        print(f'Wrote submission.csv: {len(submission)} rows x {submission.shape[1]} cols')
        print(submission.head(3).iloc[:, :6])

