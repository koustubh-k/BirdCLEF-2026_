"""models.py — PyTorch model architectures for sequence modeling and stacking.

This module contains definition for the Selective State Space Model (SSM) blocks,
prototype cross-attention networks, error-correcting residual layers, and
parallelized MLP probe stackers.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """Selective State Space Model (S4-style block) module.

    Summary:
        Implements input-dependent discretization and continuous recurrence
        for processing sequential features over time steps.

    Attributes:
        d_model (int): Feature dimensionality of sequence tokens.
        d_state (int): Dimensionality of state space latent dimensions.
        in_proj (nn.Linear): Linear projection layer to split gate/hidden flows.
        conv1d (nn.Conv1d): Depthwise Conv1D layer for local temporal pooling.
        dt_proj (nn.Linear): Linear projection layer to predict continuous time steps.
        A_log (nn.Parameter): Learnable state matrix parameters.
        D (nn.Parameter): Learnable direct residual feedthrough weights.
        B_proj (nn.Linear): State input matrix projection.
        C_proj (nn.Linear): State output matrix projection.
        out_proj (nn.Linear): Dimensionality output projection layer.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        """Initializes SelectiveSSM layer.

        Summary:
            Builds projection layers and continuous parameters (A, D) for SSM.

        Inputs:
            d_model (int): Dimensionality of input embeddings.
            d_state (int): Size of the hidden state matrix. Default is 16.
            d_conv (int): Conv1D kernel size. Default is 4.

        Outputs:
            SelectiveSSM: An instance of SelectiveSSM.

        Shapes:
            None.

        Side effects:
            None.

        Usage example:
            >>> ssm = SelectiveSSM(d_model=128, d_state=16)
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(
            d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model
        )
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies selective state space continuous transformation to input sequence.

        Summary:
            Runs discretization and sequential hidden-state transitions.

        Inputs:
            x (torch.Tensor): Input sequence tensor.

        Outputs:
            torch.Tensor: The processed output sequence.

        Shapes:
            - Input `x`: `(B, T, d_model)`
            - Output: `(B, T, d_model)`

        Side effects:
            None.

        Usage example:
            >>> x = torch.randn(4, 12, 128)
            >>> out = ssm(x)
            >>> out.shape
            (4, 12, 128)
        """
        B_sz, T, D = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_conv = F.silu(self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2))
        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)

        h = torch.zeros(B_sz, D, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            dA = torch.exp(A[None] * dt[:, t, :, None])
            dB = dt[:, t, :, None] * B[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            ys.append((h * C[:, t, None, :]).sum(-1))

        return torch.stack(ys, dim=1) + x * self.D[None, None, :]


class LightProtoSSM(nn.Module):
    """Prototype-Attention Sequence State Space Network.

    Summary:
        Projects latent Perch embeddings, fuses site/hour metadata, runs S4 bidirectional
        recurrences, and computes similarities against class prototypes.

    Attributes:
        n_classes (int): Number of target species (234).
        n_windows (int): Length of sequence in windows (12).
        use_cross_attn (bool): Whether to run multi-head attention layers.
        input_proj (nn.Sequential): Projection layers for input embedding sequences.
        pos_enc (nn.Parameter): Positional encodings.
        site_emb (nn.Embedding): Recording site embeddings.
        hour_emb (nn.Embedding): Recording hour embeddings.
        meta_proj (nn.Linear): Joint metadata projection layer.
        ssm_fwd (nn.ModuleList): Forward bidirectional SSM blocks.
        ssm_bwd (nn.ModuleList): Backward bidirectional SSM blocks.
        ssm_merge (nn.ModuleList): Merge projections.
        ssm_norm (nn.ModuleList): Layer normalization layers.
        drop (nn.Dropout): Dropout regularizer.
        cross_attn (nn.ModuleList): Multi-head attention blocks.
        cross_norm (nn.ModuleList): Attention layer norms.
        prototypes (nn.Parameter): Class prototype similarity weights.
        proto_temp (nn.Parameter): Softmax temperature weight.
        class_bias (nn.Parameter): Class specific bias values.
        fusion_alpha (nn.Parameter): Blending ratios for Perch input logits.
    """

    def __init__(
        self,
        d_input: int = 1536,
        d_model: int = 128,
        d_state: int = 16,
        n_classes: int = 234,
        n_windows: int = 12,
        dropout: float = 0.15,
        n_sites: int = 20,
        meta_dim: int = 16,
        use_cross_attn: bool = True,
        cross_attn_heads: int = 2,
        n_ssm_layers: int = 2,
    ):
        """Initializes LightProtoSSM.

        Summary:
            Instantiates embedding mappings, SSM states, prototypes, and fusion weights.

        Inputs:
            d_input (int): Dimension of input embeddings. Default is 1536 (Perch size).
            d_model (int): Hidden dimension size. Default is 128.
            d_state (int): Size of the SSM latent state. Default is 16.
            n_classes (int): Number of target classes. Default is 234.
            n_windows (int): Sequence length in windows. Default is 12.
            dropout (float): Dropout probability. Default is 0.15.
            n_sites (int): Number of sites in vocabulary. Default is 20.
            meta_dim (int): Site/hour embedding dimension. Default is 16.
            use_cross_attn (bool): Enable Multi-head cross attention. Default is True.
            cross_attn_heads (int): Number of attention heads. Default is 2.

        Outputs:
            LightProtoSSM: An instance of LightProtoSSM.

        Shapes:
            None.

        Side effects:
            None.

        Usage example:
            >>> model = LightProtoSSM(n_classes=234)
        """
        super().__init__()
        self.n_classes = n_classes
        self.n_windows = n_windows
        self.use_cross_attn = use_cross_attn

        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        self.ssm_fwd = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(n_ssm_layers)])
        self.ssm_bwd = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(n_ssm_layers)])
        self.ssm_merge = nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(n_ssm_layers)])
        self.ssm_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_ssm_layers)])
        self.drop = nn.Dropout(dropout)

        if use_cross_attn:
            self.cross_attn = nn.ModuleList([
                nn.MultiheadAttention(d_model, cross_attn_heads, dropout=dropout, batch_first=True)
                for _ in range(n_ssm_layers)
            ])
            self.cross_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_ssm_layers)])

        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        self.class_bias = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

    def init_prototypes(self, emb_tensor: torch.Tensor, labels_tensor: torch.Tensor) -> None:
        """Initializes class prototypes as normalized class centroids in latent space.

        Summary:
            Calculates centroid vectors based on labeled training embeddings.

        Inputs:
            emb_tensor (torch.Tensor): Training embeddings matrix.
            labels_tensor (torch.Tensor): Multi-hot labels targets.

        Outputs:
            None.

        Shapes:
            - emb_tensor: `(N, d_input)`
            - labels_tensor: `(N, n_classes)`

        Side effects:
            Overwrites `self.prototypes` parameters in-place.

        Usage example:
            >>> model.init_prototypes(train_embs, train_labels)
        """
        with torch.no_grad():
            h = self.input_proj(emb_tensor)
            for c in range(self.n_classes):
                mask = labels_tensor[:, c] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[c] = F.normalize(h[mask].mean(0), dim=0)

    def forward(
        self,
        emb: torch.Tensor,
        perch_logits: Optional[torch.Tensor] = None,
        site_ids: Optional[torch.Tensor] = None,
        hours: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for computing class prediction similarities.

        Summary:
            Runs projections, bidirectional SSM sequences, and prototype mappings.

        Inputs:
            emb (torch.Tensor): Latent Perch embeddings.
            perch_logits (Optional[torch.Tensor]): Mapped Perch logits. Default is None.
            site_ids (Optional[torch.Tensor]): Site integer indices. Default is None.
            hours (Optional[torch.Tensor]): Recording hour indices. Default is None.

        Outputs:
            torch.Tensor: Computed multi-class probabilities or scores.

        Shapes:
            - Input `emb`: `(B, T, d_input)`
            - Input `perch_logits`: `(B, T, n_classes)`
            - Output: `(B, T, n_classes)`

        Side effects:
            None.

        Usage example:
            >>> y_pred = model(embs, logits, sites, hours)
        """
        B, T, _ = emb.shape
        h = self.input_proj(emb) + self.pos_enc[:, :T, :]

        if site_ids is not None and hours is not None:
            meta = self.meta_proj(torch.cat([self.site_emb(site_ids), self.hour_emb(hours)], dim=-1))
            h = h + meta[:, None, :]

        for i, (fwd, bwd, merge, norm) in enumerate(
            zip(self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm)
        ):
            res = h
            hf = fwd(h)
            hb = bwd(h.flip(1)).flip(1)
            h = self.drop(merge(torch.cat([hf, hb], dim=-1)))
            h = norm(h + res)

            if self.use_cross_attn:
                attn_out, _ = self.cross_attn[i](h, h, h)
                h = self.cross_norm[i](h + attn_out)

        h_n = F.normalize(h, dim=-1)
        p_n = F.normalize(self.prototypes, dim=-1)
        sim = torch.matmul(h_n, p_n.T) * F.softplus(self.proto_temp) + self.class_bias[None, None, :]

        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            out = alpha * sim + (1 - alpha) * perch_logits
        else:
            out = sim
        return out


class ResidualSSM(nn.Module):
    """Error-Correcting Second-Pass State Space Model.

    Summary:
        Learns to correct systematic errors in the first-pass ProtoSSM predictions.

    Attributes:
        n_classes (int): Number of target classes.
        input_proj (nn.Sequential): Input projection block.
        site_emb (nn.Embedding): Site embeddings.
        hour_emb (nn.Embedding): Hour embeddings.
        meta_proj (nn.Linear): Joint meta projection layer.
        pos_enc (nn.Parameter): Learnable position encodings.
        ssm_fwd (SelectiveSSM): Forward path SSM block.
        ssm_bwd (SelectiveSSM): Backward path SSM block.
        ssm_merge (nn.Linear): Merge output layer.
        ssm_norm (nn.LayerNorm): Layer normalizer.
        ssm_drop (nn.Dropout): Dropout layers.
        output_head (nn.Linear): Output logit projector.
    """

    def __init__(
        self,
        d_input: int = 1536,
        d_scores: int = 234,
        d_model: int = 64,
        d_state: int = 8,
        n_classes: int = 234,
        n_windows: int = 12,
        dropout: float = 0.1,
        n_sites: int = 20,
        meta_dim: int = 8,
    ):
        """Initializes ResidualSSM.

        Summary:
            Builds projections for concatenated embeddings + first-pass predictions.

        Inputs:
            d_input (int): Input features dimension. Default is 1536.
            d_scores (int): Number of target class logits. Default is 234.
            d_model (int): Hidden dimension size. Default is 64.
            d_state (int): SSM latent state size. Default is 8.
            n_classes (int): Target classes count. Default is 234.
            n_windows (int): Sequence length in windows. Default is 12.
            dropout (float): Dropout probability. Default is 0.1.
            n_sites (int): Sites vocabulary size. Default is 20.
            meta_dim (int): Site/hour embedding dimension. Default is 8.

        Outputs:
            ResidualSSM: An instance of ResidualSSM.

        Shapes:
            None.

        Side effects:
            Initializes output layer weights and biases to zero.

        Usage example:
            >>> res_model = ResidualSSM(n_classes=234)
        """
        super().__init__()
        self.n_classes = n_classes

        self.input_proj = nn.Sequential(
            nn.Linear(d_input + d_scores, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)

        self.ssm_fwd = SelectiveSSM(d_model, d_state)
        self.ssm_bwd = SelectiveSSM(d_model, d_state)
        self.ssm_merge = nn.Linear(2 * d_model, d_model)
        self.ssm_norm = nn.LayerNorm(d_model)
        self.ssm_drop = nn.Dropout(dropout)

        self.output_head = nn.Linear(d_model, n_classes)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        emb: torch.Tensor,
        first_pass: torch.Tensor,
        site_ids: Optional[torch.Tensor] = None,
        hours: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculates residual error-correction values.

        Summary:
            Fuses features, runs bidirectional SSM, and projects corrections.

        Inputs:
            emb (torch.Tensor): Original Perch embeddings.
            first_pass (torch.Tensor): First pass prediction logits.
            site_ids (Optional[torch.Tensor]): Recording site indices. Default is None.
            hours (Optional[torch.Tensor]): Recording hour indices. Default is None.

        Outputs:
            torch.Tensor: Residual correction logits.

        Shapes:
            - Input `emb`: `(B, T, d_input)`
            - Input `first_pass`: `(B, T, d_scores)`
            - Output: `(B, T, n_classes)`

        Side effects:
            None.

        Usage example:
            >>> corr = res_model(embs, y_first, sites, hours)
        """
        B, T, _ = emb.shape
        x = torch.cat([emb, first_pass], dim=-1)
        h = self.input_proj(x) + self.pos_enc[:, :T, :]

        if site_ids is not None and hours is not None:
            meta = self.meta_proj(
                torch.cat(
                    [
                        self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1)),
                        self.hour_emb(hours.clamp(0, 23)),
                    ],
                    dim=-1,
                )
            )
            h = h + meta.unsqueeze(1)

        res = h
        hf = self.ssm_fwd(h)
        hb = self.ssm_bwd(h.flip(1)).flip(1)
        h = self.ssm_drop(self.ssm_merge(torch.cat([hf, hb], dim=-1)))
        h = self.ssm_norm(h + res)

        return self.output_head(h)


class VectorizedMLPProbes(nn.Module):
    """Vectorized stacking layers for multi-class MLP probes.

    Summary:
        Runs parallelized, batched MLP forward passes for a group of classes sharing
        the same layer structure.

    Attributes:
        valid_classes (List[int]): Class indices in this probe group.
        n_layers (int): Number of fully connected layers.
        weights (nn.ParameterList): Packed weight tensors.
        biases (nn.ParameterList): Packed bias tensors.
    """

    def __init__(self, probe_models: Dict[int, Any]):
        """Initializes VectorizedMLPProbes.

        Summary:
            Stacks the coefficients (weights) and intercepts (biases) of sklearn
            MLPClassifiers into parallel PyTorch parameter tensors.

        Inputs:
            probe_models (Dict[int, Any]): Maps class index to fitted sklearn MLPClassifier.

        Outputs:
            VectorizedMLPProbes: An instance of VectorizedMLPProbes.

        Shapes:
            None.

        Side effects:
            None.

        Usage example:
            >>> vec_probes = VectorizedMLPProbes(group_models_dict)
        """
        super().__init__()
        self.valid_classes = sorted(probe_models.keys())
        V = len(self.valid_classes)

        if V == 0:
            self.weights = nn.ParameterList()
            self.biases = nn.ParameterList()
            self.n_layers = 0
            return

        sample = probe_models[self.valid_classes[0]]
        self.n_layers = len(sample.coefs_)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for li in range(self.n_layers):
            W = np.stack([probe_models[c].coefs_[li] for c in self.valid_classes], axis=0)
            b = np.stack([probe_models[c].intercepts_[li] for c in self.valid_classes], axis=0)
            self.weights.append(
                nn.Parameter(torch.tensor(W, dtype=torch.float32), requires_grad=False)
            )
            self.biases.append(
                nn.Parameter(torch.tensor(b, dtype=torch.float32), requires_grad=False)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs vectorized forward predictions for all classes in this group.

        Summary:
            Applies batch matrix multiplication (bmm) and activations.

        Inputs:
            x (torch.Tensor): Pre-stacked features tensor.

        Outputs:
            torch.Tensor: Parallel predictions of class probes.

        Shapes:
            - Input `x`: `(V, N, D_features)` where `V` is classes count, `N` is windows.
            - Output: `(V, N)` containing prediction values.

        Side effects:
            None.

        Usage example:
            >>> x = torch.randn(Vg, N, D_feat)
            >>> preds_g = vec_probes(x)
        """
        h = x
        for i in range(self.n_layers):
            h = torch.bmm(h, self.weights[i]) + self.biases[i].unsqueeze(1)
            if i < self.n_layers - 1:
                h = torch.relu(h)
        return h.squeeze(-1)


class EmbeddingRetrievalHead(nn.Module):
    """Retrieval-augmented classification head for rare and unmapped bird species.

    Summary:
        Computes cosine similarities between query embeddings and a stored database
        of training embeddings to retrieve label votes for unmapped or low-sample classes.

    Attributes:
        k (int): Number of nearest neighbors to retrieve.
        retrieval_weight (float): Blending factor weight.
        train_embeddings (torch.Tensor): Registered buffer for training embeddings.
        train_labels (torch.Tensor): Registered buffer for training labels.
    """

    def __init__(self, k: int = 5, retrieval_weight: float = 0.2):
        """Initializes the retrieval head.

        Summary:
            Instantiates hyperparameter variables and buffers.

        Inputs:
            k (int): Number of nearest neighbors to query. Default is 5.
            retrieval_weight (float): Blending ratio for retrieved predictions. Default is 0.2.

        Outputs:
            EmbeddingRetrievalHead: An instance of the retrieval head.

        Shapes:
            None.

        Side effects:
            Allocates device buffers.
        """
        super().__init__()
        self.k = k
        self.retrieval_weight = retrieval_weight
        self.register_buffer("train_embeddings", None)
        self.register_buffer("train_labels", None)

    def fit(self, train_embeddings: torch.Tensor, train_labels: torch.Tensor) -> None:
        """Stores normalized training embeddings and labels for retrieval.

        Summary:
            L2-normalizes the reference embeddings and registers them as buffers.

        Inputs:
            train_embeddings (torch.Tensor): Latent feature embeddings of training samples.
            train_labels (torch.Tensor): Multi-hot labels for training samples.

        Outputs:
            None.

        Shapes:
            - Input `train_embeddings`: `(N_train, D_emb)`
            - Input `train_labels`: `(N_train, N_classes)`

        Side effects:
            Stores buffers on the module's device.
        """
        self.train_embeddings = F.normalize(train_embeddings, p=2, dim=-1)
        self.train_labels = train_labels.float()

    def forward(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        """Retrieves and weights neighbor labels to produce query predictions.

        Summary:
            Computes query-to-database cosine similarity, finds top-K, and performs
            softmax-weighted label aggregation.

        Inputs:
            query_embeddings (torch.Tensor): Feature embeddings of query windows.

        Outputs:
            torch.Tensor: Retrieval-derived class probability votes.

        Shapes:
            - Input `query_embeddings`: `(B, D_emb)`
            - Output: `(B, N_classes)`

        Side effects:
            None.

        Usage example:
            >>> head = EmbeddingRetrievalHead(k=5)
            >>> head.fit(tr_emb, tr_labels)
            >>> probs = head(te_emb)
        """
        if self.train_embeddings is None or self.train_labels is None:
            # Return zeros if not fitted yet
            n_classes = self.train_labels.shape[1] if self.train_labels is not None else 234
            return torch.zeros(query_embeddings.shape[0], n_classes, device=query_embeddings.device)

        # L2 normalize query embeddings
        q_norm = F.normalize(query_embeddings, p=2, dim=-1)

        # Cosine similarity: (B, N_train)
        sims = torch.matmul(q_norm, self.train_embeddings.T)

        # Top-K neighbors
        k = min(self.k, sims.shape[1])
        top_k_values, top_k_indices = torch.topk(sims, k, dim=-1) # (B, k)

        # Softmax-weighted votes over neighbors
        weights = F.softmax(top_k_values, dim=-1) # (B, k)

        # Retrieve labels: (B, k, C)
        retrieved_labels = self.train_labels[top_k_indices] # (B, k, C)

        # Weighted sum of retrieved labels: (B, C)
        votes = torch.sum(retrieved_labels * weights.unsqueeze(-1), dim=1)
        return torch.clamp(votes, 0.0, 1.0)
