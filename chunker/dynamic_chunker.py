"""
DynamicChunker: content-adaptive soft chunker for HD-RMT Stage 1.

Takes token embeddings x: [B, L, D] and produces:
  chunks:          [B, K, D]  — K pooled chunk representations
  boundary_scores: [B, L]     — per-position boundary probability

K (n_chunks) is fixed at construction time. Boundary *positions* are learned
end-to-end from task loss via the soft assignment matrix W. No boundary
supervision — gradients flow entirely through the downstream task.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class DynamicChunker(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_chunks: int,
        d_hidden: Optional[int] = None,
        sigma: float = 1.0,
        encoder_type: str = "conv",
    ):
        """
        Args:
            d_model:      input embedding dimension D
            n_chunks:     number of output chunks K (fixed)
            d_hidden:     boundary encoder hidden dim; defaults to D // 4
            sigma:        soft assignment temperature; smaller = sharper boundaries
            encoder_type: "conv" (depthwise-separable, preferred) or "linear" (ablation)
        """
        super().__init__()
        self.n_chunks = n_chunks
        self.sigma = sigma
        self.eps = 1e-8

        if d_hidden is None:
            d_hidden = max(d_model // 4, 1)
        self.d_hidden = d_hidden
        self.encoder_type = encoder_type

        if encoder_type == "conv":
            # Depthwise conv captures local 3-token context cheaply.
            # groups=d_model gives true depthwise (one filter per channel),
            # then pointwise 1x1 projects to d_hidden.
            self.boundary_encoder = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
                nn.GELU(),
                nn.Conv1d(d_model, d_hidden, kernel_size=1),
            )
        elif encoder_type == "linear":
            # Per-token linear: ablation baseline, no local context.
            self.boundary_encoder = nn.Linear(d_model, d_hidden)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type!r}. Use 'conv' or 'linear'.")

        self.boundary_proj = nn.Linear(d_hidden, 1)

    def forward(
        self,
        x: torch.Tensor,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    [B, L, D] token embeddings
            hard: use argmax hard boundaries (for inference); default False (soft)

        Returns:
            chunks:          [B, K, D]
            boundary_scores: [B, L]   raw sigmoid scores, not supervised
        """
        B, L, D = x.shape
        K = self.n_chunks

        # 1. boundary scoring
        if self.encoder_type == "conv":
            # Conv1d expects [B, C, L]
            h = self.boundary_encoder(x.transpose(1, 2)).transpose(1, 2)  # [B, L, d_hidden]
        else:
            h = self.boundary_encoder(x)  # [B, L, d_hidden]

        b = torch.sigmoid(self.boundary_proj(h))  # [B, L, 1]

        if hard:
            return self._hard_forward(x, b)

        # 2. soft chunk assignment via cumulative boundary mass
        # Normalize so that sum of b_norm ≈ K, giving each token a position in [0, K].
        b_norm = b * (K / (b.sum(dim=1, keepdim=True) + self.eps))  # [B, L, 1]
        cumsum = b_norm.cumsum(dim=1)  # [B, L, 1]  values in ~[0, K]

        # Soft assignment: token i → chunk j weight = Gaussian centered at j+0.5
        j_centers = torch.arange(K, device=x.device).float() + 0.5  # [K]
        W = torch.exp(
            -0.5 * ((cumsum - j_centers.view(1, 1, K)) / self.sigma) ** 2
        )  # [B, L, K]
        W = W / (W.sum(dim=1, keepdim=True) + self.eps)  # normalize over tokens → [B, L, K]

        # 3. weighted pool: each chunk = weighted average of token embeddings
        chunks = torch.einsum("blk,bld->bkd", W, x)  # [B, K, D]

        return chunks, b.squeeze(-1)  # [B, K, D], [B, L]

    def _hard_forward(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hard boundary inference: top-(K-1) boundary positions split x into K
        non-overlapping segments; each segment is mean-pooled to one vector.
        Not differentiable — use only at inference.
        """
        B, L, D = x.shape
        K = self.n_chunks
        boundary_scores = b.squeeze(-1)  # [B, L]

        # Pick K-1 highest-scoring positions as boundaries
        hard_boundaries = boundary_scores.topk(K - 1, dim=1).indices  # [B, K-1]
        hard_boundaries, _ = hard_boundaries.sort(dim=1)  # ascending

        chunks_list = []
        for i in range(B):
            splits = [0] + hard_boundaries[i].tolist() + [L]
            segs = []
            for j in range(K):
                start, end = splits[j], splits[j + 1]
                end = max(end, start + 1)  # guarantee non-empty
                segs.append(x[i, start:end].mean(dim=0))
            chunks_list.append(torch.stack(segs, dim=0))

        chunks = torch.stack(chunks_list, dim=0)  # [B, K, D]
        return chunks, boundary_scores
