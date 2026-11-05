"""
Models for weakly-supervised video anomaly detection.

``SultaniMIL``  the CVPR'18 baseline: a 3-layer MLP scoring each segment
                independently, trained with a ranking loss between the highest
                scoring segment of an anomalous bag and that of a normal bag.
                No temporal modelling at all -- this is the floor that any
                temporal contribution must clear.

``mil_ranking_loss``  the reference objective, including its sparsity and
                temporal-smoothness regularisers.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SultaniMIL(nn.Module):
    """
    Sultani et al. (CVPR 2018) scorer: Linear(D,512) -> Linear(512,32) -> Linear(32,1).

    Dropout of 0.6 is unusually strong but is what the reference uses; it matters,
    because the model is tiny and the supervision is weak.
    """

    def __init__(self, in_dim: int, dropout: float = 0.6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, 32), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [B,T,D] -> anomaly score per snippet [B,T] in (0,1)."""
        s = torch.sigmoid(self.net(x)).squeeze(-1)
        if mask is not None:
            s = s.masked_fill(~mask, 0.0)
        return s


def _masked_max(scores: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return scores.max(dim=1).values
    return scores.masked_fill(~mask, float("-inf")).max(dim=1).values


def mil_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    lambda_smooth: float = 8e-5,
    lambda_sparse: float = 8e-5,
) -> Tuple[torch.Tensor, dict]:
    """
    Reference MIL objective.

    For every (anomalous, normal) bag pair in the batch:

        hinge = max(0, 1 - max_t s_anom[t] + max_t s_norm[t])

    plus, over anomalous bags only, a temporal-smoothness term
    ``sum_t (s[t] - s[t+1])^2`` and a sparsity term ``sum_t s[t]``. The
    regularisers encode the priors that anomalies are temporally contiguous and
    rare; without them the ranking loss alone is trivially satisfiable.

    Args:
        scores: [B,T] snippet scores in (0,1)
        labels: [B] video-level labels, 1 = anomalous
        mask:   [B,T] True where a snippet is real (for padded variable-length bags)
    """
    pos = labels > 0.5
    neg = ~pos
    if not pos.any() or not neg.any():
        raise ValueError("batch must contain both anomalous and normal bags")

    max_scores = _masked_max(scores, mask)
    # All anomalous-vs-normal pairs in the batch.
    hinge = F.relu(1.0 - max_scores[pos].unsqueeze(1) + max_scores[neg].unsqueeze(0))
    rank = hinge.mean()

    s_pos = scores[pos]
    if mask is not None:
        m_pos = mask[pos]
        diff = (s_pos[:, 1:] - s_pos[:, :-1]) ** 2
        pair_valid = (m_pos[:, 1:] & m_pos[:, :-1]).float()
        smooth = (diff * pair_valid).sum(dim=1).mean()
        sparse = (s_pos * m_pos.float()).sum(dim=1).mean()
    else:
        smooth = ((s_pos[:, 1:] - s_pos[:, :-1]) ** 2).sum(dim=1).mean()
        sparse = s_pos.sum(dim=1).mean()

    total = rank + lambda_smooth * smooth + lambda_sparse * sparse
    return total, {"rank": float(rank.detach()), "smooth": float(smooth.detach()),
                   "sparse": float(sparse.detach()), "total": float(total.detach())}
