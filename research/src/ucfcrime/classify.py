"""
Anomaly-category classifier.

The detector answers "is this anomalous". This answers "of what kind", over the
13 UCF-Crime categories. It is a separate head trained on the anomalous training
videos only, because normal videos carry no category.

Supervision is video-level: a label states that a category occurs somewhere in
the video, not when. The classifier is therefore trained on an attention-pooled
video descriptor rather than per snippet, and its snippet-level output should be
read as "the category of this video's anomaly", not "the category of this
moment".

Two properties of the split limit achievable accuracy and are reported with the
results rather than hidden:
  * the training and test category balances differ sharply (Robbery 145/5,
    Shooting 27/23), so the classes best learned are the least tested;
  * several categories are visually similar under surveillance conditions
    (Robbery/Stealing/Burglary), and the label distinguishes intent, which is
    not always visible.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .splits import ANOMALY_CLASSES


class CategoryHead(nn.Module):
    """Attention-pooled video descriptor -> 13-way category logits."""

    def __init__(self, in_dim: int, d_model: int = 256, dropout: float = 0.5,
                 n_classes: int = len(ANOMALY_CLASSES)):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model),
            nn.ReLU(inplace=True), nn.Dropout(dropout))
        # Attention pooling rather than a mean: an event occupies a median 14% of
        # its video, so a mean over all snippets is dominated by ordinary footage.
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))
        self.out = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B,T,D]; mask: [B,T]; weights: optional [B,T] detector scores used to
        bias pooling toward snippets the detector considers anomalous.
        """
        h = self.embed(x)
        a = self.attn(h).squeeze(-1)
        if weights is not None:
            # log-space bias: snippets the detector scored highly get more mass.
            a = a + torch.log(weights.clamp(min=1e-4))
        if mask is not None:
            a = a.masked_fill(~mask, float("-inf"))
        a = torch.softmax(a, dim=1)
        pooled = torch.bmm(a.unsqueeze(1), h).squeeze(1)
        return self.out(pooled)


def class_weights(counts: List[int], device=None) -> torch.Tensor:
    """Inverse-frequency weights; the training split is imbalanced 145:27."""
    c = torch.tensor(counts, dtype=torch.float32, device=device)
    w = c.sum() / (len(c) * c.clamp(min=1))
    return w / w.mean()


@torch.no_grad()
def predict(head: CategoryHead, feats: torch.Tensor,
            weights: Optional[torch.Tensor] = None, top_k: int = 3
            ) -> List[Tuple[str, float]]:
    """Return the top-k (category, probability) for one video's features [T,D]."""
    head.eval()
    logits = head(feats.unsqueeze(0), weights=None if weights is None else weights.unsqueeze(0))
    p = torch.softmax(logits, dim=1)[0]
    v, i = p.topk(min(top_k, p.numel()))
    return [(ANOMALY_CLASSES[int(j)], float(s)) for s, j in zip(v, i)]
