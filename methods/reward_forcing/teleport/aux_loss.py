"""TeleportAuxLoss: Family 2 - additive penalty term for DMD loss.

Convention:
    score_map: [B, T, 1, H', W'], non-negative, GRAD-ATTACHED.
    output: scalar tensor, grad-able, non-negative.

Unlike the reweighter (which receives detached input), aux_loss expects
gradient flow: detector -> aux_loss -> backward propagates to the student
through the VAE decode path.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch


class TeleportAuxLoss(ABC):
    """Abstract base for additive teleport penalty terms (Family 2).

    Concrete subclasses implement a scalar loss that is ADDED to the
    primary DMD loss.  The score_map is expected to carry gradient.
    """

    @abstractmethod
    def __call__(self, score_map: torch.Tensor) -> torch.Tensor:
        """Compute a scalar aux loss from the score map.

        Args:
            score_map: [B, T, 1, H', W'], non-negative, grad-attached.

        Returns:
            Scalar tensor with gradient path to score_map.
        """


class MaskedMeanAuxLoss(TeleportAuxLoss):
    """Masked-mean teleport penalty.

    Computes the mean score over pixels whose score exceeds a threshold
    ``tau``, so the loss penalises only "high teleport suspicion" regions::

        mask = (score_map > tau).to(score_map.dtype)
        L_aux = (score_map * mask).sum() / mask.sum().clamp_min(1.0)

    When ``aggregation="mean"`` the mask is skipped and the plain mean
    ``score_map.mean()`` is returned instead.
    """

    def __init__(self, tau: float, aggregation: str = "masked_mean") -> None:
        if tau is None:
            raise ValueError("MaskedMeanAuxLoss requires explicit tau; got None")
        if aggregation not in {"masked_mean", "mean"}:
            raise ValueError(
                f"aggregation must be one of {{'masked_mean','mean'}}, "
                f"got {aggregation!r}"
            )
        self._tau = float(tau)
        self._aggregation = aggregation

    def __call__(self, score_map: torch.Tensor) -> torch.Tensor:
        if self._aggregation == "mean":
            return score_map.mean()

        # masked_mean path
        mask = (score_map > self._tau).to(score_map.dtype)
        denom = mask.sum().clamp_min(1.0)
        masked_sum = (score_map * mask).sum()
        # ``score_map.sum() * 0.0`` ensures the output is grad-attached
        # to score_map even when mask is all zeros (no teleport pixels).
        return masked_sum / denom + score_map.sum() * 0.0


def build_aux_loss(cfg: Optional[dict]) -> Optional[TeleportAuxLoss]:
    """Factory: create a ``TeleportAuxLoss`` instance from a config dict.

    Parameters:
        cfg:
            Dict with at least key ``type`` (str).  Additional keys are
            forwarded to the concrete constructor.

            Special values that return ``None``:
            - ``cfg is None``
            - ``cfg == {}`` (empty dict)
            - ``cfg["type"] == "disabled"``

    Returns:
        A concrete ``TeleportAuxLoss`` instance, or ``None`` when disabled.

    Raises:
        ValueError: If ``cfg["type"]`` is not recognised, or required keys
                    (e.g. ``tau`` for ``masked_mean``) are missing.
    """
    if cfg is None or cfg == {}:
        return None

    loss_type = cfg.get("type", "disabled")
    if loss_type == "disabled":
        return None

    if loss_type == "masked_mean":
        tau = cfg.get("tau")
        if tau is None:
            raise ValueError("masked_mean aux_loss requires explicit tau")
        return MaskedMeanAuxLoss(
            tau=float(tau),
            aggregation=cfg.get("aggregation", "masked_mean"),
        )

    raise ValueError(
        f"Unknown teleport.aux_loss.type: {loss_type!r}. "
        "Supported: ['masked_mean', 'disabled']"
    )
