"""TeleportReweighter: Family 1 -- multiplicative DMD reweighting.

Convention:
    student_score, teacher_score (optional): [B, T, 1, H', W'], NON-NEGATIVE, DETACHED.
    output weight_map: [B, F, 1, H_lat, W_lat], values >= 1.0.

The reweighter spatially downsamples the score map to latent resolution via
``F.interpolate`` then applies the family formula.  It does NOT track gradients
(input is detached; output is used as a multiplicative weight in DMD loss).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F


class TeleportReweighter(ABC):
    """Abstract base for multiplicative DMD reweighting strategies.

    Contract:
        - Input score maps must be **already detached** by the caller (hook layer).
        - ``compute_weight`` returns a weight map ``>= 1.0`` at every position.
        - Output shape is ``[B, F, 1, H_lat, W_lat]`` after spatial interpolation.
    """

    def __init__(self, alpha: float = 5.0, latent_h: int = 40, latent_w: int = 72) -> None:
        self.alpha = alpha
        self.latent_h = latent_h
        self.latent_w = latent_w

    @abstractmethod
    def compute_weight(
        self,
        student_score: torch.Tensor,
        teacher_score: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-pixel multiplicative weight map.

        Args:
            student_score: ``[B, T, 1, H', W']``, non-negative, detached.
            teacher_score: ``[B, T, 1, H', W']`` or ``None``, detached.

        Returns:
            weight_map: ``[B, F, 1, H_lat, W_lat]``, all values ``>= 1.0``.
        """

    def _interpolate_to_latent(self, score_map: torch.Tensor) -> torch.Tensor:
        """Spatially downsample score map to latent resolution.

        Args:
            score_map: ``[B, T, 1, H', W']``.

        Returns:
            ``[B, T, 1, H_lat, W_lat]``.
        """
        B, T, C, _H, _W = score_map.shape
        flat = score_map.reshape(B * T, C, _H, _W)
        ds = F.interpolate(
            flat,
            size=(self.latent_h, self.latent_w),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )
        return ds.reshape(B, T, C, self.latent_h, self.latent_w)

    def _clamp_floor(self, weight_map: torch.Tensor) -> torch.Tensor:
        """Defensive floor: ensure all weights are >= 1.0."""
        return weight_map.clamp_min(1.0)


class TeacherRelativeReweighter(TeleportReweighter):
    """Weight proportional to student-teacher score gap.

    Formula: ``w = 1 + alpha * relu(student_score - teacher_score)``.

    Requires ``teacher_score`` input.  When student == teacher, weight = 1.0.
    When student > teacher, weight > 1.0 (teleport-prone regions up-weighted).
    When teacher > student, weight = 1.0 (relu floor).
    """

    def compute_weight(
        self,
        student_score: torch.Tensor,
        teacher_score: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if teacher_score is None:
            raise ValueError(
                "TeacherRelativeReweighter requires teacher_score; got None."
            )
        student = self._interpolate_to_latent(student_score)
        teacher = self._interpolate_to_latent(teacher_score)
        gap = torch.relu(student - teacher)
        weight = 1.0 + self.alpha * gap
        return self._clamp_floor(weight)


class AbsoluteReweighter(TeleportReweighter):
    """Weight proportional to absolute student score.

    Formula: ``w = 1 + alpha * student_score``.

    Does NOT accept ``teacher_score`` -- raises ``ValueError`` if provided.
    """

    def compute_weight(
        self,
        student_score: torch.Tensor,
        teacher_score: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if teacher_score is not None:
            raise ValueError(
                "AbsoluteReweighter does not accept teacher_score; "
                "received a non-None value."
            )
        student = self._interpolate_to_latent(student_score)
        weight = 1.0 + self.alpha * student
        return self._clamp_floor(weight)


class ThresholdedReweighter(TeleportReweighter):
    """Weight proportional to student score above a hard threshold.

    Formula: ``w = 1 + alpha * relu(student_score - tau)``.

    ``tau`` must be provided via config (no default).  Does NOT accept
    ``teacher_score``.
    """

    def __init__(
        self, alpha: float = 5.0, latent_h: int = 40, latent_w: int = 72, tau: float = 0.0
    ) -> None:
        super().__init__(alpha=alpha, latent_h=latent_h, latent_w=latent_w)
        self.tau = tau

    def compute_weight(
        self,
        student_score: torch.Tensor,
        teacher_score: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if teacher_score is not None:
            raise ValueError(
                "ThresholdedReweighter does not accept teacher_score; "
                "received a non-None value."
            )
        student = self._interpolate_to_latent(student_score)
        above = torch.relu(student - self.tau)
        weight = 1.0 + self.alpha * above
        return self._clamp_floor(weight)


def build_reweighter(cfg: Optional[dict]) -> Optional[TeleportReweighter]:
    """Factory: create a ``TeleportReweighter`` instance from a config dict.

    Parameters:
        cfg:
            Dict with keys ``type`` (str), optionally ``alpha``, ``latent_h``,
            ``latent_w``, and ``tau`` (for thresholded).

            Special values that return ``None``:
            - ``cfg is None``
            - ``cfg == {}`` (empty dict)
            - ``cfg["type"] == "disabled"``

    Returns:
        A concrete ``TeleportReweighter`` instance, or ``None`` when disabled.

    Raises:
        ValueError: If ``cfg["type"]`` is not recognized or required key missing.
    """
    if cfg is None or cfg == {}:
        return None

    rw_type = cfg.get("type", "disabled")
    if rw_type == "disabled":
        return None

    alpha = cfg.get("alpha", 5.0)
    latent_h = cfg.get("latent_h", 40)
    latent_w = cfg.get("latent_w", 72)

    if rw_type == "teacher_relative":
        return TeacherRelativeReweighter(
            alpha=alpha, latent_h=latent_h, latent_w=latent_w
        )

    if rw_type == "absolute":
        return AbsoluteReweighter(
            alpha=alpha, latent_h=latent_h, latent_w=latent_w
        )

    if rw_type == "thresholded":
        if "tau" not in cfg:
            raise ValueError(
                "thresholded reweighter requires 'tau' in config; got missing."
            )
        return ThresholdedReweighter(
            alpha=alpha, latent_h=latent_h, latent_w=latent_w, tau=cfg["tau"]
        )

    raise ValueError(
        f"Unknown teleport.reweighter.type: {rw_type!r}. "
        "Supported: ['teacher_relative', 'absolute', 'thresholded', 'disabled']"
    )
