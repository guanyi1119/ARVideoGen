"""TeleportDetector: differentiable pixel-level score map for center-emergence
events in generated video.

Used by tel_det_regular:
- metric: aggregate score map → teleport rate per video (T7)
- D-rev aux_loss: per-pixel weight into DMD loss (T8b / T9)

Differentiable contract: all internal ops must support gradient flow so the
aux_loss path can backprop through detector → VAE → student.  Callers that
do NOT want gradients (e.g. reweight mode, metric eval) should ``.detach()``
the score map at the call site.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F


class TeleportDetector(ABC):
    """Pixel-level teleport score map detector.

    Convention:
        input  ``rgb`` shape ``[B, T, 3, H, W]``, values in **[0, 1]**
        output ``score_map`` shape ``[B, T, 1, H', W']``, non-negative.
                H'/W' may be smaller than H/W (internal downsampling).
                ``score_map[:, 0]`` is always zero (no prior frame).
        Differentiable: ``score(rgb_with_grad).sum().backward()`` must work.
    """

    @abstractmethod
    def score(self, rgb: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def is_available(self) -> bool:
        """Whether all internal backends have been successfully initialized."""


class OpticalFlowTeleportDetector(TeleportDetector):
    """Detector combining frame-diff + (optional) optical-flow residual + edge IoU.

    Score formula:
        score = alpha_diff * frame_diff + alpha_flow * flow_anomaly + alpha_edge * (1 - edge_proximity)
    Each component is in [0, 1] after normalization.
    """

    def __init__(
        self,
        downsample_factor: int = 4,
        flow_backend: str = "raft_small",       # | "none"
        alpha_diff: float = 0.3,
        alpha_flow: float = 0.5,
        alpha_edge: float = 0.2,
        edge_sigma: float = 0.15,               # Gaussian falloff width for edge mask
        flow_model_path: Optional[str] = None,  # for raft_small lazy load
    ) -> None:
        super().__init__()
        self._downsample_factor = downsample_factor
        self._flow_backend = flow_backend
        self._alpha_diff = alpha_diff
        self._alpha_flow = alpha_flow
        self._alpha_edge = alpha_edge
        self._edge_sigma = edge_sigma
        self._flow_model_path = flow_model_path
        self._flow_model: Optional[torch.nn.Module] = None
        self._centeredness_cache: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Whether all internal backends have been successfully initialized."""
        if self._flow_backend == "none":
            return True
        return self._flow_model is not None

    def score(self, rgb: torch.Tensor) -> torch.Tensor:
        """Compute score map.  See class docstring for shape contract.

        Args:
            rgb: [B, T, 3, H, W], values in [0, 1].

        Returns:
            score_map: [B, T, 1, H', W'], non-negative.  First frame is always zero.
        """
        if rgb.dim() != 5:
            raise ValueError(
                f"Expected rgb shape [B, T, 3, H, W], got {tuple(rgb.shape)}"
            )
        B, T, C, H, W = rgb.shape
        if C != 3:
            raise ValueError(f"Expected 3 channels, got {C}")

        # --- Downsample ---
        if self._downsample_factor > 1:
            # [B, T, 3, H, W] → [B*T, 3, H, W] for interpolate
            rgb_flat = rgb.reshape(B * T, C, H, W)
            rgb_ds = F.interpolate(
                rgb_flat,
                scale_factor=1.0 / self._downsample_factor,
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )
            _, _, H_ds, W_ds = rgb_ds.shape
            rgb = rgb_ds.reshape(B, T, C, H_ds, W_ds)
        else:
            H_ds, W_ds = H, W

        # --- Frame diff score ---
        fd = self._compute_frame_diff(rgb)  # [B, T, H_ds, W_ds]

        # --- Flow anomaly score ---
        fa = self._compute_flow_anomaly(rgb)  # [B, T, H_ds, W_ds]

        # --- Centeredness mask ---
        cent = self._compute_centeredness(
            H_ds, W_ds, device=rgb.device, dtype=rgb.dtype
        )  # [1, 1, H_ds, W_ds]

        # --- Weighted sum ---
        score_map = (
            self._alpha_diff * fd
            + self._alpha_flow * fa
            + self._alpha_edge * cent
        )  # [B, T, H_ds, W_ds]

        # Add channel dim → [B, T, 1, H_ds, W_ds]
        score_map = score_map.unsqueeze(2)

        # Clamp to non-negative (defensive)
        score_map = score_map.clamp(min=0.0)

        # First frame is always zero (no prior frame to compare against)
        score_map[:, 0] = 0.0

        return score_map

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_flow_model(self) -> None:
        """Lazy-load the RAFT-small optical flow model."""
        if self._flow_model is not None:
            return
        if self._flow_backend != "raft_small":
            return

        from torchvision.models.optical_flow import (
            Raft_Small_Weights,
            raft_small,
        )

        if self._flow_model_path is not None:
            model = raft_small(weights=None)
            state = torch.load(
                self._flow_model_path, map_location="cpu", weights_only=True
            )
            model.load_state_dict(state)
        else:
            model = raft_small(weights=Raft_Small_Weights.DEFAULT)

        # Move model to the same device as input tensors will be on.
        # When transfer_to_npu is active, "cuda" maps to NPU.
        self._flow_model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    def _compute_frame_diff(self, rgb: torch.Tensor) -> torch.Tensor:
        """Compute per-frame channel-mean absolute difference.

        Args:
            rgb: [B, T, 3, H, W], values in [0, 1].

        Returns:
            frame_diff: [B, T, H, W], t=0 is zero.
        """
        B, T, _C, H, W = rgb.shape
        if T <= 1:
            return torch.zeros(B, T, H, W, device=rgb.device, dtype=rgb.dtype)

        # |x_t - x_{t-1}| → [B, T-1, 3, H, W]
        diff = (rgb[:, 1:] - rgb[:, :-1]).abs()

        # Mean over channel → [B, T-1, H, W]
        diff = diff.mean(dim=2)

        # Pad t=0 with zeros
        pad = torch.zeros(B, 1, H, W, device=rgb.device, dtype=rgb.dtype)
        return torch.cat([pad, diff], dim=1)

    def _compute_flow_anomaly(self, rgb: torch.Tensor) -> torch.Tensor:
        """Compute flow-based anomaly score.

        For each consecutive frame pair, compute RAFT flow from t-1 to t,
        warp frame t-1 to t, and compute residual.

        Args:
            rgb: [B, T, 3, H, W], values in [0, 1].

        Returns:
            anomaly: [B, T, H, W], t=0 is zero.
        """
        B, T, _C, H, W = rgb.shape
        if T <= 1 or self._flow_backend == "none":
            return torch.zeros(B, T, H, W, device=rgb.device, dtype=rgb.dtype)

        self._load_flow_model()

        # RAFT internally downsamples by 8x and requires feature maps >= 16x16.
        # If our input is too small (H/8 < 16 or W/8 < 16), upsample to the
        # minimum size before feeding to RAFT, then downsample results back.
        raft_min = 128  # 8 * 16
        need_upsample = H < raft_min or W < raft_min
        if need_upsample:
            scale_h = max(raft_min / H, 1.0)
            scale_w = max(raft_min / W, 1.0)
            scale = max(scale_h, scale_w)
            new_h = int(H * scale)
            new_w = int(W * scale)
            rgb_raft = F.interpolate(
                rgb.reshape(B * T, _C, H, W),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, T, _C, new_h, new_w)
        else:
            rgb_raft = rgb
            new_h, new_w = H, W

        results: list[torch.Tensor] = []
        for t in range(T):
            if t == 0:
                results.append(
                    torch.zeros(B, H, W, device=rgb.device, dtype=rgb.dtype)
                )
                continue

            frame_prev = rgb_raft[:, t - 1]  # [B, 3, new_h, new_w]
            frame_curr = rgb_raft[:, t]       # [B, 3, new_h, new_w]

            # RAFT expects values in [0, 1]
            flow = self._flow_model(frame_prev, frame_curr)
            # flow may be a list of [B, 2, H, W] at different scales
            if isinstance(flow, list):
                flow = flow[-1]  # take finest scale

            # Warp prev frame using flow
            warped = self._warp_frame(frame_prev, flow)

            # Residual
            residual = (frame_curr - warped).abs().mean(dim=1)  # [B, new_h, new_w]

            # Downsample residual back to original H, W if we upsampled
            if need_upsample:
                residual = F.interpolate(
                    residual.unsqueeze(1), size=(H, W), mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

            results.append(residual)

        return torch.stack(results, dim=1)  # [B, T, H, W]

    @staticmethod
    def _warp_frame(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp frame using optical flow via grid_sample.

        Args:
            frame: [B, C, H, W], the source frame to warp.
            flow:  [B, 2, H, W], optical flow in pixel units (dx, dy).

        Returns:
            warped: [B, C, H, W], frame warped by flow.
        """
        B, _C, H, W = frame.shape

        # Create identity grid in [-1, 1] normalized coordinates
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=frame.device, dtype=frame.dtype),
            torch.linspace(-1, 1, W, device=frame.device, dtype=frame.dtype),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

        # Normalize flow from pixel units to [-1, 1]
        flow_perm = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
        flow_norm = flow_perm.clone()
        flow_norm[..., 0] = flow_norm[..., 0] / (W - 1) * 2.0
        flow_norm[..., 1] = flow_norm[..., 1] / (H - 1) * 2.0

        # Warp: grid + flow samples from source (prev frame) to reconstruct target
        warped = F.grid_sample(
            frame,
            grid + flow_norm,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return warped

    def _compute_centeredness(
        self, H: int, W: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Compute centeredness mask [1, 1, H, W] with Gaussian falloff from edges.

        Returns values in [0, 1] where 0 = edge, 1 = center.
        Mask is cached by (H, W, device) for reuse across forward calls.
        """
        key = (H, W, device)
        if key in self._centeredness_cache:
            return self._centeredness_cache[key]

        # Distance to nearest edge (in pixels)
        y_coords = torch.arange(H, device=device, dtype=dtype)
        x_coords = torch.arange(W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        dist_top = yy
        dist_bottom = (H - 1) - yy
        dist_left = xx
        dist_right = (W - 1) - xx

        dist_edge = torch.min(
            torch.min(dist_top, dist_bottom),
            torch.min(dist_left, dist_right),
        )  # [H, W]

        # Normalize to [0, 1] (0 = edge, 1 = center)
        half_min = min(H, W) / 2.0
        edge_proximity = dist_edge / half_min  # [H, W], in [0, 1]

        # Gaussian falloff: centeredness = 1 - exp(-edge_proximity^2 / (2 * sigma^2))
        centeredness = 1.0 - torch.exp(
            -(edge_proximity ** 2) / (2.0 * self._edge_sigma ** 2)
        )

        # Add batch and channel dims → [1, 1, H, W]
        centeredness = centeredness.unsqueeze(0).unsqueeze(0)
        self._centeredness_cache[key] = centeredness
        return centeredness


def build_teleport_detector(cfg: Optional[dict]) -> Optional[TeleportDetector]:
    """Factory: create a ``TeleportDetector`` instance from a config dict.

    Parameters:
        cfg:
            Dict with keys ``type`` (str), and optionally ``downsample_factor``,
            ``flow_backend``, ``alpha_diff``, ``alpha_flow``, ``alpha_edge``,
            ``edge_sigma``, ``flow_model_path``.

            Special values that return ``None``:
            - ``cfg is None``
            - ``cfg == {}`` (empty dict)
            - ``cfg["type"] == "disabled"``

    Returns:
        A concrete ``TeleportDetector`` instance, or ``None`` when disabled.

    Raises:
        ValueError: If ``cfg["type"]`` is not recognized.
    """
    if cfg is None or cfg == {}:
        return None

    det_type = cfg.get("type", "disabled")
    if det_type == "disabled":
        return None

    if det_type == "optical_flow_no_raft":
        return OpticalFlowTeleportDetector(
            downsample_factor=cfg.get("downsample_factor", 4),
            flow_backend="none",
            alpha_diff=cfg.get("alpha_diff", 0.3),
            alpha_flow=cfg.get("alpha_flow", 0.5),
            alpha_edge=cfg.get("alpha_edge", 0.2),
            edge_sigma=cfg.get("edge_sigma", 0.15),
        )

    if det_type == "optical_flow":
        return OpticalFlowTeleportDetector(
            downsample_factor=cfg.get("downsample_factor", 4),
            flow_backend=cfg.get("flow_backend", "raft_small"),
            alpha_diff=cfg.get("alpha_diff", 0.3),
            alpha_flow=cfg.get("alpha_flow", 0.5),
            alpha_edge=cfg.get("alpha_edge", 0.2),
            edge_sigma=cfg.get("edge_sigma", 0.15),
            flow_model_path=cfg.get("flow_model_path"),
        )

    raise ValueError(
        f"Unknown teleport.detector.type: {det_type!r}. "
        "Supported: ['optical_flow', 'optical_flow_no_raft', 'disabled']"
    )
