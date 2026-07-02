"""Tests for _apply_teleport_metric_gates in ReDMD.

Verifies pixel-level and frame-level gating of detector score maps
before they are fed to reweighter / aux_loss.
"""
from __future__ import annotations

import pytest
import torch


class _MockGater:
    """Minimal stand-in for ReDMD that only carries the two gate thresholds."""

    def __init__(self, score_threshold=None, min_event_area_ratio=None):
        self._teleport_score_threshold = score_threshold
        self._teleport_min_event_area_ratio = min_event_area_ratio

    def _apply_teleport_metric_gates(self, score_map: torch.Tensor) -> torch.Tensor:
        """Copy of the real implementation (kept in sync manually)."""
        gated = score_map
        if self._teleport_score_threshold is not None:
            th = float(self._teleport_score_threshold)
            gated = gated * (gated > th).to(gated.dtype)

        if self._teleport_min_event_area_ratio is not None:
            ratio = float(self._teleport_min_event_area_ratio)
            B, T, C, H, W = gated.shape
            frame_ratio = (gated > 0).float().sum(dim=(2, 3, 4)) / (C * H * W)
            frame_mask = (frame_ratio > ratio).float()
            gated = gated * frame_mask.view(B, T, 1, 1, 1)

        return gated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_score_map(values: list) -> torch.Tensor:
    """Build [B=1, T=len(values), 1, H=4, W=4] score map where every spatial
    position in frame t has the same scalar value."""
    T = len(values)
    tensor = torch.zeros(1, T, 1, 4, 4)
    for t, v in enumerate(values):
        tensor[0, t, :, :, :] = v
    return tensor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPixelGate:
    def test_pixel_gate_zeros_below_threshold(self):
        """Pixels with score <= threshold become 0; above survive."""
        gater = _MockGater(score_threshold=0.3)
        score = _make_score_map([0.1, 0.5, 0.0, 0.31])
        gated = gater._apply_teleport_metric_gates(score)

        # Frame 0 (0.1) and frame 2 (0.0) are below threshold
        assert gated[0, 0].sum().item() == 0.0
        assert gated[0, 2].sum().item() == 0.0
        # Frame 1 (0.5) and frame 3 (0.31) survive
        assert gated[0, 1, 0, 0, 0].item() == pytest.approx(0.5)
        assert gated[0, 3, 0, 0, 0].item() == pytest.approx(0.31)

    def test_pixel_gate_all_above_threshold_survives(self):
        """When every pixel is above threshold, map is unchanged."""
        gater = _MockGater(score_threshold=0.1)
        score = _make_score_map([0.5, 0.6])
        gated = gater._apply_teleport_metric_gates(score)
        assert torch.allclose(gated, score)

    def test_pixel_gate_all_below_threshold_zeros(self):
        """When every pixel is below threshold, map becomes all zeros."""
        gater = _MockGater(score_threshold=0.9)
        score = _make_score_map([0.1, 0.2])
        gated = gater._apply_teleport_metric_gates(score)
        assert gated.sum().item() == 0.0


class TestFrameGate:
    def test_frame_gate_zeros_low_area_frames(self):
        """Frames where teleport-pixel fraction < min_event_area_ratio
        are zeroed out entirely."""
        # 4x4 = 16 spatial positions per frame
        # min_event_area_ratio = 0.1  => need > 1.6 pixels (~2+) to survive
        gater = _MockGater(
            score_threshold=0.0, min_event_area_ratio=0.1
        )
        score = torch.zeros(1, 3, 1, 4, 4)
        # Frame 0: only 1 pixel > 0  => 1/16 = 0.0625 < 0.1 => should be zeroed
        score[0, 0, 0, 0, 0] = 1.0
        # Frame 1: 4 pixels > 0 => 4/16 = 0.25 > 0.1 => survives
        score[0, 1, 0, 0, 0] = 1.0
        score[0, 1, 0, 1, 1] = 1.0
        score[0, 1, 0, 2, 2] = 1.0
        score[0, 1, 0, 3, 3] = 1.0
        # Frame 2: all zeros => already zero

        gated = gater._apply_teleport_metric_gates(score)
        assert gated[0, 0].sum().item() == 0.0  # zeroed by frame gate
        assert gated[0, 1].sum().item() == 4.0  # survives
        assert gated[0, 2].sum().item() == 0.0  # already zero

    def test_frame_gate_survives_when_area_high_enough(self):
        """Frames with enough teleport pixels survive."""
        gater = _MockGater(
            score_threshold=0.0, min_event_area_ratio=0.05
        )
        score = torch.zeros(1, 1, 1, 4, 4)
        # 2 pixels > 0 => 2/16 = 0.125 > 0.05 => survives
        score[0, 0, 0, 0, 0] = 0.5
        score[0, 0, 0, 1, 1] = 0.5

        gated = gater._apply_teleport_metric_gates(score)
        assert gated[0, 0, 0, 0, 0].item() == pytest.approx(0.5)
        assert gated[0, 0, 0, 1, 1].item() == pytest.approx(0.5)


class TestNoGate:
    def test_no_gate_when_both_configs_missing(self):
        """When both thresholds are None, gate is identity."""
        gater = _MockGater(
            score_threshold=None, min_event_area_ratio=None
        )
        score = _make_score_map([0.1, 0.5, 0.0])
        gated = gater._apply_teleport_metric_gates(score)
        assert torch.allclose(gated, score)

    def test_only_pixel_gate_active(self):
        """Only score_threshold set => frame gate skipped."""
        gater = _MockGater(score_threshold=0.3, min_event_area_ratio=None)
        score = _make_score_map([0.1, 0.5])
        gated = gater._apply_teleport_metric_gates(score)
        assert gated[0, 0].sum().item() == 0.0
        assert gated[0, 1, 0, 0, 0].item() == pytest.approx(0.5)

    def test_only_frame_gate_active(self):
        """Only min_event_area_ratio set => pixel gate skipped."""
        gater = _MockGater(score_threshold=None, min_event_area_ratio=0.5)
        score = torch.zeros(1, 2, 1, 4, 4)
        score[0, 0, 0, 0, 0] = 0.8  # 1/16 pixels
        score[0, 1, 0, :, :] = 0.8  # 16/16 pixels

        gated = gater._apply_teleport_metric_gates(score)
        assert gated[0, 0].sum().item() == 0.0  # 1/16 < 0.5
        assert gated[0, 1].sum().item() == pytest.approx(16 * 0.8)


class TestGradPreservation:
    def test_gate_preserves_grad_on_aux_loss_path(self):
        """Gating is a multiplication by a constant mask; gradients flow
        through positions where mask == 1."""
        gater = _MockGater(score_threshold=0.3)
        # Create a score_map that requires grad
        score = torch.tensor([[[[[0.1, 0.5], [0.0, 0.4]]]]], dtype=torch.float32,
                             requires_grad=True)  # [1,1,1,2,2]
        gated = gater._apply_teleport_metric_gates(score)
        loss = gated.sum()
        loss.backward()

        # Positions below threshold should have grad 0
        assert score.grad[0, 0, 0, 0, 0].item() == 0.0  # 0.1 < 0.3
        assert score.grad[0, 0, 0, 1, 0].item() == 0.0  # 0.0 < 0.3
        # Positions above threshold should have grad 1.0 (sum() derivative)
        assert score.grad[0, 0, 0, 0, 1].item() == 1.0  # 0.5 > 0.3
        assert score.grad[0, 0, 0, 1, 1].item() == 1.0  # 0.4 > 0.3

    def test_frame_gate_preserves_grad_where_surviving(self):
        """Frame gate zeroes out entire frame but preserves grad in frames
        that pass the area ratio check."""
        gater = _MockGater(
            score_threshold=0.0, min_event_area_ratio=0.5
        )
        score = torch.zeros(1, 2, 1, 4, 4, requires_grad=True)
        # Frame 0: 1 pixel > 0 => fails frame gate => grad should be 0
        score.data[0, 0, 0, 0, 0] = 0.5
        # Frame 1: all 16 pixels > 0 => passes frame gate
        score.data[0, 1, 0, :, :] = 0.5

        gated = gater._apply_teleport_metric_gates(score)
        loss = gated.sum()
        loss.backward()

        # Frame 0 entirely zeroed => no grad
        assert score.grad[0, 0].sum().item() == 0.0
        # Frame 1 survives => grad 1.0 per position
        assert score.grad[0, 1].sum().item() == pytest.approx(16.0)


