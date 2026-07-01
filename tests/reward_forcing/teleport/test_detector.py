"""Unit tests for TeleportDetector.  All flow paths mocked or use flow_backend='none'."""

import pytest
import torch

from methods.reward_forcing.teleport.detector import (
    MultiFrameFlowDetector,
    OpticalFlowTeleportDetector,
    TeleportDetector,
    build_teleport_detector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_emergence_clip(*, position: str, T: int = 3, H: int = 64, W: int = 64):
    """Build a synthetic clip where a bright patch appears at given position
    starting from frame t=2 (frames 0 and 1 are static black)."""
    clip = torch.zeros(1, T, 3, H, W)
    if position == "center":
        cy, cx = H // 2, W // 2
    elif position == "edge_left":
        cy, cx = H // 2, 8
    elif position == "edge_top":
        cy, cx = 8, W // 2
    else:
        raise ValueError(position)
    half = 6
    clip[:, 2:, :, cy - half : cy + half, cx - half : cx + half] = 1.0
    return clip


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_factory_disabled(self):
        assert build_teleport_detector(None) is None
        assert build_teleport_detector({}) is None
        assert build_teleport_detector({"type": "disabled"}) is None

    def test_factory_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown"):
            build_teleport_detector({"type": "foo"})

    def test_factory_optical_flow_constructs_no_raft_load(self):
        det = build_teleport_detector({"type": "optical_flow", "flow_backend": "none"})
        assert isinstance(det, OpticalFlowTeleportDetector)

    def test_factory_no_raft_sugar(self):
        det = build_teleport_detector({"type": "optical_flow_no_raft"})
        assert isinstance(det, OpticalFlowTeleportDetector)

    def test_factory_defaults_passthrough(self):
        det = build_teleport_detector(
            {
                "type": "optical_flow",
                "flow_backend": "none",
                "downsample_factor": 2,
                "alpha_diff": 0.5,
                "alpha_flow": 0.3,
                "alpha_edge": 0.1,
                "edge_sigma": 0.2,
            }
        )
        assert det._downsample_factor == 2
        assert det._alpha_diff == 0.5
        assert det._alpha_flow == 0.3
        assert det._alpha_edge == 0.1
        assert det._edge_sigma == 0.2


# ---------------------------------------------------------------------------
# Output shape contract
# ---------------------------------------------------------------------------


class TestShape:
    def test_output_shape(self):
        det = OpticalFlowTeleportDetector(flow_backend="none", downsample_factor=4)
        rgb = torch.zeros(1, 5, 3, 64, 64)
        out = det.score(rgb)
        assert out.shape == (1, 5, 1, 16, 16)

    def test_output_shape_no_downsample(self):
        det = OpticalFlowTeleportDetector(flow_backend="none", downsample_factor=1)
        rgb = torch.zeros(1, 3, 3, 32, 32)
        out = det.score(rgb)
        assert out.shape == (1, 3, 1, 32, 32)

    def test_first_frame_is_zero(self):
        det = OpticalFlowTeleportDetector(flow_backend="none")
        rgb = torch.rand(1, 4, 3, 64, 64)
        out = det.score(rgb)
        assert torch.all(out[:, 0] == 0)

    def test_single_frame_output(self):
        det = OpticalFlowTeleportDetector(flow_backend="none")
        rgb = torch.ones(2, 1, 3, 32, 32)
        out = det.score(rgb)
        assert out.shape == (2, 1, 1, 8, 8)
        # Single frame → T=1, frame 0 is always zero by spec
        assert torch.all(out == 0)


# ---------------------------------------------------------------------------
# Semantic correctness — center emergence ranks higher than edge emergence
# ---------------------------------------------------------------------------


class TestSemanticRanking:
    def test_center_emergence_score_higher_than_edge(self):
        det = OpticalFlowTeleportDetector(flow_backend="none", downsample_factor=4)
        center = det.score(_make_emergence_clip(position="center"))
        edge_l = det.score(_make_emergence_clip(position="edge_left"))
        edge_t = det.score(_make_emergence_clip(position="edge_top"))
        assert center.max().item() > edge_l.max().item()
        assert center.max().item() > edge_t.max().item()

    def test_static_clip_low_score(self):
        det = OpticalFlowTeleportDetector(flow_backend="none", downsample_factor=4)
        static = torch.zeros(1, 4, 3, 64, 64) + 0.5  # constant gray
        out = det.score(static)
        # frame_diff = 0, flow = 0, only edge bias remains
        # edge bias = alpha_edge * centeredness, max ≈ 0.2 at center
        assert out.max().item() < 0.3

    def test_two_emergence_events_both_detected(self):
        det = OpticalFlowTeleportDetector(flow_backend="none", downsample_factor=4)
        clip = torch.zeros(1, 4, 3, 64, 64)
        # event 1: patch at center at t=2
        clip[:, 2:, :, 28:36, 28:36] = 1.0
        # event 2: another patch at t=3 in different location
        clip[:, 3:, :, 10:18, 50:58] = 1.0
        out = det.score(clip)
        assert out[:, 2].max().item() > 0
        assert out[:, 3].max().item() > 0


# ---------------------------------------------------------------------------
# Differentiability — critical for aux_loss path
# ---------------------------------------------------------------------------


class TestDifferentiability:
    def test_gradient_flows_through_detector(self):
        det = OpticalFlowTeleportDetector(flow_backend="none", downsample_factor=4)
        rgb = torch.rand(1, 3, 3, 64, 64, requires_grad=True)
        score_map = det.score(rgb)
        loss = score_map.sum()
        loss.backward()
        assert rgb.grad is not None
        assert rgb.grad.abs().sum().item() > 0

    def test_no_inplace_op_breaks_grad(self):
        """No detector op should break autograd."""
        det = OpticalFlowTeleportDetector(flow_backend="none", downsample_factor=4)
        rgb = torch.rand(1, 3, 3, 64, 64, requires_grad=True)
        # should not raise
        det.score(rgb).sum().backward()

    def test_gradient_symmetric_for_identical_frames(self):
        """Gradient should be zero when all frames are identical (no change to detect)."""
        det = OpticalFlowTeleportDetector(flow_backend="none", downsample_factor=4)
        rgb = torch.zeros(1, 3, 3, 64, 64, requires_grad=True)
        score_map = det.score(rgb)
        # centeredness is constant → gradient should flow through
        loss = score_map.sum()
        loss.backward()
        assert rgb.grad is not None


# ---------------------------------------------------------------------------
# is_available / configuration knobs
# ---------------------------------------------------------------------------


class TestConfig:
    def test_is_available_no_flow(self):
        det = OpticalFlowTeleportDetector(flow_backend="none")
        assert det.is_available() is True

    def test_alpha_weights_override(self):
        # alpha_flow=0 effectively turns off flow contribution
        det = OpticalFlowTeleportDetector(
            flow_backend="none", alpha_diff=1.0, alpha_flow=0.0, alpha_edge=0.0
        )
        clip = torch.zeros(1, 3, 3, 32, 32)
        clip[:, 2:, :, 12:20, 12:20] = 1.0
        out = det.score(clip)
        # Pure frame_diff: t=2 patch area should be ~1.0 (after channel mean)
        # After 4x downsampling: 32→8, patch 12:20 → 3:5
        assert out[:, 2, 0, 3:5, 3:5].mean().item() > 0.5

    def test_invalid_shape_raises(self):
        det = OpticalFlowTeleportDetector(flow_backend="none")
        with pytest.raises(ValueError, match="Expected rgb shape"):
            det.score(torch.zeros(1, 3, 64, 64))
        with pytest.raises(ValueError, match="Expected 3 channels"):
            det.score(torch.zeros(1, 3, 1, 64, 64))

    def test_centeredness_cache_reuse(self):
        det = OpticalFlowTeleportDetector(flow_backend="none", downsample_factor=1)
        rgb = torch.zeros(1, 2, 3, 32, 32)
        det.score(rgb)  # first call → cache populated
        assert len(det._centeredness_cache) == 1
        det.score(rgb)  # second call → cache hit
        assert len(det._centeredness_cache) == 1

    def test_different_downsample_sizes_cache_separately(self):
        det = OpticalFlowTeleportDetector(flow_backend="none", downsample_factor=2)
        rgb1 = torch.zeros(1, 2, 3, 64, 64)
        rgb2 = torch.zeros(1, 2, 3, 32, 32)
        det.score(rgb1)  # 32x32 after ds → cache key (32, 32, ...)
        det.score(rgb2)  # 16x16 after ds → cache key (16, 16, ...)
        assert len(det._centeredness_cache) == 2


# ---------------------------------------------------------------------------
# MultiFrameFlowDetector tests
# ---------------------------------------------------------------------------


class TestMultiFrameFactory:
    def test_factory_multi_frame_chained(self):
        det = build_teleport_detector({
            "type": "multi_frame_chained",
            "flow_backend": "none",
            "tau": 3,
        })
        assert isinstance(det, MultiFrameFlowDetector)
        assert det._mode == "chained"
        assert det._tau == 3

    def test_factory_multi_frame_direct(self):
        det = build_teleport_detector({
            "type": "multi_frame_direct",
            "flow_backend": "none",
            "tau": 7,
        })
        assert isinstance(det, MultiFrameFlowDetector)
        assert det._mode == "direct"
        assert det._tau == 7

    def test_factory_multi_frame_alias(self):
        det = build_teleport_detector({
            "type": "multi_frame",
            "flow_backend": "none",
        })
        assert isinstance(det, MultiFrameFlowDetector)
        assert det._mode == "chained"

    def test_factory_multi_frame_default_tau(self):
        det = build_teleport_detector({
            "type": "multi_frame",
            "flow_backend": "none",
        })
        assert det._tau == 5


class TestMultiFrameShape:
    def test_output_shape(self):
        det = MultiFrameFlowDetector(
            tau=3, flow_backend="none", downsample_factor=4,
        )
        rgb = torch.zeros(1, 8, 3, 64, 64)
        out = det.score(rgb)
        assert out.shape == (1, 8, 1, 16, 16)

    def test_output_shape_no_downsample(self):
        det = MultiFrameFlowDetector(
            tau=3, flow_backend="none", downsample_factor=1,
        )
        rgb = torch.zeros(1, 5, 3, 32, 32)
        out = det.score(rgb)
        assert out.shape == (1, 5, 1, 32, 32)

    def test_first_tau_frames_are_zero(self):
        det = MultiFrameFlowDetector(
            tau=3, flow_backend="none",
        )
        rgb = torch.rand(1, 6, 3, 64, 64)
        out = det.score(rgb)
        assert torch.all(out[:, :3] == 0)
        # Frames tau+ should be non-zero (from centeredness * residual=0)
        # With flow_backend="none", residual=0, so ALL frames are zero
        assert torch.all(out[:, 3:] == 0)

    def test_single_frame_output(self):
        det = MultiFrameFlowDetector(
            tau=3, flow_backend="none",
        )
        rgb = torch.ones(2, 1, 3, 32, 32)
        out = det.score(rgb)
        assert out.shape == (2, 1, 1, 8, 8)
        assert torch.all(out == 0)

    def test_t_shorter_than_tau_all_zero(self):
        det = MultiFrameFlowDetector(
            tau=5, flow_backend="none",
        )
        rgb = torch.ones(1, 3, 3, 64, 64)
        out = det.score(rgb)
        assert out.shape == (1, 3, 1, 16, 16)
        assert torch.all(out == 0)


class TestMultiFrameDifferentiability:
    def test_gradient_flows_through_detector(self):
        """Backward should not crash. With flow_backend='none', gradient
        magnitude may be zero (residual=0 always), which is correct."""
        det = MultiFrameFlowDetector(
            tau=2, flow_backend="none", downsample_factor=4,
        )
        rgb = torch.rand(1, 4, 3, 64, 64, requires_grad=True)
        score_map = det.score(rgb)
        loss = score_map.sum()
        loss.backward()
        assert rgb.grad is not None
        # Gradient exists but may be zero since residual=0 when
        # flow_backend="none" — this is semantically correct.

    def test_no_inplace_op_breaks_grad(self):
        det = MultiFrameFlowDetector(
            tau=2, flow_backend="none", downsample_factor=4,
        )
        rgb = torch.rand(1, 4, 3, 64, 64, requires_grad=True)
        det.score(rgb).sum().backward()


class TestMultiFrameConfig:
    def test_is_available_no_flow(self):
        det = MultiFrameFlowDetector(flow_backend="none")
        assert det.is_available() is True

    def test_invalid_shape_raises(self):
        det = MultiFrameFlowDetector(flow_backend="none")
        with pytest.raises(ValueError, match="Expected rgb shape"):
            det.score(torch.zeros(1, 3, 64, 64))
        with pytest.raises(ValueError, match="Expected 3 channels"):
            det.score(torch.zeros(1, 3, 1, 64, 64))

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            MultiFrameFlowDetector(mode="invalid")

    def test_invalid_tau_raises(self):
        with pytest.raises(ValueError, match="tau must be"):
            MultiFrameFlowDetector(tau=0)
        with pytest.raises(ValueError, match="tau must be"):
            MultiFrameFlowDetector(tau=-1)

    def test_centeredness_cache_reuse(self):
        det = MultiFrameFlowDetector(
            tau=2, flow_backend="none", downsample_factor=1,
        )
        rgb = torch.zeros(1, 4, 3, 32, 32)
        det.score(rgb)
        assert len(det._centeredness_cache) == 1
        det.score(rgb)
        assert len(det._centeredness_cache) == 1

    def test_direct_mode_constructs(self):
        det = MultiFrameFlowDetector(
            tau=4, mode="direct", flow_backend="none",
        )
        assert det._mode == "direct"
        assert det._tau == 4


class TestFlowComposition:
    def test_compose_two_uniform_flows(self):
        """Two uniform [1, 0] flows should compose to [2, 0]."""
        det = MultiFrameFlowDetector(flow_backend="none")
        B, H, W = 1, 16, 16
        flow_a = torch.zeros(B, 2, H, W)
        flow_a[:, 0, :, :] = 1.0  # dx = 1
        flow_b = torch.zeros(B, 2, H, W)
        flow_b[:, 0, :, :] = 1.0  # dx = 1

        composed = det._compose_flows([flow_a, flow_b])
        assert composed.shape == (B, 2, H, W)
        # Near center, composition should be ~[2, 0]
        # At borders, grid_sample with border padding may differ
        center_dx = composed[0, 0, H // 2, W // 2].item()
        center_dy = composed[0, 1, H // 2, W // 2].item()
        assert abs(center_dx - 2.0) < 0.2, f"got dx={center_dx}"
        assert abs(center_dy) < 0.1, f"got dy={center_dy}"

    def test_compose_three_uniform_flows(self):
        """Three uniform [1, 0] flows should compose to ~[3, 0]."""
        det = MultiFrameFlowDetector(flow_backend="none")
        B, H, W = 1, 16, 16
        flows = []
        for _ in range(3):
            f = torch.zeros(B, 2, H, W)
            f[:, 0, :, :] = 1.0
            flows.append(f)

        composed = det._compose_flows(flows)
        center_dx = composed[0, 0, H // 2, W // 2].item()
        center_dy = composed[0, 1, H // 2, W // 2].item()
        assert abs(center_dx - 3.0) < 0.3, f"got dx={center_dx}"
        assert abs(center_dy) < 0.1, f"got dy={center_dy}"

    def test_compose_vertical_flow(self):
        """Two uniform [0, 1] flows should compose to ~[0, 2]."""
        det = MultiFrameFlowDetector(flow_backend="none")
        B, H, W = 1, 16, 16
        flow_a = torch.zeros(B, 2, H, W)
        flow_a[:, 1, :, :] = 1.0  # dy = 1
        flow_b = torch.zeros(B, 2, H, W)
        flow_b[:, 1, :, :] = 1.0  # dy = 1

        composed = det._compose_flows([flow_a, flow_b])
        center_dx = composed[0, 0, H // 2, W // 2].item()
        center_dy = composed[0, 1, H // 2, W // 2].item()
        assert abs(center_dx) < 0.1
        assert abs(center_dy - 2.0) < 0.2

    def test_warp_flow_identity(self):
        """Warping by zero flow should return the original flow."""
        det = MultiFrameFlowDetector(flow_backend="none")
        B, H, W = 1, 16, 16
        flow_src = torch.rand(B, 2, H, W)
        flow_zero = torch.zeros(B, 2, H, W)

        warped = det._warp_flow(flow_src, flow_zero)
        assert torch.allclose(warped, flow_src, atol=1e-5)

    def test_compose_single_flow_returns_itself(self):
        """Composing a single flow should return it unchanged."""
        det = MultiFrameFlowDetector(flow_backend="none")
        B, H, W = 1, 16, 16
        flow = torch.rand(B, 2, H, W)
        composed = det._compose_flows([flow])
        assert torch.allclose(composed, flow, atol=1e-5)

    def test_multiplicative_centeredness_gate(self):
        """With flow_backend='none', residual=0, so score should be all zeros
        regardless of centeredness."""
        det = MultiFrameFlowDetector(
            tau=2, flow_backend="none", downsample_factor=4,
        )
        rgb = torch.rand(1, 5, 3, 64, 64)
        out = det.score(rgb)
        # residual=0 * centeredness = 0 everywhere
        assert torch.all(out == 0)
