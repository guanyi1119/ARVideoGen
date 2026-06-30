"""Unit tests for TeleportReweighter (Family 1 -- multiplicative DMD reweighting)."""

import pytest
import torch

from methods.reward_forcing.teleport.reweighter import (
    AbsoluteReweighter,
    TeacherRelativeReweighter,
    TeleportReweighter,
    ThresholdedReweighter,
    build_reweighter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score_map(B: int = 1, T: int = 4, H: int = 32, W: int = 32, fill: float = 0.0) -> torch.Tensor:
    """Synthetic score map [B, T, 1, H, W] with given fill value."""
    return torch.full((B, T, 1, H, W), fill)


def _rand_score_map(B: int = 1, T: int = 4, H: int = 32, W: int = 32) -> torch.Tensor:
    """Synthetic score map with uniform random values in [0, 1]."""
    return torch.rand(B, T, 1, H, W)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_factory_none_returns_none(self):
        assert build_reweighter(None) is None

    def test_factory_empty_returns_none(self):
        assert build_reweighter({}) is None

    def test_factory_disabled_returns_none(self):
        assert build_reweighter({"type": "disabled"}) is None

    def test_factory_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            build_reweighter({"type": "bad"})

    def test_factory_teacher_relative_constructs(self):
        rw = build_reweighter({"type": "teacher_relative"})
        assert isinstance(rw, TeacherRelativeReweighter)
        assert rw.alpha == 5.0

    def test_factory_absolute_constructs(self):
        rw = build_reweighter({"type": "absolute"})
        assert isinstance(rw, AbsoluteReweighter)
        assert rw.alpha == 5.0

    def test_factory_thresholded_constructs(self):
        rw = build_reweighter(
            {"type": "thresholded", "tau": 0.3, "latent_h": 8, "latent_w": 8}
        )
        assert isinstance(rw, ThresholdedReweighter)
        assert rw.tau == 0.3

    def test_factory_thresholded_missing_tau_raises(self):
        with pytest.raises(ValueError, match="requires 'tau'"):
            build_reweighter({"type": "thresholded", "latent_h": 8, "latent_w": 8})

    def test_factory_alpha_override(self):
        rw = build_reweighter({"type": "absolute", "alpha": 3.0})
        assert rw.alpha == 3.0

    def test_factory_latent_resolution_override(self):
        rw = build_reweighter(
            {"type": "absolute", "latent_h": 16, "latent_w": 32}
        )
        assert rw.latent_h == 16
        assert rw.latent_w == 32


# ---------------------------------------------------------------------------
# Zero score → unit weights (all three subclasses)
# ---------------------------------------------------------------------------


class TestZeroScoreYieldsUnitWeights:
    def test_teacher_relative_zero_yields_unit(self):
        rw = TeacherRelativeReweighter(latent_h=4, latent_w=4)
        s = _score_map(H=16, W=16)
        t = _score_map(H=16, W=16)
        w = rw.compute_weight(s, t)
        assert torch.allclose(w, torch.tensor(1.0), atol=1e-6)

    def test_absolute_zero_yields_unit(self):
        rw = AbsoluteReweighter(latent_h=4, latent_w=4)
        s = _score_map(H=16, W=16)
        w = rw.compute_weight(s)
        assert torch.allclose(w, torch.tensor(1.0), atol=1e-6)

    def test_thresholded_zero_yields_unit(self):
        rw = ThresholdedReweighter(latent_h=4, latent_w=4, tau=0.1)
        s = _score_map(H=16, W=16)
        w = rw.compute_weight(s)
        assert torch.allclose(w, torch.tensor(1.0), atol=1e-6)


# ---------------------------------------------------------------------------
# Positive score → above-unit weights
# ---------------------------------------------------------------------------


class TestPositiveScoreYieldsAboveUnit:
    def test_absolute_positive_yields_above_unit(self):
        rw = AbsoluteReweighter(alpha=2.0, latent_h=4, latent_w=4)
        s = _score_map(H=16, W=16, fill=0.5)
        w = rw.compute_weight(s)
        # w = 1 + 2 * 0.5 = 2.0 at all positions
        assert torch.allclose(w, torch.tensor(2.0), atol=1e-6)

    def test_thresholded_above_tau_yields_above_unit(self):
        rw = ThresholdedReweighter(alpha=2.0, latent_h=4, latent_w=4, tau=0.2)
        s = _score_map(H=16, W=16, fill=0.5)
        w = rw.compute_weight(s)
        # w = 1 + 2 * (0.5 - 0.2) = 1 + 2 * 0.3 = 1.6
        assert torch.allclose(w, torch.tensor(1.6), atol=1e-6)

    def test_teacher_relative_student_higher_yields_above_unit(self):
        rw = TeacherRelativeReweighter(alpha=2.0, latent_h=4, latent_w=4)
        s = _score_map(H=16, W=16, fill=0.5)
        t = _score_map(H=16, W=16, fill=0.1)
        w = rw.compute_weight(s, t)
        # w = 1 + 2 * relu(0.5 - 0.1) = 1 + 2 * 0.4 = 1.8
        assert torch.allclose(w, torch.tensor(1.8), atol=1e-6)


# ---------------------------------------------------------------------------
# TeacherRelativeReweighter edge cases
# ---------------------------------------------------------------------------


class TestTeacherRelative:
    def test_equal_student_teacher_yields_unit(self):
        rw = TeacherRelativeReweighter(latent_h=4, latent_w=4)
        s = _score_map(H=16, W=16, fill=0.3)
        t = _score_map(H=16, W=16, fill=0.3)
        w = rw.compute_weight(s, t)
        assert torch.allclose(w, torch.tensor(1.0), atol=1e-6)

    def test_teacher_higher_clamped_to_unit(self):
        rw = TeacherRelativeReweighter(alpha=10.0, latent_h=4, latent_w=4)
        s = _score_map(H=16, W=16, fill=0.1)
        t = _score_map(H=16, W=16, fill=0.5)
        w = rw.compute_weight(s, t)
        # relu(0.1 - 0.5) = 0 → weight = 1.0
        assert torch.allclose(w, torch.tensor(1.0), atol=1e-6)

    def test_requires_teacher_score(self):
        rw = TeacherRelativeReweighter(latent_h=4, latent_w=4)
        s = _score_map(H=16, W=16)
        with pytest.raises(ValueError, match="requires teacher_score"):
            rw.compute_weight(s, teacher_score=None)

    def test_spatially_heterogeneous_gap(self):
        """Verify weight is > 1.0 only where student > teacher."""
        rw = TeacherRelativeReweighter(alpha=3.0, latent_h=4, latent_w=4)
        # student: [1, 2, 1, 16, 16], left half = 0.5, right half = 0.1
        s = torch.zeros(1, 2, 1, 16, 16)
        s[:, :, :, :, :8] = 0.5
        # teacher: [1, 2, 1, 16, 16], all = 0.1
        t = torch.full((1, 2, 1, 16, 16), 0.1)
        w = rw.compute_weight(s, t)
        # left half: w = 1 + 3 * (0.5 - 0.1) = 2.2
        # right half: w = 1 + 3 * relu(0.1 - 0.1) = 1.0
        assert w.shape == (1, 2, 1, 4, 4)
        # After interpolation, check that left region > 1.0 and right = 1.0
        assert w[:, :, :, :, :2].min().item() > 1.0
        assert torch.allclose(w[:, :, :, :, 2:], torch.tensor(1.0), atol=1e-6)


# ---------------------------------------------------------------------------
# AbsoluteReweighter -- rejects teacher_score
# ---------------------------------------------------------------------------


class TestAbsoluteRejectsTeacher:
    def test_rejects_teacher_score(self):
        rw = AbsoluteReweighter(latent_h=4, latent_w=4)
        s = _score_map(H=16, W=16)
        t = _score_map(H=16, W=16)
        with pytest.raises(ValueError, match="does not accept teacher_score"):
            rw.compute_weight(s, teacher_score=t)


# ---------------------------------------------------------------------------
# ThresholdedReweighter -- rejects teacher_score
# ---------------------------------------------------------------------------


class TestThresholdedRejectsTeacher:
    def test_rejects_teacher_score(self):
        rw = ThresholdedReweighter(latent_h=4, latent_w=4, tau=0.1)
        s = _score_map(H=16, W=16)
        t = _score_map(H=16, W=16)
        with pytest.raises(ValueError, match="does not accept teacher_score"):
            rw.compute_weight(s, teacher_score=t)


# ---------------------------------------------------------------------------
# ThresholdedReweighter -- tau behavior
# ---------------------------------------------------------------------------


class TestThresholdedTau:
    def test_below_tau_yields_unit(self):
        rw = ThresholdedReweighter(latent_h=4, latent_w=4, tau=0.5)
        s = _score_map(H=16, W=16, fill=0.3)
        w = rw.compute_weight(s)
        assert torch.allclose(w, torch.tensor(1.0), atol=1e-6)

    def test_above_tau_yields_above_unit(self):
        rw = ThresholdedReweighter(alpha=4.0, latent_h=4, latent_w=4, tau=0.2)
        s = _score_map(H=16, W=16, fill=0.6)
        w = rw.compute_weight(s)
        # w = 1 + 4 * (0.6 - 0.2) = 1 + 4 * 0.4 = 2.6
        assert torch.allclose(w, torch.tensor(2.6), atol=1e-6)

    def test_exactly_at_tau_yields_unit(self):
        rw = ThresholdedReweighter(latent_h=4, latent_w=4, tau=0.3)
        s = _score_map(H=16, W=16, fill=0.3)
        w = rw.compute_weight(s)
        # relu(0.3 - 0.3) = 0 → weight = 1.0
        assert torch.allclose(w, torch.tensor(1.0), atol=1e-6)


# ---------------------------------------------------------------------------
# Output shape contract
# ---------------------------------------------------------------------------


class TestOutputShape:
    def test_shape_matches_latent_resolution(self):
        rw = build_reweighter({"type": "absolute", "latent_h": 4, "latent_w": 8})
        s = _score_map(B=2, T=8, H=32, W=32)
        w = rw.compute_weight(s)
        assert w.shape == (2, 8, 1, 4, 8)

    def test_shape_teacher_relative(self):
        rw = TeacherRelativeReweighter(latent_h=6, latent_w=10)
        s = _score_map(B=1, T=4, H=24, W=40)
        t = _score_map(B=1, T=4, H=24, W=40)
        w = rw.compute_weight(s, t)
        assert w.shape == (1, 4, 1, 6, 10)

    def test_shape_thresholded(self):
        rw = ThresholdedReweighter(latent_h=5, latent_w=9, tau=0.2)
        s = _score_map(B=3, T=2, H=20, W=36)
        w = rw.compute_weight(s)
        assert w.shape == (3, 2, 1, 5, 9)


# ---------------------------------------------------------------------------
# Output always >= 1.0 (defensive floor)
# ---------------------------------------------------------------------------


class TestOutputAlwaysGeOne:
    def test_absolute_randomized_ge_one(self):
        rw = AbsoluteReweighter(alpha=5.0, latent_h=8, latent_w=8)
        s = _rand_score_map(B=2, T=6, H=32, W=32)
        w = rw.compute_weight(s)
        assert (w >= 1.0).all()

    def test_teacher_relative_randomized_ge_one(self):
        rw = TeacherRelativeReweighter(alpha=5.0, latent_h=8, latent_w=8)
        s = _rand_score_map(B=2, T=6, H=32, W=32)
        t = _rand_score_map(B=2, T=6, H=32, W=32)
        w = rw.compute_weight(s, t)
        assert (w >= 1.0).all()

    def test_thresholded_randomized_ge_one(self):
        rw = ThresholdedReweighter(alpha=5.0, latent_h=8, latent_w=8, tau=0.3)
        s = _rand_score_map(B=2, T=6, H=32, W=32)
        w = rw.compute_weight(s)
        assert (w >= 1.0).all()

    def test_negative_score_still_clamped(self):
        """Even if (somehow) score is negative, clamp_min ensures >= 1.0."""
        rw = AbsoluteReweighter(alpha=100.0, latent_h=4, latent_w=4)
        s = _score_map(H=16, W=16, fill=-0.5)
        w = rw.compute_weight(s)
        assert (w >= 1.0).all()


# ---------------------------------------------------------------------------
# Input detached -- reweighter does not call .detach()
# ---------------------------------------------------------------------------


class TestNoInternalDetach:
    def test_grad_attached_input_passes_through(self):
        """reweighter should NOT call .detach(); if input has grad, output does too."""
        rw = AbsoluteReweighter(latent_h=4, latent_w=4)
        s = _score_map(H=16, W=16, fill=0.3).requires_grad_(True)
        w = rw.compute_weight(s)
        # F.interpolate + arithmetic on grad tensor → output should have grad_fn
        assert w.requires_grad is True

    def test_detached_input_gives_detached_output(self):
        rw = AbsoluteReweighter(latent_h=4, latent_w=4)
        s = _score_map(H=16, W=16, fill=0.3)  # no grad
        w = rw.compute_weight(s)
        assert w.requires_grad is False
