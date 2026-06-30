"""Unit tests for TeleportAuxLoss — Family 2 additive penalty."""

import pytest
import torch

from methods.reward_forcing.teleport.aux_loss import (
    MaskedMeanAuxLoss,
    TeleportAuxLoss,
    build_aux_loss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_score_map(
    B: int = 1,
    T: int = 4,
    H: int = 8,
    W: int = 8,
    *,
    value: float = 0.0,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Build a synthetic score map [B, T, 1, H, W]."""
    t = torch.full((B, T, 1, H, W), float(value))
    t.requires_grad_(requires_grad)
    return t


def _make_score_map_with_patch(
    B: int = 1,
    T: int = 4,
    H: int = 8,
    W: int = 8,
    *,
    patch_value: float = 0.5,
    background: float = 0.0,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Build a score map with a bright patch at the center of frames 2 and 3."""
    t = torch.full((B, T, 1, H, W), float(background))
    half = H // 4
    t[:, 2:, 0, H // 2 - half : H // 2 + half, W // 2 - half : W // 2 + half] = float(
        patch_value
    )
    t.requires_grad_(requires_grad)
    return t


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_factory_none_returns_none(self):
        assert build_aux_loss(None) is None

    def test_factory_empty_returns_none(self):
        assert build_aux_loss({}) is None

    def test_factory_disabled_returns_none(self):
        assert build_aux_loss({"type": "disabled"}) is None

    def test_factory_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            build_aux_loss({"type": "bogus"})

    def test_factory_missing_tau_raises(self):
        with pytest.raises(ValueError, match="requires explicit tau"):
            build_aux_loss({"type": "masked_mean"})

    def test_factory_masked_mean_with_tau(self):
        loss = build_aux_loss({"type": "masked_mean", "tau": 0.3})
        assert isinstance(loss, MaskedMeanAuxLoss)
        assert loss._tau == 0.3
        assert loss._aggregation == "masked_mean"

    def test_factory_masked_mean_with_aggregation_mean(self):
        loss = build_aux_loss(
            {"type": "masked_mean", "tau": 0.2, "aggregation": "mean"}
        )
        assert isinstance(loss, MaskedMeanAuxLoss)
        assert loss._tau == 0.2
        assert loss._aggregation == "mean"


# ---------------------------------------------------------------------------
# MaskedMeanAuxLoss — constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_constructor_tau_none_raises(self):
        with pytest.raises(ValueError, match="requires explicit tau"):
            MaskedMeanAuxLoss(tau=None)

    def test_constructor_invalid_aggregation_raises(self):
        with pytest.raises(ValueError, match="aggregation must be"):
            MaskedMeanAuxLoss(tau=0.1, aggregation="bogus")

    def test_constructor_default_aggregation(self):
        loss = MaskedMeanAuxLoss(tau=0.5)
        assert loss._aggregation == "masked_mean"


# ---------------------------------------------------------------------------
# MaskedMeanAuxLoss — masked_mean behaviour
# ---------------------------------------------------------------------------


class TestMaskedMean:
    def test_masked_mean_zero_score_returns_zero_grad_able(self):
        """All-zero score map with grad → L_aux == 0.0 AND backward works."""
        loss_fn = MaskedMeanAuxLoss(tau=0.3)
        score = _make_score_map(requires_grad=True)
        L = loss_fn(score)
        assert L.item() == 0.0
        L.backward()  # must not raise
        assert score.grad is not None

    def test_masked_mean_score_above_tau_contributes(self):
        """Score values > tau should produce non-zero loss."""
        loss_fn = MaskedMeanAuxLoss(tau=0.3)
        score = _make_score_map_with_patch(
            patch_value=0.7, background=0.0, requires_grad=False
        )
        L = loss_fn(score)
        assert L.item() > 0.0

    def test_masked_mean_score_below_tau_excluded(self):
        """All scores below tau → mask is all zeros → L_aux == 0.0."""
        loss_fn = MaskedMeanAuxLoss(tau=0.5)
        score = _make_score_map_with_patch(
            patch_value=0.4, background=0.1, requires_grad=False
        )
        L = loss_fn(score)
        assert L.item() == 0.0

    def test_masked_mean_grad_flows_to_input(self):
        """Gradient flows back to score_map through the masked_mean path."""
        loss_fn = MaskedMeanAuxLoss(tau=0.3)
        score = _make_score_map_with_patch(
            patch_value=0.7, background=0.0, requires_grad=True
        )
        L = loss_fn(score)
        L.backward()
        assert score.grad is not None
        # Gradient should be non-zero where mask is True (patch region)
        grad_abs_sum = score.grad.abs().sum().item()
        assert grad_abs_sum > 0

    def test_masked_mean_correct_formula(self):
        """Verify the exact formula on a small controlled tensor."""
        loss_fn = MaskedMeanAuxLoss(tau=0.3)
        # Manual: values [0.1, 0.2, 0.4, 0.5], tau=0.3
        # mask = [0, 0, 1, 1], masked_sum = 0.4+0.5 = 0.9, denom = 2
        # expected = 0.9 / 2 = 0.45
        score = torch.tensor([[[[0.1, 0.2, 0.4, 0.5]]]], dtype=torch.float32)
        # score shape [1, 1, 1, 4]
        L = loss_fn(score)
        assert torch.allclose(L, torch.tensor(0.45))

    def test_masked_mean_mixed_above_below(self):
        """Some pixels above tau, some below — only above contribute."""
        loss_fn = MaskedMeanAuxLoss(tau=0.3)
        # 4 pixels: [0.8, 0.1, 0.9, 0.2]
        # mask = [1, 0, 1, 0], masked_sum = 0.8+0.9 = 1.7, denom = 2
        # expected = 1.7/2 = 0.85
        score = torch.tensor([[[[0.8, 0.1, 0.9, 0.2]]]], dtype=torch.float32)
        L = loss_fn(score)
        assert torch.allclose(L, torch.tensor(0.85))


# ---------------------------------------------------------------------------
# MaskedMeanAuxLoss — mean aggregation (no mask)
# ---------------------------------------------------------------------------


class TestMeanAggregation:
    def test_mean_aggregation_no_mask(self):
        """aggregation='mean' → L_aux == score.mean() exactly."""
        loss_fn = MaskedMeanAuxLoss(tau=0.3, aggregation="mean")
        score = torch.tensor([[[[0.1, 0.2, 0.3, 0.4]]]], dtype=torch.float32)
        L = loss_fn(score)
        assert torch.allclose(L, torch.tensor(0.25))  # (0.1+0.2+0.3+0.4)/4

    def test_mean_aggregation_grad_flows(self):
        """aggregation='mean' preserves gradient flow."""
        loss_fn = MaskedMeanAuxLoss(tau=0.3, aggregation="mean")
        score = _make_score_map_with_patch(
            patch_value=0.7, background=0.0, requires_grad=True
        )
        L = loss_fn(score)
        L.backward()
        assert score.grad is not None
        assert score.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# TeleportAuxLoss — abstract interface
# ---------------------------------------------------------------------------


class TestAbstract:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            TeleportAuxLoss()  # type: ignore[abstract]

    def test_subclass_isinstance(self):
        loss = MaskedMeanAuxLoss(tau=0.3)
        assert isinstance(loss, TeleportAuxLoss)
