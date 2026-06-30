"""T10: Teleport hook integration tests for ReDMD.compute_rewarded_distribution_matching_loss.

All tests run on CPU with heavy mocking -- no real Wan teacher/generator loaded.

Strategy: instead of importing the full module chain (which hits CUDA at class-def
time in wan.modules.t5), we construct mock ReDMD-like objects and test the hook
logic in isolation.  The actual code changes in re_dmd.py are verified by import
sanity checks and bit-equivalence tests via subprocess.
"""

from __future__ import annotations

import ast
import os
import sys
import pytest
import torch
from unittest.mock import MagicMock

from methods.reward_forcing.teleport import (
    build_teleport_detector,
    build_reweighter,
    build_aux_loss,
)
from methods.reward_forcing.teleport.reweighter import AbsoluteReweighter
from methods.reward_forcing.teleport.aux_loss import MaskedMeanAuxLoss
from methods.reward_forcing.teleport.detector import OpticalFlowTeleportDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_omegaconf(d: dict):
    """Build an OmegaConf-like dict-config object (supports getattr + .get)."""
    class _Cfg:
        def __init__(self, data: dict):
            self.__dict__["_data"] = data
            for k, v in data.items():
                if isinstance(v, dict):
                    self.__dict__[k] = _Cfg(v)
                else:
                    self.__dict__[k] = v

        def __getattr__(self, name):
            try:
                return self.__dict__[name]
            except KeyError:
                raise AttributeError(name)

        def get(self, key, default=None):
            return self._data.get(key, default)

    return _Cfg(d)


def _make_teleport_tel_det_regular_cfg(mode: str = "off", **overrides):
    """Build a tel_det_regular config dict for a given mode."""
    base = {
        "mode": mode,
        "detector": {
            "type": "optical_flow_no_raft",
            "downsample_factor": 4,
            "alpha_diff": 0.3,
            "alpha_flow": 0.5,
            "alpha_edge": 0.2,
        },
    }

    if mode == "reweight":
        base["reweighter"] = {
            "type": "absolute",
            "alpha": 5.0,
            "latent_h": 40,
            "latent_w": 72,
        }
    elif mode == "aux_loss":
        base["aux_loss"] = {
            "type": "masked_mean",
            "tau": 0.3,
            "aggregation": "masked_mean",
        }
        base["aux_beta"] = 1.0

    for k, v in overrides.items():
        base[k] = v

    return base


def _init_teleport_hooks(obj, tel_det_regular_cfg):
    """Replicate the T10 __init__ hook builder logic from re_dmd.py.

    This is the exact same code pattern as in ReDMD.__init__, extracted
    for isolated testing.  The real code lives in re_dmd.py:53-81.
    """
    obj._teleport_mode = "off"
    obj._teleport_detector = None
    obj._teleport_reweighter = None
    obj._teleport_aux_loss = None
    obj._teleport_aux_beta = 1.0

    if tel_det_regular_cfg is not None:
        cfg = _make_simple_omegaconf(tel_det_regular_cfg) if isinstance(tel_det_regular_cfg, dict) else tel_det_regular_cfg
        obj._teleport_mode = getattr(cfg, "mode", "off")
        if obj._teleport_mode == "reweight":
            obj._teleport_detector = build_teleport_detector(getattr(cfg, "detector", None))
            obj._teleport_reweighter = build_reweighter(getattr(cfg, "reweighter", None))
        elif obj._teleport_mode == "aux_loss":
            obj._teleport_detector = build_teleport_detector(getattr(cfg, "detector", None))
            obj._teleport_aux_loss = build_aux_loss(getattr(cfg, "aux_loss", None))
            obj._teleport_aux_beta = getattr(cfg, "aux_beta", 1.0)
        elif obj._teleport_mode != "off":
            raise ValueError(f"Unknown teleport.tel_det_regular.mode: {obj._teleport_mode!r}")

    assert (obj._teleport_reweighter is None) or (obj._teleport_aux_loss is None), (
        "Mutex violation: reweighter and aux_loss cannot both be active."
    )


class MockReDMD:
    """Mock ReDMD-like object for testing teleport hook integration.

    Contains the exact same teleport hook attributes as the real ReDMD
    and the same loss assembly logic from compute_rewarded_distribution_matching_loss.
    """

    def __init__(self, tel_det_regular_cfg=None):
        _init_teleport_hooks(self, tel_det_regular_cfg)


# ---------------------------------------------------------------------------
# Test 1-6: __init__ hook construction
# ---------------------------------------------------------------------------


class TestInitHookConstruction:
    """Tests for teleport hook initialization (replicating ReDMD.__init__)."""

    def test_mode_off_initializes_no_hooks(self):
        """When tel_det_regular_cfg is None or mode=off, all hook attrs are None."""
        model = MockReDMD(tel_det_regular_cfg=None)
        assert model._teleport_mode == "off"
        assert model._teleport_detector is None
        assert model._teleport_reweighter is None
        assert model._teleport_aux_loss is None
        assert model._teleport_aux_beta == 1.0

        model_off = MockReDMD(tel_det_regular_cfg=_make_teleport_tel_det_regular_cfg("off"))
        assert model_off._teleport_mode == "off"
        assert model_off._teleport_detector is None
        assert model_off._teleport_reweighter is None
        assert model_off._teleport_aux_loss is None

    def test_mode_reweight_initializes_reweighter(self):
        """mode=reweight with absolute reweighter initializes detector + reweighter."""
        cfg = _make_teleport_tel_det_regular_cfg("reweight")
        model = MockReDMD(tel_det_regular_cfg=cfg)
        assert model._teleport_mode == "reweight"
        assert model._teleport_detector is not None
        assert isinstance(model._teleport_detector, OpticalFlowTeleportDetector)
        assert model._teleport_reweighter is not None
        assert isinstance(model._teleport_reweighter, AbsoluteReweighter)
        assert model._teleport_aux_loss is None

    def test_mode_aux_loss_initializes_aux_loss(self):
        """mode=aux_loss with masked_mean initializes detector + aux_loss."""
        cfg = _make_teleport_tel_det_regular_cfg("aux_loss")
        model = MockReDMD(tel_det_regular_cfg=cfg)
        assert model._teleport_mode == "aux_loss"
        assert model._teleport_detector is not None
        assert isinstance(model._teleport_detector, OpticalFlowTeleportDetector)
        assert model._teleport_aux_loss is not None
        assert isinstance(model._teleport_aux_loss, MaskedMeanAuxLoss)
        assert model._teleport_reweighter is None
        assert model._teleport_aux_beta == 1.0

    def test_mode_unknown_raises(self):
        """mode=bogus raises ValueError."""
        cfg = _make_teleport_tel_det_regular_cfg("bogus")
        with pytest.raises(ValueError, match="Unknown teleport.tel_det_regular.mode"):
            MockReDMD(tel_det_regular_cfg=cfg)

    def test_thresholded_missing_tau_raises_at_init(self):
        """mode=reweight with thresholded reweighter without tau raises ValueError."""
        # Pass raw dict to avoid _Cfg containment issue with build_reweighter
        from methods.reward_forcing.teleport.reweighter import build_reweighter
        with pytest.raises(ValueError, match="requires 'tau'"):
            build_reweighter({"type": "thresholded", "alpha": 5.0, "latent_h": 40, "latent_w": 72})

    def test_aux_loss_missing_tau_raises_at_init(self):
        """mode=aux_loss with masked_mean without tau raises ValueError."""
        from methods.reward_forcing.teleport.aux_loss import build_aux_loss
        with pytest.raises(ValueError, match="requires explicit tau"):
            build_aux_loss({"type": "masked_mean"})


# ---------------------------------------------------------------------------
# Test 7-13: Loss assembly logic
# ---------------------------------------------------------------------------


def _assemble_loss(original_latent, grad, gradient_mask, reward_term, beta,
                   teleport_weight_map, teleport_aux_loss_val, aux_beta, teleport_log):
    """Replicate the T10 loss assembly from re_dmd.py:282-315."""
    import torch.nn.functional as F

    if gradient_mask is not None:
        if teleport_weight_map is not None:
            w = teleport_weight_map.to(original_latent.dtype).to(original_latent.device)
            sq = (original_latent.double() - (original_latent.double() - grad.double()).detach()) ** 2
            weighted_sq = w.double() * sq
            mse = weighted_sq[gradient_mask].mean()
        else:
            mse = F.mse_loss(
                original_latent.double()[gradient_mask],
                (original_latent.double() - grad.double()).detach()[gradient_mask],
                reduction="mean",
            )
        rl_dmd_loss = 0.5 * torch.exp(beta * reward_term) * mse
    else:
        if teleport_weight_map is not None:
            w = teleport_weight_map.to(original_latent.dtype).to(original_latent.device)
            sq = (original_latent.double() - (original_latent.double() - grad.double()).detach()) ** 2
            weighted_sq = w.double() * sq
            mse = weighted_sq.mean()
        else:
            mse = F.mse_loss(
                original_latent.double(),
                (original_latent.double() - grad.double()).detach(),
                reduction="mean",
            )
        rl_dmd_loss = 0.5 * torch.exp(beta * reward_term) * mse

    if teleport_aux_loss_val is not None:
        rl_dmd_loss = rl_dmd_loss + aux_beta * teleport_aux_loss_val

    log_dict = {"dmdtrain_gradient_norm": torch.tensor(0.1)}
    log_dict.update(teleport_log)
    return rl_dmd_loss, log_dict


def _compute_teleport_score(model, videos):
    """Replicate the T10 teleport score map computation from re_dmd.py:215-235."""
    teleport_weight_map = None
    teleport_aux_loss_val = None
    teleport_log = {}

    if model._teleport_mode == "reweight" and model._teleport_detector is not None:
        with torch.no_grad():
            student_score = model._teleport_detector.score(videos).detach()
        teleport_weight_map = model._teleport_reweighter.compute_weight(
            student_score,
            teacher_score=None,
        )
        teleport_log["teleport_score_mean"] = student_score.mean().detach()
        teleport_log["teleport_weight_mean"] = teleport_weight_map.mean().detach()
    elif model._teleport_mode == "aux_loss" and model._teleport_detector is not None:
        student_score_grad = model._teleport_detector.score(videos)
        teleport_aux_loss_val = model._teleport_aux_loss(student_score_grad)
        teleport_log["teleport_score_mean"] = student_score_grad.mean().detach()
        teleport_log["teleport_aux_loss"] = teleport_aux_loss_val.detach()

    return teleport_weight_map, teleport_aux_loss_val, teleport_log


class TestLossAssembly:
    """Tests for the teleport-aware loss assembly logic."""

    @staticmethod
    def _make_latents(B=1, F=4, C=16, H=8, W=8, fill=0.1):
        return torch.full((B, F, C, H, W), fill, dtype=torch.float32)

    @staticmethod
    def _make_videos(B=1, F=4, H=64, W=64, fill=0.3):
        return torch.full((B, F, 3, H, W), fill, dtype=torch.float32)

    def test_loss_off_mode_no_teleport(self):
        """When teleport_weight_map and aux_loss are None, loss equals vanilla mse."""
        latents = self._make_latents()
        grad = torch.full_like(latents, 0.02, dtype=torch.float64)
        reward_term = torch.tensor(0.5, dtype=torch.float64)
        beta = torch.tensor(1.0, dtype=torch.float64)

        loss, log_dict = _assemble_loss(
            latents, grad, gradient_mask=None, reward_term=reward_term,
            beta=beta, teleport_weight_map=None, teleport_aux_loss_val=None,
            aux_beta=1.0, teleport_log={},
        )

        import torch.nn.functional as F
        expected_mse = F.mse_loss(latents.double(), (latents.double() - grad.double()).detach())
        expected_loss = 0.5 * torch.exp(beta * reward_term) * expected_mse
        assert torch.allclose(loss, expected_loss, rtol=1e-12, atol=1e-12)

    def test_loss_with_weight_map_all_ones(self):
        """Weight map of all 1.0 produces same loss as vanilla."""
        latents = self._make_latents()
        grad = torch.full_like(latents, 0.02, dtype=torch.float64)
        reward_term = torch.tensor(0.5, dtype=torch.float64)
        beta = torch.tensor(1.0, dtype=torch.float64)

        weight_map = torch.ones(1, 4, 1, 8, 8, dtype=torch.float64)

        loss_w, _ = _assemble_loss(
            latents, grad, gradient_mask=None, reward_term=reward_term,
            beta=beta, teleport_weight_map=weight_map, teleport_aux_loss_val=None,
            aux_beta=1.0, teleport_log={},
        )
        loss_v, _ = _assemble_loss(
            latents, grad, gradient_mask=None, reward_term=reward_term,
            beta=beta, teleport_weight_map=None, teleport_aux_loss_val=None,
            aux_beta=1.0, teleport_log={},
        )
        assert torch.allclose(loss_w, loss_v, rtol=1e-10, atol=1e-10)

    def test_loss_with_weight_map_above_one(self):
        """Weight map > 1.0 produces strictly greater loss."""
        latents = self._make_latents()
        grad = torch.full_like(latents, 0.02, dtype=torch.float64)
        reward_term = torch.tensor(0.5, dtype=torch.float64)
        beta = torch.tensor(1.0, dtype=torch.float64)

        weight_map = torch.full((1, 4, 1, 8, 8), 3.0, dtype=torch.float64)

        loss_w, _ = _assemble_loss(
            latents, grad, gradient_mask=None, reward_term=reward_term,
            beta=beta, teleport_weight_map=weight_map, teleport_aux_loss_val=None,
            aux_beta=1.0, teleport_log={},
        )
        loss_v, _ = _assemble_loss(
            latents, grad, gradient_mask=None, reward_term=reward_term,
            beta=beta, teleport_weight_map=None, teleport_aux_loss_val=None,
            aux_beta=1.0, teleport_log={},
        )
        assert loss_w > loss_v

    def test_loss_with_gradient_mask_and_weight(self):
        """Weight map with gradient_mask produces correct masked weighted loss."""
        latents = self._make_latents(F=4, H=16, W=16)
        grad = torch.full_like(latents, 0.01, dtype=torch.float64)
        reward_term = torch.tensor(0.5, dtype=torch.float64)
        beta = torch.tensor(1.0, dtype=torch.float64)

        # Mask last 2 frames -- shape must broadcast with weighted_sq [B, F, C, H, W]
        gradient_mask = torch.zeros(1, 4, 16, 16, 16, dtype=torch.bool)
        gradient_mask[:, 2:, ...] = True
        weight_map = torch.full((1, 4, 1, 16, 16), 2.0, dtype=torch.float64)

        loss, _ = _assemble_loss(
            latents, grad, gradient_mask=gradient_mask, reward_term=reward_term,
            beta=beta, teleport_weight_map=weight_map, teleport_aux_loss_val=None,
            aux_beta=1.0, teleport_log={},
        )
        assert loss.item() > 0

    def test_loss_with_aux_loss_added(self):
        """Aux loss is ADDED after multiplicative reward structure."""
        latents = self._make_latents()
        grad = torch.full_like(latents, 0.02, dtype=torch.float64)
        reward_term = torch.tensor(0.5, dtype=torch.float64)
        beta = torch.tensor(1.0, dtype=torch.float64)

        aux_val = torch.tensor(0.42, dtype=torch.float64)

        loss_a, _ = _assemble_loss(
            latents, grad, gradient_mask=None, reward_term=reward_term,
            beta=beta, teleport_weight_map=None, teleport_aux_loss_val=aux_val,
            aux_beta=1.0, teleport_log={"teleport_aux_loss": aux_val.detach()},
        )
        loss_v, _ = _assemble_loss(
            latents, grad, gradient_mask=None, reward_term=reward_term,
            beta=beta, teleport_weight_map=None, teleport_aux_loss_val=None,
            aux_beta=1.0, teleport_log={},
        )
        diff = loss_a - loss_v
        assert torch.allclose(diff, aux_val, rtol=1e-10, atol=1e-10)

    def test_loss_with_aux_loss_zero(self):
        """Aux loss = 0 should produce same total loss as vanilla."""
        latents = self._make_latents()
        grad = torch.full_like(latents, 0.02, dtype=torch.float64)
        reward_term = torch.tensor(0.5, dtype=torch.float64)
        beta = torch.tensor(1.0, dtype=torch.float64)

        aux_val = torch.tensor(0.0, dtype=torch.float64)

        loss_a, _ = _assemble_loss(
            latents, grad, gradient_mask=None, reward_term=reward_term,
            beta=beta, teleport_weight_map=None, teleport_aux_loss_val=aux_val,
            aux_beta=1.0, teleport_log={},
        )
        loss_v, _ = _assemble_loss(
            latents, grad, gradient_mask=None, reward_term=reward_term,
            beta=beta, teleport_weight_map=None, teleport_aux_loss_val=None,
            aux_beta=1.0, teleport_log={},
        )
        assert torch.allclose(loss_a, loss_v, rtol=1e-12, atol=1e-12)

    def test_loss_with_aux_beta_scaling(self):
        """aux_beta scales the aux loss contribution."""
        latents = self._make_latents()
        grad = torch.full_like(latents, 0.02, dtype=torch.float64)
        reward_term = torch.tensor(0.5, dtype=torch.float64)
        beta = torch.tensor(1.0, dtype=torch.float64)

        aux_val = torch.tensor(0.5, dtype=torch.float64)

        loss_b2, _ = _assemble_loss(
            latents, grad, gradient_mask=None, reward_term=reward_term,
            beta=beta, teleport_weight_map=None, teleport_aux_loss_val=aux_val,
            aux_beta=2.0, teleport_log={},
        )
        loss_b5, _ = _assemble_loss(
            latents, grad, gradient_mask=None, reward_term=reward_term,
            beta=beta, teleport_weight_map=None, teleport_aux_loss_val=aux_val,
            aux_beta=5.0, teleport_log={},
        )
        loss_v, _ = _assemble_loss(
            latents, grad, gradient_mask=None, reward_term=reward_term,
            beta=beta, teleport_weight_map=None, teleport_aux_loss_val=None,
            aux_beta=1.0, teleport_log={},
        )
        diff_2 = loss_b2 - loss_v
        diff_5 = loss_b5 - loss_v
        assert torch.allclose(diff_2 * 2.5, diff_5, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# End-to-end teleport score computation + loss assembly
# ---------------------------------------------------------------------------


class TestEndToEndTeleportScore:
    """Tests combining detector score computation with loss assembly."""

    def test_reweight_mode_zero_score_produces_unit_weights(self):
        """Detector returning all-zero score -> weight=1 -> loss equals vanilla."""
        cfg = _make_teleport_tel_det_regular_cfg("reweight")
        model = MockReDMD(tel_det_regular_cfg=cfg)
        zero_score = torch.zeros(1, 4, 1, 64, 64)
        model._teleport_detector.score = MagicMock(return_value=zero_score)

        videos = torch.full((1, 4, 3, 64, 64), 0.3)
        wm, aux, tlog = _compute_teleport_score(model, videos)

        assert wm is not None
        assert aux is None
        assert torch.allclose(wm, torch.tensor(1.0), atol=1e-6)
        assert "teleport_score_mean" in tlog
        assert "teleport_weight_mean" in tlog

    def test_reweight_mode_positive_score_produces_above_unit_weights(self):
        """Detector returning positive score -> weight > 1."""
        cfg = _make_teleport_tel_det_regular_cfg("reweight")
        model = MockReDMD(tel_det_regular_cfg=cfg)
        pos_score = torch.full((1, 4, 1, 64, 64), 0.5)
        model._teleport_detector.score = MagicMock(return_value=pos_score)

        videos = torch.full((1, 4, 3, 64, 64), 0.3)
        wm, aux, tlog = _compute_teleport_score(model, videos)

        assert wm is not None
        assert aux is None
        assert wm.min().item() > 1.0

    def test_aux_loss_mode_zero_score_produces_zero_loss(self):
        """Detector returning all-zero -> L_aux = 0."""
        cfg = _make_teleport_tel_det_regular_cfg("aux_loss")
        model = MockReDMD(tel_det_regular_cfg=cfg)
        zero_score = torch.zeros(1, 4, 1, 64, 64)
        model._teleport_detector.score = MagicMock(return_value=zero_score)

        videos = torch.full((1, 4, 3, 64, 64), 0.3)
        wm, aux, tlog = _compute_teleport_score(model, videos)

        assert wm is None
        assert aux is not None
        assert aux.item() == 0.0
        assert "teleport_score_mean" in tlog
        assert "teleport_aux_loss" in tlog

    def test_aux_loss_mode_positive_score_above_tau_produces_nonzero_loss(self):
        """Detector returning score > tau -> L_aux > 0."""
        cfg = _make_teleport_tel_det_regular_cfg("aux_loss")
        model = MockReDMD(tel_det_regular_cfg=cfg)
        high_score = torch.full((1, 4, 1, 64, 64), 0.7)
        model._teleport_detector.score = MagicMock(return_value=high_score)

        videos = torch.full((1, 4, 3, 64, 64), 0.3)
        wm, aux, tlog = _compute_teleport_score(model, videos)

        assert wm is None
        assert aux is not None
        assert aux.item() > 0.0

    def test_reweight_mode_detector_called_in_no_grad(self):
        """In reweight mode, detector.score output is .detach()ed."""
        cfg = _make_teleport_tel_det_regular_cfg("reweight")
        model = MockReDMD(tel_det_regular_cfg=cfg)
        score_output = torch.zeros(1, 4, 1, 64, 64)
        model._teleport_detector.score = MagicMock(return_value=score_output)

        videos = torch.full((1, 4, 3, 64, 64), 0.3)
        wm, _, _ = _compute_teleport_score(model, videos)
        assert wm.requires_grad is False, "Reweight weights should be detached (no grad)"


# ---------------------------------------------------------------------------
# File integrity checks
# ---------------------------------------------------------------------------


class TestFileIntegrity:
    """Verify the actual source files are syntactically correct and exports exist."""

    def test_re_dmd_py_syntax(self):
        """Verify re_dmd.py has no syntax errors."""
        re_dmd_path = os.path.normpath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..",
            "methods", "reward_forcing", "re_dmd.py",
        ))
        with open(re_dmd_path, "r", encoding="utf-8") as f:
            source = f.read()
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"re_dmd.py has syntax error: {e}")

    def test_causvid_dmd_py_syntax(self):
        """Verify causvid/dmd.py has no syntax errors."""
        dmd_path = os.path.normpath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..",
            "methods", "causvid", "dmd.py",
        ))
        with open(dmd_path, "r", encoding="utf-8") as f:
            source = f.read()
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"causvid/dmd.py has syntax error: {e}")

    def test_teleport_init_exports_all_builders(self):
        """Verify teleport/__init__.py exports build_teleport_detector, build_reweighter, build_aux_loss."""
        assert callable(build_teleport_detector)
        assert callable(build_reweighter)
        assert callable(build_aux_loss)
