"""Unit tests for StreamingTrainingModelPP pure-logic methods.

Tests only the methods that don't require GPU/model initialization:
_synced_random_int, sample_window, rollout length logic. GPU-dependent
methods (rollout_and_sample_window, compute_generator_loss) are verified
via import smoke tests.
"""
import pytest
import torch
from unittest.mock import MagicMock
from methods.reward_forcing.streaming_training_pp import StreamingTrainingModelPP


def _make_pp_without_init(**config_overrides):
    """Create a StreamingTrainingModelPP bypassing __init__ (no GPU needed)."""
    pp = StreamingTrainingModelPP.__new__(StreamingTrainingModelPP)
    pp.device = torch.device("cpu")
    pp.dtype = torch.float32
    pp.rollout_length = config_overrides.get("sfpp_rollout_length", 150)
    pp.window_size = config_overrides.get("sfpp_window_size", 21)
    pp.beta = config_overrides.get("sfpp_beta", 0.0)
    pp.chunk_size = config_overrides.get("streaming_chunk_size", 21)
    pp.num_frame_per_block = 3
    return pp


class _FakeDist:
    """Minimal fake of torch.distributed for _synced_random_int tests."""
    def __init__(self, initialized=False, rank=0):
        self._initialized = initialized
        self._rank = rank
        self.broadcast = MagicMock()

    def is_initialized(self):
        return self._initialized

    def get_rank(self):
        return self._rank


class TestSyncedRandomInt:
    def test_single_process_returns_value_in_range(self, monkeypatch):
        pp = _make_pp_without_init()
        import methods.reward_forcing.streaming_training_pp as mod
        monkeypatch.setattr(mod, "dist", _FakeDist(initialized=False))
        val = pp._synced_random_int(0, 10)
        assert 0 <= val < 10

    def test_distributed_broadcasts_from_rank0(self, monkeypatch):
        pp = _make_pp_without_init()
        import methods.reward_forcing.streaming_training_pp as mod
        fake = _FakeDist(initialized=True, rank=0)
        monkeypatch.setattr(mod, "dist", fake)
        val = pp._synced_random_int(5, 10)
        assert 5 <= val < 10
        fake.broadcast.assert_called_once()

    def test_low_ge_high_returns_low(self, monkeypatch):
        pp = _make_pp_without_init()
        import methods.reward_forcing.streaming_training_pp as mod
        monkeypatch.setattr(mod, "dist", _FakeDist(initialized=False))
        assert pp._synced_random_int(5, 5) == 5
        assert pp._synced_random_int(5, 3) == 5


class TestComputeGeneratorLossDelegation:
    def test_passes_window_directly_to_dmd_with_beta_zero(self):
        """Verify compute_generator_loss passes the window (which has grad
        from rollout_and_sample_window) directly to DMD loss with beta=0.
        No separate denoising step -- the window IS the rollout output."""
        pp = _make_pp_without_init(sfpp_beta=0.0)
        pp.base_model = MagicMock()
        pp.base_model.vae.decode_to_pixel.return_value = torch.randn(1, 21, 3, 480, 832)
        pp.base_model.compute_rewarded_distribution_matching_loss.return_value = (
            torch.tensor(0.5, requires_grad=True), {"test": True}
        )

        # Window has grad (from rollout_and_sample_window's grad-bearing chunk)
        window = torch.randn(1, 21, 16, 60, 104, requires_grad=True)
        cond = {"context": MagicMock()}
        uncond = {"context": MagicMock()}
        loss, log_dict = pp.compute_generator_loss(window, cond, uncond,
                                                    text_prompts=["test"])

        assert loss.item() == 0.5
        # DMD loss receives the window directly (NOT a separate denoised output)
        dmd_call_args = pp.base_model.compute_rewarded_distribution_matching_loss.call_args
        dmd_input = dmd_call_args.kwargs.get("image_or_video")
        assert dmd_input is window, "DMD loss should receive the window directly"

        # Verify beta=0.0 was passed (pure DMD)
        assert dmd_call_args.kwargs.get("beta", None) == 0.0
        assert dmd_call_args.kwargs.get("gradient_mask") is None

    def test_scores_forwarded_to_dmd_when_provided(self):
        """Verify compute_generator_loss forwards scores to DMD loss."""
        pp = _make_pp_without_init(sfpp_beta=0.0)
        pp.base_model = MagicMock()
        pp.base_model.vae.decode_to_pixel.return_value = torch.randn(1, 21, 3, 480, 832)
        pp.base_model.compute_rewarded_distribution_matching_loss.return_value = (
            torch.tensor(0.5, requires_grad=True), {"test": True}
        )

        window = torch.randn(1, 21, 16, 60, 104, requires_grad=True)
        cond = {"context": MagicMock()}
        uncond = {"context": MagicMock()}
        scores = torch.tensor([0.7])
        pp.compute_generator_loss(window, cond, uncond,
                                  text_prompts=["test"], scores=scores)

        dmd_call_args = pp.base_model.compute_rewarded_distribution_matching_loss.call_args
        assert dmd_call_args.kwargs.get("scores") is scores, "scores should be forwarded to DMD loss"

    def test_scores_defaults_to_none_when_not_provided(self):
        """Verify compute_generator_loss passes scores=None by default."""
        pp = _make_pp_without_init(sfpp_beta=0.0)
        pp.base_model = MagicMock()
        pp.base_model.vae.decode_to_pixel.return_value = torch.randn(1, 21, 3, 480, 832)
        pp.base_model.compute_rewarded_distribution_matching_loss.return_value = (
            torch.tensor(0.5, requires_grad=True), {"test": True}
        )

        window = torch.randn(1, 21, 16, 60, 104, requires_grad=True)
        cond = {"context": MagicMock()}
        uncond = {"context": MagicMock()}
        pp.compute_generator_loss(window, cond, uncond, text_prompts=["test"])

        dmd_call_args = pp.base_model.compute_rewarded_distribution_matching_loss.call_args
        assert dmd_call_args.kwargs.get("scores") is None, "scores should default to None"


class TestClearCacheGradients:
    def test_detaches_kv_cache_tensors(self):
        pp = _make_pp_without_init()
        grad_t = torch.randn(2, 2, requires_grad=True)
        no_grad_t = torch.randn(2, 2)
        pp.inference_pipeline = MagicMock()
        pp.inference_pipeline.kv_cache1 = [{"k": grad_t.clone(), "v": no_grad_t.clone()}]
        pp.inference_pipeline.crossattn_cache = None

        pp._clear_cache_gradients()

        assert not pp.inference_pipeline.kv_cache1[0]["k"].requires_grad

    def test_no_error_when_caches_none(self):
        pp = _make_pp_without_init()
        pp.inference_pipeline = MagicMock()
        pp.inference_pipeline.kv_cache1 = None
        pp.inference_pipeline.crossattn_cache = None
        pp._clear_cache_gradients()
