"""Unit tests for StreamingTrainingModelPP pure-logic methods.

Tests only the methods that don't require GPU/model initialization:
_synced_random_int, sample_window. GPU-dependent methods (rollout_long,
compute_generator_loss) are verified via import smoke tests.

NOTE: We import the module directly and patch its `dist` attribute in-place
via `monkeypatch.setattr` rather than `unittest.mock.patch` with a dotted
path. The latter forces import of `methods.reward_forcing.__init__`, which
imports ReDMD -> triggers CUDA/torch_npu dependencies unavailable in CPU-only
test envs. Patching the already-imported module's attribute avoids that.
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


class TestSampleWindow:
    def test_returns_correct_shape(self):
        pp = _make_pp_without_init(sfpp_window_size=21)
        V = torch.randn(1, 150, 16, 60, 104)
        W = pp.sample_window(V)
        assert W.shape == (1, 21, 16, 60, 104)

    def test_window_is_detached(self):
        pp = _make_pp_without_init(sfpp_window_size=21)
        V = torch.randn(1, 150, 16, 60, 104, requires_grad=True)
        W = pp.sample_window(V)
        assert not W.requires_grad

    def test_custom_K(self):
        pp = _make_pp_without_init(sfpp_window_size=21)
        V = torch.randn(1, 150, 16, 60, 104)
        W = pp.sample_window(V, K=10)
        assert W.shape == (1, 10, 16, 60, 104)

    def test_assert_when_V_too_short(self):
        pp = _make_pp_without_init(sfpp_window_size=21)
        V = torch.randn(1, 10, 16, 60, 104)
        with pytest.raises(AssertionError):
            pp.sample_window(V)


class TestRolloutLengthTruncation:
    """Test that rollout_long respects rollout_length even when it's not
    a multiple of chunk_size. We test the truncation logic by mocking
    generate_chunk_with_cache to return deterministic chunks."""

    def test_rollout_respects_length_when_not_multiple_of_chunk(self):
        pp = _make_pp_without_init(
            sfpp_rollout_length=25,
            streaming_chunk_size=21,
        )
        pp.num_frame_per_block = 3
        pp.image_or_video_shape = [1, 25, 16, 60, 104]

        # Mock inference_pipeline
        pp.inference_pipeline = MagicMock()
        pp.inference_pipeline.kv_cache1 = None
        pp.inference_pipeline.crossattn_cache = None

        def mock_generate(noise, **kwargs):
            frames = noise.shape[1]
            return torch.randn(1, frames, 16, 60, 104), None, None

        def mock_init_kv(**kwargs):
            pp.inference_pipeline.kv_cache1 = [MagicMock()]
        def mock_init_cross(**kwargs):
            pp.inference_pipeline.crossattn_cache = [MagicMock()]

        pp.inference_pipeline._initialize_kv_cache = mock_init_kv
        pp.inference_pipeline._initialize_crossattn_cache = mock_init_cross
        pp.inference_pipeline.clear_kv_cache = MagicMock()
        pp.inference_pipeline.generate_chunk_with_cache = mock_generate

        V = pp.rollout_long(
            conditional_dict={"context": MagicMock()},
            unconditional_dict={"context": MagicMock()},
        )
        # 25 = 21 + 4 (4 is not multiple of 3, so truncated to 3)
        # So V should be 21+3=24 frames (<= 25)
        assert V.shape[1] <= pp.rollout_length
        assert V.shape[1] > 0


class TestComputeGeneratorLossDelegation:
    def test_calls_base_model_with_beta_zero(self):
        pp = _make_pp_without_init(sfpp_beta=0.0)
        pp.base_model = MagicMock()
        pp.base_model.vae.decode_to_pixel.return_value = torch.randn(1, 21, 3, 480, 832)
        pp.base_model.compute_rewarded_distribution_matching_loss.return_value = (
            torch.tensor(0.5, requires_grad=True), {"test": True}
        )

        window = torch.randn(1, 21, 16, 60, 104)
        cond = {"context": MagicMock()}
        uncond = {"context": MagicMock()}
        loss, log_dict = pp.compute_generator_loss(window, cond, uncond,
                                                    text_prompts=["test"])

        assert loss.item() == 0.5
        # Verify beta=0.0 was passed (pure DMD)
        call_kwargs = pp.base_model.compute_rewarded_distribution_matching_loss.call_args
        assert call_kwargs.kwargs.get("beta", None) == 0.0
        assert call_kwargs.kwargs.get("gradient_mask") is None


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
        # v had no grad, stays as-is

    def test_no_error_when_caches_none(self):
        pp = _make_pp_without_init()
        pp.inference_pipeline = MagicMock()
        pp.inference_pipeline.kv_cache1 = None
        pp.inference_pipeline.crossattn_cache = None
        # Should not raise
        pp._clear_cache_gradients()
