"""Unit tests for the KV-cache random-zeroing augmentation in streaming training.

These tests exercise the logic added to ``methods/reward_forcing/streaming_training.py``
without requiring a real Wan model or GPU.  They verify:

1. When both probabilities are 0, ``_generate_chunk`` is a passthrough (no
   save/restore, no zeroing, no extra context-update calls).
2. ``_maybe_zero_kv_cache`` zeros only the sink region when ``zero_sink`` fires.
3. ``_maybe_zero_kv_cache`` zeros only the window region when ``zero_window`` fires.
4. ``_maybe_zero_kv_cache`` zeros both regions when both fire.
5. ``_save_kv_cache`` / ``_restore_kv_cache`` round-trip preserves the cache.
6. After ``_generate_chunk`` with zeroing, the KV cache equals the restored
   history plus the new tokens written by ``_context_update_for_chunk`` — i.e.
   the next chunk still sees the full history.
7. The first chunk (``chunk_start_frame == 0``) is never zeroed even when
   probabilities are 1.0.
8. Cross-rank synchronisation path is exercised (decisions tensor is broadcast).
"""

import importlib
import sys
import types
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
#  Stub out heavy modules so importing streaming_training doesn't pull in
#  CUDA-dependent code paths (wan.modules.t5 calls torch.cuda.current_device
#  at class-definition time).
# ---------------------------------------------------------------------------
# Map of package module name -> real filesystem path (so submodule imports work).
# None means a pure stub with no real path.
import os as _os
_pkg_specs = {
    "core": None,
    "core.misc": None,
    "core.misc.debug_option": None,
    "core.misc.memory": None,
    "core.wan_wrapper": None,
    "core.scheduler": None,
    "core.loss": None,
    "core.data": None,
    "core.distributed": None,
    "methods": "methods",
    "methods.reward_forcing": "methods/reward_forcing",
    "methods.reward_forcing.pipelines": "methods/reward_forcing/pipelines",
    "methods.reward_forcing.pipelines.streaming_switch_training": None,
    "methods.reward_forcing.pipelines.reward_forcing_training": None,
    "methods.reward_forcing.pipelines.self_forcing_training": None,
    "methods.base": "methods/base",
    "methods.base.base_reward_forcing": None,
    "wan": None,
    "wan.modules": None,
    "wan.utils": None,
}
for _mod_name, _fs_path in _pkg_specs.items():
    if _mod_name not in sys.modules:
        _m = types.ModuleType(_mod_name)
        if _fs_path is not None and _os.path.isdir(_fs_path):
            _m.__path__ = [_os.path.abspath(_fs_path)]
        else:
            _m.__path__ = []
        sys.modules[_mod_name] = _m

# debug_option stubs
sys.modules["core.misc.debug_option"].DEBUG = False
sys.modules["core.misc.debug_option"].LOG_GPU_MEMORY = False
sys.modules["core.misc.debug_option"].DEBUG_GRADIENT = False
# memory stub
sys.modules["core.misc.memory"].log_gpu_memory = lambda *a, **kw: None
# streaming_switch_training stub
class _DummySwitchPipeline:  # noqa: E302
    pass
sys.modules["methods.reward_forcing.pipelines.streaming_switch_training"].StreamingSwitchTrainingPipeline = _DummySwitchPipeline
# Stub the pipeline classes and base class to avoid heavy imports.
sys.modules["methods.reward_forcing.pipelines.reward_forcing_training"].RewardForcingTrainingPipeline = type("_Stub", (), {})
sys.modules["methods.reward_forcing.pipelines.self_forcing_training"].SelfForcingTrainingPipeline = type("_Stub", (), {})
class _StubRewardForcingModel:  # noqa: E302
    pass
sys.modules["methods.base.base_reward_forcing"].RewardForcingModel = _StubRewardForcingModel
# core.loss stub
sys.modules["core.loss"].get_denoising_loss = lambda name: lambda: (lambda **kw: torch.tensor(0.0))

# Now import the modules under test via importlib so they pick up the stubs.
_streaming_mod = importlib.import_module("methods.reward_forcing.streaming_training")
StreamingTrainingModel = _streaming_mod.StreamingTrainingModel
_re_dmd_mod = importlib.import_module("methods.reward_forcing.re_dmd")
ReDMD = _re_dmd_mod.ReDMD

# Register submodule attributes on the stub parent packages so that
# ``mock.patch("methods.reward_forcing.streaming_training.dist")`` can resolve.
sys.modules["methods"].reward_forcing = sys.modules["methods.reward_forcing"]
sys.modules["methods.reward_forcing"].streaming_training = _streaming_mod
sys.modules["methods.reward_forcing"].re_dmd = _re_dmd_mod
# Ensure dist attribute exists on the module for patching.
if not hasattr(_streaming_mod, "dist"):
    _streaming_mod.dist = torch.distributed


# ---------------------------------------------------------------------------
#  Mock helpers
# ---------------------------------------------------------------------------

class _FakeScheduler:
    """Minimal scheduler that implements ``add_noise`` like FlowMatchScheduler."""

    def add_noise(self, x0, noise, timesteps):
        # Return a deterministic combination so tests can inspect it.
        return x0 + noise * 0.0  # just x0 (noise neutralised)

    def convert_x0_to_noise(self, x0, xt, timestep):
        return x0 - xt


class _FakeSelfAttn(nn.Module):
    def __init__(self, sink_size):
        super().__init__()
        self.sink_size = sink_size


class _FakeBlock(nn.Module):
    def __init__(self, sink_size):
        super().__init__()
        self.self_attn = _FakeSelfAttn(sink_size)


class _FakeGeneratorModel(nn.Module):
    """Holds ``blocks`` so ``_get_sink_size`` works and records forward calls."""

    def __init__(self, sink_size, num_blocks=2):
        super().__init__()
        self.local_attn_size = -1
        self.max_attention_size = 32760
        self.blocks = nn.ModuleList([_FakeBlock(sink_size) for _ in range(num_blocks)])
        self.forward_call_count = 0
        # Record (input_sum, current_start) for each call so tests can inspect.
        self.call_log = []

    def forward(self, noisy_image_or_video, conditional_dict, timestep,
                kv_cache=None, crossattn_cache=None, current_start=0):
        self.forward_call_count += 1
        # Sum the input so we can later verify which tokens were "written".
        self.call_log.append({
            "input": noisy_image_or_video.detach().clone(),
            "current_start": current_start,
        })
        # Write the *input* tokens into the KV cache at the right slot so that
        # the context-update simulation actually modifies the cache.
        # kv_cache may be a list of per-block dicts or a single dict.
        if kv_cache is not None:
            cache_list = kv_cache if isinstance(kv_cache, list) else [kv_cache]
            for _ci, kv_cache_blk in enumerate(cache_list):
                frame_seq_length = 1560
                num_new = noisy_image_or_video.shape[1] * frame_seq_length
                local_end = kv_cache_blk["local_end_index"].item()
                new_local_start = local_end
                new_local_end = local_end + num_new
                cache_size = kv_cache_blk["k"].shape[1]
                if new_local_end > cache_size:
                    new_local_end = cache_size
                    new_local_start = max(0, new_local_end - num_new)
                B = noisy_image_or_video.shape[0]
                token_repr = noisy_image_or_video.reshape(B, -1)
                actual_new = new_local_end - new_local_start
                if actual_new > 0:
                    tile = token_repr.mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
                    tile = tile.expand(B, actual_new, 12, 128)
                    kv_cache_blk["k"][:, new_local_start:new_local_end] = tile
                    kv_cache_blk["v"][:, new_local_start:new_local_end] = tile
                kv_cache_blk["local_end_index"].fill_(new_local_end)
                kv_cache_blk["global_end_index"].fill_(current_start + num_new)
        return None, noisy_image_or_video


class _FakeGenerator(nn.Module):
    """Wraps the fake model — mirrors WanDiffusionWrapper.model access."""

    def __init__(self, sink_size):
        super().__init__()
        self.model = _FakeGeneratorModel(sink_size)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class _FakePipeline:
    """Stand-in for StreamingTrainingPipeline with the methods we need."""

    def __init__(self, sink_size, num_transformer_blocks=4, kv_cache_size=20000,
                 frame_seq_length=1560, num_frame_per_block=3, context_noise=0):
        self.generator = _FakeGenerator(sink_size)
        self.scheduler = _FakeScheduler()
        self.num_transformer_blocks = num_transformer_blocks
        self.kv_cache_size = kv_cache_size
        self.frame_seq_length = frame_seq_length
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.kv_cache1 = None
        self.crossattn_cache = None
        self.denoising_step_list = [999, 500, 0]
        # Track calls to generate_chunk_with_cache
        self.gen_chunk_calls = []

    def _initialize_kv_cache(self, batch_size, dtype, device):
        self.kv_cache1 = []
        for _ in range(self.num_transformer_blocks):
            self.kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        self.crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            self.crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False,
            })

    def clear_kv_cache(self):
        if self.kv_cache1 is not None:
            for blk in self.kv_cache1:
                blk["k"].zero_()
                blk["v"].zero_()
                blk["global_end_index"].zero_()
                blk["local_end_index"].zero_()
        if self.crossattn_cache is not None:
            for blk in self.crossattn_cache:
                blk["k"].zero_()
                blk["v"].zero_()
                blk["is_init"] = False

    def generate_chunk_with_cache(self, noise, conditional_dict, *,
                                  current_start_frame=0, requires_grad=True,
                                  return_sim_step=False, **extra):
        """Fake chunk generation: returns the noise as 'output' and writes tokens
        into the KV cache via the fake generator forward."""
        self.gen_chunk_calls.append({
            "current_start_frame": current_start_frame,
            "noise_shape": tuple(noise.shape),
        })
        # Simulate the generation: call generator forward for each block so the
        # cache gets populated (this mirrors the real pipeline's behaviour).
        num_frames = noise.shape[1]
        local_start = 0
        for bi in range(num_frames // self.num_frame_per_block):
            cur = self.num_frame_per_block
            block_in = noise[:, local_start:local_start + cur]
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=block_in,
                    conditional_dict=conditional_dict,
                    timestep=torch.zeros([block_in.shape[0], cur], dtype=torch.int64, device=noise.device),
                    kv_cache=self.kv_cache1,  # pass the full list (matches real pipeline)
                    crossattn_cache=self.crossattn_cache,
                    current_start=(current_start_frame + local_start) * self.frame_seq_length,
                )
            local_start += cur
        return noise, 999, 500


class _FakeBaseModel:
    """Minimal stand-in for ReDMD used by StreamingTrainingModel.__init__."""

    def __init__(self, sink_size, device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.num_frame_per_block = 3
        self.scheduler = _FakeScheduler()
        self.denoising_loss_func = lambda **kw: torch.tensor(0.0)
        self.generator = _FakeGenerator(sink_size)
        self.fake_score = _FakeGenerator(sink_size)
        self.inference_pipeline = _FakePipeline(sink_size)
        self.vae = SimpleNamespace(decode_to_pixel=lambda x, **kw: x)

    def _initialize_inference_pipeline(self):
        # Already set in __init__; no-op.
        pass


def _make_config(sink_prob=0.0, window_prob=0.0, chunk_size=21, max_length=63):
    return SimpleNamespace(
        streaming_chunk_size=chunk_size,
        streaming_max_length=max_length,
        streaming_possible_max_length=None,
        streaming_min_new_frame=18,
        image_or_video_shape=[1, 21, 16, 60, 104],
        sink_kv_cache_zero_prob=sink_prob,
        window_kv_cache_zero_prob=window_prob,
    )


def _make_model(sink_size=2, sink_prob=0.0, window_prob=0.0):
    """Create a StreamingTrainingModel with all mocked dependencies."""
    base = _FakeBaseModel(sink_size=sink_size)
    cfg = _make_config(sink_prob=sink_prob, window_prob=window_prob)
    # Ensure pipeline cache is initialised before model construction.
    base.inference_pipeline._initialize_kv_cache(1, torch.float32, "cpu")
    base.inference_pipeline._initialize_crossattn_cache(1, torch.float32, "cpu")
    model = StreamingTrainingModel(base, cfg)
    # Set up a dummy sequence state so _generate_chunk can proceed.
    model.state["conditional_info"] = {
        "conditional_dict": {"context": torch.zeros(1, 10)},
        "unconditional_dict": {"context": torch.zeros(1, 10)},
        "text_prompts": ["test"],
        "scores": None,
    }
    model.state["temp_max_length"] = 63
    return model, base


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------

def test_prob_zero_is_passthrough():
    """Both probabilities 0 → _generate_chunk calls pipeline exactly once, no
    save/restore overhead, cache unchanged by zeroing logic, output_zeroed=None."""
    model, base = _make_model(sink_size=2, sink_prob=0.0, window_prob=0.0)
    pipe = base.inference_pipeline

    # Pre-fill the cache with sentinel values to detect any zeroing.
    for blk in pipe.kv_cache1:
        blk["k"].fill_(7.0)
        blk["v"].fill_(7.0)
        blk["local_end_index"].fill_(50)
        blk["global_end_index"].fill_(50)

    noise = torch.randn(1, 3, 16, 60, 104)
    initial_gen_chunk_calls = len(pipe.gen_chunk_calls)

    with mock.patch("methods.reward_forcing.streaming_training.dist") as dist_mock:
        dist_mock.is_initialized.return_value = False
        out_full, ts_from, ts_to, out_zeroed = model._generate_chunk(
            noise_chunk=noise, chunk_start_frame=3, requires_grad=True
        )

    # generate_chunk_with_cache called exactly once (no second zeroed generation).
    assert len(pipe.gen_chunk_calls) == initial_gen_chunk_calls + 1
    # output_zeroed must be None (zeroing disabled).
    assert out_zeroed is None, "output_zeroed should be None when zero prob is 0"
    # The sentinel values must be untouched (no zeroing).
    for blk in pipe.kv_cache1:
        assert torch.all(blk["k"][:, :50] == 7.0), "Cache was modified despite zero prob"
        assert torch.all(blk["v"][:, :50] == 7.0), "Cache was modified despite zero prob"
    print("[PASS] test_prob_zero_is_passthrough")


def test_maybe_zero_sink_only():
    """zero_sink=True, zero_window=False → only sink region zeroed."""
    model, base = _make_model(sink_size=2, sink_prob=1.0, window_prob=0.0)
    pipe = base.inference_pipeline
    frame_seq = model.frame_seq_length  # 1560
    total_sink = 2 * frame_seq

    # Fill cache with sentinels; set local_end beyond sink.
    for blk in pipe.kv_cache1:
        blk["k"].fill_(5.0)
        blk["v"].fill_(5.0)
        blk["local_end_index"].fill_(total_sink + 30)
        blk["global_end_index"].fill_(total_sink + 30)

    with mock.patch("methods.reward_forcing.streaming_training.dist") as dist_mock:
        dist_mock.is_initialized.return_value = False
        info = model._maybe_zero_kv_cache(device=torch.device("cpu"))

    assert info["zero_sink"] is True
    assert info["zero_window"] is False
    for blk in pipe.kv_cache1:
        # Sink region zeroed.
        assert torch.all(blk["k"][:, :total_sink] == 0.0)
        assert torch.all(blk["v"][:, :total_sink] == 0.0)
        # Window region (after sink, up to local_end) untouched.
        assert torch.all(blk["k"][:, total_sink:total_sink + 30] == 5.0)
        assert torch.all(blk["v"][:, total_sink:total_sink + 30] == 5.0)
    print("[PASS] test_maybe_zero_sink_only")


def test_maybe_zero_window_only():
    """zero_sink=False, zero_window=True → only window region zeroed."""
    model, base = _make_model(sink_size=2, sink_prob=0.0, window_prob=1.0)
    pipe = base.inference_pipeline
    frame_seq = model.frame_seq_length
    total_sink = 2 * frame_seq

    for blk in pipe.kv_cache1:
        blk["k"].fill_(9.0)
        blk["v"].fill_(9.0)
        blk["local_end_index"].fill_(total_sink + 40)
        blk["global_end_index"].fill_(total_sink + 40)

    with mock.patch("methods.reward_forcing.streaming_training.dist") as dist_mock:
        dist_mock.is_initialized.return_value = False
        info = model._maybe_zero_kv_cache(device=torch.device("cpu"))

    assert info["zero_sink"] is False
    assert info["zero_window"] is True
    for blk in pipe.kv_cache1:
        # Sink untouched.
        assert torch.all(blk["k"][:, :total_sink] == 9.0)
        assert torch.all(blk["v"][:, :total_sink] == 9.0)
        # Window zeroed.
        assert torch.all(blk["k"][:, total_sink:total_sink + 40] == 0.0)
        assert torch.all(blk["v"][:, total_sink:total_sink + 40] == 0.0)
    print("[PASS] test_maybe_zero_window_only")


def test_maybe_zero_both():
    """Both flags True → both regions zeroed."""
    model, base = _make_model(sink_size=2, sink_prob=1.0, window_prob=1.0)
    pipe = base.inference_pipeline
    total_sink = 2 * model.frame_seq_length

    for blk in pipe.kv_cache1:
        blk["k"].fill_(3.0)
        blk["v"].fill_(3.0)
        blk["local_end_index"].fill_(total_sink + 20)
        blk["global_end_index"].fill_(total_sink + 20)

    with mock.patch("methods.reward_forcing.streaming_training.dist") as dist_mock:
        dist_mock.is_initialized.return_value = False
        info = model._maybe_zero_kv_cache(device=torch.device("cpu"))

    assert info["zero_sink"] is True
    assert info["zero_window"] is True
    for blk in pipe.kv_cache1:
        assert torch.all(blk["k"][:, :total_sink + 20] == 0.0)
        assert torch.all(blk["v"][:, :total_sink + 20] == 0.0)
    print("[PASS] test_maybe_zero_both")


def test_save_restore_roundtrip():
    """_save_kv_cache then _restore_kv_cache reproduces the original exactly."""
    model, base = _make_model(sink_size=2, sink_prob=0.0, window_prob=0.0)
    pipe = base.inference_pipeline

    # Fill with random data.
    for blk in pipe.kv_cache1:
        blk["k"].copy_(torch.randn_like(blk["k"]))
        blk["v"].copy_(torch.randn_like(blk["v"]))
        blk["local_end_index"].fill_(77)
        blk["global_end_index"].fill_(88)

    saved = model._save_kv_cache()

    # Mutate the cache.
    for blk in pipe.kv_cache1:
        blk["k"].fill_(0.0)
        blk["v"].fill_(0.0)
        blk["local_end_index"].fill_(0)
        blk["global_end_index"].fill_(0)

    model._restore_kv_cache(saved)

    for blk, sb in zip(pipe.kv_cache1, saved):
        assert torch.equal(blk["k"], sb["k"])
        assert torch.equal(blk["v"], sb["v"])
        assert torch.equal(blk["local_end_index"], sb["local_end_index"])
        assert torch.equal(blk["global_end_index"], sb["global_end_index"])
    print("[PASS] test_save_restore_roundtrip")


def test_generate_chunk_restores_history_and_writes_new_tokens():
    """Full _generate_chunk with zeroing (requires_grad=True, prob=1.0):
    1. output_zeroed is produced (not None)
    2. After the call, the cache equals the post-full-cache state (history +
       full-chunk tokens), NOT the zeroed state — the next chunk sees full history.
    3. generate_chunk_with_cache was called twice (full + zeroed)."""
    model, base = _make_model(sink_size=2, sink_prob=1.0, window_prob=1.0)
    pipe = base.inference_pipeline
    total_sink = 2 * model.frame_seq_length

    # Simulate a history: fill sink and window with distinct sentinels.
    for blk in pipe.kv_cache1:
        blk["k"].fill_(0.0)
        blk["v"].fill_(0.0)
        # Sink = 1.0
        blk["k"][:, :total_sink].fill_(1.0)
        blk["v"][:, :total_sink].fill_(1.0)
        # Window history = 2.0 for 30 tokens after sink
        blk["k"][:, total_sink:total_sink + 30].fill_(2.0)
        blk["v"][:, total_sink:total_sink + 30].fill_(2.0)
        blk["local_end_index"].fill_(total_sink + 30)
        blk["global_end_index"].fill_(total_sink + 30)

    noise = torch.randn(1, 3, 16, 60, 104)  # one block of 3 frames
    chunk_start = 30  # non-zero so zeroing is eligible
    initial_gen_chunk_calls = len(pipe.gen_chunk_calls)

    with mock.patch("methods.reward_forcing.streaming_training.dist") as dist_mock:
        dist_mock.is_initialized.return_value = False
        out_full, _, _, out_zeroed = model._generate_chunk(
            noise_chunk=noise, chunk_start_frame=chunk_start, requires_grad=True
        )

    # 1. output_zeroed must be produced.
    assert out_zeroed is not None, "output_zeroed should be produced when zeroing fires"
    assert out_zeroed.shape == out_full.shape, "output_zeroed shape must match output_full"

    # 2. generate_chunk_with_cache was called twice (full + zeroed).
    assert len(pipe.gen_chunk_calls) == initial_gen_chunk_calls + 2, \
        "Expected 2 generate_chunk_with_cache calls (full + zeroed)"

    # 3. After the call, the cache should be in the post-full state:
    #    - Sink history restored to 1.0 (NOT zero — the zeroed generation's
    #      cache modifications were overwritten by the post-full restore).
    #    - Window history restored to 2.0.
    #    - New tokens written by the FULL-cache generation (not the zeroed one).
    for blk in pipe.kv_cache1:
        assert torch.all(blk["k"][:, :total_sink] == 1.0), \
            "Sink history was not restored to post-full state!"
        assert torch.all(blk["v"][:, :total_sink] == 1.0), \
            "Sink history was not restored to post-full state!"
        # The window history before the new chunk should be restored to 2.0.
        assert torch.all(blk["k"][:, total_sink:total_sink + 30] == 2.0), \
            "Window history was not restored to post-full state!"
        # The new tokens (3 frames * 1560 = 4680 tokens) should have been
        # written after the existing history by the FULL-cache generation.
        new_token_start = total_sink + 30
        new_token_end = new_token_start + 3 * model.frame_seq_length
        new_tokens = blk["k"][:, new_token_start:new_token_end]
        assert new_tokens.abs().sum() > 0, "New tokens from full generation were not written!"
    print("[PASS] test_generate_chunk_restores_history_and_writes_new_tokens")


def test_first_chunk_not_zeroed():
    """chunk_start_frame == 0 → no zeroing even with prob=1.0.
    Also: requires_grad=False → no zeroing (critic path)."""
    model, base = _make_model(sink_size=2, sink_prob=1.0, window_prob=1.0)
    pipe = base.inference_pipeline
    total_sink = 2 * model.frame_seq_length

    for blk in pipe.kv_cache1:
        blk["k"].fill_(42.0)
        blk["v"].fill_(42.0)
        blk["local_end_index"].fill_(total_sink + 10)
        blk["global_end_index"].fill_(total_sink + 10)

    noise = torch.randn(1, 3, 16, 60, 104)
    initial_gen_chunk_calls = len(pipe.gen_chunk_calls)

    with mock.patch("methods.reward_forcing.streaming_training.dist") as dist_mock:
        dist_mock.is_initialized.return_value = False
        out_full, _, _, out_zeroed = model._generate_chunk(
            noise_chunk=noise, chunk_start_frame=0, requires_grad=True
        )

    # output_zeroed must be None (first chunk, no history to zero).
    assert out_zeroed is None, "First chunk should not produce zeroed output"
    # Only one generation call (no second zeroed pass).
    assert len(pipe.gen_chunk_calls) == initial_gen_chunk_calls + 1
    # The sentinel 42.0 in the sink must survive (no zeroing on first chunk).
    for blk in pipe.kv_cache1:
        assert torch.all(blk["k"][:, :total_sink] == 42.0), \
            "First chunk was zeroed despite chunk_start_frame == 0!"
    print("[PASS] test_first_chunk_not_zeroed")


def test_critic_path_not_zeroed():
    """requires_grad=False (critic path) → no zeroing even with prob=1.0."""
    model, base = _make_model(sink_size=2, sink_prob=1.0, window_prob=1.0)
    pipe = base.inference_pipeline
    total_sink = 2 * model.frame_seq_length

    for blk in pipe.kv_cache1:
        blk["k"].fill_(99.0)
        blk["v"].fill_(99.0)
        blk["local_end_index"].fill_(total_sink + 10)
        blk["global_end_index"].fill_(total_sink + 10)

    noise = torch.randn(1, 3, 16, 60, 104)
    initial_gen_chunk_calls = len(pipe.gen_chunk_calls)

    with mock.patch("methods.reward_forcing.streaming_training.dist") as dist_mock:
        dist_mock.is_initialized.return_value = False
        out_full, _, _, out_zeroed = model._generate_chunk(
            noise_chunk=noise, chunk_start_frame=30, requires_grad=False
        )

    # Critic path: no zeroing, no second generation.
    assert out_zeroed is None, "Critic path should not produce zeroed output"
    assert len(pipe.gen_chunk_calls) == initial_gen_chunk_calls + 1
    # Cache sentinel preserved.
    for blk in pipe.kv_cache1:
        assert torch.all(blk["k"][:, :total_sink] == 99.0), \
            "Critic path cache was modified!"
    print("[PASS] test_critic_path_not_zeroed")


def test_cross_rank_sync():
    """When dist is initialised, the random decision is broadcast from rank 0."""
    model, base = _make_model(sink_size=2, sink_prob=1.0, window_prob=1.0)
    pipe = base.inference_pipeline

    with mock.patch("methods.reward_forcing.streaming_training.dist") as dist_mock:
        dist_mock.is_initialized.return_value = True
        dist_mock.get_rank.return_value = 0
        # Capture the tensor passed to broadcast.
        broadcasted = {}

        def fake_broadcast(tensor, src):
            broadcasted["tensor"] = tensor.clone()
            return tensor

        dist_mock.broadcast.side_effect = fake_broadcast

        info = model._maybe_zero_kv_cache(device=torch.device("cpu"))

        # broadcast was called with a 2-element tensor.
        assert "tensor" in broadcasted
        assert broadcasted["tensor"].shape == (2,)
        assert info["zero_sink"] is True  # prob=1.0 → always zero
        assert info["zero_window"] is True
    print("[PASS] test_cross_rank_sync")


def test_prob_between_zero_and_one_can_skip():
    """With probability < 1.0, it is possible (though rare) to skip zeroing.
    Use a very small probability and a fixed seed to get the skip case."""
    model, base = _make_model(sink_size=2, sink_prob=0.01, window_prob=0.01)
    pipe = base.inference_pipeline
    total_sink = 2 * model.frame_seq_length

    for blk in pipe.kv_cache1:
        blk["k"].fill_(1.0)
        blk["v"].fill_(1.0)
        blk["local_end_index"].fill_(total_sink + 10)
        blk["global_end_index"].fill_(total_sink + 10)

    # Force random.random() to return a value > 0.01 so zeroing is skipped.
    with mock.patch("methods.reward_forcing.streaming_training.dist") as dist_mock:
        dist_mock.is_initialized.return_value = False
        with mock.patch("methods.reward_forcing.streaming_training._random") as rnd_mock:
            rnd_mock.random.return_value = 0.5
            info = model._maybe_zero_kv_cache(device=torch.device("cpu"))

    assert info["zero_sink"] is False
    assert info["zero_window"] is False
    for blk in pipe.kv_cache1:
        assert torch.all(blk["k"][:, :total_sink + 10] == 1.0), "Cache zeroed when it should have been skipped"
    print("[PASS] test_prob_between_zero_and_one_can_skip")


def test_generate_chunk_zeroed_output_is_detached():
    """When zeroing fires, output_zeroed must be detached (no grad)."""
    model, base = _make_model(sink_size=2, sink_prob=1.0, window_prob=1.0)
    pipe = base.inference_pipeline
    total_sink = 2 * model.frame_seq_length

    for blk in pipe.kv_cache1:
        blk["k"].fill_(1.0)
        blk["v"].fill_(1.0)
        blk["local_end_index"].fill_(total_sink + 30)
        blk["global_end_index"].fill_(total_sink + 30)

    noise = torch.randn(1, 3, 16, 60, 104)

    with mock.patch("methods.reward_forcing.streaming_training.dist") as dist_mock:
        dist_mock.is_initialized.return_value = False
        out_full, _, _, out_zeroed = model._generate_chunk(
            noise_chunk=noise, chunk_start_frame=30, requires_grad=True
        )

    assert out_zeroed is not None
    assert not out_zeroed.requires_grad, "output_zeroed must be detached (no gradients)"
    print("[PASS] test_generate_chunk_zeroed_output_is_detached")


def test_generate_next_chunk_includes_chunk_zeroed_in_info():
    """generate_next_chunk should include 'chunk_zeroed' in the returned info dict,
    and it should be None when zeroing doesn't fire (prob=0)."""
    model, base = _make_model(sink_size=2, sink_prob=0.0, window_prob=0.0)

    with mock.patch("methods.reward_forcing.streaming_training.dist") as dist_mock:
        dist_mock.is_initialized.return_value = False
        # Mock can_generate_more to always allow.
        model.can_generate_more = lambda: True
        chunk, info = model.generate_next_chunk(requires_grad=True)

    assert "chunk_zeroed" in info, "info dict must contain 'chunk_zeroed' key"
    assert info["chunk_zeroed"] is None, "chunk_zeroed should be None when prob=0"
    print("[PASS] test_generate_next_chunk_includes_chunk_zeroed_in_info")


def test_unified_cfg_uses_zeroed_noisy_for_uncond():
    """Verify that _compute_kl_grad uses noisy_image_or_video_zeroed for the
    uncond path when provided, and falls back to noisy_image_or_video when None.

    We monkey-patch the fake_score and real_score callables to record which
    noisy tensor is passed with which conditional_dict.
    """
    model, base = _make_model(sink_size=2, sink_prob=0.0, window_prob=0.0)

    # Create a fake ReDMD-like object with instrumented score functions.
    call_log = []

    class _FakeScore(nn.Module):
        def __init__(self, name):
            super().__init__()
            self._name = name
        def forward(self, noisy_image_or_video, conditional_dict, timestep):
            is_cond = "cond" if conditional_dict.get("is_cond", False) else "uncond"
            call_log.append({
                "score": self._name,
                "is_cond": is_cond,
                "noisy_id": id(noisy_image_or_video),
            })
            return None, noisy_image_or_video

    class _FakeReDMD:
        fake_guidance_scale = 1.0
        real_guidance_scale = 1.0
        def __init__(self):
            self.fake_score = _FakeScore("fake")
            self.real_score = _FakeScore("real")

    fake_dmd = _FakeReDMD()

    # We need to call _compute_kl_grad which is a method on ReDMD.
    # Import the unbound method and bind it to our fake object.
    from methods.reward_forcing.re_dmd import ReDMD
    compute_kl_grad = ReDMD._compute_kl_grad.__get__(fake_dmd, _FakeReDMD)

    noisy_full = torch.randn(1, 3, 4, 4, 4)
    noisy_zeroed = torch.randn(1, 3, 4, 4, 4)
    cond = {"is_cond": True}
    uncond = {"is_cond": False}
    timestep = torch.zeros(1, 3, dtype=torch.int64)

    # Case 1: with noisy_zeroed → uncond should use noisy_zeroed
    call_log.clear()
    grad, _ = compute_kl_grad(
        noisy_image_or_video=noisy_full,
        estimated_clean_image_or_video=noisy_full,
        timestep=timestep,
        conditional_dict=cond,
        unconditional_dict=uncond,
        noisy_image_or_video_zeroed=noisy_zeroed,
    )

    # Check: fake_score called twice (cond with noisy_full, uncond with noisy_zeroed)
    fake_calls = [c for c in call_log if c["score"] == "fake"]
    assert len(fake_calls) == 2
    cond_call = [c for c in fake_calls if c["is_cond"] == "cond"][0]
    uncond_call = [c for c in fake_calls if c["is_cond"] == "uncond"][0]
    assert cond_call["noisy_id"] == id(noisy_full), "cond path should use noisy_full"
    assert uncond_call["noisy_id"] == id(noisy_zeroed), "uncond path should use noisy_zeroed"

    # Check: real_score same pattern
    real_calls = [c for c in call_log if c["score"] == "real"]
    assert len(real_calls) == 2
    cond_call = [c for c in real_calls if c["is_cond"] == "cond"][0]
    uncond_call = [c for c in real_calls if c["is_cond"] == "uncond"][0]
    assert cond_call["noisy_id"] == id(noisy_full), "real cond should use noisy_full"
    assert uncond_call["noisy_id"] == id(noisy_zeroed), "real uncond should use noisy_zeroed"

    # Case 2: without noisy_zeroed (None) → uncond should use noisy_full (original behaviour)
    call_log.clear()
    grad, _ = compute_kl_grad(
        noisy_image_or_video=noisy_full,
        estimated_clean_image_or_video=noisy_full,
        timestep=timestep,
        conditional_dict=cond,
        unconditional_dict=uncond,
        noisy_image_or_video_zeroed=None,
    )

    fake_calls = [c for c in call_log if c["score"] == "fake"]
    uncond_call = [c for c in fake_calls if c["is_cond"] == "uncond"][0]
    assert uncond_call["noisy_id"] == id(noisy_full), \
        "When noisy_zeroed is None, uncond should fall back to noisy_full"

    print("[PASS] test_unified_cfg_uses_zeroed_noisy_for_uncond")


def test_compute_rewarded_loss_accepts_image_or_video_zeroed():
    """compute_rewarded_distribution_matching_loss should accept the
    image_or_video_zeroed parameter and pass noisy_zeroed to _compute_kl_grad.
    When image_or_video_zeroed is None, it should behave as before."""
    # We test the parameter acceptance and noisy creation logic by
    # instrumenting _compute_kl_grad to capture what it receives.
    captured = {}

    class _FakeScheduler:
        def add_noise(self, x0, noise, timesteps):
            return x0 + noise * 0.001  # deterministic small perturbation

    class _FakeReDMD:
        fake_guidance_scale = 0.0
        real_guidance_scale = 0.0
        ts_schedule = False
        ts_schedule_max = False
        min_score_timestep = 0
        num_train_timestep = 1000
        timestep_shift = 1.0
        min_step = 20
        max_step = 980
        num_frame_per_block = 3
        scheduler = _FakeScheduler()
        inferencer = None
        def _get_timestep(self, *args, **kwargs):
            return torch.ones(1, 3, dtype=torch.int64) * 500
        def _compute_kl_grad(self, noisy_image_or_video, estimated_clean_image_or_video,
                             timestep, conditional_dict, unconditional_dict,
                             noisy_image_or_video_zeroed=None, normalization=True):
            captured["noisy_full"] = noisy_image_or_video
            captured["noisy_zeroed"] = noisy_image_or_video_zeroed
            return torch.zeros_like(noisy_image_or_video), {}

    # We also need a fake inferencer for reward_from_frames.
    fake_dmd = _FakeReDMD()
    fake_dmd.inferencer = SimpleNamespace(
        reward_from_frames=lambda *a, **kw: {"MQ": torch.tensor(1.0), "VQ": torch.tensor(1.0)}
    )

    from methods.reward_forcing.re_dmd import ReDMD
    loss_fn = ReDMD.compute_rewarded_distribution_matching_loss.__get__(fake_dmd, _FakeReDMD)

    x_full = torch.randn(1, 3, 4, 4, 4)
    x_zeroed = torch.randn(1, 3, 4, 4, 4)
    pixels = torch.randn(1, 3, 4, 4, 4)

    # Case 1: with image_or_video_zeroed
    captured.clear()
    loss, _ = loss_fn(
        image_or_video=x_full,
        pixels=pixels,
        text_prompts=["test"],
        conditional_dict={"is_cond": True},
        unconditional_dict={"is_cond": False},
        image_or_video_zeroed=x_zeroed,
    )
    assert captured["noisy_zeroed"] is not None, "noisy_zeroed should be created and passed"
    assert captured["noisy_zeroed"].shape == x_zeroed.shape

    # Case 2: without image_or_video_zeroed (None)
    captured.clear()
    loss, _ = loss_fn(
        image_or_video=x_full,
        pixels=pixels,
        text_prompts=["test"],
        conditional_dict={"is_cond": True},
        unconditional_dict={"is_cond": False},
        image_or_video_zeroed=None,
    )
    assert captured["noisy_zeroed"] is None, "noisy_zeroed should be None when not provided"

    print("[PASS] test_compute_rewarded_loss_accepts_image_or_video_zeroed")


if __name__ == "__main__":
    test_prob_zero_is_passthrough()
    test_maybe_zero_sink_only()
    test_maybe_zero_window_only()
    test_maybe_zero_both()
    test_save_restore_roundtrip()
    test_generate_chunk_restores_history_and_writes_new_tokens()
    test_first_chunk_not_zeroed()
    test_critic_path_not_zeroed()
    test_cross_rank_sync()
    test_prob_between_zero_and_one_can_skip()
    test_generate_chunk_zeroed_output_is_detached()
    test_generate_next_chunk_includes_chunk_zeroed_in_info()
    test_unified_cfg_uses_zeroed_noisy_for_uncond()
    test_compute_rewarded_loss_accepts_image_or_video_zeroed()
    print("\nAll tests passed!")
