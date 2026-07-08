# Self-Forcing++ post-training streaming model wrapper.
#
# Wraps a ReDMD/DMD base_model to implement the Self-Forcing++ training flow:
#   1. No-grad long rollout (N >> teacher horizon) with rolling KV cache
#   2. Uniformly sample a K-length contiguous window from the rollout
#   3. DMD loss on the window (backward noise init is inside the loss function)
#
# Paper: Self-Forcing++: Towards Minute-Scale High-Quality Video Generation
# arXiv: 2510.02283
#
# SPDX-License-Identifier: Apache-2.0
import time
import random as _random
from typing import Tuple, Dict, Any, Optional, List

import torch
import torch.distributed as dist

from core.misc.debug_option import DEBUG, LOG_GPU_MEMORY
from core.misc.memory import log_gpu_memory


class StreamingTrainingModelPP:
    """Self-Forcing++ post-training model wrapper.

    Wraps a ReDMD/DMD base_model. Reuses base_model.inference_pipeline
    for rolling-KV-cache chunk generation and base_model's DMD loss,
    but orchestrates them into SF++'s no-grad-rollout -> sample-window -> DMD flow.

    Unlike StreamingTrainingModel, this class does NOT do per-chunk gradient
    updates. Instead, it rolls out N frames without gradient, samples one
    K-length window, and computes DMD loss on that window.
    """

    def __init__(self, base_model, config):
        self.base_model = base_model
        self.config = config
        self.device = base_model.device
        self.dtype = base_model.dtype
        self.image_or_video_shape = getattr(config, 'image_or_video_shape', None)

        # SF++ configuration
        self.rollout_length = getattr(config, "sfpp_rollout_length", 150)
        self.window_size = getattr(config, "sfpp_window_size", 21)
        self.beta = getattr(config, "sfpp_beta", 0.0)
        self.chunk_size = getattr(config, "streaming_chunk_size", 21)

        # Get required components from the underlying model
        self.generator = base_model.generator
        self.fake_score = base_model.fake_score
        self.scheduler = base_model.scheduler
        self.denoising_loss_func = base_model.denoising_loss_func
        self.inference_pipeline = base_model.inference_pipeline

        # Model config
        self.num_frame_per_block = base_model.num_frame_per_block
        self.frame_seq_length = getattr(
            base_model.inference_pipeline, 'frame_seq_length', 1560
        )

        # Validate
        assert self.rollout_length >= self.window_size, \
            f"sfpp_rollout_length ({self.rollout_length}) must be >= " \
            f"sfpp_window_size ({self.window_size})"

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SF++-Model] Initialized: rollout_length={self.rollout_length}, "
                  f"window_size={self.window_size}, beta={self.beta}, "
                  f"chunk_size={self.chunk_size}")

    def _synced_random_int(self, low: int, high: int) -> int:
        """Random int in [low, high), synced across distributed ranks.

        Rank 0 samples and broadcasts to all other ranks so every rank
        picks the same value. Uses the same pattern as
        StreamingTrainingModel._decide_cfg_degradations.
        """
        if high <= low:
            return low
        if dist.is_initialized():
            if dist.get_rank() == 0:
                val = _random.randint(low, high - 1)
                tensor_val = torch.tensor(val, device=self.device, dtype=torch.int64)
            else:
                tensor_val = torch.tensor(0, device=self.device, dtype=torch.int64)
            dist.broadcast(tensor_val, src=0)
            return tensor_val.item()
        else:
            return _random.randint(low, high - 1)

    def sample_window(self, V: torch.Tensor, K: int = None) -> torch.Tensor:
        """Random contiguous window [B, K, C, H, W] detached.

        Random start index is synced across ranks via _synced_random_int.
        """
        K = K if K is not None else self.window_size
        N = V.shape[1]
        assert N >= K, f"V length ({N}) must be >= window_size ({K})"

        i = self._synced_random_int(0, N - K + 1)
        window = V[:, i:i + K, ...].detach()

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SF++-Model] Sampled window: start={i}, K={K}, shape={window.shape}")
        return window

    def setup_sequence(self, conditional_dict, unconditional_dict,
                       initial_latent=None, text_prompts=None, scores=None):
        """Initialize KV cache and crossattn cache, prepare for rollout.

        Mirrors StreamingTrainingModel.setup_sequence but simplified:
        no prompt switching, no temp_max_length, no previous_frames.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        batch_size = self.image_or_video_shape[0]

        # Initialize KV cache if needed
        if self.inference_pipeline.kv_cache1 is None:
            self.inference_pipeline._initialize_kv_cache(
                batch_size=batch_size,
                dtype=self.dtype,
                device=self.device
            )
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SF++-Model] Initialized kv_cache1")

        if self.inference_pipeline.crossattn_cache is None:
            self.inference_pipeline._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=self.dtype,
                device=self.device
            )
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SF++-Model] Initialized crossattn_cache")

        # Clear existing cache state
        self.inference_pipeline.clear_kv_cache()

        # Prime cache with initial_latent if provided (e.g. for i2v)
        if initial_latent is not None:
            timestep = torch.zeros(
                [batch_size, initial_latent.shape[1]],
                device=self.device, dtype=torch.int64
            )
            with torch.no_grad():
                self.inference_pipeline.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.inference_pipeline.kv_cache1,
                    crossattn_cache=self.inference_pipeline.crossattn_cache,
                    current_start=0
                )
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SF++-Model] Primed cache with initial_latent shape={initial_latent.shape}")

    def rollout_long(self, conditional_dict, unconditional_dict,
                     initial_latent=None, text_prompts=None) -> torch.Tensor:
        """No-grad long rollout. Returns [B, N, C, H, W] detached.

        Generates chunks sequentially using the rolling KV cache,
        collecting all clean frames. No gradient is computed during rollout.
        """
        self.setup_sequence(conditional_dict, unconditional_dict, initial_latent,
                            text_prompts)

        batch_size = self.image_or_video_shape[0]
        C, H, W = self.image_or_video_shape[2:]
        V_chunks = []
        current_length = 0

        with torch.no_grad():
            while current_length < self.rollout_length:
                # Compute chunk_frames: min(chunk_size, remaining)
                remaining = self.rollout_length - current_length
                chunk_frames = min(self.chunk_size, remaining)
                # Truncate to a multiple of num_frame_per_block
                chunk_frames = (chunk_frames // self.num_frame_per_block) * self.num_frame_per_block
                # If remaining frames are fewer than one block, stop —
                # generating a partial block would exceed rollout_length and
                # generate_chunk_with_cache requires block-aligned frames.
                if chunk_frames == 0:
                    break

                noise_chunk = torch.randn(
                    [batch_size, chunk_frames, C, H, W],
                    device=self.device,
                    dtype=self.dtype
                )

                if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                    log_gpu_memory(
                        f"[SF++-Model] Rollout before chunk at frame {current_length}",
                        device=self.device,
                        rank=dist.get_rank() if dist.is_initialized() else 0
                    )

                chunk, _, _ = self.inference_pipeline.generate_chunk_with_cache(
                    noise=noise_chunk,
                    conditional_dict=conditional_dict,
                    current_start_frame=current_length,
                    requires_grad=False,
                    return_sim_step=False,
                )
                V_chunks.append(chunk.detach())
                current_length += chunk_frames

                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SF++-Model] Rollout progress: {current_length}/{self.rollout_length}")

        V = torch.cat(V_chunks, dim=1)
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SF++-Model] Rollout complete: V shape={V.shape}")
        return V
