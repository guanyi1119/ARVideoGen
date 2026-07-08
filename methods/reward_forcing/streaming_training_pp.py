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
