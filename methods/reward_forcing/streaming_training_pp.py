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

        # Generator-level enlarged CFG (step-level CFG inside generate_chunk_with_cache).
        # Three independent probability switches control which uncond degradations are
        # applied (sink zero / window zero / neg prompt). kv_cache_cfg_scale controls
        # the CFG strength. All default to 0.0 = disabled.
        self.kv_cache_cfg_scale = float(getattr(config, "kv_cache_cfg_scale", 0.0))
        self.sink_kv_cache_zero_prob = float(getattr(config, "sink_kv_cache_zero_prob", 0.0))
        self.window_kv_cache_zero_prob = float(getattr(config, "window_kv_cache_zero_prob", 0.0))
        self.neg_prompt_zero_prob = float(getattr(config, "neg_prompt_zero_prob", 0.0))

        # Get required components from the underlying model
        self.generator = base_model.generator
        self.fake_score = base_model.fake_score
        self.scheduler = base_model.scheduler
        self.denoising_loss_func = base_model.denoising_loss_func

        # Ensure we have a StreamingTrainingPipeline (has generate_chunk_with_cache,
        # _initialize_kv_cache, etc.). ReDMD._initialize_inference_pipeline() creates
        # a SelfForcingTrainingPipeline which lacks these streaming methods.
        # If the existing pipeline is already a StreamingTrainingPipeline (e.g.
        # created by ReDMDSwitch), keep it; otherwise create a new one.
        from methods.reward_forcing.pipelines.streaming_training import StreamingTrainingPipeline

        self.inference_pipeline = base_model.inference_pipeline
        if self.inference_pipeline is None or not isinstance(
            self.inference_pipeline, StreamingTrainingPipeline
        ):
            self.inference_pipeline = StreamingTrainingPipeline(
                denoising_step_list=base_model.denoising_step_list,
                scheduler=base_model.scheduler,
                generator=base_model.generator,
                num_frame_per_block=base_model.num_frame_per_block,
                same_step_across_blocks=getattr(
                    base_model.args, 'same_step_across_blocks', True
                ),
                last_step_only=getattr(base_model.args, 'last_step_only', False),
                context_noise=getattr(base_model.args, 'context_noise', 0),
                local_attn_size=getattr(config, 'model_kwargs', {}).get(
                    'local_attn_size', -1
                ),
                slice_last_frames=getattr(config, 'slice_last_frames', 21),
            )
            # Update base_model so any code that reads base_model.inference_pipeline
            # sees the streaming-capable pipeline.
            base_model.inference_pipeline = self.inference_pipeline

        # Model config
        self.num_frame_per_block = base_model.num_frame_per_block
        self.frame_seq_length = getattr(
            self.inference_pipeline, 'frame_seq_length', 1560
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

    def _decide_cfg_degradations(self) -> Tuple[bool, bool, bool]:
        """Decide which CFG degradations fire this rollout (synced across ranks).

        Returns (zero_sink, zero_window, use_neg_prompt).
        Same logic as StreamingTrainingModel._decide_cfg_degradations.
        """
        decisions = torch.zeros(3, device=self.device, dtype=torch.float32)
        if dist.is_initialized():
            if dist.get_rank() == 0:
                if self.sink_kv_cache_zero_prob > 0:
                    decisions[0] = _random.random()
                if self.window_kv_cache_zero_prob > 0:
                    decisions[1] = _random.random()
                if self.neg_prompt_zero_prob > 0:
                    decisions[2] = _random.random()
            dist.broadcast(decisions, src=0)
        else:
            if self.sink_kv_cache_zero_prob > 0:
                decisions[0] = _random.random()
            if self.window_kv_cache_zero_prob > 0:
                decisions[1] = _random.random()
            if self.neg_prompt_zero_prob > 0:
                decisions[2] = _random.random()

        zero_sink = decisions[0].item() < self.sink_kv_cache_zero_prob
        zero_window = decisions[1].item() < self.window_kv_cache_zero_prob
        use_neg = decisions[2].item() < self.neg_prompt_zero_prob
        return zero_sink, zero_window, use_neg

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

        # Clear existing cache contents (zeros out tensors, preserves allocation).
        # StreamingTrainingPipeline.clear_kv_cache() is available now that we
        # use StreamingTrainingPipeline instead of SelfForcingTrainingPipeline.
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

    def rollout_and_sample_window(self, conditional_dict, unconditional_dict,
                                  initial_latent=None, text_prompts=None,
                                  requires_grad: bool = True,
                                  switch_conditional_dict=None,
                                  switch_frame_index=None) -> Tuple[torch.Tensor, dict, int]:
        """Rollout context (no-grad) then window (grad controlled by requires_grad).

        Returns ``(window, window_conditional_dict)`` where window is
        [B, window_size, C, H, W] and window_conditional_dict is the
        conditional dict that applies to the window's first block.

        The rollout loops at **block granularity** (``num_frame_per_block``
        frames, typically 3) so the window can start at any block boundary,
        not just at multiples of ``chunk_size``. For a rollout of 150 frames
        with 3 frames/block and a 21-frame (7-block) window, there are
        50 - 7 + 1 = 44 possible window positions instead of just 7.

        When requires_grad=True, each window block has gradient through its
        final denoising step. KV cache updates (context_noise rerun) are
        always no-grad, so previous blocks' cache entries are detached --
        gradient flows only through the current window blocks' generation.

        When ``switch_conditional_dict`` and ``switch_frame_index`` are
        provided, blocks at/after the switch frame use the switch dict.
        The switch frame is rounded down to the nearest block boundary.
        If the window spans the switch, the loss uses the first block's
        dict (an approximation noted here for completeness).

        When ``kv_cache_cfg_scale > 0`` and this is a generator step with
        window not at the very start, step-level enlarged CFG is applied
        to window blocks: each denoising step runs twice (cond + uncond
        clone with optional zeroed sink/window + neg prompt) and predictions
        are CFG-combined. Context blocks (before the window) never get CFG.

        Flow:
          1. Compute block counts; sample window start block (synced across ranks)
          2. Generate context blocks 0..i-1 (no-grad, build KV cache)
          3. Generate window blocks i..i+W-1 (grad per requires_grad, +CFG if enabled)
          4. Concatenate window block outputs -> window tensor
          5. Return (window, window_conditional_dict)
        """
        self.setup_sequence(conditional_dict, unconditional_dict, initial_latent,
                            text_prompts)

        batch_size = self.image_or_video_shape[0]
        C, H, W = self.image_or_video_shape[2:]
        bsz = self.num_frame_per_block  # frames per block (typically 3)

        # Block-based configuration
        assert self.rollout_length % bsz == 0, \
            f"sfpp_rollout_length ({self.rollout_length}) must be divisible by " \
            f"num_frame_per_block ({bsz})"
        assert self.window_size % bsz == 0, \
            f"sfpp_window_size ({self.window_size}) must be divisible by " \
            f"num_frame_per_block ({bsz})"

        num_blocks_total = self.rollout_length // bsz       # e.g. 150 // 3 = 50
        window_num_blocks = self.window_size // bsz         # e.g.  21 // 3 = 7
        assert num_blocks_total >= window_num_blocks, \
            f"rollout_length ({self.rollout_length}) too small for window_size " \
            f"({self.window_size}) at {bsz} frames/block"

        # Sample window start block (synced across ranks).
        # Window can start at any block boundary: 0, 1, 2, ..., num_blocks_total - window_num_blocks
        window_start_block = self._synced_random_int(
            0, num_blocks_total - window_num_blocks + 1
        )
        window_start_frame = window_start_block * bsz

        # Determine the switch block boundary (first block that uses switch dict)
        switch_block_idx = None
        if switch_conditional_dict is not None and switch_frame_index is not None:
            switch_block_idx = switch_frame_index // bsz
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SF++-Model] Switch at frame {switch_frame_index} -> "
                      f"block {switch_block_idx}/{num_blocks_total}")

        # Helper: pick the right conditional dict for a given block index
        def _dict_for_block(block_idx: int) -> dict:
            if switch_block_idx is not None and block_idx >= switch_block_idx:
                return switch_conditional_dict
            return conditional_dict

        # The conditional dict that applies to the window's first block (for losses)
        window_conditional_dict = _dict_for_block(window_start_block)

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SF++-Model] Window at block {window_start_block}/{num_blocks_total}, "
                  f"frames {window_start_frame}-{window_start_frame + self.window_size} "
                  f"({window_num_blocks} blocks), "
                  f"using_switch={window_conditional_dict is not conditional_dict}")

        current_length = 0
        block_idx = 0

        # --- Step 2: Generate context blocks (no-grad, build KV cache) ---
        with torch.no_grad():
            while block_idx < window_start_block:
                noise_block = torch.randn(
                    [batch_size, bsz, C, H, W],
                    device=self.device,
                    dtype=self.dtype
                )

                if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                    log_gpu_memory(
                        f"[SF++-Model] Context block at frame {current_length}",
                        device=self.device,
                        rank=dist.get_rank() if dist.is_initialized() else 0
                    )

                _, _, _ = self.inference_pipeline.generate_chunk_with_cache(
                    noise=noise_block,
                    conditional_dict=_dict_for_block(block_idx),
                    current_start_frame=current_length,
                    requires_grad=False,
                    return_sim_step=False,
                )
                current_length += bsz
                block_idx += 1

                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SF++-Model] Context rollout: block {block_idx}/{window_start_block}")

        # --- Step 3: Generate window blocks (grad per requires_grad) ---
        # NOTE: We call generate_chunk_with_cache once per block (3 frames)
        # instead of once per chunk (21 frames). This means each block
        # independently samples its denoising exit_flag, whereas a single
        # 21-frame call with same_step_across_blocks=True would use the
        # same exit_flag for all 7 blocks. Independent exit flags give
        # more training diversity and match inference behavior (where each
        # block goes through the full denoising_step_list independently).

        # Determine whether step-level CFG should fire for window blocks.
        # CFG only applies on generator steps (not critic), when the window
        # is not at the very start (needs history in KV cache), and when
        # at least one degradation probability is set.
        apply_cfg = (
            requires_grad
            and window_start_block > 0
            and self.kv_cache_cfg_scale > 0
            and (
                self.sink_kv_cache_zero_prob > 0
                or self.window_kv_cache_zero_prob > 0
                or self.neg_prompt_zero_prob > 0
            )
            and self.inference_pipeline.kv_cache1 is not None
        )

        cfg_kwargs = {}
        if apply_cfg:
            zero_sink, zero_window, use_neg = self._decide_cfg_degradations()
            if zero_sink or zero_window or use_neg:
                cfg_kwargs["cfg_uncond_dict"] = (
                    unconditional_dict if use_neg else window_conditional_dict
                )
                cfg_kwargs["cfg_zero_sink"] = zero_sink
                cfg_kwargs["cfg_zero_window"] = zero_window
                cfg_kwargs["cfg_scale"] = self.kv_cache_cfg_scale
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(
                        f"[SF++-Model] Step-level CFG on window: sink={zero_sink}, "
                        f"window={zero_window}, neg_prompt={use_neg}, "
                        f"cfg_scale={self.kv_cache_cfg_scale}"
                    )

        window_outputs = []
        for _ in range(window_num_blocks):
            noise_block = torch.randn(
                [batch_size, bsz, C, H, W],
                device=self.device,
                dtype=self.dtype
            )

            if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                log_gpu_memory(
                    f"[SF++-Model] Window block at frame {current_length}",
                    device=self.device,
                    rank=dist.get_rank() if dist.is_initialized() else 0
                )

            block_output, _, _ = self.inference_pipeline.generate_chunk_with_cache(
                noise=noise_block,
                conditional_dict=_dict_for_block(block_idx),
                current_start_frame=current_length,
                requires_grad=requires_grad,
                return_sim_step=False,
                **cfg_kwargs,
            )
            window_outputs.append(block_output)
            current_length += bsz
            block_idx += 1

        # Concatenate window blocks into a single [B, window_size, C, H, W] tensor
        window = torch.cat(window_outputs, dim=1)

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SF++-Model] Window generated: shape={window.shape}, "
                  f"requires_grad={window.requires_grad}")

        return window, window_conditional_dict, window_start_frame

    def compute_generator_loss(
        self,
        window: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        text_prompts: Optional[list] = None,
        beta: Optional[float] = None,
        scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """DMD loss on the window (which has gradient through rollout generation).

        The window comes from rollout_and_sample_window, which generates it
        WITH gradient through the actual rollout process (KV cache context
        from previous detached chunks). The DMD loss
        (compute_rewarded_distribution_matching_loss) computes:
          1. scheduler.add_noise(window, noise, timestep)  <- backward noise init
          2. _compute_kl_grad(noised_window)               <- student vs teacher KL
          3. 0.5 * exp(beta * reward) * mse                <- DMD loss

        Gradient flows: MSE -> window -> generator (through rollout) -> θ.
        With beta=0.0 (default), this is pure DMD: 0.5 * mse.

        When ``scores`` is provided and ``beta > 0``, the reward term blends
        MQ (motion quality) and VQ (visual quality) per the score value.
        With beta=0.0, scores has no effect (exp(0)=1).
        """
        # NOTE: must use `is not None`, NOT `or`, because beta=0.0 is falsy
        beta = beta if beta is not None else self.beta

        with torch.no_grad():
            pixels = self.base_model.vae.decode_to_pixel(window).to(self.dtype)

        loss, log_dict = self.base_model.compute_rewarded_distribution_matching_loss(
            image_or_video=window,
            pixels=pixels,
            text_prompts=text_prompts,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=None,
            beta=beta,
            scores=scores,
        )
        return loss, log_dict

    def _clear_cache_gradients(self):
        """Detach gradient references in KV cache and cross-attention cache.

        Same logic as StreamingTrainingModel._clear_cache_gradients.
        Important for preventing memory leaks before critic training.
        """
        if hasattr(self.inference_pipeline, 'kv_cache1') and \
           self.inference_pipeline.kv_cache1 is not None:
            for cache_block in self.inference_pipeline.kv_cache1:
                if 'k' in cache_block and cache_block['k'].requires_grad:
                    cache_block['k'] = cache_block['k'].detach()
                if 'v' in cache_block and cache_block['v'].requires_grad:
                    cache_block['v'] = cache_block['v'].detach()

        if hasattr(self.inference_pipeline, 'crossattn_cache') and \
           self.inference_pipeline.crossattn_cache is not None:
            for cache_block in self.inference_pipeline.crossattn_cache:
                if 'k' in cache_block and cache_block['k'].requires_grad:
                    cache_block['k'] = cache_block['k'].detach()
                if 'v' in cache_block and cache_block['v'].requires_grad:
                    cache_block['v'] = cache_block['v'].detach()

    def compute_critic_loss(
        self,
        window: torch.Tensor,
        conditional_dict: dict,
        chunk_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Critic (fake_score) denoising loss on window.

        Logic pattern copied from StreamingTrainingModel.compute_critic_loss
        (NOT inherited — standalone implementation in this class).
        """
        _t_loss_start = time.time()

        # Critical: ensure window has no gradient connections
        if window.requires_grad:
            window = window.detach()

        # Clear gradient references in caches
        self._clear_cache_gradients()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        batch_size, num_frame = window.shape[:2]

        # Sample timestep (same logic as StreamingTrainingModel.compute_critic_loss)
        min_timestep = getattr(self.base_model, 'min_score_timestep', 0)
        max_timestep = getattr(self.base_model, 'num_train_timestep', 1000)

        critic_timestep = self.base_model._get_timestep(
            min_timestep=min_timestep,
            max_timestep=max_timestep,
            batch_size=batch_size,
            num_frame=num_frame,
            num_frame_per_block=self.num_frame_per_block,
            uniform_timestep=True,
        ).to(self.device)

        # Apply timestep shift
        if getattr(self.base_model, 'timestep_shift', 1) > 1:
            timestep_shift = self.base_model.timestep_shift
            critic_timestep = timestep_shift * \
                (critic_timestep / 1000) / \
                (1 + (timestep_shift - 1) * (critic_timestep / 1000)) * 1000
        critic_timestep = critic_timestep.clamp(
            getattr(self.base_model, 'min_step', 0),
            getattr(self.base_model, 'max_step', 1000),
        )

        # Add noise to window
        critic_noise = torch.randn_like(window)
        noisy_window = self.scheduler.add_noise(
            window.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1),
        ).unflatten(0, (batch_size, num_frame))

        # Fake score prediction
        _, pred_fake = self.fake_score(
            noisy_image_or_video=noisy_window,
            conditional_dict=conditional_dict,
            timestep=critic_timestep,
        )

        # Compute denoising loss (flow or mse)
        denoising_loss_type = getattr(
            self.base_model.args, 'denoising_loss_type', 'mse'
        )
        if denoising_loss_type == "flow":
            from core.wan_wrapper import get_wan_wrapper_classes
            _, _, WanDiffusionWrapper = get_wan_wrapper_classes('reward_forcing')
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake.flatten(0, 1),
                xt=noisy_window.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x=window.flatten(0, 1),
                xt=noisy_window.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            ).unflatten(0, (batch_size, num_frame))

        denoising_loss = self.denoising_loss_func(
            x=window.flatten(0, 1),
            x_pred=pred_fake.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred,
            gradient_mask=None,
        )

        # Cleanup intermediate variables
        del conditional_dict, critic_noise, noisy_window, pred_fake
        if 'flow_pred' in locals():
            del flow_pred
        if 'pred_fake_noise' in locals():
            del pred_fake_noise

        log_dict = {
            "loss_time": time.time() - _t_loss_start,
            "critic_loss": denoising_loss.detach(),
        }
        return denoising_loss, log_dict
