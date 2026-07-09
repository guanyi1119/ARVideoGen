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
                                  switch_frame_index=None) -> Tuple[torch.Tensor, dict]:
        """Rollout context (no-grad) then window (grad controlled by requires_grad).

        Returns ``(window, window_conditional_dict)`` where window is
        [B, K, C, H, W] and window_conditional_dict is the conditional dict
        that applies to the window chunk (equals ``conditional_dict`` when no
        switch, or ``switch_conditional_dict`` when the window falls after
        the switch frame).

        When requires_grad=True, the window has gradient through the actual
        rollout generation process (with KV cache context from preceding
        chunks). When requires_grad=False, the entire rollout runs under
        torch.no_grad() -- use this for critic-only steps to avoid building
        an unused generator graph that would leak memory.

        When ``switch_conditional_dict`` and ``switch_frame_index`` are
        provided, context chunks at/after the switch frame use the switch
        dict instead of the original dict. The switch frame is rounded down
        to the nearest chunk boundary so the switch always happens between
        chunks (never within a single chunk).

        Flow:
          1. Sample window chunk index (synced across ranks)
          2. Generate context chunks 0..i-1 (no-grad, build KV cache),
             using switch_conditional_dict for chunks at/after switch
          3. Generate window chunk i (grad per requires_grad) -- the actual rollout output
          4. Return (window, window_conditional_dict)
        """
        self.setup_sequence(conditional_dict, unconditional_dict, initial_latent,
                            text_prompts)

        batch_size = self.image_or_video_shape[0]
        C, H, W = self.image_or_video_shape[2:]

        # Number of complete chunks that fit in rollout_length
        num_chunks = self.rollout_length // self.chunk_size
        assert num_chunks >= 1, \
            f"rollout_length ({self.rollout_length}) must be >= chunk_size ({self.chunk_size})"

        # Sample window chunk index (synced across ranks)
        window_chunk_idx = self._synced_random_int(0, num_chunks)
        window_start_frame = window_chunk_idx * self.chunk_size

        # Determine the switch chunk boundary (first chunk that uses switch dict)
        switch_chunk_idx = None
        if switch_conditional_dict is not None and switch_frame_index is not None:
            switch_chunk_idx = switch_frame_index // self.chunk_size
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SF++-Model] Switch at frame {switch_frame_index} -> "
                      f"chunk {switch_chunk_idx}/{num_chunks}")

        # Helper: pick the right conditional dict for a given chunk index
        def _dict_for_chunk(chunk_idx: int) -> dict:
            if switch_chunk_idx is not None and chunk_idx >= switch_chunk_idx:
                return switch_conditional_dict
            return conditional_dict

        # The conditional dict that applies to the window chunk (for losses)
        window_conditional_dict = _dict_for_chunk(window_chunk_idx)

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SF++-Model] Window at chunk {window_chunk_idx}/{num_chunks}, "
                  f"frames {window_start_frame}-{window_start_frame + self.chunk_size}, "
                  f"using_switch={window_conditional_dict is not conditional_dict}")

        current_length = 0
        chunk_idx = 0

        # --- Step 2: Generate context chunks (no-grad, build KV cache) ---
        with torch.no_grad():
            while current_length < window_start_frame:
                noise_chunk = torch.randn(
                    [batch_size, self.chunk_size, C, H, W],
                    device=self.device,
                    dtype=self.dtype
                )

                if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                    log_gpu_memory(
                        f"[SF++-Model] Context chunk at frame {current_length}",
                        device=self.device,
                        rank=dist.get_rank() if dist.is_initialized() else 0
                    )

                _, _, _ = self.inference_pipeline.generate_chunk_with_cache(
                    noise=noise_chunk,
                    conditional_dict=_dict_for_chunk(chunk_idx),
                    current_start_frame=current_length,
                    requires_grad=False,
                    return_sim_step=False,
                )
                current_length += self.chunk_size
                chunk_idx += 1

                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SF++-Model] Context rollout: {current_length}/{window_start_frame}")

        # --- Step 3: Generate window chunk (grad controlled by requires_grad) ---
        noise_chunk = torch.randn(
            [batch_size, self.chunk_size, C, H, W],
            device=self.device,
            dtype=self.dtype
        )

        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(
                f"[SF++-Model] Window chunk at frame {current_length}",
                device=self.device,
                rank=dist.get_rank() if dist.is_initialized() else 0
            )

        window, _, _ = self.inference_pipeline.generate_chunk_with_cache(
            noise=noise_chunk,
            conditional_dict=window_conditional_dict,
            current_start_frame=current_length,
            requires_grad=requires_grad,
            return_sim_step=False,
        )

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SF++-Model] Window generated: shape={window.shape}, "
                  f"requires_grad={window.requires_grad}")

        return window, window_conditional_dict

    def compute_generator_loss(
        self,
        window: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        text_prompts: Optional[list] = None,
        beta: Optional[float] = None,
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
