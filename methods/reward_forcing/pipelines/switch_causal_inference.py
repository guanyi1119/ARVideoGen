# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional
import torch
from omegaconf import OmegaConf

from core.wan_wrapper.wan_wrapper_reward_forcing import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from core.misc.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from .streaming_causal_inference import StreamingCausalInferencePipeline
import torch.distributed as dist
from core.misc.debug_option import DEBUG


class SwitchCausalInferencePipeline(StreamingCausalInferencePipeline):

    def __init__(
        self,
        args,
        device,
        *,
        generator: WanDiffusionWrapper | None = None,
        text_encoder: WanTextEncoder | None = None,
        vae: WanVAEWrapper | None = None,
    ):
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)
        self.global_sink = getattr(args, "global_sink", False)
        # Recache window size: must match training (slice_last_frames, default 21).
        # Falls back to local_attn_size if slice_last_frames is not set.
        self.switch_recache_frames = int(getattr(args, "slice_last_frames", 0))
        if self.switch_recache_frames <= 0:
            self.switch_recache_frames = int(self.local_attn_size) if int(self.local_attn_size) > 0 else 21

        # ---- Teleport mitigation / prompt_rewrite hook — optional, off by default ----
        prompt_rewrite_cfg = OmegaConf.select(args, "teleport.prompt_rewrite")
        self._teleport_hook = None
        if prompt_rewrite_cfg is not None:
            from methods.reward_forcing.teleport import build_teleport_switch_hook

            def _text_encode(prompts):
                return self.text_encoder(text_prompts=prompts)

            def _frame_decode(latent):
                # latent: [B, T, C, H, W] → pixels [B, T, 3, H', W'] in [-1, 1]
                # In low_memory mode ``output`` lives on CPU while ``self.vae``
                # lives on GPU/NPU, so we MUST move the latent to the VAE's
                # device before decoding — mirrors the main inference path
                # at the end of ``inference()`` which does
                # ``output.to(noise.device)`` before ``decode_to_pixel``.
                vae_device = next(self.vae.parameters()).device
                if latent.device != vae_device:
                    latent = latent.to(vae_device)
                pixels = self.vae.decode_to_pixel(latent, use_cache=False)
                # Convert to [0, 1] for PIL / VLM consumption
                return (pixels * 0.5 + 0.5).clamp(0.0, 1.0)

            self._teleport_hook = build_teleport_switch_hook(
                prompt_rewrite_cfg,
                text_encoder_fn=_text_encode,
                frame_decoder_fn=_frame_decode,
            )

    def _recache_after_switch(self, output, current_start_frame, new_conditional_dict):
        """Rebuild KV cache and cross-attn cache after a prompt switch.

        This mirrors the training implementation in
        ``StreamingSwitchTrainingPipeline._recache_after_switch``:
          - Uses ``slice_last_frames`` (21) as the recache window, NOT
            ``local_attn_size`` (12).
          - Recaches **block-by-block** (``num_frame_per_block`` frames at a
            time), so each block's forward sees the previous block's freshly
            written KV — identical to training.
          - Does NOT touch ``block_mask`` (the attention function does not use
            it for the reward-forcing causal model).
        """
        # 1. Reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

        if current_start_frame == 0:
            return

        # 2. Determine recache window — use slice_last_frames (matches training)
        num_recache_frames = min(current_start_frame, self.switch_recache_frames)
        recache_start_frame = current_start_frame - num_recache_frames
        recache_start_token = recache_start_frame * self.frame_seq_length

        # Ensure divisibility by num_frame_per_block
        if num_recache_frames % self.num_frame_per_block != 0:
            # Trim to the nearest multiple
            num_recache_frames = (num_recache_frames // self.num_frame_per_block) * self.num_frame_per_block
            if num_recache_frames <= 0:
                return
            recache_start_frame = current_start_frame - num_recache_frames
            recache_start_token = recache_start_frame * self.frame_seq_length

        frames_to_recache = output[:, recache_start_frame:current_start_frame]
        if frames_to_recache.device.type == 'cpu':
            target_device = next(self.generator.parameters()).device
            frames_to_recache = frames_to_recache.to(target_device)

        batch_size = frames_to_recache.shape[0]
        device = frames_to_recache.device
        print(f"[switch-recache] num_recache_frames={num_recache_frames}, "
              f"recache_start_frame={recache_start_frame}, current_start_frame={current_start_frame}")

        # 3. If not global_sink, zero the KV cache and reset indices to the
        #    recache window start (same as training).  When global_sink is
        #    True we leave the KV and indices untouched — again matching
        #    training — because the increased KV cache size (slice_last_frames
        #    now included) is large enough to absorb the negative delta_tokens.
        if not self.global_sink:
            for block_idx in range(self.num_transformer_blocks):
                cache = self.kv_cache1[block_idx]
                cache["k"].zero_()
                cache["v"].zero_()
                cache["global_end_index"].fill_(recache_start_token)
                cache["local_end_index"].zero_()

        # 4. Recache block-by-block (num_frame_per_block frames at a time),
        #    matching the training implementation exactly.
        context_timestep_val = self.args.context_noise
        with torch.no_grad():
            for start in range(0, num_recache_frames, self.num_frame_per_block):
                end = start + self.num_frame_per_block
                recache_input = frames_to_recache[:, start:end]
                context_timestep = torch.ones(
                    [batch_size, end - start],
                    device=device, dtype=torch.int64,
                ) * context_timestep_val
                self.generator(
                    noisy_image_or_video=recache_input,
                    conditional_dict=new_conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=(recache_start_frame + start) * self.frame_seq_length,
                )

        # 5. Reset cross-attention cache again after recaching
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_first: List[str],
        text_prompts_second: List[str],
        switch_frame_index: int,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        low_memory: bool = False,
    ):
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        # Reset teleport registry at the start of every inference call
        if self._teleport_hook is not None and hasattr(self._teleport_hook, "reset"):
            self._teleport_hook.reset()

        cond_first = self.text_encoder(text_prompts=text_prompts_first)
        # cond_second encoding deferred to switch trigger point:
        # if hook is enabled and needs to rewrite, it re-encodes the
        # rewritten prompt; otherwise the original prompt is encoded below.
        cond_second = None  # placeholder, encoded just-in-time

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation
            )

        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        local_attn_cfg = self.local_attn_size  # already resolved by __init__
        kv_policy = ""
        if local_attn_cfg != -1:
            # Match the training KV cache size: local_attn_size + slice_last_frames
            slice_last = self.slice_last_frames if getattr(self, 'slice_last_frames', 0) > 0 else 21
            kv_cache_size = (local_attn_cfg + slice_last) * self.frame_seq_length
            kv_policy = f"local, size={local_attn_cfg}, slice_last={slice_last}"
        else:
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        self._initialize_kv_cache(
            batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        all_num_frames = [self.num_frame_per_block] * num_blocks

        using_second = False
        for current_num_frames in all_num_frames:
            if (not using_second) and (current_start_frame >= switch_frame_index):
                # === prompt_rewrite teleport hook at switch boundary (no-op if disabled) ===
                if self._teleport_hook is not None:
                    with torch.no_grad():
                        cond_second = self._teleport_hook.maybe_rewrite(
                            output, current_start_frame, text_prompts_second
                        )
                if cond_second is None:
                    # hook disabled or returned None → encode original prompt
                    cond_second = self.text_encoder(text_prompts=text_prompts_second)
                # === end switch-boundary hook ===
                self._recache_after_switch(output, current_start_frame, cond_second)
                cond_in_use = cond_second
                using_second = True
                print("switch_frame_index", switch_frame_index)
                print("current_start_frame", current_start_frame)
            else:
                if using_second:
                    cond_in_use = cond_second
                    # === per-chunk prompt_rewrite (active only when trigger=chunk) ===
                    if (
                        self._teleport_hook is not None
                        and self._teleport_hook.per_chunk
                    ):
                        with torch.no_grad():
                            refreshed = self._teleport_hook.maybe_rewrite(
                                output, current_start_frame, text_prompts_second
                            )
                        if refreshed is not None:
                            cond_second = refreshed     # 持续替换，下个 chunk 用最新的
                            cond_in_use = refreshed
                    # === end per-chunk hook ===
                else:
                    cond_in_use = cond_first

            noisy_input = noise[:, current_start_frame - num_input_frames : current_start_frame + current_num_frames - num_input_frames]

            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.int64) * current_timestep
                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )

            output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output.device)

            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_in_use,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            current_start_frame += current_num_frames

        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        return video
