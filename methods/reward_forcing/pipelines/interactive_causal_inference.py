# SPDX-License-Identifier: Apache-2.0
# Multi-segment streaming inference for reward_forcing.
# Extends SwitchCausalInferencePipeline to support N (>=2) prompt segments.
from typing import List
import torch

from core.wan_wrapper.wan_wrapper_reward_forcing import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from core.misc.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from core.misc.debug_option import DEBUG
from .switch_causal_inference import SwitchCausalInferencePipeline


class InteractiveCausalInferencePipeline(SwitchCausalInferencePipeline):
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

    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        return_latents: bool = False,
        low_memory: bool = False,
    ):
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert len(text_prompts_list) >= 1, "text_prompts_list must not be empty"
        assert len(switch_frame_indices) == len(text_prompts_list) - 1, (
            "length of switch_frame_indices should be one less than text_prompts_list"
        )
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        if DEBUG:
            print(f"[InteractiveInference] num_segments={len(text_prompts_list)}, switch_at={switch_frame_indices}")
        # Reset teleport registry at the start of every inference call
        if self._teleport_hook is not None and hasattr(self._teleport_hook, "reset"):
            self._teleport_hook.reset()
        # Segment 0: encode immediately (needed for first segment generation).
        # Segments 1+: defer encoding to switch point so hook can rewrite prompts.
        cond_list: List = [self.text_encoder(text_prompts=text_prompts_list[0])]
        cond_list.extend([None] * (len(text_prompts_list) - 1))

        # Encode negative prompt for enlarged CFG (zeroed cache + neg prompt)
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * batch_size
        )
        # Enlarged CFG scale (0.0 = disabled, 1.0 = standard). Controlled by --enlarged_cfg_scale CLI.
        KV_CACHE_CFG_SCALE = float(getattr(self.args, "enlarged_cfg_scale", 0.0))

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation
            )

        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype,
        )

        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(
            f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, "
            f"frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})"
        )

        self._initialize_kv_cache(
            batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size,
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
        )

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[InteractiveInference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        all_num_frames = [self.num_frame_per_block] * num_blocks
        segment_idx = 0
        next_switch_pos = (
            switch_frame_indices[segment_idx]
            if segment_idx < len(switch_frame_indices)
            else None
        )

        for current_num_frames in all_num_frames:
            if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                segment_idx += 1
                # === prompt_rewrite teleport hook at switch boundary (no-op if disabled) ===
                if self._teleport_hook is not None:
                    with torch.no_grad():
                        cond_list[segment_idx] = self._teleport_hook.maybe_rewrite(
                            output, current_start_frame, text_prompts_list[segment_idx]
                        )
                if cond_list[segment_idx] is None:
                    # hook disabled or returned None → encode original prompt
                    cond_list[segment_idx] = self.text_encoder(text_prompts=text_prompts_list[segment_idx])
                # === end switch-boundary hook ===
                self._recache_after_switch(output, current_start_frame, cond_list[segment_idx])
                print(f"[InteractiveInference] switch to segment {segment_idx} at frame {current_start_frame}")
                next_switch_pos = (
                    switch_frame_indices[segment_idx]
                    if segment_idx < len(switch_frame_indices)
                    else None
                )
            else:
                # === per-chunk prompt_rewrite (active only when trigger=chunk and past first switch) ===
                if (
                    self._teleport_hook is not None
                    and self._teleport_hook.per_chunk
                    and segment_idx >= 1
                ):
                    with torch.no_grad():
                        refreshed = self._teleport_hook.maybe_rewrite(
                            output, current_start_frame, text_prompts_list[segment_idx]
                        )
                    if refreshed is not None:
                        cond_list[segment_idx] = refreshed
                # === end per-chunk hook ===
            cond_in_use = cond_list[segment_idx]

            noisy_input = noise[:, current_start_frame : current_start_frame + current_num_frames]

            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [batch_size, current_num_frames], device=noise.device, dtype=torch.int64
                ) * current_timestep

                # --- Enlarged CFG (conditionally enabled) ---
                use_cfg = KV_CACHE_CFG_SCALE > 0
                if use_cfg:
                    pre_step_cache = self._save_kv_cache()
                    pre_step_ca = self._save_crossattn_cache()

                if index < len(self.denoising_step_list) - 1:
                    # Full cache + positive prompt (cond_in_use)
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )

                    if use_cfg:
                        post_step_cache = self._save_kv_cache()
                        post_step_ca = self._save_crossattn_cache()
                        self._restore_kv_cache(pre_step_cache)
                        self._zero_kv_cache()
                        for blk in self.crossattn_cache:
                            blk["is_init"] = False
                        with torch.no_grad():
                            _, denoised_pred_zeroed = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=unconditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                            )
                        denoised_pred = denoised_pred_zeroed + KV_CACHE_CFG_SCALE * (
                            denoised_pred - denoised_pred_zeroed
                        )
                        self._restore_kv_cache(post_step_cache)
                        self._restore_crossattn_cache(post_step_ca)

                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # Full cache + positive prompt (cond_in_use)
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )

                    if use_cfg:
                        post_step_cache = self._save_kv_cache()
                        post_step_ca = self._save_crossattn_cache()
                        self._restore_kv_cache(pre_step_cache)
                        self._zero_kv_cache()
                        for blk in self.crossattn_cache:
                            blk["is_init"] = False
                        with torch.no_grad():
                            _, denoised_pred_zeroed = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=unconditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                            )
                        denoised_pred = denoised_pred_zeroed + KV_CACHE_CFG_SCALE * (
                            denoised_pred - denoised_pred_zeroed
                        )
                        self._restore_kv_cache(post_step_cache)
                        self._restore_crossattn_cache(post_step_ca)

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
