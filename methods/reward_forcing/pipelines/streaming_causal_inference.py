# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional
import torch
import os

from core.wan_wrapper.wan_wrapper_reward_forcing import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from core.misc.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation, log_gpu_memory
from core.misc.debug_option import DEBUG
from .causal_inference import CausalInferencePipeline
import torch.distributed as dist


class StreamingCausalInferencePipeline(CausalInferencePipeline):

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
        text_prompts: List[str],
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        # Encode negative prompt for enlarged CFG (zeroed cache + neg prompt)
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * batch_size
        )
        # Enlarged CFG scale (0.0 = disabled, 1.0 = standard). Controlled by --enlarged_cfg_scale CLI.
        KV_CACHE_CFG_SCALE = float(getattr(self.args, "enlarged_cfg_scale", 0.0))

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
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

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        all_num_frames = [self.num_frame_per_block] * num_blocks

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        for current_num_frames in all_num_frames:
            if profile:
                block_start.record()

            noisy_input = noise[:, current_start_frame:current_start_frame + current_num_frames]

            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.int64) * current_timestep

                # --- Step-level enlarged CFG (clone + zero, no save/restore) ---
                use_cfg = KV_CACHE_CFG_SCALE > 0
                if use_cfg:
                    uncond_kv = self._clone_kv_cache(zero_sink=True, zero_window=True)
                    uncond_ca = self._clone_crossattn_cache()

                if index < len(self.denoising_step_list) - 1:
                    # Cond: full cache + positive prompt
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )

                    if use_cfg:
                        with torch.no_grad():
                            _, pred_uncond = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=unconditional_dict,
                                timestep=timestep,
                                kv_cache=uncond_kv,
                                crossattn_cache=uncond_ca,
                                current_start=current_start_frame * self.frame_seq_length,
                            )
                        denoised_pred = denoised_pred + KV_CACHE_CFG_SCALE * (
                            denoised_pred - pred_uncond
                        )

                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # Cond: full cache + positive prompt
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )

                    if use_cfg:
                        with torch.no_grad():
                            _, pred_uncond = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=unconditional_dict,
                                timestep=timestep,
                                kv_cache=uncond_kv,
                                crossattn_cache=uncond_ca,
                                current_start=current_start_frame * self.frame_seq_length,
                            )
                        denoised_pred = denoised_pred + KV_CACHE_CFG_SCALE * (
                            denoised_pred - pred_uncond
                        )

            output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output.device)

            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            current_start_frame += current_num_frames

        if profile:
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        if getattr(self.args.model_kwargs, "use_infinite_attention", False):
            video = self.vae.decode_to_pixel_chunk(output.to(noise.device), use_cache=False)
        else:
            video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time
            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output.to(noise.device)
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size_override: int | None = None):
        kv_cache1 = []
        if kv_cache_size_override is not None:
            kv_cache_size = kv_cache_size_override
        else:
            if self.local_attn_size != -1:
                slice_last = getattr(self, 'slice_last_frames', 21)
                kv_cache_size = (self.local_attn_size + slice_last) * self.frame_seq_length
            else:
                kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        if local_attn_size_value == -1:
            target_size = 32760
            policy = "global"
        else:
            target_size = int(local_attn_size_value) * self.frame_seq_length
            policy = "local"

        updated_modules = []
        if hasattr(self.generator.model, "max_attention_size"):
            try:
                prev = getattr(self.generator.model, "max_attention_size")
            except Exception:
                prev = None
            setattr(self.generator.model, "max_attention_size", target_size)
            updated_modules.append("<root_model>")

        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    prev = getattr(module, "max_attention_size")
                except Exception:
                    prev = None
                try:
                    setattr(module, "max_attention_size", target_size)
                    updated_modules.append(name if name else module.__class__.__name__)
                except Exception:
                    pass
