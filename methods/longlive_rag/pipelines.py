"""LongLive-RAG inference pipeline.

Wraps the Reward-Forcing causal inference pipeline with retrieval-augmented
generation: a frozen LatentAE encodes each completed block to an embedding;
before generating a new block, top-K past embeddings (by cosine similarity)
are retrieved and their KV cache entries passed to the generator as
``memory_indices``.

When ``use_latentmem`` is False or the AE checkpoint is missing, behavior
falls through to the parent CausalInferencePipeline unchanged.
"""
from typing import List, Optional
import os
import dataclasses
import torch
import torch.distributed as dist
import torch.nn.functional as F

from methods.reward_forcing.pipelines import CausalInferencePipeline
from methods.longlive_rag.ae.model import LatentAE
from methods.longlive_rag.ae.config import AEConfig


def _is_rank0() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


class LatentMemCausalInferencePipeline(CausalInferencePipeline):
    """Causal inference pipeline with latent-memory retrieval augmentation.

    Inherits generator, text_encoder, vae, scheduler, KV-cache, and the
    block-by-block denoising loop from ``CausalInferencePipeline``, then
    layers retrieval on top: after each block is denoised, a frozen
    LatentAE encodes it to an L2-normalized embedding.  Before generating
    the next block, top-K past embeddings (by cosine similarity) are
    selected and their frame indices are passed to the generator as
    ``memory_indices`` so the CausalWanModel can attend to the most
    relevant long-range context.
    """

    def __init__(
        self,
        args,
        device,
        *,
        generator=None,
        text_encoder=None,
        vae=None,
    ):
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)

        # ── Read RAG config knobs from model_kwargs ───────────────────────
        mk = getattr(args, "model_kwargs", {}) or {}
        self.use_latentmem = bool(mk.get("use_latentmem", False))
        self.compression_method = mk.get("compression_method", "avg_pool")
        self.memory_size = int(mk.get("memory_size", 0))
        self.recent_exclude = int(mk.get("recent_exclude", 0))
        self.ae_ckpt = mk.get("ae_ckpt", None)

        # ── Swap generator to longlive_rag wrapper if RAG is enabled ──────
        # The parent __init__ created self.generator from the reward_forcing
        # wrapper, which does *not* accept ``memory_indices``.  Replace it
        # with the longlive_rag variant when retrieval is active.
        if self.use_latentmem and generator is None:
            from core.wan_wrapper.wan_wrapper_longlive_rag import (
                WanDiffusionWrapper as WanDiffusionWrapperRAG,
            )
            self.generator = WanDiffusionWrapperRAG(
                **getattr(args, "model_kwargs", {}), is_causal=True,
            )
            # Re-apply local_attn_size override (parent did this too)
            if self.num_frame_per_block > 1:
                self.generator.model.num_frame_per_block = self.num_frame_per_block

        # ── Load retrieval AE ─────────────────────────────────────────────
        self.ae_model = None
        if self.use_latentmem and self.ae_ckpt and os.path.exists(self.ae_ckpt):
            ckpt = torch.load(self.ae_ckpt, map_location="cpu")
            ae_cfg_dict = ckpt.get("config", {})
            valid_keys = {f.name for f in dataclasses.fields(AEConfig)}
            ae_cfg_dict = {k: v for k, v in ae_cfg_dict.items() if k in valid_keys}
            ae_cfg = AEConfig(**ae_cfg_dict)
            self.ae_model = LatentAE(ae_cfg).to(device)
            self.ae_model.load_state_dict(ckpt["model"], strict=False)
            self.ae_model.eval()
            if _is_rank0():
                print(f"[LongLive-RAG] Loaded LatentAE from {self.ae_ckpt}")
        elif self.use_latentmem:
            if _is_rank0():
                print(
                    f"[LongLive-RAG] WARNING: ae_ckpt {self.ae_ckpt!r} not found; "
                    "falling back to native inference (no retrieval)"
                )
            self.use_latentmem = False

        # Per-inference state — reset at the start of each inference() call
        self.latent_descriptors: List[torch.Tensor] = []

    # ── Retrieval helper ────────────────────────────────────────────────────
    def _retrieve_memory_indices(
        self, current_block_latent: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Compute top-K past indices by cosine similarity.

        Returns:
            memory_indices: ``[B, K]`` long tensor on the same device as
                ``current_block_latent``, or ``None`` if there aren't enough
                descriptors yet.
        """
        if self.ae_model is None or self.memory_size <= 0:
            return None
        if len(self.latent_descriptors) <= self.recent_exclude:
            return None
        # query: [B, D]
        with torch.no_grad():
            query = self.ae_model.encode(current_block_latent, normalize=True)
        # Candidates = all descriptors except the most recent recent_exclude
        # Each descriptor is [B, D]; stack -> [N, B, D]
        candidates = torch.stack(
            self.latent_descriptors[: -self.recent_exclude], dim=0
        )  # [N, B, D]
        # Cosine sim: since both are L2-normalized, sim = dot product
        sims = torch.einsum("bd,nbd->bn", query, candidates)  # [B, N]
        K = min(self.memory_size, sims.shape[1])
        topk = sims.topk(K, dim=1).indices  # [B, K]
        return topk.long()

    # ── Inference (RAG-aware override) ──────────────────────────────────────
    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        """Standard inference with RAG retrieval turned on (when configured).

        When ``use_latentmem`` is False or ``ae_model`` is None, falls
        through to ``super().inference()`` unchanged.
        """
        if not self.use_latentmem or self.ae_model is None:
            return super().inference(
                noise=noise,
                text_prompts=text_prompts,
                initial_latent=initial_latent,
                return_latents=return_latents,
                profile=profile,
                low_memory=low_memory,
            )

        # ── RAG branch ── (adapted from parent, with retrieval hooks) ─────
        from core.misc.memory import (
            gpu,
            get_cuda_free_memory_gb,
            move_model_to_device_with_memory_preservation,
        )

        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (
            self.independent_first_frame and initial_latent is not None
        ):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        conditional_dict = self.text_encoder(text_prompts=text_prompts)

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder, target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype,
        )

        # Profiling
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

        # Step 1: Initialise / reset KV cache
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
            )
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device
                )
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device
                )

        # Step 2: Cache context from initial_latent
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones(
                [batch_size, 1], device=noise.device, dtype=torch.int64
            ) * 0
            if self.independent_first_frame:
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = initial_latent[
                    :,
                    current_start_frame : current_start_frame
                    + self.num_frame_per_block,
                ]
                output[
                    :,
                    current_start_frame : current_start_frame
                    + self.num_frame_per_block,
                ] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # ── Step 3: Temporal denoising loop (RAG hooks added) ──────────
        # Reset per-inference retrieval state
        self.latent_descriptors = []

        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames

        for current_num_frames in all_num_frames:
            if profile:
                block_start.record()

            noisy_input = noise[
                :,
                current_start_frame
                - num_input_frames : current_start_frame
                + current_num_frames
                - num_input_frames,
            ]

            # ── RAG: compute memory_indices for this block ──────────
            # Use the *previous* block's last clean frame as query, or
            # None for the very first block.
            memory_indices = None
            if current_start_frame > 0:
                # last block's final clean latent is in output at position
                # (current_start_frame - self.num_frame_per_block) or at
                # position 0 if we had independent first frame.
                if self.independent_first_frame and initial_latent is None:
                    # first block is 1-frame, query from that
                    query_block = output[
                        :,
                        current_start_frame - 1 : current_start_frame,
                    ]
                else:
                    query_block = output[
                        :,
                        current_start_frame
                        - self.num_frame_per_block : current_start_frame,
                    ]
                memory_indices = self._retrieve_memory_indices(query_block)

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                print(f"current_timestep: {current_timestep}")
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64,
                ) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        memory_indices=memory_indices,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones(
                            [batch_size * current_num_frames],
                            device=noise.device,
                            dtype=torch.long,
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        memory_indices=memory_indices,
                    )

            # Step 3.2: record the model's output
            output[
                :,
                current_start_frame : current_start_frame + current_num_frames,
            ] = denoised_pred

            # ── RAG: encode the clean latent block as descriptors ─────
            for f_idx in range(current_num_frames):
                frame = denoised_pred[:, f_idx]  # [B, C, H, W]
                desc = self.ae_model.encode(frame, normalize=True).detach().cpu()
                self.latent_descriptors.append(desc)  # [B, D]

            # Step 3.3: rerun with timestep zero to update KV cache
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                memory_indices=memory_indices,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: advance frame pointer
            current_start_frame += current_num_frames

        if profile:
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(
                f"  - Initialization/caching time: {init_time:.2f} ms "
                f"({100 * init_time / total_time:.2f}%)"
            )
            print(
                f"  - Diffusion generation time: {diffusion_time:.2f} ms "
                f"({100 * diffusion_time / total_time:.2f}%)"
            )
            for i, block_time in enumerate(block_times):
                print(
                    f"    - Block {i} generation time: {block_time:.2f} ms "
                    f"({100 * block_time / diffusion_time:.2f}% of diffusion)"
                )
            print(
                f"  - VAE decoding time: {vae_time:.2f} ms "
                f"({100 * vae_time / total_time:.2f}%)"
            )
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video
