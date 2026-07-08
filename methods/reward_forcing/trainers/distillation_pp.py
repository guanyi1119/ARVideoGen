# Self-Forcing++ post-training trainer.
#
# Inherits from StreamingDistillationTrainer to reuse all initialization
# (model setup, FSDP wrapping, LoRA, optimizer, dataloader, checkpoint
# loading, EMA, visualization) and overrides only:
#   - __init__: swap StreamingTrainingModel -> StreamingTrainingModelPP
#   - train: SF++ flow (rollout -> sample window -> DMD -> update)
#
# Paper: Self-Forcing++: Towards Minute-Scale High-Quality Video Generation
# arXiv: 2510.02283
#
# SPDX-License-Identifier: Apache-2.0
import time
from datetime import datetime

import torch
import torch.distributed as dist

from core.misc import merge_dict_list
from core.misc.debug_option import DEBUG, LOG_GPU_MEMORY
from core.misc.memory import log_gpu_memory
from methods.reward_forcing.streaming_training_pp import StreamingTrainingModelPP
from methods.reward_forcing.trainers.streaming_distillation import Trainer as StreamingDistillationTrainer


class PPTrainer(StreamingDistillationTrainer):
    """Self-Forcing++ post-training trainer.

    Reuses all infrastructure from StreamingDistillationTrainer (model init,
    FSDP, LoRA, optimizers, dataloader, checkpointing) but replaces the
    per-chunk streaming training loop with SF++'s:
      1. No-grad long rollout (N=150 frames) with rolling KV cache
      2. Uniformly sample a K=21 window from the rollout
      3. DMD loss on the window (backward noise init is inside the loss)
      4. One gradient update per rollout
    """

    def __init__(self, config):
        # Temporarily disable streaming_training in config so the parent
        # __init__ skips creating a StreamingTrainingModel (which calls
        # reset_state() -> clear_kv_cache() on a pipeline that lacks that
        # method). We create our own StreamingTrainingModelPP after super
        # returns. The parent only uses this flag in __init__ to decide
        # whether to build the streaming model; our train() override never
        # reads self.streaming_training, so restoring it is safe.
        _original_streaming_training = getattr(config, "streaming_training", False)
        config.streaming_training = False

        # Call parent __init__ — sets up everything:
        # model, FSDP, LoRA, optimizers, dataloader, EMA, checkpoint loading
        super().__init__(config)

        # Restore the original value for any downstream code that reads it
        config.streaming_training = _original_streaming_training

        # Create SF++ streaming model (replaces the None the parent left)
        self.streaming_model = StreamingTrainingModelPP(self.model, config)

        # SF++ does not use the streaming_active / sequence state machine
        self.streaming_active = False

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SF++-Trainer] Initialized PPTrainer with "
                  f"rollout_length={self.streaming_model.rollout_length}, "
                  f"window_size={self.streaming_model.window_size}")

    def _get_batch_and_encode(self):
        """Get next batch and encode text prompts into conditional dicts.

        Caches the unconditional dict (negative prompt encoding) since it
        is identical across steps — only the conditional dict changes.
        """
        batch = next(self.dataloader)
        text_prompts = batch["prompts"]

        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
            if not hasattr(self, "_cached_uncond"):
                uncond_prompts = [self.config.negative_prompt] * len(text_prompts)
                unconditional_dict = self.model.text_encoder(text_prompts=uncond_prompts)
                self._cached_uncond = {k: v.detach() for k, v in unconditional_dict.items()}

        return text_prompts, conditional_dict, self._cached_uncond

    def train(self):
        """SF++ training loop: rollout -> sample -> DMD -> update."""
        self.start_step = self.step
        self.start_time = time.time()

        try:
            while True:
                if self.step % 20 == 0:
                    torch.cuda.empty_cache()

                # Determine if this is a generator step
                TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

                if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                    log_gpu_memory(f"[SF++-Trainer] Before step {self.step}",
                                   device=self.device,
                                   rank=dist.get_rank() if dist.is_initialized() else 0)

                # Step 1: Get batch and encode text
                text_prompts, conditional_dict, unconditional_dict = \
                    self._get_batch_and_encode()

                # Step 2: SF++ core — no-grad long rollout
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SF++-Trainer] Step {self.step}: starting rollout "
                          f"(N={self.streaming_model.rollout_length})")

                V = self.streaming_model.rollout_long(
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    initial_latent=None,
                    text_prompts=text_prompts,
                )

                # Step 3: Sample random window
                W = self.streaming_model.sample_window(V)

                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SF++-Trainer] Step {self.step}: sampled window "
                          f"shape={W.shape}")

                # Zero gradients
                if TRAIN_GENERATOR:
                    self.generator_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)

                # Step 4: Generator loss + backward (if generator step)
                generator_log_dict = {}
                if TRAIN_GENERATOR:
                    gen_loss, gen_log = self.streaming_model.compute_generator_loss(
                        window=W,
                        conditional_dict=conditional_dict,
                        unconditional_dict=unconditional_dict,
                        text_prompts=text_prompts,
                    )
                    scaled_gen_loss = gen_loss / self.gradient_accumulation_steps
                    scaled_gen_loss.backward()
                    generator_log_dict = {
                        "generator_loss": gen_loss.detach(),
                    }

                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SF++-Trainer] Generator loss: {gen_loss.item():.4f}")

                # Step 5: Critic loss + backward (every step)
                critic_loss, critic_log = self.streaming_model.compute_critic_loss(
                    window=W.detach(),
                    conditional_dict=conditional_dict,
                )
                scaled_critic_loss = critic_loss / self.gradient_accumulation_steps
                scaled_critic_loss.backward()
                critic_log_dict = {
                    "critic_loss": critic_loss.detach(),
                }

                # Step 6: Optimizer steps
                if TRAIN_GENERATOR:
                    gen_grad_norm = self.model.generator.clip_grad_norm_(
                        self.max_grad_norm_generator
                    )
                    self.generator_optimizer.step()
                    if self.generator_ema is not None:
                        self.generator_ema.update(self.model.generator)
                    generator_log_dict["generator_grad_norm"] = gen_grad_norm

                critic_grad_norm = self.model.fake_score.clip_grad_norm_(
                    self.max_grad_norm_critic
                )
                self.critic_optimizer.step()
                critic_log_dict["critic_grad_norm"] = critic_grad_norm

                # Step 7: Logging
                self.step += 1

                if self.is_main_process and not self.disable_logging:
                    log_data = {**generator_log_dict, **critic_log_dict}
                    self.writer.log(log_data, step=self.step)

                # Throughput + progress logging (tokens/s/npu, aligned with
                # streaming_distillation.train and rewarded_distillation.train)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    batch_size = getattr(self.config, "batch_size", 1)
                    end_time = time.time()
                    end_step = self.step
                    step_diff = max(end_step - self.start_step, 1)
                    time_diff = end_time - self.start_time
                    seconds_per_iter = time_diff / step_diff
                    throughput = 1560 * 12 * batch_size / max(seconds_per_iter, 1e-6)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    gen_loss_val = (
                        gen_loss.item()
                        if TRAIN_GENERATOR else 0.0
                    )
                    print(
                        f"{timestamp}: [SF++ step {self.step}] "
                        f"generator_loss: {gen_loss_val:.4f} "
                        f"critic_loss: {critic_loss.item():.4f} "
                        f"DI_throughput: {throughput:.2f} tokens/s/npu"
                    )
                    if not self.disable_logging:
                        self.writer.log({"DI_throughput": throughput}, step=self.step)
                    self.start_time = time.time()
                    self.start_step = end_step

                # Checkpoint
                if (not getattr(self.config, 'no_save', False)) and \
                   self.step % self.config.log_iters == 0:
                    torch.cuda.empty_cache()
                    self.save()
                    torch.cuda.empty_cache()

                if dist.is_initialized():
                    dist.barrier()

        except KeyboardInterrupt:
            if self.is_main_process:
                print("Training interrupted by user")
            if not getattr(self.config, 'no_save', False):
                self.save()
