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
import gc
import time
from datetime import datetime

import torch
import torch.distributed as dist

from core.data.dataset import TextDataset, TwoTextDataset, TextScoreDataset, TwoTextScoreDataset, cycle
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

        # If switch_prompt_path is configured, rebuild the dataloader with a
        # TwoText variant so each batch also carries "switch_prompts". The
        # parent Trainer only selects TwoText datasets when
        # distribution_loss == "dmd_switch", but SF++ uses "dmd" (ReDMD), so
        # we rebuild here to support prompt switching in the long rollout.
        switch_prompt_path = getattr(config, "switch_prompt_path", None)
        if switch_prompt_path:
            use_score = getattr(config, "use_score", False)
            if use_score:
                dataset = TwoTextScoreDataset(config.data_path, switch_prompt_path)
            else:
                dataset = TwoTextDataset(config.data_path, switch_prompt_path)
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=True, drop_last=True)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.batch_size,
                sampler=sampler,
                num_workers=8)
            if dist.get_rank() == 0:
                print(f"[SF++-Trainer] Switch-prompt dataset loaded: "
                      f"{len(dataset)} samples, switch_prompt_path={switch_prompt_path}")
            self.dataloader = cycle(dataloader)

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SF++-Trainer] Initialized PPTrainer with "
                  f"rollout_length={self.streaming_model.rollout_length}, "
                  f"window_size={self.streaming_model.window_size}")

    def _get_switch_frame_index(self, max_length=None):
        """Override parent to default to half the rollout length.

        The parent defaults ``fixed_switch_index`` to 21, which was
        appropriate for a 42-frame rollout but is far too early for SF++'s
        150-frame rollout. When the config does not explicitly set
        ``fixed_switch_index``, we default to ``max_length // 2``. For
        ``random`` / ``random_choice`` modes the parent logic is unchanged.
        """
        if getattr(self.config, "switch_mode", "fixed") == "fixed":
            switch_idx = getattr(self.config, "fixed_switch_index", None)
            if switch_idx is None:
                switch_idx = (max_length or 42) // 2
            if max_length is not None:
                assert max_length > switch_idx, \
                    f"max_length {max_length} is not greater than switch_idx {switch_idx}"
            return switch_idx
        return super()._get_switch_frame_index(max_length)

    def _get_batch_and_encode(self):
        """Get next batch and encode text prompts into conditional dicts.

        Caches the unconditional dict (negative prompt encoding) since it
        is identical across steps. When the batch contains "switch_prompts"
        (TwoText dataset), also encodes the switch conditional dict and
        computes a synced switch_frame_index via the parent's
        _get_switch_frame_index.

        Returns a 7-tuple:
          (text_prompts, conditional_dict, unconditional_dict,
           switch_prompts, switch_conditional_dict, switch_frame_index,
           scores)
        where switch_prompts/switch_conditional_dict/switch_frame_index
        are None when no switch prompt is configured, and scores is None
        when the dataset does not provide per-prompt dynamism scores.
        """
        batch = next(self.dataloader)
        text_prompts = batch["prompts"]
        switch_prompts = batch.get("switch_prompts", None)
        scores = batch.get("scores", None)

        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
            if not hasattr(self, "_cached_uncond"):
                uncond_prompts = [self.config.negative_prompt] * len(text_prompts)
                unconditional_dict = self.model.text_encoder(text_prompts=uncond_prompts)
                self._cached_uncond = {k: v.detach() for k, v in unconditional_dict.items()}

        # Encode switch prompts if present
        switch_conditional_dict = None
        switch_frame_index = None
        if switch_prompts is not None:
            with torch.no_grad():
                switch_conditional_dict = self.model.text_encoder(
                    text_prompts=switch_prompts
                )
            # Compute switch frame index (synced across ranks) for the
            # rollout length. The parent's _get_switch_frame_index handles
            # fixed/random/random_choice modes and broadcasts from rank 0.
            switch_frame_index = self._get_switch_frame_index(
                self.streaming_model.rollout_length
            )
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SF++-Trainer] Switch at frame {switch_frame_index} "
                      f"(rollout_length={self.streaming_model.rollout_length})")

        return text_prompts, conditional_dict, self._cached_uncond, \
            switch_prompts, switch_conditional_dict, switch_frame_index, scores

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

                # Step 1: Get batch and encode text (+ switch prompts if any)
                text_prompts, conditional_dict, unconditional_dict, \
                    switch_prompts, switch_conditional_dict, switch_frame_index, scores = \
                    self._get_batch_and_encode()

                # Step 2: SF++ core - long rollout + window sampling
                # Only build the generator computation graph on TRAIN_GENERATOR
                # steps. Critic-only steps use requires_grad=False to avoid
                # building an unused graph (saved transformer activations) that
                # would leak across iterations and cause OOM.
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SF++-Trainer] Step {self.step}: starting rollout "
                          f"(N={self.streaming_model.rollout_length}, "
                          f"requires_grad={TRAIN_GENERATOR})")

                W, window_conditional_dict = self.streaming_model.rollout_and_sample_window(
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    initial_latent=None,
                    text_prompts=text_prompts,
                    requires_grad=TRAIN_GENERATOR,
                    switch_conditional_dict=switch_conditional_dict,
                    switch_frame_index=switch_frame_index,
                )

                # When the window falls after the switch frame, use the switch
                # prompt for the loss computation so the score models and reward
                # see the correct prompt for the window content.
                window_text_prompts = (
                    switch_prompts
                    if window_conditional_dict is not conditional_dict
                    else text_prompts
                )

                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SF++-Trainer] Step {self.step}: window "
                          f"shape={W.shape}, requires_grad={W.requires_grad}, "
                          f"using_switch={window_conditional_dict is not conditional_dict}")

                # Zero gradients
                if TRAIN_GENERATOR:
                    self.generator_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)

                # Step 4: Generator loss + backward (if generator step)
                generator_log_dict = {}
                if TRAIN_GENERATOR:
                    gen_loss, gen_log = self.streaming_model.compute_generator_loss(
                        window=W,
                        conditional_dict=window_conditional_dict,
                        unconditional_dict=unconditional_dict,
                        text_prompts=window_text_prompts,
                        scores=scores,
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
                    conditional_dict=window_conditional_dict,
                )
                scaled_critic_loss = critic_loss / self.gradient_accumulation_steps
                scaled_critic_loss.backward()
                critic_log_dict = {
                    "critic_loss": critic_loss.detach(),
                }

                # Free window tensor + any residual graph before optimizer steps
                # and next iteration. On critic-only steps W has no graph
                # (requires_grad=False), but del is still good hygiene; on
                # generator steps backward() already freed the graph, and del
                # drops the last Python reference so CUDA memory is returned.
                del W

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
                        f"switch_idx: {switch_frame_index if switch_frame_index is not None else '-'} "
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

                # Periodic garbage collection (aligns with streaming_distillation)
                gc_interval = getattr(self.config, 'gc_interval', 100)
                if self.step % gc_interval == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

                if dist.is_initialized():
                    dist.barrier()

                # Termination check
                if self.step > self.config.max_iters:
                    if self.is_main_process:
                        print(f"[SF++-Trainer] Reached max_iters "
                              f"({self.config.max_iters}), stopping training")
                    break

        except KeyboardInterrupt:
            if self.is_main_process:
                print("Training interrupted by user")
            if not getattr(self.config, 'no_save', False):
                self.save()
        except Exception as e:
            if self.is_main_process:
                print(f"[ERROR] Training crashed at step {self.step} "
                      f"with exception: {e}")
                import traceback
                traceback.print_exc()
            raise
        finally:
            if hasattr(self, 'writer') and self.writer is not None:
                try:
                    self.writer.close()
                except Exception as cleanup_e:
                    if self.is_main_process:
                        print(f"[WARNING] Failed to close TensorBoard "
                              f"writer: {cleanup_e}")
