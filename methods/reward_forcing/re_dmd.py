from methods.reward_forcing.pipelines.reward_forcing_training import RewardForcingTrainingPipeline
import torch.nn.functional as F
from typing import Optional, Tuple
import torch

from methods.base.base_reward_forcing import RewardForcingModel

class ReDMD(RewardForcingModel):
    def __init__(self, args, device):
        """
        Initialize the Re-DMD (Rewarded Distribution Matching Distillation) module.
        This class is self-contained and compute rewarded generator and fake score losses
        in the forward pass.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: RewardForcingTrainingPipeline = None

        # Step 2: Initialize all dmd hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

        # === T10: teleport tel_det_regular hook (mutex-gated: off | reweight | aux_loss) ===
        from methods.reward_forcing.teleport import (
            build_teleport_detector,
            build_reweighter,
            build_aux_loss,
        )
        teleport_cfg = getattr(args, "teleport", None)
        tel_det_regular_cfg = getattr(teleport_cfg, "tel_det_regular", None) if teleport_cfg is not None else None
        self._teleport_mode = "off"
        self._teleport_detector = None
        self._teleport_reweighter = None
        self._teleport_aux_loss = None
        self._teleport_aux_beta = 1.0
        if tel_det_regular_cfg is not None:
            self._teleport_mode = getattr(tel_det_regular_cfg, "mode", "off")
            if not self._teleport_mode:
                self._teleport_mode = "off"
            if self._teleport_mode == "reweight":
                self._teleport_detector = build_teleport_detector(getattr(tel_det_regular_cfg, "detector", None))
                self._teleport_reweighter = build_reweighter(getattr(tel_det_regular_cfg, "reweighter", None))
            elif self._teleport_mode == "aux_loss":
                self._teleport_detector = build_teleport_detector(getattr(tel_det_regular_cfg, "detector", None))
                self._teleport_aux_loss = build_aux_loss(getattr(tel_det_regular_cfg, "aux_loss", None))
                self._teleport_aux_beta = getattr(tel_det_regular_cfg, "aux_beta", 1.0)
            elif self._teleport_mode != "off":
                raise ValueError(f"Unknown teleport.tel_det_regular.mode: {self._teleport_mode!r}")

        # Mutex assertion (defensive)
        assert (self._teleport_reweighter is None) or (self._teleport_aux_loss is None), (
            "Mutex violation: reweighter and aux_loss cannot both be active."
        )

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        noisy_image_or_video_zeroed: Optional[torch.Tensor] = None,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: noisy version of the full-cache generation [B, F, C, H, W].
            - estimated_clean_image_or_video: the estimated clean image or video.
            - timestep: timestep tensor [B, F].
            - conditional_dict: conditional information (positive prompt).
            - unconditional_dict: unconditional information (negative prompt).
            - noisy_image_or_video_zeroed: optional noisy version of the zeroed-cache
              generation.  When provided, the uncond path of CFG uses this *plus*
              the negative prompt, creating a unified CFG that combines prompt and
              cache-quality guidance.  When None, falls back to the original
              prompt-only CFG behaviour.
            - normalization: whether to normalize the gradient.
        Output:
            - kl_grad: the KL gradient.
            - kl_log_dict: intermediate tensors for logging.
        """
        # If no zeroed generation is provided, use the same noisy for uncond (original behaviour)
        noisy_for_uncond = noisy_image_or_video_zeroed if noisy_image_or_video_zeroed is not None else noisy_image_or_video

        # Step 1: Compute the fake score
        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        if self.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_for_uncond,
                conditional_dict=unconditional_dict,
                timestep=timestep
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_for_uncond,
            conditional_dict=unconditional_dict,
            timestep=timestep
        )

        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        grad = (pred_fake_image - pred_real_image)

        if normalization:
            p_real = (estimated_clean_image_or_video - pred_real_image)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach()
        }

    def compute_rewarded_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        pixels: torch.Tensor,
        text_prompts: list,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        beta: float = 1.0,
        scores: Optional[torch.Tensor] = None,
        image_or_video_zeroed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: full-cache generation [B, F, C, H, W].
            - pixels: decoded pixel representation for reward computation.
            - text_prompts: list of text prompts.
            - conditional_dict: conditional information (positive prompt).
            - unconditional_dict: unconditional information (negative prompt).
            - gradient_mask: boolean mask indicating which pixels to compute loss on.
            - scores: optional per-prompt dynamism score in [0, 1].
            - image_or_video_zeroed: optional zeroed-cache generation [B, F, C, H, W].
              When provided, the uncond path of the CFG in ``_compute_kl_grad`` uses
              ``noisy(image_or_video_zeroed)`` + ``unconditional_dict`` instead of
              ``noisy(image_or_video)`` + ``unconditional_dict``, creating a unified
              CFG that combines prompt and cache-quality guidance.  When None, the
              behaviour is identical to the original prompt-only CFG.
        Output:
            - dmd_loss: the DMD loss.
            - dmd_log_dict: intermediate tensors for logging.
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        videos = (1 + pixels) / 2.0  # -1~1 to 0~1

        reward = self.inferencer.reward_from_frames(
            [videos[0]],
            [text_prompts[0]],
            use_norm=True,
        ) 

        if scores is not None:
            if isinstance(scores, torch.Tensor):
                score_val = scores.flatten()[0].to(reward['MQ'].device, reward['MQ'].dtype)
            else:
                score_val = torch.tensor(float(scores), device=reward['MQ'].device, dtype=reward['MQ'].dtype)
            reward_term = score_val * reward['MQ'] + (1.0 - score_val) * reward['VQ']
        else:
            reward_term = reward['MQ']

        # === T10: teleport score map computation ===
        teleport_weight_map = None
        teleport_aux_loss_val = None
        teleport_log = {}
        if self._teleport_mode == "reweight" and self._teleport_detector is not None:
            with torch.no_grad():
                # detector input: pixels in [0,1], shape [B,T,3,H,W]
                # `videos` is already (1+pixels)/2.0 in [0,1] shape [B,T,3,H,W]
                student_score = self._teleport_detector.score(videos).detach()
            teleport_weight_map = self._teleport_reweighter.compute_weight(
                student_score,
                teacher_score=None,  # teacher path is tel_det_regular+ optional extension
            )
            teleport_log["teleport_score_mean"] = student_score.mean().detach()
            teleport_log["teleport_weight_mean"] = teleport_weight_map.mean().detach()
        elif self._teleport_mode == "aux_loss" and self._teleport_detector is not None:
            # grad-attached path for aux_loss
            student_score_grad = self._teleport_detector.score(videos)
            teleport_aux_loss_val = self._teleport_aux_loss(student_score_grad)
            teleport_log["teleport_score_mean"] = student_score_grad.mean().detach()
            teleport_log["teleport_aux_loss"] = teleport_aux_loss_val.detach()

        with torch.no_grad():
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            # Create noisy version of the zeroed-cache generation using the SAME
            # noise and timestep, so the CFG difference is purely due to cache
            # quality rather than noise variance.
            noisy_latent_zeroed = None
            if image_or_video_zeroed is not None:
                noisy_latent_zeroed = self.scheduler.add_noise(
                    image_or_video_zeroed.flatten(0, 1),
                    noise.flatten(0, 1),
                    timestep.flatten(0, 1)
                ).detach().unflatten(0, (batch_size, num_frame))

            grad, rl_dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                noisy_image_or_video_zeroed=noisy_latent_zeroed,
            )

        if gradient_mask is not None:
            if teleport_weight_map is not None:
                # weight is [B, F, 1, H, W], original_latent is [B, F, C, H, W]
                w = teleport_weight_map.to(original_latent.dtype).to(original_latent.device)
                sq = (original_latent.double() - (original_latent.double() - grad.double()).detach()) ** 2
                weighted_sq = w.double() * sq
                mse = weighted_sq[gradient_mask].mean()
            else:
                mse = F.mse_loss(
                    original_latent.double()[gradient_mask],
                    (original_latent.double() - grad.double()).detach()[gradient_mask],
                    reduction="mean",
                )
            rl_dmd_loss = 0.5 * torch.exp(beta * reward_term) * mse
        else:
            if teleport_weight_map is not None:
                w = teleport_weight_map.to(original_latent.dtype).to(original_latent.device)
                sq = (original_latent.double() - (original_latent.double() - grad.double()).detach()) ** 2
                weighted_sq = w.double() * sq
                mse = weighted_sq.mean()
            else:
                mse = F.mse_loss(
                    original_latent.double(),
                    (original_latent.double() - grad.double()).detach(),
                    reduction="mean",
                )
            rl_dmd_loss = 0.5 * torch.exp(beta * reward_term) * mse

        # Aux loss adds AFTER the multiplicative reward structure
        if teleport_aux_loss_val is not None:
            rl_dmd_loss = rl_dmd_loss + self._teleport_aux_beta * teleport_aux_loss_val

        rl_dmd_log_dict.update(teleport_log)
        return rl_dmd_loss, rl_dmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        text_prompts: list,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        beta: float = 1.0,
        scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Unroll generator to obtain fake videos
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to, pixels = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent
        )

        # Step 2: Compute the DMD loss
        rl_dmd_loss, rl_dmd_log_dict = self.compute_rewarded_distribution_matching_loss(   
            image_or_video=pred_image,
            pixels = pixels,
            text_prompts=text_prompts,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            beta = beta,
            scores=scores,
        )

        return rl_dmd_loss, rl_dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # Step 1: Run generator on backward simulated noisy input
        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to, _ = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent
            )

        # Step 2: Compute the fake prediction
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=True
        )

        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )

        # Step 3: Compute the denoising loss for the fake critic
        if self.args.denoising_loss_type == "flow":
            from core.wan_wrapper.wan_wrapper_reward_forcing import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        # Step 5: Debugging Log
        critic_log_dict = {
            "critic_timestep": critic_timestep.detach()
        }

        return denoising_loss, critic_log_dict
