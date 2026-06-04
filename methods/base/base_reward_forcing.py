# Reward-Forcing version of base model classes
# Key features:
#   - BaseModel: shared base class
#   - RewardForcingModel: SelfForcingModel with VideoVLMRewardInference for reward-guided training
#   - denoising_step_list/timesteps follow the same convention as base_causal_forcing
from typing import Tuple
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch

from core.loss.loss import get_denoising_loss
from core.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from methods.reward_forcing.pipelines.reward_forcing_training import RewardForcingTrainingPipeline
from methods.reward_forcing.pipelines.self_forcing_training import SelfForcingTrainingPipeline
from methods.reward_forcing.videoalign.wan_inference import VideoVLMRewardInference


class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).cuda()
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def _initialize_models(self, args, device):
        self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
        self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)

        self.real_score = WanDiffusionWrapper(model_name=self.real_model_name, is_causal=False)
        self.real_score.model.requires_grad_(False)

        self.fake_score = WanDiffusionWrapper(model_name=self.fake_model_name, is_causal=False)
        self.fake_score.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

        # Reward-Forcing specific parameters
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)
        self.independent_first_frame = getattr(args, "independent_first_frame", False)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )
            if self.independent_first_frame:
                timestep_from_second = timestep[:, 1:]
                tfs_reshaped = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block)
                timestep_from_second = tfs_reshaped
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                tfs_reshaped2 = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1)
                timestep_from_second = tfs_reshaped2
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                ts_reshaped = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep = ts_reshaped
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                ts_reshaped2 = timestep.reshape(timestep.shape[0], -1)
                timestep = ts_reshaped2
            return timestep


class RewardForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

        # Reward-Forcing specific: initialize reward inference
        if hasattr(args, "reward_model_path") and args.reward_model_path:
            self.inferencer = VideoVLMRewardInference(
                load_from_pretrained=args.reward_model_path,
                device=device,
                dtype=self.dtype
            )
            self.inferencer.model.requires_grad_(False)
        else:
            self.inferencer = None

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        """
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        if getattr(self.args, "i2v", False):
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]
        else:
            noise_shape = image_or_video_shape.copy()

        min_num_frames = 20 if self.args.independent_first_frame else 21
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        noise_shape[1] = num_generated_frames

        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            noise=torch.randn(noise_shape,
                              device=self.device, dtype=self.dtype),
            **conditional_dict,
        )
        # Slice last 21 frames
        if pred_image_or_video.shape[1] > 21:
            with torch.no_grad():
                latent_to_decode = pred_image_or_video[:, :-20, ...]
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            pred_image_or_video_last_21 = torch.cat([image_latent, pred_image_or_video[:, -20:, ...]], dim=1)
        else:
            with torch.no_grad():
                latent_to_decode = pred_image_or_video[:, :-20, ...]
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            pred_image_or_video_last_21 = pred_image_or_video

        if num_generated_frames != min_num_frames:
            gradient_mask = torch.ones_like(pred_image_or_video_last_21, dtype=torch.bool)
            if self.args.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, :self.num_frame_per_block] = False
        else:
            gradient_mask = None

        pred_image_or_video_last_21 = pred_image_or_video_last_21.to(self.dtype)
        with torch.no_grad():
            pred_video_pixel = self.vae.decode_to_pixel(pred_image_or_video_last_21).to(self.dtype)
        return pred_image_or_video_last_21, gradient_mask, denoised_timestep_from, denoised_timestep_to, pred_video_pixel

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        **conditional_dict: dict
    ) -> torch.Tensor:
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise, **conditional_dict
        )

    def _initialize_inference_pipeline(self):
        self.inference_pipeline = SelfForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=getattr(self.args, "last_step_only", False),
            num_max_frames=self.num_training_frames,
            context_noise=getattr(self.args, "context_noise", 0)
        )

    def _initialize_inference_pipeline_reward(self):
        self.inference_pipeline = RewardForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=getattr(self.args, "last_step_only", False),
            num_max_frames=self.num_training_frames,
            context_noise=getattr(self.args, "context_noise", 0),
            spatial_self=True
        )
