# Abstract interfaces for diffusion models, VAE, and text encoders.
# Adopted from CausVid's model_interface.py — used by CausVid's inference pipeline.
from core.scheduler.scheduler import SchedulerInterface
from abc import abstractmethod, ABC
from typing import List, Optional
import torch
import types


class DiffusionModelInterface(ABC, torch.nn.Module):
    scheduler: SchedulerInterface

    @abstractmethod
    def forward(
        self, noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None
    ) -> torch.Tensor:
        pass

    def get_scheduler(self) -> SchedulerInterface:
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        self.get_scheduler()

    def set_module_grad(self, module_grad: dict) -> None:
        for k, is_trainable in module_grad.items():
            getattr(self, k).requires_grad_(is_trainable)

    @abstractmethod
    def enable_gradient_checkpointing(self) -> None:
        pass


class VAEInterface(ABC, torch.nn.Module):
    @abstractmethod
    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        pass


class TextEncoderInterface(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, text_prompts: List[str]) -> dict:
        pass


class InferencePipelineInterface(ABC):
    @abstractmethod
    def inference_with_trajectory(self, noise: torch.Tensor, conditional_dict: dict) -> torch.Tensor:
        pass
