# CausVid version of wan_wrapper
# Key features vs base (wan_wrapper.py):
#   - Uses DiffusionModelInterface/VAEInterface/TextEncoderInterface abstract base classes
#   - WanDiffusionWrapper: uses current_end instead of cache_start, no clean_x/aug_t/classify_mode,
#     forward returns pred_x0 only (not tuple), CausalWanDiffusionWrapper subclass for causal mode
#   - WanVAEWrapper: no encode_to_latent, no use_cache support, simpler decode_to_pixel
#   - WanTextEncoder: uses next(self.parameters()).device instead of torch.cuda.current_device()
#   - [DIFF-Base] Base (CF/SF) uses cache_start, clean_x/aug_t, classify_mode, returns (flow_pred, pred_x0)
#   - [DIFF-LongLive] LongLive adds sink_recache_after_switch, dynamic seq_len, decode_to_pixel_chunk
import os
from typing import List, Optional
import torch

from core.scheduler.scheduler import FlowMatchScheduler
from core.wan_wrapper.model_interface import (
    DiffusionModelInterface,
    TextEncoderInterface,
    VAEInterface
)
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model_causvid import CausalWanModel


class WanTextEncoder(TextEncoderInterface):
    def __init__(self) -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(f"{os.environ.get('MODELS_DIR', 'wan_models')}/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=f"{os.environ.get('MODELS_DIR', 'wan_models')}/Wan2.1-T2V-1.3B/google/umt5-xxl/", seq_len=512, clean='whitespace')

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(VAEInterface):
    def __init__(self):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        self.model = _video_vae(
            pretrained_path=f"{os.environ.get('MODELS_DIR', 'wan_models')}/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
            z_dim=16,
        ).eval().requires_grad_(False)

    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.decode(u.unsqueeze(0),
                              scale).float().clamp_(-1, 1).squeeze(0)
            for u in zs
        ]
        output = torch.stack(output, dim=0)
        output_p = output.permute(0, 2, 1, 3, 4)
        return output_p


class WanDiffusionWrapper(DiffusionModelInterface):
    def __init__(self):
        super().__init__()

        self.model = WanModel.from_pretrained(f"{os.environ.get('MODELS_DIR', 'wan_models')}/Wan2.1-T2V-1.3B/")
        self.model.eval()

        self.uniform_timestep = True

        self.scheduler = FlowMatchScheduler(
            shift=8.0, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760
        super().post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self, noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                current_end=current_end
            ).permute(0, 2, 1, 3, 4)
        else:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len
            ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        return pred_x0


class CausalWanDiffusionWrapper(WanDiffusionWrapper):
    def __init__(self):
        super().__init__()

        self.model = CausalWanModel.from_pretrained(
            f"{os.environ.get('MODELS_DIR', 'wan_models')}/Wan2.1-T2V-1.3B/")
        self.model.eval()

        self.uniform_timestep = False
