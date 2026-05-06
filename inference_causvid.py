import argparse
import os
import numpy as np
import torch

DEVICE_TYPE = os.environ.get('DEVICE_TYPE', 'cuda')
if DEVICE_TYPE == "npu":
    from torch_npu.contrib import transfer_to_npu

from omegaconf import OmegaConf
from tqdm import tqdm
from diffusers.utils import export_to_video

from methods.causvid.pipelines.bidirectional_inference import BidirectionalInferencePipeline
from methods.causvid.pipelines.causal_inference import CausalInferencePipeline
from methods.causvid.data import TextDataset
from core.misc import set_seed


def encode_latents(vae, videos):
    device, dtype = videos[0].device, videos[0].dtype
    scale = [vae.mean.to(device=device, dtype=dtype),
             1.0 / vae.std.to(device=device, dtype=dtype)]
    output = [
        vae.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]
    return torch.stack(output, dim=0)


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["bidirectional", "autoregressive", "longvideo"],
                    default="autoregressive",
                    help="Inference mode: bidirectional, autoregressive, or longvideo (multi-rollout)")
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--prompt_file_path", type=str, required=True)
parser.add_argument("--output_folder", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_output_frames", type=int, default=21)
parser.add_argument("--num_rollout", type=int, default=3,
                    help="Number of rollouts for longvideo mode")
parser.add_argument("--num_overlap_frames", type=int, default=3,
                    help="Number of overlap frames for longvideo mode")
args = parser.parse_args()

if args.output_folder:
    output_root = os.environ.get('OUTPUT_URL', '.')
    args.output_folder = os.path.join(output_root, args.output_folder)

set_seed(args.seed)
torch.set_grad_enabled(False)

if args.mode == "bidirectional":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

config = OmegaConf.load(args.config_path)

if args.mode == "bidirectional":
    pipeline = BidirectionalInferencePipeline(config, device="cuda")
else:
    pipeline = CausalInferencePipeline(config, device="cuda")

pipeline.to(device="cuda", dtype=torch.bfloat16)

state_dict = torch.load(os.path.join(args.checkpoint_path, "model.pt"), map_location="cpu")
pipeline.generator.load_state_dict(state_dict['generator'], strict=True)

dataset = TextDataset(args.prompt_file_path)
os.makedirs(args.output_folder, exist_ok=True)

sampled_noise = torch.randn(
    [1, args.num_output_frames, 16, 60, 104], device="cuda", dtype=torch.bfloat16
)

if args.mode == "longvideo":
    assert args.num_overlap_frames % pipeline.num_frame_per_block == 0, \
        "num_overlap_frames must be divisible by num_frame_per_block"

for prompt_index in tqdm(range(len(dataset))):
    prompts = [dataset[prompt_index]]

    if args.mode in ("bidirectional", "autoregressive"):
        video = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts
        )[0].permute(0, 2, 3, 1).cpu().numpy()

        export_to_video(
            video, os.path.join(args.output_folder, f"output_{prompt_index:03d}.mp4"), fps=16)

    elif args.mode == "longvideo":
        start_latents = None
        all_video = []
        overlap = args.num_overlap_frames

        for rollout_index in range(args.num_rollout):
            rollout_noise = torch.randn(
                [1, args.num_output_frames, 16, 60, 104], device="cuda", dtype=torch.bfloat16
            )

            video, latents = pipeline.inference(
                noise=rollout_noise,
                text_prompts=prompts,
                return_latents=True,
                start_latents=start_latents
            )

            current_video = video[0].permute(0, 2, 3, 1).cpu().numpy()

            start_frame = encode_latents(pipeline.vae, (
                video[:, -4 * (overlap - 1) - 1:-4 * (overlap - 1), :] * 2.0 - 1.0
            ).transpose(2, 1).to(torch.bfloat16)).transpose(2, 1).to(torch.bfloat16)

            start_latents = torch.cat(
                [start_frame, latents[:, -(overlap - 1):]], dim=1
            )

            all_video.append(current_video[:-(4 * (overlap - 1) + 1)])

        video = np.concatenate(all_video, axis=0)
        export_to_video(
            video, os.path.join(args.output_folder, f"long_video_output_{prompt_index:03d}.mp4"), fps=16)