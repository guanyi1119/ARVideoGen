# CausVid inference entry point
# Based on archive/CausVid/minimal_inference/ scripts
import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from diffusers.utils import export_to_video
import torch.distributed as dist

from methods.causvid.pipelines.causal_inference import CausalInferencePipeline
from methods.causvid.data import TextDataset
from core.misc import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--prompt_file_path", type=str, required=True)
parser.add_argument("--output_folder", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_output_frames", type=int, default=21)
args = parser.parse_args()

set_seed(args.seed)
torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)

pipeline = CausalInferencePipeline(config, device="cuda")
pipeline.to(device="cuda", dtype=torch.bfloat16)

state_dict = torch.load(os.path.join(args.checkpoint_path, "model.pt"), map_location="cpu")
pipeline.generator.load_state_dict(state_dict['generator'], strict=True)

dataset = TextDataset(args.prompt_file_path)

sampled_noise = torch.randn(
    [1, args.num_output_frames, 16, 60, 104], device="cuda", dtype=torch.bfloat16
)

os.makedirs(args.output_folder, exist_ok=True)

for prompt_index in tqdm(range(len(dataset))):
    prompts = [dataset[prompt_index]]

    video = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts
    )[0].permute(0, 2, 3, 1).cpu().numpy()

    export_to_video(
        video, os.path.join(args.output_folder, f"output_{prompt_index:03d}.mp4"), fps=16)
