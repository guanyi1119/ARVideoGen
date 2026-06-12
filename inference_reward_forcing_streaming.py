# Reward-Forcing Streaming Inference entry point
# Based on inference_longlive.py, adapted for reward_forcing
import argparse
import torch
import os
DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "cuda")
if DEVICE_TYPE == "npu":
    from torch_npu.contrib import transfer_to_npu

import os
from omegaconf import OmegaConf
from core.config import load_config
from tqdm import tqdm
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from methods.reward_forcing.pipelines import CausalInferencePipeline
from core.data.dataset import TextDataset
from core.misc import set_seed
from core.misc.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint file")
parser.add_argument("--data_path", type=str, default=None, help="Path to the prompt file")
parser.add_argument("--output_folder", type=str, default=None, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=None, help="Number of output frames")
parser.add_argument("--global_sink", type=lambda x: x.lower() == "true", default=True, help="Use global sink (default: True)")
parser.add_argument("--use_ema", action="store_true", help="Use EMA weights")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--save_with_index", action="store_true")
args_cli = parser.parse_args()

default_config_path = os.path.join(os.path.dirname(args_cli.config_path), "default_config.yaml")
config = load_config(args_cli.config_path, default_config_path=default_config_path)

# Override config with CLI args
config.global_sink = args_cli.global_sink
if args_cli.num_output_frames is not None:
    config.num_output_frames = args_cli.num_output_frames
elif not hasattr(config, "num_output_frames"):
    config.num_output_frames = getattr(config, "streaming_max_length", 240)
if args_cli.checkpoint_path is not None:
    config.generator_ckpt = args_cli.checkpoint_path
if args_cli.data_path is not None:
    config.data_path = args_cli.data_path
if args_cli.output_folder is not None:
    output_root = os.environ.get("OUTPUT_URL", ".")
    config.output_folder = os.path.join(output_root, args_cli.output_folder)
config.use_ema = args_cli.use_ema
config.seed = args_cli.seed
config.num_samples = args_cli.num_samples
config.save_with_index = args_cli.save_with_index

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    os.environ["NCCL_CROSS_NIC"] = "1"
    os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
    os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout,
        )
    set_seed(config.seed + local_rank)
    config.distributed = True
    if rank == 0:
        print(f"[Rank {rank}] Initialized distributed processing on device {device}")
else:
    local_rank = 0
    rank = 0
    device = torch.device("cuda")
    set_seed(config.seed)
    config.distributed = False
    print(f"Single GPU mode on device {device}")

print(f"Free VRAM {get_cuda_free_memory_gb(device)} GB")
low_memory = get_cuda_free_memory_gb(device) < 40
low_memory = True

torch.set_grad_enabled(False)

# Initialize pipeline
pipeline = CausalInferencePipeline(config, device=device)

# Load generator checkpoint
if getattr(config, "generator_ckpt", None):
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    if "generator" in state_dict or "generator_ema" in state_dict:
        raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
    elif "model" in state_dict:
        raw_gen_state_dict = state_dict["model"]
    else:
        raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
    if config.use_ema:
        def _clean_key(name: str) -> str:
            name = name.replace("_fsdp_wrapped_module.", "")
            return name
        cleaned_state_dict = {_clean_key(k): v for k, v in raw_gen_state_dict.items()}
        missing, unexpected = pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
        if local_rank == 0:
            if len(missing) > 0:
                print(f"[Warning] {len(missing)} parameters missing: {missing[:8]} ...")
            if len(unexpected) > 0:
                print(f"[Warning] {len(unexpected)} unexpected parameters: {unexpected[:8]} ...")
    else:
        pipeline.generator.load_state_dict(raw_gen_state_dict)

# LoRA support (optional)
from core.misc.lora_utils import configure_lora_for_model
import peft

pipeline.is_lora_enabled = False
if getattr(config, "adapter", None) and configure_lora_for_model is not None:
    if local_rank == 0:
        print(f"LoRA enabled with config: {config.adapter}")
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        if local_rank == 0:
            print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
    pipeline.is_lora_enabled = True

# Move pipeline to appropriate dtype and device
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

extended_prompt_path = getattr(config, "data_path", None)
dataset = TextDataset(prompt_path=config.data_path, extended_prompt_path=extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

if local_rank == 0:
    os.makedirs(config.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data["idx"].item()

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]

    all_video = []

    prompt = batch["prompts"][0]
    extended_prompt = batch["extended_prompts"][0] if "extended_prompts" in batch else None
    if extended_prompt is not None:
        prompts = [extended_prompt] * config.num_samples
    else:
        prompts = [prompt] * config.num_samples

    sampled_noise = torch.randn(
        [config.num_samples, config.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
    )

    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        low_memory=low_memory,
        profile=False,
    )
    current_video = rearrange(video, "b t c h w -> b t h w c").cpu()
    all_video.append(current_video)

    video = 255.0 * torch.cat(all_video, dim=1)
    pipeline.vae.model.clear_cache()

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if idx < num_prompts:
        if hasattr(pipeline, "is_lora_enabled") and pipeline.is_lora_enabled:
            model_type = "lora"
        elif config.use_ema:
            model_type = "ema"
        else:
            model_type = "regular"

        for seed_idx in range(config.num_samples):
            if config.save_with_index:
                output_path = os.path.join(config.output_folder, f"rank{rank}-{idx}-{seed_idx}_{model_type}.mp4")
            else:
                output_path = os.path.join(config.output_folder, f"rank{rank}-{prompt[:100]}-{seed_idx}.mp4")
            write_video(output_path, video[seed_idx], fps=16)

    if getattr(config, "inference_iter", -1) != -1 and i >= config.inference_iter:
        break

if dist.is_initialized():
    dist.destroy_process_group()