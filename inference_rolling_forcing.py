import argparse
import torch
DEVICE_TYPE = os.environ.get('DEVICE_TYPE', 'cuda')
if DEVICE_TYPE == "npu":
    from torch_npu.contrib import transfer_to_npu

import os
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
import imageio
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from methods.rolling_forcing.pipelines import CausalInferencePipeline
from core.data.dataset import TextDataset, TextImagePairDataset
from core.misc import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
    parser.add_argument("--data_path", type=str, help="Path to the dataset")
    parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
    parser.add_argument("--output_folder", type=str, help="Output folder")
    parser.add_argument("--num_output_frames", type=int, default=21,
                        help="Number of overlap frames between sliding windows")
    parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
    parser.add_argument("--save_with_index", action="store_true",
                        help="Whether to save the video using the index or prompt as the filename")
    args = parser.parse_args()

    # Initialize distributed inference
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        set_seed(args.seed + local_rank)
    else:
        device = torch.device("cuda")
        local_rank = 0
        world_size = 1
        set_seed(args.seed)

    torch.set_grad_enabled(False)

    config = OmegaConf.load(args.config_path)
    default_config_path = os.path.join(os.path.dirname(args.config_path), "default_config.yaml")
    if os.path.exists(default_config_path):
        default_config = OmegaConf.load(default_config_path)
        config = OmegaConf.merge(default_config, config)

    # Initialize pipeline
    if hasattr(config, 'denoising_step_list'):
        # Few-step inference
        pipeline = CausalInferencePipeline(config, device=device)
    else:
        raise ValueError("denoising_step_list is required for rolling forcing inference")

    if args.checkpoint_path:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        if args.use_ema:
            state_dict_to_load = state_dict['generator_ema']
            def remove_fsdp_prefix(state_dict):
                new_state_dict = OrderedDict()
                for key, value in state_dict.items():
                    if "_fsdp_wrapped_module." in key:
                        new_key = key.replace("_fsdp_wrapped_module.", "")
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                return new_state_dict
            state_dict_to_load = remove_fsdp_prefix(state_dict_to_load)
        else:
            state_dict_to_load = state_dict['generator']
        pipeline.generator.load_state_dict(state_dict_to_load)

    pipeline = pipeline.to(device=device, dtype=torch.bfloat16)

    # Create dataset
    if args.i2v:
        assert not dist.is_initialized(), "I2V does not support distributed inference yet"
        transform = transforms.Compose([
            transforms.Resize((480, 832)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset = TextImagePairDataset(args.data_path, transform=transform)
    else:
        dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
    num_prompts = len(dataset)
    print(f"Number of prompts: {num_prompts}")

    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

    # Create output directory (only on main process to avoid race conditions)
    if local_rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
        idx = batch_data['idx'].item()

        # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
        if isinstance(batch_data, dict):
            batch = batch_data
        elif isinstance(batch_data, list):
            batch = batch_data[0]

        all_video = []
        num_generated_frames = 0

        if args.i2v:
            prompt = batch['prompts'][0]
            prompts = [prompt] * args.num_samples

            image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
            initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

            sampled_noise = torch.randn(
                [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
            )
        else:
            prompt = batch['prompts'][0]
            extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
            if extended_prompt is not None:
                prompts = [extended_prompt] * args.num_samples
            else:
                prompts = [prompt] * args.num_samples
            initial_latent = None

            sampled_noise = torch.randn(
                [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
            )

        # Generate video using rolling forcing
        video, latents = pipeline.inference_rolling_forcing(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent,
        )
        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        all_video.append(current_video)
        num_generated_frames += latents.shape[1]

        video = 255.0 * torch.cat(all_video, dim=1)

        pipeline.vae.model.clear_cache()

        if idx < num_prompts:
            model = "regular" if not args.use_ema else "ema"
            for seed_idx in range(args.num_samples):
                if args.save_with_index:
                    output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
                else:
                    output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')
                write_video(output_path, video[seed_idx], fps=16)


if __name__ == "__main__":
    main()
