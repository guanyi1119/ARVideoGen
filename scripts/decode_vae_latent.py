"""
Decode VAE latents from LMDB dataset to videos.

Usage:
    # Decode specific samples
    python scripts/decode_vae_latent.py \
        --data_path <lmdb_dir> \
        --output_dir <output_dir> \
        --indices 0,1,5

    # Decode all samples
    python scripts/decode_vae_latent.py \
        --data_path <lmdb_dir> \
        --output_dir <output_dir>

    # Specify folder pattern and FPS
    python scripts/decode_vae_latent.py \
        --data_path <lmdb_dir> \
        --folder_name_pattern shard \
        --output_dir <output_dir> \
        --fps 16
"""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'methods', 'anyflow'))

import argparse
import torch
import numpy as np
from tqdm import tqdm

from core.wan_wrapper.wan_wrapper import WanVAEWrapper
from far.data.lmdb_dataset import MultiODERegressionLMDBDataset

torch.set_grad_enabled(False)


def save_video(tensor, path, fps=16):
    """Save a (T, C, H, W) tensor in [-1, 1] as mp4 video."""
    import imageio
    frames = ((tensor + 1.0) / 2.0 * 255).clamp(0, 255).to(torch.uint8)
    frames = frames.permute(0, 2, 3, 1).cpu().numpy()
    writer = imageio.get_writer(path, fps=fps, codec='libx264',
                                output_params=['-pix_fmt', 'yuv420p'])
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def main():
    parser = argparse.ArgumentParser(description='Decode VAE latents from LMDB to videos')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to LMDB data directory')
    parser.add_argument('--folder_name_pattern', type=str, default='',
                        help='Folder name pattern to filter LMDB shards')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save decoded videos')
    parser.add_argument('--indices', type=str, default=None,
                        help='Comma-separated sample indices to decode (e.g. "0,1,5"). Default: all')
    parser.add_argument('--max_pair', type=int, default=int(1e8),
                        help='Max number of pairs loaded by dataset')
    parser.add_argument('--fps', type=int, default=16,
                        help='Output video FPS')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for VAE decoding')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = MultiODERegressionLMDBDataset(
        data_path=args.data_path,
        folder_name_pattern=args.folder_name_pattern,
        max_pair=args.max_pair,
    )
    print(f"Dataset size: {len(dataset)}")

    if args.indices is not None:
        indices = [int(i) for i in args.indices.split(',')]
    else:
        indices = list(range(len(dataset)))

    vae = WanVAEWrapper().to(device=args.device, dtype=torch.float32).eval()

    for idx in tqdm(indices):
        sample = dataset[idx]
        latents = sample['latents']   # (C, T, H, W)
        prompt = sample['prompts']

        latents = latents.unsqueeze(0).to(device=args.device, dtype=torch.float32)

        with torch.no_grad():
            pixels = vae.decode_to_pixel(latents)  # (1, C, T, H, W)

        pixels = pixels.squeeze(0).cpu()  # (C, T, H, W)
        save_path = os.path.join(args.output_dir, f'{idx:06d}.mp4')
        save_video(pixels, save_path, fps=args.fps)

        tqdm.write(f'[{idx}] Saved {save_path} | prompt: {str(prompt)[:80]}')

    print(f'Done. Saved {len(indices)} videos to {args.output_dir}')


if __name__ == '__main__':
    main()
