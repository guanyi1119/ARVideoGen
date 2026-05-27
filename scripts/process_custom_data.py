"""
Process custom video data: read JSONL, resize, extract VAE latents, save to directory.

Features:
1. Read from JSONL (video_fn, long_prompt)
2. Resize to 81 frames, 480x848
3. Extract VAE latents using WanVAEWrapper
4. Save latents and prompts to directory (one .npz per video)
5. Supports moxing for remote paths (obs://)
6. Optional: save resized videos

Usage:
    # Single GPU (local files)
    python scripts/process_custom_data.py \
        --jsonl_path data.jsonl \
        --output_dir processed_data

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 scripts/process_custom_data.py \
        --jsonl_path data.jsonl \
        --output_dir processed_data

    # With moxing for remote videos (obs://)
    python scripts/process_custom_data.py \
        --jsonl_path data.jsonl \
        --output_dir obs://bucket/processed_data \
        --use_moxing
"""
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'methods', 'anyflow'))

import time
import argparse
import json
import tempfile
from io import BytesIO
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

import torch
# NPU support
DEVICE_TYPE = os.environ.get('DEVICE_TYPE', 'cuda')
if DEVICE_TYPE == "npu":
    from torch_npu.contrib import transfer_to_npu
import torch.distributed as dist
import imageio
import imageio.v3 as iio
from skimage.transform import resize

from core.wan_wrapper.wan_wrapper import WanVAEWrapper
from core.distributed.distributed import launch_distributed_job


# Optional moxing import
mox = None
def try_import_moxing():
    global mox
    try:
        import moxing as mox
        return True
    except ImportError:
        return False


def read_jsonl(jsonl_path):
    """Read JSONL file and return list of (video_fn, long_prompt)."""
    data = []
    open_fn = open
    if mox is not None and (jsonl_path.startswith('obs://') or jsonl_path.startswith('s3://')):
        open_fn = lambda p, mode: mox.file.File(p, mode)
    with open_fn(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            video_fn = item.get('video_fn', '')
            prompt = item.get('long_prompt', '')
            if video_fn and prompt:
                data.append((video_fn, prompt))
    return data


def get_video_reader(video_path, use_moxing):
    """Get a video reader that works with local or remote (moxing) paths."""
    # Prepend VIDEO_DIR environment variable if exists
    video_dir = os.environ.get('VIDEO_DIR', '')
    if video_dir:
        video_path = os.path.join(video_dir, video_path)

    if use_moxing and mox is not None and (video_path.startswith('obs://') or video_path.startswith('s3://')):
        # Read directly into memory via mox.file.File
        with mox.file.File(video_path, 'rb') as f:
            video_bytes = f.read()
        video_buffer = BytesIO(video_bytes)
        return iio.imread(video_buffer, plugin='pyav')
    else:
        # Local read
        return iio.imread(video_path, plugin='pyav')


def save_resized_video(video_np, save_path, fps=16, use_moxing=False):
    """Save resized video (T, H, W, 3) as mp4, supports moxing remote paths."""
    # Check if remote path
    is_remote = use_moxing and (save_path.startswith('obs://') or save_path.startswith('s3://'))

    if is_remote and mox is not None:
        # Write to local temp file first, then copy to remote
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            # Write to temp
            writer = imageio.get_writer(tmp_path, fps=fps, codec='libx264', output_params=['-pix_fmt', 'yuv420p'])
            for frame in video_np:
                writer.append_data(frame)
            writer.close()
            # Copy to remote
            try:
                mox.file.make_dirs(os.path.dirname(save_path))
            except Exception:
                pass  # Directory might already exist
            mox.file.copy(tmp_path, save_path)
        finally:
            os.unlink(tmp_path)
    else:
        # Local write
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        writer = imageio.get_writer(save_path, fps=fps, codec='libx264', output_params=['-pix_fmt', 'yuv420p'])
        for frame in video_np:
            writer.append_data(frame)
        writer.close()


def resize_video(video_np, target_frames=81, target_height=480, target_width=848, fps=16):
    """
    Resize video: take first 5 seconds (81 frames @16fps), then resize spatially.
    Input shape: (T, H, W, 3)
    Output shape: (target_frames, target_height, target_width, 3)
    """
    T, H, W, C = video_np.shape

    # Step 1: Temporal crop - take first 5 seconds (81 frames @16fps)
    max_frames = int(fps * (81 / fps))  # exactly 81 frames
    resized_t = video_np[:min(T, max_frames)]

    # Pad with last frame if too short
    if resized_t.shape[0] < max_frames:
        pad_frames = max_frames - resized_t.shape[0]
        last_frame = resized_t[-1:]
        padding = np.tile(last_frame, (pad_frames, 1, 1, 1))
        resized_t = np.concatenate([resized_t, padding], axis=0)

    # Step 2: Spatial resize (height, width) - same as process_mixkit.py
    resized = []
    for frame in resized_t:
        frame_float = frame.astype(float) / 255.0
        resized_frame = resize(
            frame_float,
            (target_height, target_width, 3),
            mode='reflect',
            anti_aliasing=True,
            preserve_range=True
        )
        resized.append((resized_frame * 255).astype(np.uint8))
    resized_spatial = np.stack(resized, axis=0)

    return resized_spatial


def encode_video(vae, video_np, device):
    """
    Encode video using VAE (exact same logic as compute_vae_latent.py).
    Input: (T, H, W, 3) numpy array (uint8)
    Output: (1, T, C, H/8, W/8) tensor (exact same as compute_vae_latent.py saves)
    """
    # Normalize to [-1, 1] and reshape - same as compute_vae_latent.py line 93-97
    video_tensor = torch.tensor(video_np, dtype=torch.float32, device=device).unsqueeze(0).permute(0, 4, 1, 2, 3) / 255.0
    video_tensor = video_tensor * 2 - 1  # [0,1] -> [-1,1]
    video_tensor = video_tensor.to(torch.bfloat16)

    # Encode - same as compute_vae_latent.py line 36-46 encode() function
    def encode_fn(vae_model, videos):
        device, dtype = videos[0].device, videos[0].dtype
        scale = [vae_model.mean.to(device=device, dtype=dtype),
                 1.0 / vae_model.std.to(device=device, dtype=dtype)]
        output = [
            vae_model.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in videos
        ]
        output = torch.stack(output, dim=0)
        return output

    with torch.no_grad():
        encoded_latents = encode_fn(vae, video_tensor).transpose(2, 1)  # line 98: (1, C, T, H, W) -> (1, T, C, H, W)

    return encoded_latents.cpu().detach()  # keep batch dim (1, T, C, H, W), same as .pt files


def save_latent_and_prompt(latent_np, prompt, output_path, use_moxing):
    """Save latent and prompt to .npz file, supports moxing remote paths."""
    is_remote = use_moxing and (output_path.startswith('obs://') or output_path.startswith('s3://'))

    if is_remote and mox is not None:
        # Save to BytesIO first, then write to remote
        bio = BytesIO()
        np.savez(bio, latent=latent_np, prompt=prompt)
        bio.seek(0)
        # Create remote directory
        try:
            mox.file.make_dirs(os.path.dirname(output_path))
        except Exception:
            pass  # Directory might already exist
        with mox.file.File(output_path, 'wb') as f:
            f.write(bio.read())
    else:
        # Local save
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        np.savez(output_path, latent=latent_np, prompt=prompt)


def main():
    parser = argparse.ArgumentParser(description="Process custom video data and save latents to directory")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed latents")
    parser.add_argument("--save_video_dir", type=str, default=None, help="Directory to save resized videos (optional)")
    parser.add_argument("--use_moxing", action="store_true", help="Enable moxing for remote file access")
    parser.add_argument("--target_frames", type=int, default=81, help="Target number of frames (default: 81)")
    parser.add_argument("--target_height", type=int, default=480, help="Target height (default: 480)")
    parser.add_argument("--target_width", type=int, default=832, help="Target width (default: 832)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for VAE (default: cuda)")
    parser.add_argument("--deduplicate_prompts", action="store_true", help="Deduplicate prompts (same as create_lmdb_iterative.py)")
    args = parser.parse_args()

    # Prepend OUTPUT_URL environment variable if exists
    output_url = os.environ.get('OUTPUT_URL', '')
    if output_url:
        args.output_dir = os.path.join(output_url, args.output_dir)
        if args.save_video_dir:
            args.save_video_dir = os.path.join(output_url, args.save_video_dir)

    # Import moxing if needed
    if args.use_moxing:
        if not try_import_moxing():
            print("Warning: moxing not available, falling back to local file access only")
            args.use_moxing = False

    # Setup distributed
    launch_distributed_job()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_main = rank == 0

    if is_main:
        print(f"Processing data from: {args.jsonl_path}")
        print(f"Saving latents to: {args.output_dir}")
        print(f"Target size: {args.target_frames} frames, {args.target_height}x{args.target_width}")

    # Read all data on main rank
    all_data = []
    if is_main:
        all_data = read_jsonl(args.jsonl_path)
        print(f"Total videos in JSONL: {len(all_data)}")

        # Deduplicate prompts if needed
        if args.deduplicate_prompts:
            seen_prompts = set()
            deduplicated = []
            for video_fn, prompt in all_data:
                if prompt not in seen_prompts:
                    seen_prompts.add(prompt)
                    deduplicated.append((video_fn, prompt))
            all_data = deduplicated
            print(f"Deduplicated to {len(all_data)} videos")

    # Broadcast data to all ranks
    if dist.is_initialized():
        import pickle
        data_obj = pickle.dumps(all_data) if is_main else None
        data_obj = [data_obj]
        dist.broadcast_object_list(data_obj, src=0)
        if not is_main:
            all_data = pickle.loads(data_obj[0])

    # Determine this rank's share of work
    num_total = len(all_data)
    start_idx = rank * (num_total // world_size) + min(rank, num_total % world_size)
    end_idx = start_idx + (num_total // world_size) + (1 if rank < num_total % world_size else 0)
    local_data = all_data[start_idx:end_idx]

    print(f"Rank {rank}: processing videos {start_idx} to {end_idx - 1} ({len(local_data)} videos)")

    # Load VAE
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16).eval()

    # Process local videos
    start_step = 0
    start_time = time.time()
    seq_len = 1560
    num_heads = 12
    local_success_count = 0

    for idx, (video_fn, prompt) in enumerate(tqdm(local_data, desc=f"Rank {rank} processing", disable=not is_main)):
        try:
            # Read video
            video_np = get_video_reader(video_fn, args.use_moxing)

            # Resize
            resized_np = resize_video(
                video_np,
                target_frames=args.target_frames,
                target_height=args.target_height,
                target_width=args.target_width,
                fps=16
            )

            # Save resized video if save_video_dir is set
            if args.save_video_dir:
                video_basename = os.path.basename(video_fn)
                video_name, _ = os.path.splitext(video_basename)
                save_path = os.path.join(args.save_video_dir, f"{video_name}_{start_idx + idx:08d}.mp4")
                save_resized_video(resized_np, save_path, 16, args.use_moxing)

            # Encode
            latent = encode_video(vae, resized_np, device)

            # Save to output directory
            latent_np = latent.numpy().astype(np.float16).squeeze(0)  # (T, C, H, W) float16
            output_path = os.path.join(args.output_dir, f"latent_{start_idx + idx:08d}.npz")
            save_latent_and_prompt(latent_np, prompt, output_path, args.use_moxing)

            local_success_count += 1

            end_time = time.time()
            end_step = idx + 1

            # Calculate throughput
            step_diff = end_step - start_step
            time_diff = end_time - start_time
            seconds_per_iter = time_diff / step_diff if step_diff > 0 else 0
            throughput = 1 * seq_len * num_heads / seconds_per_iter if seconds_per_iter > 0 else 0

            # Print log
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"{timestamp}: [step {idx}] " \
                f"DI_throughput: {throughput:.2f} tokens/s/npu"
            )
            start_time = time.time()
            start_step = end_step

        except Exception as e:
            print(f"Rank {rank}: failed to process {video_fn}: {e}")
            continue

    # Wait for all ranks to finish
    if dist.is_initialized():
        dist.barrier()

    if is_main:
        print(f"Done! Successfully processed videos. Latents saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
