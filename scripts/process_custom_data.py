"""
Process custom video data: read JSONL(s), resize, extract VAE latents, save to directory.

Features:
1. Read from one or more JSONL files (video_fn, long_prompt)
2. Resize to 81 frames, 480x848 (GPU accelerated)
3. Extract VAE latents using WanVAEWrapper (supports batching)
4. Save latents and prompts to directory (one .npz per video)
5. Supports moxing for remote file access (obs://)
6. Optional: save resized videos alongside latents in the same directory
7. Pipeline parallelism: IO workers + prefetch queue + async writers for high GPU utilization

Optimizations (A + C + D + optional B):
- A. Pipeline parallelism with ThreadPoolExecutor + Queue
- C. GPU resize using torch.nn.functional.interpolate
- D. Async writing to disk/moxing
- B. Optional VAE batching (--vae_batch_size)

Usage:
    # Single GPU, single JSONL
    python scripts/process_custom_data.py \
        --jsonl_paths data.jsonl \
        --output_dir processed_data

    # Multiple JSONL files
    python scripts/process_custom_data.py \
        --jsonl_paths data1.jsonl data2.jsonl data3.jsonl \
        --output_dir processed_data

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 scripts/process_custom_data.py \
        --jsonl_paths data.jsonl \
        --output_dir processed_data

    # With moxing for remote videos (obs://)
    python scripts/process_custom_data.py \
        --jsonl_paths data.jsonl \
        --output_dir obs://bucket/processed_data \
        --use_moxing

    # Also save resized videos in a separate directory
    python scripts/process_custom_data.py \
        --jsonl_paths data.jsonl \
        --output_dir processed_data/latents \
        --save_video_dir processed_data/videos

    # With optimizations enabled
    python scripts/process_custom_data.py \
        --jsonl_paths data.jsonl \
        --output_dir processed_data \
        --num_io_workers 8 \
        --num_write_workers 4 \
        --prefetch 16 \
        --vae_batch_size 2
"""
import sys
import os
# Disable moxing cache BEFORE any moxing imports to avoid "database is locked" issues
os.environ['MOX_FILE_CACHE_ENABLE'] = '0'
os.environ['MOX_ENABLE_CACHE'] = '0'
os.environ['MOX_DISABLE_CACHE'] = '1'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'methods', 'anyflow'))

import argparse
import json
import tempfile
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
from queue import Queue
import threading
import hashlib
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
# NPU support
DEVICE_TYPE = os.environ.get('DEVICE_TYPE', 'cuda')
if DEVICE_TYPE == "npu":
    from torch_npu.contrib import transfer_to_npu
import torch.distributed as dist
import imageio
import imageio.v3 as iio

from core.wan_wrapper.wan_wrapper import WanVAEWrapper
from core.distributed.distributed import launch_distributed_job


VIDEO_FN_KEY = 'video_fn'
PROMPT_KEY = 'Qwen3VL_32B_General_Caption_Level_2'

# Optional moxing import
mox = None
# Global lock for ALL moxing operations to avoid "database is locked" errors
_mox_lock = threading.Lock()

def try_import_moxing():
    global mox
    try:
        import moxing as mox
        return True
    except ImportError:
        return False


@dataclass
class WorkItem:
    """Item passed through the pipeline queues."""
    idx: int
    video_fn: str
    prompt: str
    file_hash: str  # Hash of video_fn for unique filename
    resized_np: Optional[np.ndarray] = None  # (T, H, W, 3) uint8
    latent_np: Optional[np.ndarray] = None   # (T, C, H, W) float16
    success: bool = True
    error_msg: str = ""
    skip_existing: bool = False  # Whether to skip because file already exists


def get_video_hash(video_fn: str) -> str:
    """Generate a short hash from video filename for unique filenames."""
    hash_bytes = hashlib.md5(video_fn.encode('utf-8')).digest()
    # Use base64-like encoding but filename-safe (replace /+ with -_)
    import base64
    hash_str = base64.urlsafe_b64encode(hash_bytes).decode('ascii')[:12]
    return hash_str


def read_jsonl(jsonl_path):
    """Read JSONL file and return list of (VIDEO_FN_KEY, PROMPT_KEY)."""
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
            video_fn = item.get(VIDEO_FN_KEY, '')
            prompt = item.get(PROMPT_KEY, '')
            if video_fn and prompt:
                data.append((video_fn, prompt))
    return data


def read_multiple_jsonl(jsonl_paths):
    """Read multiple JSONL files and return combined list of (VIDEO_FN_KEY, PROMPT_KEY)."""
    all_data = []
    for path in jsonl_paths:
        print(f"  Reading: {path}")
        data = read_jsonl(path)
        print(f"    Found {len(data)} items")
        all_data.extend(data)
    return all_data


def get_video_reader(video_path, use_moxing):
    """Get a video reader that works with local or remote (moxing) paths."""
    # Prepend VIDEO_DIR environment variable if exists
    video_dir = os.environ.get('VIDEO_DIR', '')
    if video_dir:
        video_path = os.path.join(video_dir, video_path)

    if use_moxing and mox is not None and (video_path.startswith('obs://') or video_path.startswith('s3://')):
        with _mox_lock:
            with mox.file.File(video_path, 'rb') as f:
                video_bytes = f.read()
        video_buffer = BytesIO(video_bytes)
        return iio.imread(video_buffer, plugin='pyav')
    else:
        return iio.imread(video_path, plugin='pyav')


def resize_video_gpu(video_np, target_frames=81, target_height=480, target_width=832, fps=16, device='cuda'):
    """
    Resize video using GPU: take first 5 seconds (81 frames @16fps), then resize spatially.
    Input shape: (T, H, W, 3) numpy uint8
    Output shape: (target_frames, target_height, target_width, 3) numpy uint8
    """
    T, H, W, C = video_np.shape

    # Temporal trim/pad on CPU first (lightweight)
    max_frames = int(fps * (81 / fps))
    resized_t = video_np[:min(T, max_frames)]

    if resized_t.shape[0] < max_frames:
        pad_frames = max_frames - resized_t.shape[0]
        last_frame = resized_t[-1:]
        padding = np.tile(last_frame, (pad_frames, 1, 1, 1))
        resized_t = np.concatenate([resized_t, padding], axis=0)

    # Spatial resize on GPU
    # (T, H, W, C) -> (T, C, H, W) for torch
    video_tensor = torch.from_numpy(resized_t).permute(0, 3, 1, 2).float() / 255.0
    video_tensor = video_tensor.to(device)

    # Resize with bilinear, align_corners=False (matches skimage behavior approximately)
    resized_tensor = F.interpolate(
        video_tensor,
        size=(target_height, target_width),
        mode='bilinear',
        align_corners=False,
        antialias=True
    )

    # Back to uint8 numpy, (T, H, W, C)
    resized_tensor = (resized_tensor.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    resized_np = resized_tensor.permute(0, 2, 3, 1).cpu().numpy()

    return resized_np


def resize_video(video_np, target_frames=81, target_height=480, target_width=832, fps=16):
    """
    Resize video (CPU fallback, kept for backward compatibility).
    Prefer resize_video_gpu for performance.
    """
    T, H, W, C = video_np.shape

    max_frames = int(fps * (81 / fps))
    resized_t = video_np[:min(T, max_frames)]

    if resized_t.shape[0] < max_frames:
        pad_frames = max_frames - resized_t.shape[0]
        last_frame = resized_t[-1:]
        padding = np.tile(last_frame, (pad_frames, 1, 1, 1))
        resized_t = np.concatenate([resized_t, padding], axis=0)

    # Import skimage here only if needed
    from skimage.transform import resize
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


def io_worker_thread(item: WorkItem, use_moxing: bool, target_frames: int, target_height: int, target_width: int, device: str, use_gpu_resize: bool, gpu_resize_lock: Optional[threading.Lock] = None) -> WorkItem:
    """IO worker: read video + resize (on GPU if available)."""
    try:
        video_np = get_video_reader(item.video_fn, use_moxing)
        if use_gpu_resize and device is not None and 'cpu' not in str(device):
            if gpu_resize_lock is not None:
                with gpu_resize_lock:
                    item.resized_np = resize_video_gpu(video_np, target_frames, target_height, target_width, fps=16, device=device)
            else:
                item.resized_np = resize_video_gpu(video_np, target_frames, target_height, target_width, fps=16, device=device)
        else:
            item.resized_np = resize_video(video_np, target_frames, target_height, target_width, fps=16)
        item.success = True
    except Exception as e:
        item.success = False
        item.error_msg = str(e)
    return item


def encode_batch(vae, resized_np_list: List[np.ndarray], device):
    """
    Encode a batch of videos using VAE.
    Input: list of (T, H, W, 3) numpy uint8
    Output: list of (T, C, H/8, W/8) numpy float16
    """
    # Build batch tensor: (B, C, T, H, W) bf16
    batch_tensors = []
    for resized_np in resized_np_list:
        video_tensor = torch.tensor(resized_np, dtype=torch.float32, device=device).permute(3, 0, 1, 2) / 255.0
        video_tensor = video_tensor * 2 - 1
        batch_tensors.append(video_tensor)

    batch_tensor = torch.stack(batch_tensors, dim=0).to(device=device, dtype=torch.bfloat16)  # (B, C, T, H, W)

    # Ensure vae is on the correct device (safety check)
    vae = vae.to(device)

    # Use WanVAEWrapper's encode_to_latent which supports batch
    with torch.no_grad():
        encoded_latents = vae.encode_to_latent(batch_tensor)  # (B, T, C, H/8, W/8)

    # Convert to list of numpy arrays
    latent_np_list = []
    for i in range(encoded_latents.shape[0]):
        latent_np = encoded_latents[i].cpu().numpy().astype(np.float16)  # (T, C, H, W)
        latent_np_list.append(latent_np)

    return latent_np_list


def encode_video(vae, video_np, device):
    """
    Encode single video using VAE (backward compatibility).
    Input: (T, H, W, 3) numpy array (uint8)
    Output: (1, T, C, H/8, W/8) tensor
    """
    # Use same logic as encode_batch for consistency
    latent_np_list = encode_batch(vae, [video_np], device)
    return torch.tensor(latent_np_list[0]).unsqueeze(0)


def save_resized_video(video_np, save_path, fps=16, use_moxing=False):
    """Save resized video (T, H, W, 3) as mp4, supports moxing remote paths."""
    is_remote = use_moxing and (save_path.startswith('obs://') or save_path.startswith('s3://'))

    if is_remote and mox is not None:
        # Write to BytesIO first, then write to remote with mox.file.File
        bio = BytesIO()
        writer = imageio.get_writer(bio, format='ffmpeg', fps=fps, codec='libx264', output_params=['-pix_fmt', 'yuv420p'])
        for frame in video_np:
            writer.append_data(frame)
        writer.close()
        bio.seek(0)
        with _mox_lock:
            with mox.file.File(save_path, 'wb') as f:
                f.write(bio.read())
    else:
        writer = imageio.get_writer(save_path, fps=fps, codec='libx264', output_params=['-pix_fmt', 'yuv420p'])
        for frame in video_np:
            writer.append_data(frame)
        writer.close()


def check_file_exists(file_path: str, use_moxing: bool) -> bool:
    """Check if a file exists, supporting local and moxing paths."""
    if use_moxing and mox is not None and (file_path.startswith('obs://') or file_path.startswith('s3://')):
        return mox.file.exists(file_path)
    else:
        return Path(file_path).exists()


def save_latent_and_prompt(latent_np, prompt, output_path, use_moxing):
    """Save latent and prompt to .npz file, supports moxing remote paths."""
    is_remote = use_moxing and (output_path.startswith('obs://') or output_path.startswith('s3://'))

    if is_remote and mox is not None:
        bio = BytesIO()
        np.savez(bio, latent=latent_np, prompt=prompt)
        bio.seek(0)
        with _mox_lock:
            with mox.file.File(output_path, 'wb') as f:
                f.write(bio.read())
    else:
        np.savez(output_path, latent=latent_np, prompt=prompt)


def write_worker_thread(item: WorkItem, output_dir: str, save_video_dir: Optional[str], use_moxing: bool):
    """Write worker: save latents and videos to disk/moxing."""
    try:
        # Use hash-based filename
        latent_path = os.path.join(output_dir, f"latent_{item.file_hash}.npz")

        # Save latent (no existence check - already pre-checked on rank0)
        save_latent_and_prompt(item.latent_np, item.prompt, latent_path, use_moxing)

        # Save video if requested
        if save_video_dir is not None and item.resized_np is not None:
            video_basename = os.path.basename(item.video_fn)
            video_name, _ = os.path.splitext(video_basename)
            video_save_path = os.path.join(save_video_dir, f"{video_name}_{item.file_hash}.mp4")
            save_resized_video(item.resized_np, video_save_path, 16, use_moxing)

        item.success = True
    except Exception as e:
        item.success = False
        item.error_msg = str(e)
    return item


def main():
    parser = argparse.ArgumentParser(description="Process custom video data and save latents to directory")
    parser.add_argument("--jsonl_paths", type=str, nargs='+', required=True, help="Path(s) to JSONL file(s)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed latents")
    parser.add_argument("--save_video_dir", type=str, default=None, help="Directory to save resized videos (optional)")
    parser.add_argument("--use_moxing", action="store_true", help="Enable moxing for remote file access")
    parser.add_argument("--target_frames", type=int, default=81, help="Target number of frames (default: 81)")
    parser.add_argument("--target_height", type=int, default=480, help="Target height (default: 480)")
    parser.add_argument("--target_width", type=int, default=832, help="Target width (default: 832)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for VAE (default: cuda)")
    parser.add_argument("--deduplicate_prompts", action="store_true", help="Deduplicate prompts")

    # New optimization flags
    parser.add_argument("--num_io_workers", type=int, default=8, help="Number of IO/resize worker threads (default: 8)")
    parser.add_argument("--num_write_workers", type=int, default=4, help="Number of write worker threads (default: 4)")
    parser.add_argument("--prefetch", type=int, default=16, help="Max prefetch items in queue (default: 16)")
    parser.add_argument("--vae_batch_size", type=int, default=1, help="VAE batch size (default: 1, no batching)")
    parser.add_argument("--cpu_resize_only", action="store_true", help="Force use CPU resize (default: GPU resize if available)")

    args = parser.parse_args()

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
        print(f"Processing data from: {args.jsonl_paths}")
        print(f"Saving to: {args.output_dir}")
        if args.save_video_dir:
            print(f"Videos will be saved to: {args.save_video_dir}")
        print(f"Target size: {args.target_frames} frames, {args.target_height}x{args.target_width}")
        print(f"Optimizations: num_io_workers={args.num_io_workers}, num_write_workers={args.num_write_workers}, "
              f"prefetch={args.prefetch}, vae_batch_size={args.vae_batch_size}, "
              f"gpu_resize={not args.cpu_resize_only}")

    # Read all data on main rank and pre-check existing files
    all_data = []
    filtered_data = []
    if is_main:
        all_data = read_multiple_jsonl(args.jsonl_paths)
        print(f"Total videos across all JSONLs: {len(all_data)}")

        if args.deduplicate_prompts:
            seen_prompts = set()
            deduplicated = []
            for video_fn, prompt in all_data:
                if prompt not in seen_prompts:
                    seen_prompts.add(prompt)
                    deduplicated.append((video_fn, prompt))
            all_data = deduplicated
            print(f"Deduplicated to {len(all_data)} videos")

        # Pre-check existing files (only on rank0 to avoid sqlite3 lock issues)
        skipped_precheck = 0
        filtered_data = []
        for video_fn, prompt in all_data:
            file_hash = get_video_hash(video_fn)
            latent_path = os.path.join(args.output_dir, f"latent_{file_hash}.npz")
            if check_file_exists(latent_path, args.use_moxing):
                skipped_precheck += 1
            else:
                filtered_data.append((video_fn, prompt))
        print(f"Pre-skipped {skipped_precheck} already existing videos, processing {len(filtered_data)} videos")

        # Create output directories upfront on rank0 to avoid concurrent creation issues
        try:
            if args.use_moxing and mox is not None:
                mox.file.make_dirs(args.output_dir)
                if args.save_video_dir:
                    mox.file.make_dirs(args.save_video_dir)
            else:
                os.makedirs(args.output_dir, exist_ok=True)
                if args.save_video_dir:
                    os.makedirs(args.save_video_dir, exist_ok=True)
        except Exception:
            pass
    else:
        filtered_data = []

    # Broadcast filtered data to all ranks
    if dist.is_initialized():
        import pickle
        data_obj = pickle.dumps(filtered_data) if is_main else None
        data_obj = [data_obj]
        dist.broadcast_object_list(data_obj, src=0)
        if not is_main:
            filtered_data = pickle.loads(data_obj[0])

    # Determine this rank's share of work
    num_total = len(filtered_data)
    rank_start = rank * (num_total // world_size) + min(rank, num_total % world_size)
    rank_end = rank_start + (num_total // world_size) + (1 if rank < num_total % world_size else 0)
    local_data = filtered_data[rank_start:rank_end]

    print(f"Rank {rank}: processing {len(local_data)} videos (hash-based filenames)")

    if len(local_data) == 0:
        print(f"Rank {rank}: No videos to process.")
        if dist.is_initialized():
            dist.barrier()
        return

    # Load VAE
    if torch.cuda.is_available():
        if dist.is_initialized():
            # Use the device already set by launch_distributed_job (local_rank)
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16).eval()

    # Disable GPU resize by default to avoid OOM from multiple IO workers
    # User can still enable with --cpu_resize_only=False if needed
    use_gpu_resize = not args.cpu_resize_only and 'cpu' not in str(device)
    gpu_resize_lock = threading.Lock() if use_gpu_resize else None
    if use_gpu_resize and args.num_io_workers > 1:
        if is_main:
            print(f"Warning: GPU resize with {args.num_io_workers} IO workers may cause OOM. "
                  f"Consider using --cpu_resize_only or reducing --num_io_workers.")

    # Create work items for this rank (with hash-based filenames)
    work_items = []
    for idx, (video_fn, prompt) in enumerate(local_data):
        item = WorkItem(
            idx=idx,
            video_fn=video_fn,
            prompt=prompt,
            file_hash=get_video_hash(video_fn)
        )
        work_items.append(item)

    if len(work_items) == 0:
        print(f"Rank {rank}: No videos to process.")
        if dist.is_initialized():
            dist.barrier()
        return

    # ========== Stream processing with queues ==========
    io_queue = Queue(maxsize=args.prefetch)  # Holds WorkItem
    resized_queue = Queue(maxsize=args.prefetch)  # Holds WorkItem with resized_np
    write_queue = Queue(maxsize=args.prefetch)  # Holds WorkItem with latent_np

    # Counters and flags
    total = len(work_items)
    write_ok = 0
    skipped = 0
    failed = 0

    pbar = tqdm(total=total, desc=f"Rank {rank} processing", disable=not is_main)

    # ---------- IO Workers (read + resize) ----------
    io_tasks_submitted = 0
    io_tasks_completed = 0
    io_lock = threading.Lock()
    all_io_submitted = threading.Event()
    all_work_done = threading.Event()

    def io_worker():
        nonlocal io_tasks_completed
        while True:
            try:
                item = io_queue.get(timeout=0.1)
            except:
                if all_io_submitted.is_set() and io_queue.empty():
                    break
                continue

            try:
                result_item = io_worker_thread(item, args.use_moxing, args.target_frames,
                                               args.target_height, args.target_width, device, use_gpu_resize, gpu_resize_lock)
                resized_queue.put(result_item)
            except Exception as e:
                item.success = False
                item.error_msg = str(e)
                resized_queue.put(item)

            with io_lock:
                io_tasks_completed += 1

            # Check if all done
            with io_lock:
                if all_io_submitted.is_set() and io_tasks_completed >= io_tasks_submitted:
                    all_work_done.set()

    # Start IO workers
    io_threads = []
    for _ in range(args.num_io_workers):
        t = threading.Thread(target=io_worker, daemon=True)
        t.start()
        io_threads.append(t)

    # ---------- VAE Worker (batch encode) ----------
    vae_completed = threading.Event()
    items_encoded = 0
    start_time = time.time()
    start_step = 0
    last_printed_step = 0

    def vae_worker():
        nonlocal failed, items_encoded, start_time, start_step, last_printed_step
        vae_batch_size = max(1, args.vae_batch_size)
        batch_buffer = []
        items_received = 0

        while True:
            # Try to fill batch buffer
            try:
                item = resized_queue.get(timeout=0.1)
            except:
                if all_work_done.is_set() and items_received >= total:
                    break
                if len(batch_buffer) > 0 and all_work_done.is_set():
                    # Flush remaining items
                    pass
                else:
                    continue

            items_received += 1

            if item.success:
                batch_buffer.append(item)
            else:
                # Failed, send directly to write queue for error handling
                write_queue.put(item)
                pbar.update(1)
                failed += 1

            # Process batch if buffer full or all items received
            if len(batch_buffer) >= vae_batch_size or (all_work_done.is_set() and len(batch_buffer) > 0):
                try:
                    with torch.no_grad():
                        if vae_batch_size == 1:
                            # Single item
                            single_item = batch_buffer[0]
                            latent = encode_video(vae, single_item.resized_np, device)
                            single_item.latent_np = latent.numpy().astype(np.float16).squeeze(0)
                            write_queue.put(single_item)
                        else:
                            resized_list = [i.resized_np for i in batch_buffer]
                            latent_np_list = encode_batch(vae, resized_list, device)
                            for i, latent_np in zip(batch_buffer, latent_np_list):
                                i.latent_np = latent_np
                                write_queue.put(i)

                    pbar.update(len(batch_buffer))
                    items_encoded += len(batch_buffer)

                    # Print throughput periodically (only from main rank, with lock protection)
                    current_step = items_encoded
                    end_time = time.time()
                    end_step = current_step
                    step_diff = end_step - start_step
                    time_diff = end_time - start_time
                    throughput = step_diff / time_diff if time_diff > 0 else 0
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"{timestamp}: [processed {current_step}/{total}] "
                        f"DI_throughput: {throughput:.2f} samples/s/npu"
                    )
                    start_time = end_time
                    start_step = end_step
                    last_printed_step = current_step
                except Exception as e:
                    import traceback
                    print(f"Rank {rank}: VAE encode failed for batch: {e}")
                    print(f"Rank {rank}: Stack trace: {traceback.format_exc()}")
                    for i in batch_buffer:
                        i.success = False
                        i.error_msg = f"VAE encode failed: {e}"
                        write_queue.put(i)
                        failed += 1
                    pbar.update(len(batch_buffer))

                batch_buffer = []

            if all_work_done.is_set() and items_received >= total:
                break

        vae_completed.set()

    # Start VAE worker
    vae_thread = threading.Thread(target=vae_worker, daemon=True)
    vae_thread.start()

    # ---------- Feeder Thread (feeds IO queue without blocking) ----------
    def feeder():
        nonlocal io_tasks_submitted
        for item in work_items:
            io_queue.put(item)  # This blocks if queue is full (backpressure)
            with io_lock:
                io_tasks_submitted += 1
        all_io_submitted.set()

    feeder_thread = threading.Thread(target=feeder, daemon=True)
    feeder_thread.start()

    # ---------- Write Workers (save to disk) ----------
    items_to_write = total

    def write_worker():
        nonlocal write_ok, skipped, failed
        written = 0
        while True:
            try:
                item = write_queue.get(timeout=0.1)
            except:
                if vae_completed.is_set() and written >= items_to_write:
                    break
                continue

            written += 1

            try:
                result_item = write_worker_thread(item, args.output_dir, args.save_video_dir, args.use_moxing)
                if result_item.success:
                    write_ok += 1
                else:
                    failed += 1
                    print(f"Rank {rank}: Write failed for {result_item.video_fn}: {result_item.error_msg}")
            except Exception as e:
                failed += 1
                print(f"Rank {rank}: Write worker exception for {item.video_fn}: {e}")

    # Start write workers
    write_threads = []
    for _ in range(args.num_write_workers):
        t = threading.Thread(target=write_worker, daemon=True)
        t.start()
        write_threads.append(t)

    # Wait for all threads to finish
    feeder_thread.join()
    for t in io_threads:
        t.join()
    vae_thread.join()
    for t in write_threads:
        t.join()

    pbar.close()

    # Print final throughput on main rank
    if is_main:
        total_time = time.time() - start_time
        if items_encoded > 0 and total_time > 0:
            overall_throughput = items_encoded / total_time
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n{timestamp}: Done! Total processed {items_encoded} videos in {total_time:.1f}s, "
                  f"overall throughput: {overall_throughput:.2f} videos/s")

    print(f"Rank {rank}: Done! Processed {write_ok}/{len(local_data)} videos (pre-skipped {skipped_precheck}, runtime-skipped {skipped}, failed {failed}).")

    # Wait for all ranks to finish
    if dist.is_initialized():
        dist.barrier()

    if is_main:
        print(f"Done! Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
