"""
Filter JSONL data items by video resolution, aspect ratio, frame count, and fps.

Features:
1. Read JSONL lines, check width/height (16:9, height > 720p)
2. Check video file existence under a configurable root directory (supports moxing)
3. Check video frame count and fps fall within acceptable ranges
4. Write passing lines to a new JSONL file
5. Multi-threaded video reading for faster filtering
6. Optional sharding: flush to a new file every N passed items

Usage:
    # Local files
    python scripts/filter_data_items.py \
        --input_jsonl data.jsonl \
        --output_jsonl filtered.jsonl \
        --video_root /path/to/videos

    # With moxing for remote videos
    python scripts/filter_data_items.py \
        --input_jsonl data.jsonl \
        --output_jsonl filtered.jsonl \
        --video_root obs://bucket/videos \
        --use_moxing

    # Custom thresholds + multi-threaded
    python scripts/filter_data_items.py \
        --input_jsonl data.jsonl \
        --output_jsonl filtered.jsonl \
        --video_root /path/to/videos \
        --min_frames 81 \
        --fps_min 24 \
        --fps_max 28 \
        --num_workers 8

    # Shard output: flush every 1000 passed items into separate files
    python scripts/filter_data_items.py \
        --input_jsonl data.jsonl \
        --output_jsonl filtered.jsonl \
        --video_root /path/to/videos \
        --max_per_file 1000
"""
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

import argparse
import json
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from tqdm import tqdm

# Optional moxing import
mox = None
def try_import_moxing():
    global mox
    try:
        import moxing as mox
        return True
    except ImportError:
        return False


def video_exists(video_path, use_moxing):
    """Check if a video file exists, supports moxing remote paths."""
    if use_moxing and mox is not None and (video_path.startswith('obs://') or video_path.startswith('s3://')):
        return mox.file.exists(video_path)
    else:
        return os.path.isfile(video_path)


def read_video_info(video_path, use_moxing):
    """
    Read video metadata (num_frames, fps) without decoding pixel data.
    Uses PyAV directly for reliable metadata access.
    Returns (num_frames, fps) or None if reading fails.
    """
    import av
    try:
        if use_moxing and mox is not None and (video_path.startswith('obs://') or video_path.startswith('s3://')):
            with mox.file.File(video_path, 'rb') as f:
                video_bytes = f.read()
            container = av.open(BytesIO(video_bytes))
        else:
            container = av.open(video_path)

        with container:
            stream = container.streams.video[0]
            num_frames = stream.frames
            fps = float(stream.average_rate)
            return num_frames, fps
    except Exception:
        return None


def is_16by9(width, height, tolerance=0.02):
    """Check if the aspect ratio is approximately 16:9 within tolerance."""
    if height == 0:
        return False
    ratio = width / height
    target = 16.0 / 9.0
    return abs(ratio - target) / target <= tolerance


# Reason constants for filter results
REASON_PASS = 'pass'
REASON_NO_WIDTH_HEIGHT = 'no_width_height'
REASON_HEIGHT_TOO_SMALL = 'height_too_small'
REASON_NOT_16BY9 = 'not_16by9'
REASON_FILE_NOT_FOUND = 'file_not_found'
REASON_TOO_FEW_FRAMES = 'too_few_frames'
REASON_FPS_OUT_OF_RANGE = 'fps_out_of_range'
REASON_DURATION_TOO_LONG = 'duration_too_long'
REASON_READ_ERROR = 'read_error'


def filter_one(line, args):
    """
    Filter a single JSONL line. Returns (line, reason).
    reason is REASON_PASS if the item passes, otherwise a failure reason string.
    """
    item = json.loads(line)
    width = item.get('width')
    height = item.get('height')
    video_fn = item.get('video_fn', '')

    if width is None or height is None:
        return line, REASON_NO_WIDTH_HEIGHT

    if height < args.min_height:
        return line, REASON_HEIGHT_TOO_SMALL

    if not is_16by9(width, height, args.aspect_tolerance):
        return line, REASON_NOT_16BY9

    if args.video_root:
        full_video_path = os.path.join(args.video_root, video_fn)
    else:
        full_video_path = video_fn

    if not video_exists(full_video_path, args.use_moxing):
        return line, REASON_FILE_NOT_FOUND

    info = read_video_info(full_video_path, args.use_moxing)
    if info is None:
        return line, REASON_READ_ERROR

    num_frames, fps = info

    if num_frames < args.min_frames:
        return line, REASON_TOO_FEW_FRAMES

    if fps < args.fps_min or fps > args.fps_max:
        return line, REASON_FPS_OUT_OF_RANGE

    if args.max_duration > 0:
        duration = num_frames / fps
        if duration > args.max_duration:
            return line, REASON_DURATION_TOO_LONG

    return line, REASON_PASS


def write_jsonl_lines(lines, output_path, is_remote):
    """Write a list of JSONL strings to a file, supports moxing remote paths."""
    if is_remote and mox is not None:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as tmp:
            tmp_path = tmp.name
            for line in lines:
                tmp.write(line + '\n')
        try:
            try:
                mox.file.make_dirs(os.path.dirname(output_path))
            except Exception:
                pass
            mox.file.copy(tmp_path, output_path)
        finally:
            os.unlink(tmp_path)
    else:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')


def get_shard_path(base_path, shard_idx, total_shards):
    """Get output path for a shard. If only 1 shard, use base_path directly."""
    if total_shards == 1:
        return base_path
    root, ext = os.path.splitext(base_path)
    return f"{root}_shard_{shard_idx}{ext}"


def main():
    parser = argparse.ArgumentParser(description="Filter JSONL data items by video properties")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output JSONL file path for filtered items")
    parser.add_argument("--video_root", type=str, default="", help="Root directory for video files (prepended to video_fn)")
    parser.add_argument("--min_height", type=int, default=720, help="Minimum video height (default: 720)")
    parser.add_argument("--min_frames", type=int, default=81, help="Minimum number of frames (default: 81)")
    parser.add_argument("--fps_min", type=float, default=24.0, help="Minimum acceptable fps (default: 24)")
    parser.add_argument("--fps_max", type=float, default=28.0, help="Maximum acceptable fps (default: 28)")
    parser.add_argument("--aspect_tolerance", type=float, default=0.02, help="Relative tolerance for 16:9 check (default: 0.02)")
    parser.add_argument("--max_duration", type=float, default=0, help="Maximum video duration in seconds (0 = no limit, default: 0)")
    parser.add_argument("--max_per_file", type=int, default=0, help="Max passed items per output file; flush to a new shard when reached (0 = single file, default: 0)")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker threads for parallel video reading (default: 1)")
    parser.add_argument("--use_moxing", action="store_true", help="Enable moxing for remote file access")
    args = parser.parse_args()

    # Import moxing if needed
    if args.use_moxing:
        if not try_import_moxing():
            print("Warning: moxing not available, falling back to local file access only")
            args.use_moxing = False

    # Read input JSONL
    print(f"Reading input JSONL: {args.input_jsonl}")
    open_fn = open
    if args.use_moxing and mox is not None and (args.input_jsonl.startswith('obs://') or args.input_jsonl.startswith('s3://')):
        open_fn = lambda p, mode: mox.file.File(p, mode)

    lines = []
    with open_fn(args.input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    print(f"Total items in input: {len(lines)}")

    # Ensure output directory exists
    is_output_remote = args.use_moxing and (args.output_jsonl.startswith('obs://') or args.output_jsonl.startswith('s3://'))

    # Filter with thread pool
    num_workers = max(1, args.num_workers)
    max_per_file = max(0, args.max_per_file)

    # Accumulate passed items; flush to file when batch reaches max_per_file
    batch = []
    shard_idx = 0
    total_passed = 0
    shard_counts = []
    stats = Counter()

    def flush_batch():
        """Write current batch to a shard file and reset."""
        nonlocal batch, shard_idx, total_passed
        if not batch:
            return
        shard_path = get_shard_path(args.output_jsonl, shard_idx, -1 if max_per_file > 0 else 1)
        write_jsonl_lines(batch, shard_path, is_output_remote)
        print(f"  Flushed shard {shard_idx}: {len(batch)} items -> {shard_path}")
        shard_counts.append(len(batch))
        total_passed += len(batch)
        batch = []
        shard_idx += 1

    if num_workers == 1:
        # Sequential path (no threading overhead)
        pbar = tqdm(lines, desc="Filtering")
        for line in pbar:
            line_out, reason = filter_one(line, args)
            stats[reason] += 1
            if reason == REASON_PASS:
                batch.append(line_out)
                if max_per_file > 0 and len(batch) >= max_per_file:
                    flush_batch()
            pbar.set_postfix(passed=total_passed + len(batch))
    else:
        # Parallel path
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(filter_one, line, args): i for i, line in enumerate(lines)}
            results = [None] * len(lines)
            pbar = tqdm(total=len(lines), desc=f"Filtering ({num_workers} workers)")
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                pbar.update(1)
                # Approximate count for display (some results may not be counted yet)
                passed_so_far = sum(1 for r in results if r is not None and r[1] == REASON_PASS)
                pbar.set_postfix(passed=passed_so_far)
            pbar.close()

            for line_out, reason in results:
                stats[reason] += 1
                if reason == REASON_PASS:
                    batch.append(line_out)
                    if max_per_file > 0 and len(batch) >= max_per_file:
                        flush_batch()

    # Flush remaining items
    flush_batch()

    # Print summary
    total = len(lines)
    num_shards = len(shard_counts)
    print(f"\nFiltering summary:")
    print(f"  Total input items:    {total}")
    print(f"  Missing width/height: {stats[REASON_NO_WIDTH_HEIGHT]}")
    print(f"  Not 16:9:             {stats[REASON_NOT_16BY9]}")
    print(f"  Height < {args.min_height}:           {stats[REASON_HEIGHT_TOO_SMALL]}")
    print(f"  File not found:       {stats[REASON_FILE_NOT_FOUND]}")
    print(f"  Frames < {args.min_frames}:            {stats[REASON_TOO_FEW_FRAMES]}")
    print(f"  FPS out of range:     {stats[REASON_FPS_OUT_OF_RANGE]}")
    if args.max_duration > 0:
        print(f"  Duration > {args.max_duration}s:       {stats[REASON_DURATION_TOO_LONG]}")
    print(f"  Video read error:     {stats[REASON_READ_ERROR]}")
    print(f"  Passed:               {total_passed}")
    if num_shards > 1:
        print(f"  Shards:               {num_shards} ({shard_counts})")
    else:
        print(f"  Output:               {args.output_jsonl}")


if __name__ == "__main__":
    main()
