"""
Filter JSONL data items by video resolution, aspect ratio, frame count, and fps.

Features:
1. Read JSONL lines, check width/height (16:9, height > 720p)
2. Check video file existence under a configurable root directory (supports moxing)
3. Check video frame count and fps fall within acceptable ranges
4. Write passing lines to a new JSONL file

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

    # Custom thresholds
    python scripts/filter_data_items.py \
        --input_jsonl data.jsonl \
        --output_jsonl filtered.jsonl \
        --video_root /path/to/videos \
        --min_frames 81 \
        --fps_min 24 \
        --fps_max 28
"""
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

import argparse
import json
from io import BytesIO
from pathlib import Path

import imageio.v3 as iio
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
    Read video metadata (num_frames, fps) without decoding all pixel data.
    Returns (num_frames, fps) or None if reading fails.
    """
    try:
        if use_moxing and mox is not None and (video_path.startswith('obs://') or video_path.startswith('s3://')):
            with mox.file.File(video_path, 'rb') as f:
                video_bytes = f.read()
            video_buffer = BytesIO(video_bytes)
            # Read only metadata via pyav backend
            reader = iio.imopen(video_buffer, 'r', plugin='pyav')
        else:
            reader = iio.imopen(video_path, 'r', plugin='pyav')

        with reader:
            # pyav provides properties on the internal container
            container = reader._video
            stream = container.streams.video[0]
            num_frames = stream.frames
            # fps may be a Fraction
            fps = float(stream.average_rate)
            return num_frames, fps
    except Exception as e:
        print(f"  Warning: failed to read video info from {video_path}: {e}")
        return None


def is_16by9(width, height, tolerance=0.02):
    """Check if the aspect ratio is approximately 16:9 within tolerance."""
    if height == 0:
        return False
    ratio = width / height
    target = 16.0 / 9.0
    return abs(ratio - target) / target <= tolerance


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
    if not is_output_remote:
        os.makedirs(os.path.dirname(args.output_jsonl) or '.', exist_ok=True)

    # Filter
    passed = []
    stats = {
        'total': len(lines),
        'no_width_height': 0,
        'not_16by9': 0,
        'height_too_small': 0,
        'file_not_found': 0,
        'too_few_frames': 0,
        'fps_out_of_range': 0,
        'duration_too_long': 0,
        'read_error': 0,
    }

    for line in tqdm(lines, desc="Filtering"):
        item = json.loads(line)
        width = item.get('width')
        height = item.get('height')
        video_fn = item.get('video_fn', '')

        # Check width/height exist
        if width is None or height is None:
            stats['no_width_height'] += 1
            continue

        # Check height >= min_height
        if height < args.min_height:
            stats['height_too_small'] += 1
            continue

        # Check 16:9 aspect ratio
        if not is_16by9(width, height, args.aspect_tolerance):
            stats['not_16by9'] += 1
            continue

        # Resolve full video path
        if args.video_root:
            full_video_path = os.path.join(args.video_root, video_fn)
        else:
            full_video_path = video_fn

        # Check file existence
        if not video_exists(full_video_path, args.use_moxing):
            stats['file_not_found'] += 1
            continue

        # Read video info (frame count, fps)
        info = read_video_info(full_video_path, args.use_moxing)
        if info is None:
            stats['read_error'] += 1
            continue

        num_frames, fps = info

        # Check frame count
        if num_frames < args.min_frames:
            stats['too_few_frames'] += 1
            continue

        # Check fps range
        if fps < args.fps_min or fps > args.fps_max:
            stats['fps_out_of_range'] += 1
            continue

        # Check duration
        if args.max_duration > 0:
            duration = num_frames / fps
            if duration > args.max_duration:
                stats['duration_too_long'] += 1
                continue

        passed.append(line)

    # Write output JSONL
    if is_output_remote:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as tmp:
            tmp_path = tmp.name
            for line in passed:
                tmp.write(line + '\n')
        try:
            try:
                mox.file.make_dirs(os.path.dirname(args.output_jsonl))
            except Exception:
                pass
            mox.file.copy(tmp_path, args.output_jsonl)
        finally:
            os.unlink(tmp_path)
    else:
        with open(args.output_jsonl, 'w', encoding='utf-8') as f:
            for line in passed:
                f.write(line + '\n')

    # Print summary
    print(f"\nFiltering summary:")
    print(f"  Total input items:    {stats['total']}")
    print(f"  Missing width/height: {stats['no_width_height']}")
    print(f"  Not 16:9:             {stats['not_16by9']}")
    print(f"  Height < {args.min_height}:           {stats['height_too_small']}")
    print(f"  File not found:       {stats['file_not_found']}")
    print(f"  Frames < {args.min_frames}:            {stats['too_few_frames']}")
    print(f"  FPS out of range:     {stats['fps_out_of_range']}")
    if args.max_duration > 0:
        print(f"  Duration > {args.max_duration}s:       {stats['duration_too_long']}")
    print(f"  Video read error:     {stats['read_error']}")
    print(f"  Passed:               {len(passed)}")
    print(f"\nOutput written to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
