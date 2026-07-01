"""Standalone teleport-rate evaluation: load MP4s -> run detector -> aggregate.

CLI:
    python scripts/eval_teleport_rate.py --video_dir outputs/sampleA --output report.json

The script is deterministic (no random sampling) and uses the no-RAFT detector
backend so it runs on CPU.  GPU/NPU is auto-detected when available.

NPU support: set environment variable ``DEVICE_TYPE=npu`` before running.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# NPU device transfer (must happen before any torch.cuda call).
# Mirrors train_reward_forcing.py: check DEVICE_TYPE env var, import
# torch_npu.contrib.transfer_to_npu which patches torch.cuda.* -> NPU.
# ---------------------------------------------------------------------------
import os as _os

_DEVICE_TYPE = _os.environ.get("DEVICE_TYPE", "cuda")
if _DEVICE_TYPE == "npu":
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

# ---------------------------------------------------------------------------
# CPU-only guard: stub methods.reward_forcing to prevent CUDA import cascade
# when running outside pytest (conftest.py handles this inside pytest).
# ---------------------------------------------------------------------------
import sys as _sys
import types as _types
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
_RF_PATH = str(_PROJECT_ROOT / "methods" / "reward_forcing")

if "methods.reward_forcing" not in _sys.modules:
    _rf_stub = _types.ModuleType("methods.reward_forcing")
    _rf_stub.__path__ = [_RF_PATH]
    _rf_stub.__all__ = []
    _sys.modules["methods.reward_forcing"] = _rf_stub

if "methods.reward_forcing.pipelines" not in _sys.modules:
    _sys.modules["methods.reward_forcing.pipelines"] = _types.ModuleType(
        "methods.reward_forcing.pipelines"
    )

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

from methods.reward_forcing.teleport import build_teleport_detector

logger = logging.getLogger("eval_teleport_rate")


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------


def load_video(path: str) -> torch.Tensor:
    """Load an MP4 video as a float tensor in [0, 1].

    Returns:
        Tensor of shape [1, T, 3, H, W], values in [0, 1].
    """
    import imageio.v3 as iio

    frames: Any = iio.imread(path)  # np.ndarray, [T, H, W, 3] or [H, W, 3]
    if frames.ndim == 3:
        frames = frames[None, ...]  # add time dim for single-frame images
    # frames: [T, H, W, 3] uint8
    tensor = torch.from_numpy(frames.copy()).float() / 255.0
    # [T, H, W, 3] -> [1, T, 3, H, W]
    tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0)
    return tensor


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def compute_teleport_rate(
    rgb: torch.Tensor,
    detector: Any,
    threshold: float,
    min_event_area_ratio: float = 0.001,
) -> dict[str, float | int]:
    """Run detector on a video tensor and aggregate score map into a rate.

    A frame is counted as a teleport event only when the *fraction* of
    spatial positions whose score exceeds *threshold* is larger than
    *min_event_area_ratio*.  This replaces the previous per-pixel-max
    aggregation that caused every frame with any motion or noise to be
    counted as an event when RAFT was not used.

    Args:
        rgb: [1, T, 3, H, W] float tensor in [0, 1].
        detector: A TeleportDetector instance.
        threshold: Score threshold for a single spatial position.
        min_event_area_ratio: Minimum fraction of spatial positions above
            *threshold* for the frame to count as a teleport event.

    Returns:
        Dict with keys ``teleport_rate``, ``frame_count``, ``event_count``.
    """
    # no_grad is critical: without it, RAFT forward passes inside score()
    # retain computation graphs, causing OOM on long videos (240+ frames).
    with torch.no_grad():
        score_map = detector.score(rgb)  # [1, T, 1, H', W']
    T = rgb.shape[1]

    # Per-frame fraction of spatial positions exceeding the threshold
    score_2d = score_map[0, :, 0, :, :]  # [T, H', W']
    frame_area_ratio = (score_2d > threshold).float().mean(dim=(1, 2))  # [T]
    event_count = int((frame_area_ratio > min_event_area_ratio).sum().item())
    teleport_rate = event_count / T if T > 0 else 0.0

    return {
        "teleport_rate": round(teleport_rate, 6),
        "frame_count": T,
        "event_count": event_count,
    }


def evaluate_directory(
    video_dir: str,
    output_path: str,
    max_videos: int | None = None,
    threshold: float = 0.3,
    min_event_area_ratio: float = 0.001,
    device: str = "auto",
    use_raft: bool = False,
    flow_mode: str = "chained",
    tau: int = 5,
    flow_model_path: str | None = None,
) -> dict[str, Any]:
    """Evaluate all MP4 files in a directory and write a JSON report.

    Args:
        video_dir: Path to directory containing .mp4 files.
        output_path: Where to write the JSON report.
        max_videos: Limit to the first N videos (sorted alphabetically).
        threshold: Score threshold for a single spatial position.
        min_event_area_ratio: Minimum fraction of spatial positions above
            *threshold* for a frame to count as a teleport event.
        device: ``"auto"``, ``"cpu"``, or ``"cuda"``.
        use_raft: If True, use RAFT optical flow backend (requires GPU).
        flow_mode: ``"adjacent"``, ``"direct"``, or ``"chained"``
            (default: ``"chained"``). Only used when ``use_raft`` is True.
        tau: Multi-frame lookback distance (default: 5). Only used when
            ``use_raft`` is True and ``flow_mode`` is not ``"adjacent"``.
        flow_model_path: Optional path to offline RAFT weights.

    Returns:
        The report dict (also written to ``output_path``).

    Raises:
        ValueError: If ``video_dir`` is not a directory or contains no MP4 files.
    """
    dir_path = Path(video_dir)
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {video_dir}")

    # Collect MP4 files, sorted alphabetically for determinism
    mp4_files = sorted(
        p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"
    )
    if not mp4_files:
        raise ValueError(f"No MP4 files found in {video_dir}")

    if max_videos is not None:
        mp4_files = mp4_files[:max_videos]

    # Resolve device
    if device == "auto":
        if _DEVICE_TYPE == "npu" and torch.cuda.is_available():
            device = "cuda"  # transfer_to_npu maps cuda -> npu
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    elif device == "npu":
        if _DEVICE_TYPE != "npu":
            raise ValueError("--device npu requires environment variable DEVICE_TYPE=npu")
        device = "cuda"  # transfer_to_npu maps cuda -> npu

    # Build detector
    if use_raft:
        if flow_mode == "adjacent":
            detector = build_teleport_detector({
                "type": "optical_flow",
                "downsample_factor": 4,
                "flow_model_path": flow_model_path,
            })
        else:
            det_type = f"multi_frame_{flow_mode}"
            detector = build_teleport_detector({
                "type": det_type,
                "tau": tau,
                "downsample_factor": 4,
                "flow_backend": "raft_small",
                "flow_model_path": flow_model_path,
            })
    else:
        detector = build_teleport_detector({
            "type": "optical_flow_no_raft",
            "downsample_factor": 4,
        })

    per_video: dict[str, dict] = {}
    for mp4_path in mp4_files:
        logger.info("Processing %s ...", mp4_path.name)
        try:
            rgb = load_video(str(mp4_path))
        except Exception:
            logger.warning("Failed to load %s, skipping", mp4_path.name, exc_info=True)
            continue
        rgb = rgb.to(device)
        result = compute_teleport_rate(rgb, detector, threshold, min_event_area_ratio)
        per_video[mp4_path.name] = result

    if not per_video:
        raise RuntimeError("No videos could be loaded successfully")

    # Aggregate
    total_frames = sum(v["frame_count"] for v in per_video.values())
    total_events = sum(v["event_count"] for v in per_video.values())
    total_videos = len(per_video)
    rates = [v["teleport_rate"] for v in per_video.values()]
    mean_rate = sum(rates) / total_videos if total_videos > 0 else 0.0

    report: dict[str, Any] = {
        "per_video": per_video,
        "aggregate": {
            "mean_teleport_rate": round(mean_rate, 6),
            "total_videos": total_videos,
            "total_frames": total_frames,
            "flow_mode": flow_mode,
            "tau": tau,
        },
    }

    # Write JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate teleport rate for MP4 videos in a directory.",
    )
    parser.add_argument(
        "--video_dir",
        required=True,
        help="Directory containing .mp4 video files to evaluate.",
    )
    parser.add_argument(
        "--output",
        default="eval_report.json",
        help="Path to write the JSON report (default: eval_report.json).",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Limit evaluation to the first N videos (sorted alphabetically).",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.3,
        help="Score threshold for counting a frame as a teleport event (default: 0.3).",
    )
    parser.add_argument(
        "--min_event_area_ratio",
        type=float,
        default=0.001,
        help=(
            "Minimum fraction of spatial positions above score_threshold for a "
            "frame to count as a teleport event (default: 0.001). "
            "Increase to reduce false positives from noise / normal motion."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "npu"],
        help="Device to run detector on (default: auto). For NPU, set DEVICE_TYPE=npu env var.",
    )
    parser.add_argument(
        "--use_raft",
        action="store_true",
        default=False,
        help="Use RAFT optical flow backend (requires GPU and torchvision RAFT weights).",
    )
    parser.add_argument(
        "--flow_mode",
        default="chained",
        choices=["adjacent", "direct", "chained"],
        help=(
            "Flow mode when --use_raft is set: 'adjacent' (original adjacent-frame), "
            "'direct' (multi-frame single RAFT call), 'chained' (multi-frame composed flows). "
            "Default: chained."
        ),
    )
    parser.add_argument(
        "--tau",
        type=int,
        default=5,
        help="Multi-frame lookback distance (default: 5). Only used when --use_raft and flow_mode != adjacent.",
    )
    parser.add_argument(
        "--flow_model_path",
        type=str,
        default=None,
        help="Optional path to offline RAFT model weights (.pth).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    report = evaluate_directory(
        video_dir=args.video_dir,
        output_path=args.output,
        max_videos=args.max_videos,
        threshold=args.score_threshold,
        min_event_area_ratio=args.min_event_area_ratio,
        device=args.device,
        use_raft=args.use_raft,
        flow_mode=args.flow_mode,
        tau=args.tau,
        flow_model_path=args.flow_model_path,
    )

    agg = report["aggregate"]
    print(
        f"Evaluated {agg['total_videos']} video(s), "
        f"{agg['total_frames']} total frames. "
        f"Mean teleport rate: {agg['mean_teleport_rate']:.6f}"
    )
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
