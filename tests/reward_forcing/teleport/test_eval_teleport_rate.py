"""Tests for scripts/eval_teleport_rate.py — CLI, compute_teleport_rate, evaluate_directory.

All tests mock ``load_video`` to return synthetic tensors; no real MP4 decoding.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from methods.reward_forcing.teleport.detector import OpticalFlowTeleportDetector

# ---------------------------------------------------------------------------
# Load the eval script module via importlib (scripts/ is not a package)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SCRIPT_PATH = _PROJECT_ROOT / "scripts" / "eval_teleport_rate.py"
_spec = importlib.util.spec_from_file_location("eval_teleport_rate", str(_SCRIPT_PATH))
assert _spec is not None
_eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)  # type: ignore[arg-type]

compute_teleport_rate = _eval_mod.compute_teleport_rate
evaluate_directory = _eval_mod.evaluate_directory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_static_clip(T: int = 8, H: int = 64, W: int = 64) -> torch.Tensor:
    """All-zero frames (static black).  Shape [1, T, 3, H, W]."""
    return torch.zeros(1, T, 3, H, W)


def _make_emergence_clip(
    *,
    position: str = "center",
    T: int = 8,
    H: int = 64,
    W: int = 64,
    onset_frame: int = 5,
) -> torch.Tensor:
    """Synthetic clip: black until onset_frame, then a bright square appears.

    Args:
        position: ``"center"``, ``"edge_left"``, or ``"edge_top"``.
        T: Total frames.
        H, W: Frame height and width.
        onset_frame: First frame where the bright patch appears (0-indexed).

    Returns:
        Tensor [1, T, 3, H, W] in [0, 1].
    """
    clip = torch.zeros(1, T, 3, H, W)
    if position == "center":
        cy, cx = H // 2, W // 2
    elif position == "edge_left":
        cy, cx = H // 2, 1  # touches the very left edge → centeredness ~0
    elif position == "edge_top":
        cy, cx = 1, W // 2  # touches the very top edge → centeredness ~0
    else:
        raise ValueError(f"Unknown position: {position}")
    half = 6
    clip[:, onset_frame:, :, cy - half : cy + half, cx - half : cx + half] = 1.0
    return clip


def _make_smooth_motion_clip(
    T: int = 16, H: int = 64, W: int = 64
) -> torch.Tensor:
    """Synthetic clip: a bright Gaussian blob moves 1 px/frame right.

    The Gaussian blur ensures frame-to-frame differences are moderate,
    unlike hard-edged squares that produce artificially high frame diffs.
    This simulates normal smooth motion that should NOT produce a high
    teleport rate.

    Returns:
        Tensor [1, T, 3, H, W] in [0, 1].
    """
    clip = torch.zeros(1, T, 3, H, W)
    cy, cx = H // 2, W // 4  # start left of center
    radius = 6

    for t in range(T):
        # Gaussian-weighted circle
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij",
        )
        dist = ((yy - cy) ** 2 + (xx - (cx + t)) ** 2).sqrt()  # move right
        gauss = torch.exp(-(dist**2) / (2.0 * radius**2))
        # Clamp to [0, 1] and set all 3 channels
        gauss = gauss.clamp(0.0, 1.0)
        clip[0, t, 0] = gauss
        clip[0, t, 1] = gauss
        clip[0, t, 2] = gauss

    return clip


def _make_noisy_clip(
    T: int = 16, H: int = 64, W: int = 64, noise_std: float = 0.01
) -> torch.Tensor:
    """Synthetic clip: constant gray with small Gaussian noise per frame.

    Simulates camera sensor noise or mild compression artifacts.  With the
    old max-pixel aggregation every frame would be an event; with the
    area-aware aggregation this should produce zero events.

    Returns:
        Tensor [1, T, 3, H, W] in [0, 1].
    """
    clip = torch.full((1, T, 3, H, W), 0.5)
    clip += torch.randn(1, T, 3, H, W) * noise_std
    return clip.clamp(0.0, 1.0)


def _make_detector() -> OpticalFlowTeleportDetector:
    """Create a CPU-safe no-RAFT detector."""
    return OpticalFlowTeleportDetector(flow_backend="none", downsample_factor=4)


def _fake_load_video(tensor: torch.Tensor):
    """Return a callable that ignores its path arg and returns *tensor*."""
    def _loader(_path: str) -> torch.Tensor:
        return tensor.clone()
    return _loader


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_help_lists_required_args(self):
        """``--help`` prints usage and lists ``--video_dir``, ``--output``, ``--max_videos``."""
        env = {**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)}
        result = subprocess.run(
            [sys.executable, str(_SCRIPT_PATH), "--help"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(_PROJECT_ROOT),
            env=env,
        )
        assert result.returncode == 0
        stdout = result.stdout
        assert "--video_dir" in stdout
        assert "--output" in stdout
        assert "--max_videos" in stdout
        assert "--score_threshold" in stdout
        assert "--min_event_area_ratio" in stdout
        assert "--device" in stdout


# ---------------------------------------------------------------------------
# compute_teleport_rate unit tests
# ---------------------------------------------------------------------------


class TestComputeTeleportRate:
    def test_static_video_zero_rate(self):
        """Static (all-zero) video has teleport_rate == 0.0."""
        det = _make_detector()
        rgb = _make_static_clip(T=8)
        result = compute_teleport_rate(rgb, det, threshold=0.3)
        assert result["teleport_rate"] == 0.0
        assert result["frame_count"] == 8
        assert result["event_count"] == 0

    def test_center_emergence_detected(self):
        """Bright square at center on frame 5 produces at least one event."""
        det = _make_detector()
        rgb = _make_emergence_clip(position="center", T=8, onset_frame=5)
        result = compute_teleport_rate(rgb, det, threshold=0.3)
        assert result["event_count"] >= 1
        assert result["teleport_rate"] > 0.0
        assert result["frame_count"] == 8

    def test_edge_emergence_lower_than_center(self):
        """Edge emergence has strictly lower teleport_rate than center emergence."""
        det = _make_detector()
        center = _make_emergence_clip(position="center", T=8, onset_frame=5)
        edge = _make_emergence_clip(position="edge_left", T=8, onset_frame=5)
        rate_center = compute_teleport_rate(center, det, threshold=0.3)["teleport_rate"]
        rate_edge = compute_teleport_rate(edge, det, threshold=0.3)["teleport_rate"]
        assert rate_edge < rate_center

    def test_determinism(self):
        """Same input twice returns byte-identical dict (floats via round)."""
        det = _make_detector()
        rgb = _make_emergence_clip(position="center", T=8, onset_frame=5)
        torch.manual_seed(42)
        r1 = compute_teleport_rate(rgb, det, threshold=0.3)
        torch.manual_seed(42)
        r2 = compute_teleport_rate(rgb, det, threshold=0.3)
        assert r1 == r2

    def test_single_frame_video_zero_rate(self):
        """T=1 video: first frame is always zero, teleport_rate == 0.0."""
        det = _make_detector()
        rgb = torch.ones(1, 1, 3, 64, 64)  # bright, but T=1 → first frame = 0
        result = compute_teleport_rate(rgb, det, threshold=0.3)
        assert result["teleport_rate"] == 0.0
        assert result["event_count"] == 0

    def test_threshold_too_high_no_events(self):
        """threshold=1.0 should yield zero events even for emergence."""
        det = _make_detector()
        rgb = _make_emergence_clip(position="center", T=8, onset_frame=5)
        result = compute_teleport_rate(rgb, det, threshold=1.0)
        assert result["event_count"] == 0
        assert result["teleport_rate"] == 0.0

    # ------------------------------------------------------------------
    # Area-aware aggregation — the bugfix
    # ------------------------------------------------------------------

    def test_smooth_motion_produces_low_rate(self):
        """Smooth Gaussian blob motion should NOT produce a high teleport rate.

        With the old max-pixel aggregation this would be near 1.0 because
        any moving edge pixel crosses the threshold.  Area-aware aggregation
        (default min_event_area_ratio=0.001) should filter these out.
        """
        det = _make_detector()
        rgb = _make_smooth_motion_clip(T=16)
        result = compute_teleport_rate(rgb, det, threshold=0.3)
        # With smooth Gaussian motion, frame_diff at edges is moderate,
        # so the area above threshold should be very small.
        assert result["teleport_rate"] <= 0.3, (
            f"Smooth motion should have low teleport_rate, got {result['teleport_rate']}"
        )

    def test_noisy_clip_zero_events(self):
        """Noisy gray clip (sensor noise) should yield zero events.

        Small per-frame noise should not cause teleport detection when
        using area-aware aggregation.
        """
        det = _make_detector()
        torch.manual_seed(42)
        rgb = _make_noisy_clip(T=16, noise_std=0.02)
        result = compute_teleport_rate(rgb, det, threshold=0.3)
        assert result["event_count"] == 0, (
            f"Noise alone should not trigger events, got {result['event_count']}"
        )
        assert result["teleport_rate"] == 0.0

    def test_min_event_area_ratio_controls_sensitivity(self):
        """Higher min_event_area_ratio reduces event count."""
        det = _make_detector()
        rgb = _make_emergence_clip(position="center", T=8, onset_frame=5)

        # With default ratio → should detect the emergence
        r_default = compute_teleport_rate(rgb, det, threshold=0.3)
        assert r_default["event_count"] >= 1

        # With very high ratio → should detect nothing
        r_strict = compute_teleport_rate(
            rgb, det, threshold=0.3, min_event_area_ratio=1.0
        )
        assert r_strict["event_count"] == 0

    def test_min_event_area_ratio_zero_matches_old_behavior(self):
        """min_event_area_ratio=0.0 is equivalent to max-pixel aggregation."""
        det = _make_detector()
        # Use a hard-edged moving square that produces frame_max > 0.3
        # at the leading/trailing edge (like real video motion)
        T, H, W = 16, 64, 64
        rgb = torch.zeros(1, T, 3, H, W)
        for t in range(T):
            x_start = t  # moves 1 px/frame right
            rgb[0, t, :, 28:36, x_start : x_start + 8] = 1.0

        # With ratio=0.0 → every frame where any pixel exceeds threshold is an event
        r_old = compute_teleport_rate(
            rgb, det, threshold=0.3, min_event_area_ratio=0.0
        )
        # Hard-edge moving square: some frames have frame_max > 0.3
        assert r_old["teleport_rate"] >= 0.3, (
            f"ratio=0.0 should match old max-pixel behavior, got {r_old}"
        )

        # With ratio=1.0 → nothing can possibly trigger
        r_strict = compute_teleport_rate(
            rgb, det, threshold=0.3, min_event_area_ratio=1.0
        )
        assert r_strict["event_count"] == 0, (
            f"ratio=1.0 should block all events, got {r_strict}"
        )


# ---------------------------------------------------------------------------
# evaluate_directory integration tests
# ---------------------------------------------------------------------------


class TestEvaluateDirectory:
    def test_empty_dir_raises_value_error(self, tmp_path: Path):
        """Directory with no .mp4 files raises ValueError."""
        with pytest.raises(ValueError, match="No MP4 files"):
            evaluate_directory(
                str(tmp_path),
                str(tmp_path / "report.json"),
                device="cpu",
            )

    def test_not_a_directory_raises_value_error(self, tmp_path: Path):
        """Passing a file path instead of a directory raises ValueError."""
        fake_file = tmp_path / "not_a_dir"
        fake_file.write_text("nope")
        with pytest.raises(ValueError, match="Not a directory"):
            evaluate_directory(
                str(fake_file),
                str(tmp_path / "report.json"),
                device="cpu",
            )

    def test_max_videos_limits_count(self, tmp_path: Path, monkeypatch):
        """max_videos=2 processes only 2 of 5 .mp4 files."""
        for i in range(5):
            (tmp_path / f"vid_{i:02d}.mp4").write_text("")

        static = _make_static_clip(T=4)
        monkeypatch.setattr(_eval_mod, "load_video", _fake_load_video(static))

        report = evaluate_directory(
            str(tmp_path),
            str(tmp_path / "report.json"),
            max_videos=2,
            device="cpu",
        )
        assert report["aggregate"]["total_videos"] == 2

    def test_skip_non_mp4(self, tmp_path: Path, monkeypatch):
        """Only .mp4 files are processed; .txt and .png are ignored."""
        (tmp_path / "video.mp4").write_text("")
        (tmp_path / "notes.txt").write_text("")
        (tmp_path / "thumb.png").write_text("")

        static = _make_static_clip(T=4)
        monkeypatch.setattr(_eval_mod, "load_video", _fake_load_video(static))

        report = evaluate_directory(
            str(tmp_path),
            str(tmp_path / "report.json"),
            device="cpu",
        )
        assert report["aggregate"]["total_videos"] == 1
        assert "video.mp4" in report["per_video"]
        assert "notes.txt" not in report["per_video"]
        assert "thumb.png" not in report["per_video"]

    def test_output_json_schema(self, tmp_path: Path, monkeypatch):
        """Report JSON has top-level keys per_video and aggregate with correct structure."""
        (tmp_path / "a.mp4").write_text("")
        static = _make_static_clip(T=4)
        monkeypatch.setattr(_eval_mod, "load_video", _fake_load_video(static))

        output_path = str(tmp_path / "report.json")
        evaluate_directory(str(tmp_path), output_path, device="cpu")

        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "per_video" in data
        assert "aggregate" in data
        assert isinstance(data["per_video"], dict)
        agg = data["aggregate"]
        assert "mean_teleport_rate" in agg
        assert "total_videos" in agg
        assert "total_frames" in agg

        entry = data["per_video"]["a.mp4"]
        assert "teleport_rate" in entry
        assert "frame_count" in entry
        assert "event_count" in entry

    def test_aggregate_mean_is_average(self, tmp_path: Path, monkeypatch):
        """Aggregate mean_teleport_rate equals arithmetic mean of per-video rates."""
        for name in ("a.mp4", "b.mp4", "c.mp4"):
            (tmp_path / name).write_text("")

        call_count = [0]
        rates = [0.1, 0.2, 0.3]

        def _fake_compute(rgb, detector, threshold, min_event_area_ratio=0.001):
            idx = call_count[0]
            call_count[0] += 1
            return {"teleport_rate": rates[idx], "frame_count": 10, "event_count": 1}

        monkeypatch.setattr(_eval_mod, "load_video", _fake_load_video(_make_static_clip(T=10)))
        monkeypatch.setattr(_eval_mod, "compute_teleport_rate", _fake_compute)

        report = evaluate_directory(
            str(tmp_path),
            str(tmp_path / "report.json"),
            device="cpu",
        )
        expected_mean = round(sum(rates) / len(rates), 6)
        assert report["aggregate"]["mean_teleport_rate"] == expected_mean
        assert report["aggregate"]["total_videos"] == 3

    def test_alphabetical_sort_deterministic(self, tmp_path: Path, monkeypatch):
        """Files are processed in alphabetical order (deterministic output)."""
        # Create files in reverse alphabetical order
        for name in ("z.mp4", "m.mp4", "a.mp4"):
            (tmp_path / name).write_text("")

        static = _make_static_clip(T=4)
        monkeypatch.setattr(_eval_mod, "load_video", _fake_load_video(static))

        report = evaluate_directory(
            str(tmp_path),
            str(tmp_path / "report.json"),
            device="cpu",
        )
        keys = list(report["per_video"].keys())
        assert keys == ["a.mp4", "m.mp4", "z.mp4"]
