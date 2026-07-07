"""TDD tests for the multi-node latent corpus generator (RED phase).

These tests exercise the pure functions in `scripts/generate_longlive_rag_latent.py`
that are deterministic and CPU-only (sharding logic, filename formatting,
sampling / striping). The full `main()` / `build_pipeline()` paths depend
on the inference pipeline and generator checkpoints, so they are NOT unit-tested
here; they are exercised by the P5 end-to-end smoke run.
"""
import os
import sys
import random
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import target — module under scripts/ must be importable as a module.
# We add scripts/ to sys.path so `import generate_longlive_rag_latent` works
# even when tests are run from repo root.
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))


def test_module_importable():
    import generate_longlive_rag_latent as g  # noqa: F401
    assert hasattr(g, "compute_shard_pairs")
    assert hasattr(g, "latent_path_for")
    assert hasattr(g, "prompt_id")
    assert hasattr(g, "main")


# ---------------------------------------------------------------------------
# compute_shard_pairs — multi-node sharding contract
# ---------------------------------------------------------------------------

class _FakeDataset:
    """Stand-in for TextDataset that returns prompts with given indices."""
    def __init__(self, n):
        self.prompts = [f"prompt_{i}" for i in range(n)]
    def __len__(self):
        return len(self.prompts)


def test_compute_shard_pairs_no_overlap(monkeypatch):
    """The union of all ranks' shard pairs has no duplicates across RANK ∈ [0, W)."""
    import generate_longlive_rag_latent as g

    n_prompts = 100
    # Align with the new default (0.1) so the test no longer relies on a
    # historical 0.5 default. See test_compute_shard_pairs_default_sample_ratio_is_0_1
    # for the contract on the default itself.
    sample_ratio = 0.1
    SEED = 42  # Avoid Python 3 class-scope gotcha: literal in class body
    world_size = 8

    # Patch TextDataset to return our fake dataset
    class _Cfg:
        data_path = "ignored"
        seed = SEED  # literal, not outer-scope reference
    monkeypatch.setattr(g, "TextDataset", lambda prompt_path=None, extended_prompt_path=None: _FakeDataset(n_prompts))

    all_pos = []
    per_rank_sizes = []
    for rank in range(world_size):
        full, pairs = g.compute_shard_pairs(_Cfg, rank, world_size, reverse=False)
        positions = [pos for pos, _ in pairs]
        all_pos.extend(positions)
        per_rank_sizes.append(len(positions))

    # No overlap → unique set size equals sum of per-rank sizes
    assert len(set(all_pos)) == len(all_pos), (
        f"sharding overlaps: {len(set(all_pos))} unique vs {len(all_pos)} total"
    )
    # Each rank handles the fair share (10 prompts sampled / 8 ranks ≈ 1-2 each)
    sampled = max(1, int(n_prompts * sample_ratio))
    assert sum(per_rank_sizes) == sampled
    # Roughly balanced (rounding within 1)
    assert max(per_rank_sizes) - min(per_rank_sizes) <= 1


def test_compute_shard_pairs_default_sample_ratio_is_0_1(monkeypatch):
    """When the config does not carry sample_ratio, compute_shard_pairs
    must fall back to the documented default 0.1 (matches README and the
    shipped generate_latent_*.yaml files)."""
    import generate_longlive_rag_latent as g

    SEED = 42
    n_prompts = 100

    class _Cfg:
        data_path = "ignored"
        # No sample_ratio attribute on purpose → must hit the default
        seed = SEED
    monkeypatch.setattr(g, "TextDataset", lambda prompt_path=None, extended_prompt_path=None: _FakeDataset(n_prompts))

    _, pairs = g.compute_shard_pairs(_Cfg, 0, 1, reverse=False)
    # With sample_ratio=0.1 and 100 prompts, exactly 10 prompts should be sampled.
    assert len(pairs) == 10, (
        f"default sample_ratio should give 10 sampled prompts (n_prompts=100, 0.1), got {len(pairs)}"
    )


def test_compute_shard_pairs_explicit_sample_ratio_overrides_default(monkeypatch):
    """An explicit config.sample_ratio must override the 0.1 default.

    This guards the "yaml-provided sample_ratio actually wins" contract —
    exactly the regression that triggered P5.fix-1.
    """
    import generate_longlive_rag_latent as g

    SEED = 42
    n_prompts = 100

    class _Cfg:
        # Use a non-default ratio so the test catches both "default wins"
        # and "explicit wins" regressions.
        data_path = "ignored"
        sample_ratio = 0.05
        seed = SEED
    monkeypatch.setattr(g, "TextDataset", lambda prompt_path=None, extended_prompt_path=None: _FakeDataset(n_prompts))

    _, pairs = g.compute_shard_pairs(_Cfg, 0, 1, reverse=False)
    # 0.05 * 100 = 5 — proves the explicit field took effect over the default 0.1.
    assert len(pairs) == 5, (
        f"explicit sample_ratio=0.05 must override default 0.1; expected 5 sampled, got {len(pairs)}"
    )


def test_compute_shard_pairs_same_seed_same_sample(monkeypatch):
    """Calling compute_shard_pairs with SAME config.seed produces SAME sampled_indices order
    on every rank — guarantees the stripe boundary is consistent across nodes."""
    import generate_longlive_rag_latent as g

    class _Cfg:
        data_path = "ignored"
        seed = 42
    monkeypatch.setattr(g, "TextDataset", lambda prompt_path=None, extended_prompt_path=None: _FakeDataset(50))

    _, pairs_rank0 = g.compute_shard_pairs(_Cfg, 0, 4, reverse=False)
    _, pairs_rank1 = g.compute_shard_pairs(_Cfg, 1, 4, reverse=False)

    # Rank 1's positions should NOT collide with rank 0's positions
    pos0 = {p for p, _ in pairs_rank0}
    pos1 = {p for p, _ in pairs_rank1}
    assert not (pos0 & pos1), "ranks with same seed should not collide on sharding boundary"


def test_compute_shard_pairs_reverse_inverts_order(monkeypatch):
    """reverse=True inverts the shard's iteration order without changing its membership."""
    import generate_longlive_rag_latent as g

    class _Cfg:
        data_path = "ignored"
        seed = 42
    monkeypatch.setattr(g, "TextDataset", lambda prompt_path=None, extended_prompt_path=None: _FakeDataset(40))

    _, pairs = g.compute_shard_pairs(_Cfg, 0, 4, reverse=False)
    _, pairs_rev = g.compute_shard_pairs(_Cfg, 0, 4, reverse=True)

    pos = [p for p, _ in pairs]
    pos_rev = [p for p, _ in pairs_rev]
    assert pos == pos_rev[::-1], "reverse=True should invert the order"


# ---------------------------------------------------------------------------
# prompt_id — deterministic hash of prompt string
# ---------------------------------------------------------------------------

def test_prompt_id_is_deterministic():
    """Same prompt string must always produce the same prompt_id."""
    import generate_longlive_rag_latent as g
    pid1 = g.prompt_id("a cat playing piano")
    pid2 = g.prompt_id("a cat playing piano")
    assert pid1 == pid2


def test_prompt_id_different_prompts_differ():
    """Different prompts must produce different prompt_ids (no collision in practice)."""
    import generate_longlive_rag_latent as g
    pid1 = g.prompt_id("a cat playing piano")
    pid2 = g.prompt_id("a dog chasing a ball")
    assert pid1 != pid2


def test_prompt_id_is_hex_string():
    """prompt_id must be a hex string (lowercase, no dashes) so it's filename-safe."""
    import generate_longlive_rag_latent as g
    pid = g.prompt_id("test prompt 123")
    assert isinstance(pid, str)
    assert all(c in "0123456789abcdef" for c in pid), f"non-hex char in {pid!r}"
    assert len(pid) == 16, f"expected 16 hex chars, got {len(pid)}"


def test_prompt_id_stable_across_runs():
    """prompt_id must not depend on random seed, world_size, or rank — only on prompt text."""
    import generate_longlive_rag_latent as g
    # Hardcode the expected sha1 prefix so a future refactor can't silently
    # change the hash function without breaking tests.
    pid = g.prompt_id("hello world")
    expected = __import__("hashlib").sha1(b"hello world").hexdigest()[:16]
    assert pid == expected, f"prompt_id changed hash; expected {expected}, got {pid}"


# ---------------------------------------------------------------------------
# latent_path_for — filename contract (prompt_id based, not global_pos)
# ---------------------------------------------------------------------------

def test_latent_path_for_single_sample():
    """num_samples == 1 → latent_{pid}.pt (no seed suffix)."""
    import generate_longlive_rag_latent as g
    pid = g.prompt_id("a cat playing piano")
    p = g.latent_path_for("/tmp/foo", num_samples=1, pid=pid, seed_idx=0)
    assert p == os.path.join("/tmp/foo", f"latent_{pid}.pt")


def test_latent_path_for_multi_sample():
    """num_samples > 1 → latent_{pid}_s{seed_idx}.pt."""
    import generate_longlive_rag_latent as g
    pid = g.prompt_id("a cat playing piano")
    p = g.latent_path_for("/tmp/foo", num_samples=3, pid=pid, seed_idx=2)
    assert p == os.path.join("/tmp/foo", f"latent_{pid}_s2.pt")


def test_latent_path_for_same_prompt_same_filename():
    """The same prompt always produces the same latent filename — this is the
    core guarantee that makes --skip_existing safe across sample_ratio / world_size changes."""
    import generate_longlive_rag_latent as g
    pid = g.prompt_id("stable prompt")
    p1 = g.latent_path_for("/tmp", num_samples=1, pid=pid, seed_idx=0)
    p2 = g.latent_path_for("/tmp", num_samples=1, pid=pid, seed_idx=0)
    assert p1 == p2


# ---------------------------------------------------------------------------
# Smoke config sanity (not requiring generator ckpt)
# ---------------------------------------------------------------------------

def test_generate_latent_yaml_files_exist():
    """The two RAG corpus generation configs must be on disk after P2 lands."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cfg1 = os.path.join(repo_root, "configs", "longlive_rag", "generate_latent_reward_forcing.yaml")
    cfg2 = os.path.join(repo_root, "configs", "longlive_rag", "generate_latent_longlive.yaml")
    assert os.path.isfile(cfg1), f"missing: {cfg1}"
    assert os.path.isfile(cfg2), f"missing: {cfg2}"


def test_generate_latent_yaml_has_required_fields(tmp_path):
    """Each generate_latent yaml must specify the keys consumed by the script."""
    import yaml
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for name in ("generate_latent_reward_forcing.yaml", "generate_latent_longlive.yaml"):
        p = os.path.join(repo_root, "configs", "longlive_rag", name)
        with open(p) as f:
            cfg = yaml.safe_load(f)
        # Required top-level fields
        for field in ("data_path", "output_folder", "num_output_frames",
                      "sample_ratio", "num_samples", "generator_ckpt"):
            assert field in cfg, f"{name} missing field {field}"
        # Required model_kwargs block
        assert "model_kwargs" in cfg
        for field in ("local_attn_size", "timestep_shift", "sink_size"):
            assert field in cfg["model_kwargs"], f"{name} missing model_kwargs.{field}"


def test_generate_latent_yaml_lora_only_on_longlive():
    """generate_latent_longlive.yaml must include LoRA adapter; reward_forcing.yaml must NOT."""
    import yaml
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    with open(os.path.join(repo_root, "configs", "longlive_rag", "generate_latent_longlive.yaml")) as f:
        cfg_ll = yaml.safe_load(f)
    with open(os.path.join(repo_root, "configs", "longlive_rag", "generate_latent_reward_forcing.yaml")) as f:
        cfg_rf = yaml.safe_load(f)
    # LongLive path needs lora_ckpt + adapter
    assert cfg_ll.get("lora_ckpt") or cfg_ll.get("adapter"), (
        "generate_latent_longlive.yaml must have lora_ckpt + adapter block"
    )
    # RF-only path must NOT have lora_ckpt
    assert cfg_rf.get("lora_ckpt") is None, (
        "generate_latent_reward_forcing.yaml must NOT have lora_ckpt"
    )


def test_generate_latent_sh_launcher_exists():
    """The multi-node torchrun launcher sh must exist."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    p = os.path.join(repo_root, "generate_longlive_rag_latent.sh")
    assert os.path.isfile(p), f"missing launcher: {p}"


# ---------------------------------------------------------------------------
# parse_args — --sample_ratio CLI override
# ---------------------------------------------------------------------------

class _Namespace:
    """Stand-in namespace object compatible with the original argparse return."""
    pass


def test_parse_args_has_sample_ratio_flag():
    """parse_args must accept --sample_ratio and parse it as float."""
    import generate_longlive_rag_latent as g
    saved = sys.argv
    try:
        sys.argv = ["prog", "--config_path", "X.yaml", "--sample_ratio", "0.005"]
        args = g.parse_args()
        assert hasattr(args, "sample_ratio")
        assert args.sample_ratio == 0.005, (
            f"--sample_ratio 0.005 should parse to 0.005, got {args.sample_ratio}"
        )
        assert args.config_path == "X.yaml"
    finally:
        sys.argv = saved


def test_parse_args_sample_ratio_default_is_none():
    """When --sample_ratio is omitted, args.sample_ratio must be None
    (so main() can detect "no CLI override" and fall back to yaml/merged config).
    """
    import generate_longlive_rag_latent as g
    saved = sys.argv
    try:
        sys.argv = ["prog", "--config_path", "X.yaml"]
        args = g.parse_args()
        assert args.sample_ratio is None, (
            f"default args.sample_ratio should be None, got {args.sample_ratio!r}"
        )
    finally:
        sys.argv = saved