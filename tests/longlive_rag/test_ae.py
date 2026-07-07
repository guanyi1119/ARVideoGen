"""TDD tests for LongLive-RAG AE module (RED phase).

These tests exercise the public API contract that will be implemented in
`methods/longlive_rag/ae/`. They are written FIRST per the test-driven
development discipline and MUST fail before P1.1-P1.4 implementation lands.

Test surface:
  - Module import path:  methods.longlive_rag.ae.model / .config / .dataset / .train
  - LatentAE shape contract:  encoder [N, C, H, W] -> [N, latent_dim]; decoder reverses
  - LatentAE.encode (no_grad + L2-normalize) returns unit-norm embeddings
  - TemporalDeltaLoss hinge shape and margin behavior
  - AEConfig.from_yaml loads known fields with type coercion
  - LatentFrameDataset scans a directory and returns [S, C, H, W] chunks
"""
import os
import sys
import tempfile
import numpy as np
import pytest
import torch

# Make repo root importable when run from anywhere.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Imports - the contract under test
# ---------------------------------------------------------------------------

def test_imports():
    """All four AE submodules must be importable under methods.longlive_rag.ae."""
    from methods.longlive_rag.ae.model import LatentAE, Encoder, Decoder, TemporalDeltaLoss  # noqa: F401
    from methods.longlive_rag.ae.config import AEConfig  # noqa: F401
    from methods.longlive_rag.ae.dataset import LatentFrameDataset  # noqa: F401
    from methods.longlive_rag.ae.train import compute_losses, save_checkpoint  # noqa: F401


def test_default_data_dir_relocated():
    """AEConfig default data_dir must be 'datasets/longlive_rag' (P1.2 change)."""
    from methods.longlive_rag.ae.config import AEConfig
    cfg = AEConfig()
    assert cfg.data_dir == "datasets/longlive_rag", (
        f"default data_dir should be 'datasets/longlive_rag', got {cfg.data_dir!r}"
    )


# ---------------------------------------------------------------------------
# LatentAE forward shape contract
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_ae():
    """Build a tiny LatentAE so tests run on CPU in seconds."""
    from methods.longlive_rag.ae.config import AEConfig
    from methods.longlive_rag.ae.model import LatentAE
    cfg = AEConfig(
        in_channels=4,
        spatial_h=8,
        spatial_w=8,
        latent_dim=16,
        hidden_dims=[8, 16],
    )
    return LatentAE(cfg)


def test_encoder_output_shape(small_ae):
    """encoder([N, C, H, W]) -> [N, latent_dim]."""
    x = torch.randn(7, 4, 8, 8)
    emb = small_ae.encoder(x)
    assert emb.shape == (7, 16), f"expected (7, 16), got {tuple(emb.shape)}"


def test_decoder_output_shape(small_ae):
    """decoder([N, latent_dim]) -> [N, C, H, W] matching input spatial."""
    z = torch.randn(3, 16)
    out = small_ae.decoder(z)
    assert out.shape == (3, 4, 8, 8), f"expected (3, 4, 8, 8), got {tuple(out.shape)}"


def test_latentae_forward_returns_recon_and_embed(small_ae):
    """LatentAE.forward returns (recon[N,C,H,W], embed[N,D])."""
    x = torch.randn(2, 4, 8, 8)
    recon, embed = small_ae(x)
    assert recon.shape == x.shape
    assert embed.shape == (2, 16)


def test_encode_l2_normalized(small_ae):
    """LatentAE.encode(..., normalize=True) must return unit-norm rows."""
    x = torch.randn(5, 4, 8, 8)
    emb = small_ae.encode(x, normalize=True)
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-5), (
        f"expected unit-norm, got norms={norms.tolist()}"
    )


def test_encode_no_normalization(small_ae):
    """encode(..., normalize=False) returns raw embeddings (not necessarily unit)."""
    x = torch.randn(5, 4, 8, 8)
    emb = small_ae.encode(x, normalize=False)
    # Just check shape; norms need not be 1.
    assert emb.shape == (5, 16)
    # And it should be no_grad'd (we don't strictly assert grad absence
    # since the encoder is already decorated, but shape check at least
    # confirms the path works).


# ---------------------------------------------------------------------------
# TemporalDeltaLoss
# ---------------------------------------------------------------------------

def test_temporal_delta_loss_shape():
    from methods.longlive_rag.ae.model import TemporalDeltaLoss
    loss_fn = TemporalDeltaLoss(margin=0.85, weight=1.0)
    v_t = torch.randn(4, 16)
    v_ref = torch.randn(4, 16)
    loss = loss_fn(v_t, v_ref)
    assert loss.dim() == 0, f"expected scalar, got dim={loss.dim()}"
    assert loss.item() >= 0, "hinge loss must be non-negative"


def test_temporal_delta_loss_zero_when_orthogonal():
    """When embeddings are orthogonal (cosine sim ≈ 0), hinge(0-0.85)=0."""
    from methods.longlive_rag.ae.model import TemporalDeltaLoss
    loss_fn = TemporalDeltaLoss(margin=0.85, weight=1.0)
    # Construct explicitly orthogonal vectors via Gram-Schmidt on eye.
    v_t = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    v_ref = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    loss = loss_fn(v_t, v_ref)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_temporal_delta_loss_positive_when_too_similar():
    """When cosine sim > margin, loss = sim - margin > 0."""
    from methods.longlive_rag.ae.model import TemporalDeltaLoss
    loss_fn = TemporalDeltaLoss(margin=0.5, weight=1.0)
    v = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    loss = loss_fn(v, v)  # cosine sim = 1 > 0.5
    assert loss.item() == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# AEConfig YAML loading
# ---------------------------------------------------------------------------

def test_aeconfig_from_yaml(tmp_path):
    """AEConfig.from_yaml coerces string types and respects declared fields."""
    yaml_text = """
data_dir: datasets/longlive_rag
in_channels: 4
spatial_h: 8
spatial_w: 8
latent_dim: 16
hidden_dims: [8, 16]
batch_size: 4
epochs: 2
delta_weight: 1.0
delta_margin: 0.85
delta_window: 3
smooth_weight: 1.0
"""
    p = tmp_path / "ae.yaml"
    p.write_text(yaml_text)
    from methods.longlive_rag.ae.config import AEConfig
    cfg = AEConfig.from_yaml(str(p))
    assert cfg.in_channels == 4
    assert cfg.spatial_h == 8
    assert cfg.spatial_w == 8
    assert cfg.latent_dim == 16
    assert cfg.hidden_dims == [8, 16]
    assert cfg.batch_size == 4
    assert cfg.epochs == 2
    # type coercion: yaml may parse 4 as int already; float types must coerce
    assert isinstance(cfg.lr, float)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def test_latent_frame_dataset(tmp_path):
    """LatentFrameDataset reads .pt files and returns [S, C, H, W] chunks."""
    from methods.longlive_rag.ae.dataset import LatentFrameDataset
    # write 3 dummy latents of shape [12, 4, 8, 8]
    for i in range(3):
        torch.save(torch.randn(12, 4, 8, 8), tmp_path / f"latent_{i:04d}.pt")
    ds = LatentFrameDataset(str(tmp_path), seq_len=8)
    assert len(ds) == 3
    chunk = ds[0]
    assert chunk.shape == (8, 4, 8, 8), f"expected (8, 4, 8, 8), got {tuple(chunk.shape)}"


def test_latent_frame_dataset_padding_short_video(tmp_path):
    """A video shorter than seq_len is padded by repeating its last frame."""
    from methods.longlive_rag.ae.dataset import LatentFrameDataset
    torch.save(torch.randn(3, 4, 8, 8), tmp_path / "short.pt")  # T=3 < seq_len=8
    ds = LatentFrameDataset(str(tmp_path), seq_len=8)
    chunk = ds[0]
    assert chunk.shape == (8, 4, 8, 8)


# ---------------------------------------------------------------------------
# compute_losses (train loop core)
# ---------------------------------------------------------------------------

def test_compute_losses_returns_4_tensors(small_ae):
    """train.compute_losses returns (total, recon, delta, smooth) all scalar."""
    from methods.longlive_rag.ae.config import AEConfig
    from methods.longlive_rag.ae.model import TemporalDeltaLoss
    from methods.longlive_rag.ae.train import compute_losses
    cfg = AEConfig(
        in_channels=4, spatial_h=8, spatial_w=8, latent_dim=16,
        hidden_dims=[8, 16], delta_window=2, smooth_weight=1.0,
    )
    device = torch.device("cpu")
    chunk = torch.randn(2, 4, 4, 8, 8)  # [B, S, C, H, W]
    loss_fn = TemporalDeltaLoss(margin=0.85, weight=1.0)
    total, recon, delta, smooth = compute_losses(small_ae, chunk, cfg, loss_fn, device)
    assert all(isinstance(t, torch.Tensor) for t in (total, recon, delta, smooth))
    assert total.dim() == 0
    assert recon.dim() == 0
    assert delta.dim() == 0
    assert smooth.dim() == 0
    assert total.item() == pytest.approx(recon.item() + delta.item() + smooth.item(), rel=1e-4)