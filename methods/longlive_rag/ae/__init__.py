"""Retrieval autoencoder for compressing Wan latent frames to 1-D embeddings.

Design points (see ``model.py`` for the implementation):
- GroupNorm (no train/eval distribution shift), residual blocks, global average pool
- Encoder/decoder symmetric via hidden_dims, no spatial attention
- Frame-wise independent compression (each frame -> one embedding vector)
"""
from .model import LatentAE, Encoder, Decoder, TemporalDeltaLoss
from .config import AEConfig

__all__ = ["LatentAE", "Encoder", "Decoder", "TemporalDeltaLoss", "AEConfig"]