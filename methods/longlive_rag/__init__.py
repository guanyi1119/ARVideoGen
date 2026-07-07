"""LongLive-RAG: a retrieval-augmented long-video generation method.

This package contains the retrieval autoencoder training (``ae/``) and the
RAG-enabled inference pipeline. The AR generator backbone stays frozen;
only the small retrieval encoder is trained.
"""


def __getattr__(name):
    if name == "LatentMemCausalInferencePipeline":
        from .pipelines import LatentMemCausalInferencePipeline

        return LatentMemCausalInferencePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LatentMemCausalInferencePipeline"]