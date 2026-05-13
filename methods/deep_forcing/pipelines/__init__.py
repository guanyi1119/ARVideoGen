from .causal_inference import CausalInferencePipeline
try:
    from .causal_diffusion_inference import CausalDiffusionInferencePipeline
except ImportError:
    # If not available, fall back to causal_forcing version
    from methods.causal_forcing.pipelines import CausalDiffusionInferencePipeline

__all__ = [
    "CausalInferencePipeline",
    "CausalDiffusionInferencePipeline",
]
