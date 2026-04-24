# CausVid factory functions for model instantiation
# Maps model_name strings to core.wan_wrapper classes
from core.wan_wrapper.wan_wrapper_causvid import (
    WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper, CausalWanDiffusionWrapper
)


_DIFFUSION_WRAPPERS = {
    "wan": WanDiffusionWrapper,
    "causal_wan": CausalWanDiffusionWrapper,
}


def get_diffusion_wrapper(model_name):
    return _DIFFUSION_WRAPPERS[model_name]


def get_text_encoder_wrapper(model_name):
    return WanTextEncoder


def get_vae_wrapper(model_name):
    return WanVAEWrapper


def get_inference_pipeline_wrapper(model_name, **kwargs):
    raise NotImplementedError("Use methods.causvid.pipelines directly")
