from .wan_wrapper import WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper
from .model_interface import (
    DiffusionModelInterface,
    VAEInterface,
    TextEncoderInterface,
    InferencePipelineInterface
)


def get_wan_wrapper_classes(method='default'):
    """Return (WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper) for the given method.

    Args:
        method: One of 'default' (CF/SF), 'causvid', 'longlive', 'deepforcing', 'rolling_forcing', 'reward_forcing'

    Returns:
        Tuple of (WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper) classes
    """
    if method == 'default':
        from .wan_wrapper import WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper
        return WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper
    elif method == 'causvid':
        from .wan_wrapper_causvid import WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper, CausalWanDiffusionWrapper
        return WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper, CausalWanDiffusionWrapper
    elif method == 'longlive':
        from .wan_wrapper_longlive import WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper
        return WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper
    elif method == 'deepforcing':
        from .wan_wrapper_deepforcing import WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper
        return WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper
    elif method == 'rolling_forcing':
        from .wan_wrapper_rollingforcing import WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper
        return WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper
    elif method == 'reward_forcing':
        from .wan_wrapper_reward_forcing import WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper
        return WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper
    else:
        raise ValueError(f"Unknown wan_wrapper method '{method}'. Choose from: 'default', 'causvid', 'longlive', 'deepforcing', 'rolling_forcing', 'reward_forcing'")
