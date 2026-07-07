from .attention import attention
from .npu_attention import chunked_flex_attention, npu_flex_attention, npu_flex_attention_v2
from .npu_flex_attention import FlexAttentionNPU
from .model import WanModel
from .t5 import T5Decoder, T5Encoder, T5EncoderModel, T5Model
from .tokenizers import HuggingfaceTokenizer
from .vae import WanVAE

__all__ = [
    'WanVAE',
    'WanModel',
    'T5Model',
    'T5Encoder',
    'T5Decoder',
    'T5EncoderModel',
    'HuggingfaceTokenizer',
    'attention',
    'get_causal_model_class',
    'chunked_flex_attention',
    'npu_flex_attention',
    'npu_flex_attention_v2',
    'FlexAttentionNPU'
]


def get_causal_model_class(model_type='default'):
    """Factory function to return the appropriate CausalWanModel class.

    Args:
        model_type: One of 'default', 'causvid', 'longlive', 'infinity', 'rolling_forcing', 'latentmem'.
            - 'default': Causal-Forcing/Self-Forcing version (teacher forcing, I2V, cache_start)
            - 'causvid': CausVid version (window_size, simplified KV cache, current_end)
            - 'longlive': LongLive version (deferred cache update, sink_recache_after_switch)
            - 'infinity': LongLive Infinity version (infinite length attention)
            - 'rolling_forcing': RollingForcing version (rolling window training, updating_cache)
            - 'latentmem': LongLive-RAG latentmem version (memory retrieval, EMA sink, CPU offload)

    Returns:
        CausalWanModel class from the corresponding module.
    """
    if model_type == 'default':
        from .causal_model import CausalWanModel
        return CausalWanModel
    elif model_type == 'causvid':
        from .causal_model_causvid import CausalWanModel
        return CausalWanModel
    elif model_type == 'longlive':
        from .causal_model_longlive import CausalWanModel
        return CausalWanModel
    elif model_type == 'infinity':
        from .causal_model_infinity import CausalWanModel
        return CausalWanModel
    elif model_type == 'rolling_forcing':
        from .causal_model_rolling_forcing import CausalWanModel
        return CausalWanModel
    elif model_type == 'latentmem':
        from .causal_model_latentmem import CausalWanModel
        return CausalWanModel
    else:
        raise ValueError(
            f"Unknown causal_model_type '{model_type}'. "
            f"Choose from: 'default', 'causvid', 'longlive', 'infinity', 'rolling_forcing', 'latentmem'"
        )
