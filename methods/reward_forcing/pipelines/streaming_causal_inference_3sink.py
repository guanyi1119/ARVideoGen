# SPDX-License-Identifier: Apache-2.0
"""
StreamingCausalInferencePipeline3Sink — 3-sink parallel variant for inference.

This is a lightweight subclass of StreamingCausalInferencePipeline that
inherits ALL core logic (inference, _initialize_kv_cache, etc.) unchanged.
The only addition is a debug marker printed at construction time that
reports the resolved 3-sink layout.

kv_cache_size is NOT modified manually — the parent class computes it
from local_attn_size (set to 18 in the config: 3 long + 6 mid + 9 rolling).
"""

from methods.reward_forcing.pipelines.streaming_causal_inference import (
    StreamingCausalInferencePipeline,
)


class StreamingCausalInferencePipeline3Sink(StreamingCausalInferencePipeline):
    """3-sink parallel variant of StreamingCausalInferencePipeline.

    Behavior is identical to the parent.  The 3-sink layout is configured
    through the model kwargs (local_attn_size=18, long_sink_size=3,
    mid_sink_size=6) and does not require any code changes in the pipeline.
    """

    def __init__(self, args, device, *, generator=None, text_encoder=None, vae=None):
        if generator is None:
            from core.wan_wrapper import get_wan_wrapper_classes
            _, _, WanDiffusionWrapper3Sink = get_wan_wrapper_classes("reward_forcing_3sink")
            generator = WanDiffusionWrapper3Sink(
                **getattr(args, "model_kwargs", {}), is_causal=True
            )
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)
        long_sink = getattr(args, "long_sink_size", 3)
        mid_sink = getattr(args, "mid_sink_size", 6)
        rolling = self.local_attn_size - long_sink - mid_sink
        print(
            f"[StreamingCausalInferencePipeline3Sink] 3-sink layout: "
            f"long={long_sink} "
            f"mid={mid_sink} "
            f"rolling={rolling} "
            f"total_attn={self.local_attn_size}"
        )
