# SPDX-License-Identifier: Apache-2.0
"""
InteractiveCausalInferencePipeline3Sink — 3-sink variant for multi-segment inference.

Inherits ALL logic from InteractiveCausalInferencePipeline (N-segment prompt
switching, switch_frame_indices, etc.).  The only addition is that __init__
creates a 3-sink WanDiffusionWrapper when generator is None, instead of
falling through to the 2-sink wrapper hardcoded in CausalInferencePipeline.
"""

from methods.reward_forcing.pipelines.interactive_causal_inference import (
    InteractiveCausalInferencePipeline,
)


class InteractiveCausalInferencePipeline3Sink(InteractiveCausalInferencePipeline):
    """3-sink variant of InteractiveCausalInferencePipeline.

    Behavior is identical to the parent except the generator is a 3-sink
    WanDiffusionWrapper when generator=None.
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
            f"[InteractiveCausalInferencePipeline3Sink] 3-sink layout: "
            f"long={long_sink} "
            f"mid={mid_sink} "
            f"rolling={rolling} "
            f"total_attn={self.local_attn_size}"
        )
