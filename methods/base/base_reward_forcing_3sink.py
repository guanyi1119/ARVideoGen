# SPDX-License-Identifier: Apache-2.0
"""
BaseModel3SinkMixin — injects 3-sink WanDiffusionWrapper into model initialization.

The base reward-forcing model (methods/base/base_reward_forcing.py:19) hardcodes
    WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper = get_wan_wrapper_classes('reward_forcing')
at module level, so BaseModel._initialize_models always creates a 2-sink generator.

This mixin overrides _initialize_models to:
1. Filter out 3-sink specific kwargs that the 2-sink wrapper does not accept.
2. Call super()._initialize_models() with filtered kwargs.
3. Replace self.generator with the 3-sink wrapper using the ORIGINAL kwargs.
4. Reconfigure num_frame_per_block / independent_first_frame on the new generator.
5. Update self.scheduler from the new generator.
"""

from core.wan_wrapper import get_wan_wrapper_classes

_3SINK_ONLY_KEYS = frozenset({
    "long_sink_size",
    "mid_sink_size",
    "long_compression_alpha",
    "mid_compression_alpha",
})


class BaseModel3SinkMixin:
    """Mixin that replaces the 2-sink generator with a 3-sink generator after base init.

    Place FIRST in the MRO:
        class ReDMD3Sink(BaseModel3SinkMixin, ReDMD):
    """

    def _initialize_models(self, args, device):
        original_model_kwargs = getattr(args, "model_kwargs", {})
        filtered_kwargs = {
            k: v
            for k, v in dict(original_model_kwargs).items()
            if k not in _3SINK_ONLY_KEYS
        }

        args.model_kwargs = filtered_kwargs
        super()._initialize_models(args, device)
        args.model_kwargs = original_model_kwargs

        _, _, WanDiffusionWrapper3Sink = get_wan_wrapper_classes("reward_forcing_3sink")
        self.generator = WanDiffusionWrapper3Sink(
            **dict(original_model_kwargs), is_causal=True
        )
        self.generator.model.requires_grad_(True)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)
