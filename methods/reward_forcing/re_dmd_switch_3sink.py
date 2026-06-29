# SPDX-License-Identifier: Apache-2.0
"""
ReDMDSwitch3Sink — ReDMDSwitch variant that uses the 3-sink switch training pipeline.

Inherits all reward-forcing and prompt-switching logic from ReDMDSwitch and
only replaces the inference pipeline with StreamingSwitchTrainingPipeline3Sink
to enable 3-sink (long/mid/rolling) KV-cache layout during streaming
generation with mid-video prompt switching.
"""

from methods.base.base_reward_forcing_3sink import BaseModel3SinkMixin
from methods.reward_forcing.re_dmd_switch import ReDMDSwitch
from methods.reward_forcing.pipelines.streaming_switch_training_3sink import (
    StreamingSwitchTrainingPipeline3Sink,
)


class ReDMDSwitch3Sink(BaseModel3SinkMixin, ReDMDSwitch):
    """ReDMDSwitch variant with 3-sink parallel KV-cache training pipeline.

    Behavior is identical to ReDMDSwitch except the inference pipeline uses a
    3-sink attention layout (long + mid + rolling sinks) configured through
    model kwargs (local_attn_size, long_sink_size, mid_sink_size).
    """

    def _initialize_inference_pipeline(self):
        self.inference_pipeline = StreamingSwitchTrainingPipeline3Sink(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            context_noise=self.args.context_noise,
            local_attn_size=getattr(self.args, "model_kwargs", {}).get("local_attn_size", -1),
            slice_last_frames=getattr(self.args, "slice_last_frames", 21),
            global_sink=getattr(self.args, "global_sink", False),
            long_sink_size=getattr(self.args, "model_kwargs", {}).get("long_sink_size", 3),
            mid_sink_size=getattr(self.args, "model_kwargs", {}).get("mid_sink_size", 6),
        )
