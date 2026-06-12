from methods.reward_forcing.re_dmd import ReDMD
from methods.reward_forcing.pipelines.streaming_switch_training import StreamingSwitchTrainingPipeline


class ReDMDSwitch(ReDMD):
    """ReDMD variant that supports streaming training with prompt switching.

    Inherits all reward-forcing logic from ReDMD (reward computation, DMD loss, etc.)
    and replaces the inference pipeline with StreamingSwitchTrainingPipeline to enable
    chunk-by-chunk generation with KV cache reuse and mid-video prompt switching.
    """

    def _initialize_inference_pipeline(self):
        self.inference_pipeline = StreamingSwitchTrainingPipeline(
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
        )
