# SPDX-License-Identifier: Apache-2.0
"""
StreamingSwitchTrainingPipeline3Sink — 3-sink parallel variant with prompt switching.

This is a lightweight subclass of StreamingSwitchTrainingPipeline that
inherits ALL core logic (generate_chunk_with_cache, _recache_after_switch,
etc.) unchanged.  The only addition is a debug marker printed at
construction time that reports the resolved 3-sink layout.

kv_cache_size is NOT modified manually — the parent class computes it
from local_attn_size (set to 18 in the config: 3 long + 6 mid + 9 rolling).
"""

from methods.reward_forcing.pipelines.streaming_switch_training import (
    StreamingSwitchTrainingPipeline,
)


class StreamingSwitchTrainingPipeline3Sink(StreamingSwitchTrainingPipeline):
    """3-sink parallel variant of StreamingSwitchTrainingPipeline.

    Behavior is identical to the parent.  The 3-sink layout is configured
    through the model kwargs (local_attn_size=18, long_sink_size=3,
    mid_sink_size=6) and does not require any code changes in the pipeline.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        long_sink = kwargs.get("long_sink_size", 3)
        mid_sink = kwargs.get("mid_sink_size", 6)
        rolling = self.local_attn_size - long_sink - mid_sink
        print(
            f"[StreamingSwitchTrainingPipeline3Sink] 3-sink layout: "
            f"long={long_sink} "
            f"mid={mid_sink} "
            f"rolling={rolling} "
            f"total_attn={self.local_attn_size} "
            f"kv_cache_size={self.kv_cache_size}"
        )
