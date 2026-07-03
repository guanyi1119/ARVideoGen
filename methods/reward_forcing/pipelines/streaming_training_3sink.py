# SPDX-License-Identifier: Apache-2.0
"""
StreamingTrainingPipeline3Sink — 3-sink parallel variant.

This is a lightweight subclass of StreamingTrainingPipeline that inherits
ALL core logic (generate_chunk_with_cache, _initialize_kv_cache, etc.)
unchanged.  The only addition is a debug marker printed at construction
time that reports the resolved 3-sink layout (long / mid / rolling).

kv_cache_size is NOT modified manually — the parent class computes it
from local_attn_size (set to 18 in the config: 3 long + 6 mid + 9 rolling).
"""

from methods.reward_forcing.pipelines.streaming_training import StreamingTrainingPipeline


class StreamingTrainingPipeline3Sink(StreamingTrainingPipeline):
    """3-sink parallel variant of StreamingTrainingPipeline.

    Behavior is identical to the parent except _clone_kv_cache also copies
    the 3-sink-specific cache keys (mid_ring_idx, long_ring_idx, mid_sink_filled).
    The 3-sink layout is configured through the model kwargs (local_attn_size=18,
    long_sink_size=3, mid_sink_size=6) and does not require other code changes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        long_sink = kwargs.get("long_sink_size", 3)
        mid_sink = kwargs.get("mid_sink_size", 6)
        rolling = self.local_attn_size - long_sink - mid_sink
        print(
            f"[StreamingTrainingPipeline3Sink] 3-sink layout: "
            f"long={long_sink} "
            f"mid={mid_sink} "
            f"rolling={rolling} "
            f"total_attn={self.local_attn_size} "
            f"kv_cache_size={self.kv_cache_size}"
        )

    def _clone_kv_cache(self, zero_sink: bool = False, zero_window: bool = False) -> list:
        """Clone KV cache with 3-sink extra keys, optionally zeroing sink/window."""
        sink_size = self._get_sink_size()
        total_sink_tokens = sink_size * self.frame_seq_length
        cloned = []
        for blk in self.kv_cache1:
            k = blk["k"].clone()
            v = blk["v"].clone()
            local_end = blk["local_end_index"].item()
            if zero_sink and total_sink_tokens > 0:
                k[:, :total_sink_tokens].zero_()
                v[:, :total_sink_tokens].zero_()
            if zero_window and local_end > total_sink_tokens:
                k[:, total_sink_tokens:local_end].zero_()
                v[:, total_sink_tokens:local_end].zero_()
            entry = {
                "k": k,
                "v": v,
                "global_end_index": blk["global_end_index"].clone(),
                "local_end_index": blk["local_end_index"].clone(),
            }
            for key in ("mid_ring_idx", "long_ring_idx", "mid_sink_filled"):
                if key in blk:
                    entry[key] = blk[key].clone()
            cloned.append(entry)
        return cloned
