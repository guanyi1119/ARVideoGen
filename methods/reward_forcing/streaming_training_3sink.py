# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""
StreamingTrainingModel3Sink — 3-sink KV cache variant.

Extends StreamingTrainingModel with a 3-sink attention layout (long / mid / rolling)
instead of the default 2-sink (sink / rolling).  The KV cache zeroing augmentation
inherited from the parent already covers the combined long+mid sink head because
_get_sink_size reads attn.sink_size from the model, which we configure as long+mid.

This class is a pure organizational marker — all generation, loss computation,
and cache zeroing logic is inherited unchanged from StreamingTrainingModel.
"""

from methods.reward_forcing.streaming_training import StreamingTrainingModel


class StreamingTrainingModel3Sink(StreamingTrainingModel):
    """
    Streaming training model with a 3-sink KV cache layout.

    The attention window is split into three regions:
      - long sink:  persistent prefix tokens (e.g. first 3 frames)
      - mid sink:   medium-term prefix tokens (e.g. next 6 frames)
      - rolling:    sliding window of recent tokens

    All training logic (_generate_chunk, generate_next_chunk, compute_*_loss,
    KV cache zeroing) is inherited unchanged from StreamingTrainingModel.
    """

    def __init__(self, base_model, config):
        super().__init__(base_model, config)

        model_kwargs = getattr(config, 'model_kwargs', {})
        long_sink = getattr(model_kwargs, 'long_sink_size', 3)
        mid_sink = getattr(model_kwargs, 'mid_sink_size', 6)
        local_attn = getattr(model_kwargs, 'local_attn_size', 18)
        rolling = local_attn - long_sink - mid_sink

        print(
            f"[StreamingTrainingModel3Sink] 3-sink layout: "
            f"long={long_sink} mid={mid_sink} rolling={rolling} "
            f"total_attn={local_attn} chunk_size={self.chunk_size}"
        )

    def _save_kv_cache(self) -> list:
        saved = []
        for blk in self.inference_pipeline.kv_cache1:
            entry = {
                "k": blk["k"].clone(),
                "v": blk["v"].clone(),
                "global_end_index": blk["global_end_index"].clone(),
                "local_end_index": blk["local_end_index"].clone(),
            }
            for key in ("mid_ring_idx", "long_ring_idx", "mid_sink_filled"):
                if key in blk:
                    entry[key] = blk[key].clone()
            saved.append(entry)
        return saved

    def _restore_kv_cache(self, saved: list):
        for blk, sb in zip(self.inference_pipeline.kv_cache1, saved):
            blk["k"].copy_(sb["k"])
            blk["v"].copy_(sb["v"])
            blk["global_end_index"].copy_(sb["global_end_index"])
            blk["local_end_index"].copy_(sb["local_end_index"])
            for key in ("mid_ring_idx", "long_ring_idx", "mid_sink_filled"):
                if key in sb:
                    if key not in blk:
                        blk[key] = sb[key].clone()
                    else:
                        blk[key].copy_(sb[key])
