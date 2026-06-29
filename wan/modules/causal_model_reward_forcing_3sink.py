# 3-Sink variant of causal_model_reward_forcing: long_sink + mid_sink + rolling window.
# Cloned from wan/modules/causal_model_reward_forcing.py with the following changes:
#   - CausalWanSelfAttention3Sink accepts long_sink_size, mid_sink_size, long_compression_alpha, mid_compression_alpha
#     instead of sink_size, compression_alpha.
#   - KV cache layout: [long_sink (L frames) | mid_sink (M frames) | rolling window + slack].
#   - Cascade EMA eviction: evicted rolling tokens → EMA-update mid_sink ring buffer →
#     if mid ring wraps, cascade overwritten mid frame → EMA-update long_sink ring buffer.
#   - Backward-compat aliases: self.sink_size = long_sink_size + mid_sink_size,
#     self.compression_alpha = mid_compression_alpha.
#   - local_attn_size is total attention window frames (sink + rolling), unchanged semantics.
# DO NOT merge with the default or reward-forcing variants.
from wan.modules.attention import attention
from wan.modules.npu_attention import chunked_flex_attention, npu_flex_attention, npu_flex_attention_v2
from wan.modules.npu_flex_attention import FlexAttentionNPU
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention as _flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import copy
import random
import torch.distributed as dist


_IS_NPU = os.environ.get('DEVICE_TYPE', 'cuda') == 'npu'
_USE_NPU_FLEX_ATTENTION_VERSION = os.environ.get('USE_NPU_FLEX_ATTENTION_VERSION', '1')

# torch.compile relies on Triton/CUDA backends which are not supported on NPU
if _IS_NPU:
    if _USE_NPU_FLEX_ATTENTION_VERSION == '0':
        flex_attention = chunked_flex_attention
    elif _USE_NPU_FLEX_ATTENTION_VERSION == '1':
        flex_attention = npu_flex_attention
    elif _USE_NPU_FLEX_ATTENTION_VERSION == '2':
        flex_attention = npu_flex_attention_v2
    else:
        flex_attention = _flex_attention
else:
    # wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
    # see https://github.com/pytorch/pytorch/issues/133254
    flex_attention = torch.compile(
        _flex_attention,
        dynamic=False,
        mode="max-autotune-no-cudagraphs"
    )


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)

class CausalWanSelfAttention3Sink(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 long_sink_size=0,
                 mid_sink_size=0,
                 qk_norm=True,
                 eps=1e-6,
                 long_compression_alpha=1.0,
                 mid_compression_alpha=0.999):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.long_sink_size = long_sink_size
        self.mid_sink_size = mid_sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.long_compression_alpha = long_compression_alpha
        self.mid_compression_alpha = mid_compression_alpha

        # Backward-compat aliases for trainer code that reads attn.sink_size / attn.compression_alpha
        self.sink_size = long_sink_size + mid_sink_size
        self.compression_alpha = mid_compression_alpha

        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def incremental_update(self, new_k: torch.Tensor, new_v: torch.Tensor,
                         current_sink_k: torch.Tensor, current_sink_v: torch.Tensor,
                         alpha: float = None) -> tuple:
        """
        Args:
            new_k: [B, num_tokens, num_heads, head_dim]
            new_v:  [B, num_tokens, num_heads, head_dim]
            current_sink_k:  [B, num_tokens, num_heads, head_dim]
            current_sink_v:  [B, num_tokens, num_heads, head_dim]
            alpha: compression alpha (defaults to self.compression_alpha)
        Returns:
            updated_sink_k, updated_sink_v: [B, num_tokens, num_heads, head_dim]
        """
        if alpha is None:
            alpha = self.compression_alpha

        # updated = α * current + (1-α) * new
        updated_sink_k = alpha * current_sink_k + (1 - alpha) * new_k
        updated_sink_v = alpha * current_sink_v + (1 - alpha) * new_v

        return updated_sink_k, updated_sink_v

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        current_start_frame = current_start // frame_seqlen
        current_end = current_start + q.shape[1]
        total_sink_tokens = (self.long_sink_size + self.mid_sink_size) * frame_seqlen
        long_sink_tokens = self.long_sink_size * frame_seqlen

        # KV cache management
        kv_cache_size = kv_cache["k"].shape[1]
        num_new_tokens = q.shape[1]

        evicted_tokens_exist = False
        evicted_k = None
        evicted_v = None

        global_end_index = kv_cache["global_end_index"].item()
        local_end_before = kv_cache["local_end_index"].item()
        delta_tokens = current_end - global_end_index
        overwrite_cache = delta_tokens <= 0

        if overwrite_cache:
            local_end_index = local_end_before + delta_tokens
            local_start_index = local_end_index - num_new_tokens
            # Guard: when recaching after a prompt switch, delta_tokens can be
            # very negative because global_end_index is far ahead of
            # current_start.  Reset both indices so that subsequent blocks
            # write sequentially in non-overwrite mode.
            if local_start_index < 0 or local_end_index > kv_cache_size:
                local_start_index = total_sink_tokens
                local_end_index = total_sink_tokens + num_new_tokens
                kv_cache["global_end_index"].fill_(current_start)
                kv_cache["local_end_index"].fill_(local_end_index)
            if local_start_index < 0 or local_end_index > kv_cache_size:
                raise RuntimeError(
                    f"Invalid KV overwrite range: local_start={local_start_index}, "
                    f"local_end={local_end_index}, num_new_tokens={num_new_tokens}, "
                    f"local_end_before={local_end_before}, current_start={current_start}, "
                    f"current_end={current_end}, global_end_index={global_end_index}"
                )
            kv_cache["k"][:, local_start_index:local_end_index] = k
            kv_cache["v"][:, local_start_index:local_end_index] = v
        elif self.local_attn_size != -1 and (num_new_tokens + local_end_before > kv_cache_size):
            num_evicted_tokens = num_new_tokens + local_end_before - kv_cache_size
            num_rolled_tokens = local_end_before - num_evicted_tokens - total_sink_tokens

            evicted_start = total_sink_tokens
            evicted_end = total_sink_tokens + num_evicted_tokens
            evicted_k = kv_cache["k"][:, evicted_start:evicted_end].clone()
            evicted_v = kv_cache["v"][:, evicted_start:evicted_end].clone()
            evicted_tokens_exist = True

            kv_cache["k"][:, total_sink_tokens:total_sink_tokens + num_rolled_tokens] = \
                kv_cache["k"][:, total_sink_tokens + num_evicted_tokens:total_sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            kv_cache["v"][:, total_sink_tokens:total_sink_tokens + num_rolled_tokens] = \
                kv_cache["v"][:, total_sink_tokens + num_evicted_tokens:total_sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

            local_end_index = local_end_before + delta_tokens - num_evicted_tokens
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = k
            kv_cache["v"][:, local_start_index:local_end_index] = v
        else:
            local_end_index = local_end_before + delta_tokens
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = k
            kv_cache["v"][:, local_start_index:local_end_index] = v

        # ---- Cascade EMA sink update (3-sink variant) ----
        if evicted_tokens_exist and evicted_k is not None and evicted_k.shape[1] > 0:
            num_evicted_total = evicted_k.shape[1]

            # Initialize ring-buffer state in kv_cache on first use
            if "mid_ring_idx" not in kv_cache:
                kv_cache["mid_ring_idx"] = torch.zeros(1, device=evicted_k.device, dtype=torch.long)
            if "long_ring_idx" not in kv_cache:
                kv_cache["long_ring_idx"] = torch.zeros(1, device=evicted_k.device, dtype=torch.long)
            if "mid_sink_filled" not in kv_cache:
                kv_cache["mid_sink_filled"] = torch.zeros(1, device=evicted_k.device, dtype=torch.bool)

            mid_sink_capacity = self.mid_sink_size * frame_seqlen
            long_sink_capacity = self.long_sink_size * frame_seqlen
            mid_region_start = long_sink_tokens

            remaining = num_evicted_total
            evicted_offset = 0

            while remaining > 0:
                chunk = min(mid_sink_capacity, remaining)
                mid_ring_idx = kv_cache["mid_ring_idx"].item()

                # How many tokens fit before ring wrap
                space_before_wrap = mid_sink_capacity - mid_ring_idx
                part1 = min(chunk, space_before_wrap)
                part2 = chunk - part1

                # Read current mid_sink content before overwrite (for cascade)
                overwritten_mid_k_parts = []
                overwritten_mid_v_parts = []

                # Part 1: before wrap
                mid_start1 = mid_region_start + mid_ring_idx
                mid_end1 = mid_start1 + part1
                current_mid_k1 = kv_cache["k"][:, mid_start1:mid_end1].clone()
                current_mid_v1 = kv_cache["v"][:, mid_start1:mid_end1].clone()
                overwritten_mid_k_parts.append(current_mid_k1)
                overwritten_mid_v_parts.append(current_mid_v1)

                evicted_k_part1 = evicted_k[:, evicted_offset:evicted_offset + part1]
                evicted_v_part1 = evicted_v[:, evicted_offset:evicted_offset + part1]
                updated_mid_k1, updated_mid_v1 = self.incremental_update(
                    evicted_k_part1, evicted_v_part1,
                    current_mid_k1, current_mid_v1,
                    alpha=self.mid_compression_alpha
                )
                kv_cache["k"][:, mid_start1:mid_end1] = updated_mid_k1
                kv_cache["v"][:, mid_start1:mid_end1] = updated_mid_v1

                # Part 2: after wrap (if any)
                if part2 > 0:
                    mid_start2 = mid_region_start
                    mid_end2 = mid_region_start + part2
                    current_mid_k2 = kv_cache["k"][:, mid_start2:mid_end2].clone()
                    current_mid_v2 = kv_cache["v"][:, mid_start2:mid_end2].clone()
                    overwritten_mid_k_parts.append(current_mid_k2)
                    overwritten_mid_v_parts.append(current_mid_v2)

                    evicted_k_part2 = evicted_k[:, evicted_offset + part1:evicted_offset + chunk]
                    evicted_v_part2 = evicted_v[:, evicted_offset + part1:evicted_offset + chunk]
                    updated_mid_k2, updated_mid_v2 = self.incremental_update(
                        evicted_k_part2, evicted_v_part2,
                        current_mid_k2, current_mid_v2,
                        alpha=self.mid_compression_alpha
                    )
                    kv_cache["k"][:, mid_start2:mid_end2] = updated_mid_k2
                    kv_cache["v"][:, mid_start2:mid_end2] = updated_mid_v2

                new_mid_ring_idx = (mid_ring_idx + chunk) % mid_sink_capacity
                wrapped = (part2 > 0) or (new_mid_ring_idx <= mid_ring_idx and chunk > 0)

                # Cascade overwritten mid content into long_sink if wrapped and mid already filled
                if wrapped and kv_cache["mid_sink_filled"].item():
                    overwritten_mid_k = torch.cat(overwritten_mid_k_parts, dim=1)
                    overwritten_mid_v = torch.cat(overwritten_mid_v_parts, dim=1)
                    # long_sink may also need chunking if overwritten content > long_sink_capacity
                    long_remaining = overwritten_mid_k.shape[1]
                    long_offset = 0
                    while long_remaining > 0:
                        long_chunk = min(long_sink_capacity, long_remaining)
                        long_ring_idx = kv_cache["long_ring_idx"].item()
                        long_space_before_wrap = long_sink_capacity - long_ring_idx
                        long_part1 = min(long_chunk, long_space_before_wrap)
                        long_part2 = long_chunk - long_part1

                        # Part 1
                        long_start1 = long_ring_idx
                        long_end1 = long_start1 + long_part1
                        current_long_k1 = kv_cache["k"][:, long_start1:long_end1].clone()
                        current_long_v1 = kv_cache["v"][:, long_start1:long_end1].clone()
                        long_src_k1 = overwritten_mid_k[:, long_offset:long_offset + long_part1]
                        long_src_v1 = overwritten_mid_v[:, long_offset:long_offset + long_part1]
                        updated_long_k1, updated_long_v1 = self.incremental_update(
                            long_src_k1, long_src_v1,
                            current_long_k1, current_long_v1,
                            alpha=self.long_compression_alpha
                        )
                        kv_cache["k"][:, long_start1:long_end1] = updated_long_k1
                        kv_cache["v"][:, long_start1:long_end1] = updated_long_v1

                        # Part 2
                        if long_part2 > 0:
                            long_start2 = 0
                            long_end2 = long_part2
                            current_long_k2 = kv_cache["k"][:, long_start2:long_end2].clone()
                            current_long_v2 = kv_cache["v"][:, long_start2:long_end2].clone()
                            long_src_k2 = overwritten_mid_k[:, long_offset + long_part1:long_offset + long_chunk]
                            long_src_v2 = overwritten_mid_v[:, long_offset + long_part1:long_offset + long_chunk]
                            updated_long_k2, updated_long_v2 = self.incremental_update(
                                long_src_k2, long_src_v2,
                                current_long_k2, current_long_v2,
                                alpha=self.long_compression_alpha
                            )
                            kv_cache["k"][:, long_start2:long_end2] = updated_long_k2
                            kv_cache["v"][:, long_start2:long_end2] = updated_long_v2

                        kv_cache["long_ring_idx"].fill_((long_ring_idx + long_chunk) % long_sink_capacity)
                        long_remaining -= long_chunk
                        long_offset += long_chunk

                kv_cache["mid_ring_idx"].fill_(new_mid_ring_idx)
                if wrapped:
                    kv_cache["mid_sink_filled"].fill_(True)

                remaining -= chunk
                evicted_offset += chunk

        # Prepare key and value tensors for attention
        kv_start_index = max(total_sink_tokens, local_end_index - self.max_attention_size + total_sink_tokens)
        kv_end_index = local_end_index

        # Calculate the actual number of tokens to use
        actual_kv_tokens = kv_end_index - kv_start_index

        # Ensure we have valid dimensions for RoPE application
        k_segment = kv_cache["k"][:, kv_start_index:kv_end_index]
        v_segment = kv_cache["v"][:, kv_start_index:kv_end_index]

        # Use all sink tokens (long + mid) for attention
        if self.sink_size > 0:
            sink_k = kv_cache["k"][:, :total_sink_tokens]
            sink_v = kv_cache["v"][:, :total_sink_tokens]

            k_combined = torch.cat([sink_k, k_segment], dim=1)
            v_combined = torch.cat([sink_v, v_segment], dim=1)

            # Update grid_sizes_kv to reflect the correct frame count
            grid_sizes_kv = copy.deepcopy(grid_sizes)
            total_combined_tokens = total_sink_tokens + actual_kv_tokens
            total_frames = total_combined_tokens // frame_seqlen
            grid_sizes_kv[:, 0] = total_frames

            # Calculate start frame for query RoPE
            query_start_frame = (total_sink_tokens +
                               (local_end_index - max(total_sink_tokens, local_end_index - self.max_attention_size + total_sink_tokens)) -
                               q.shape[1]) // frame_seqlen

            x = attention(
                causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=query_start_frame
                ).type_as(v),
                causal_rope_apply(
                    k_combined, grid_sizes_kv, freqs, start_frame=0
                ).type_as(v),
                v_combined,
            )
        else:
            # No sink tokens case
            grid_sizes_kv = copy.deepcopy(grid_sizes)
            grid_sizes_kv[:, 0] = actual_kv_tokens // frame_seqlen

            query_start_frame = (local_end_index - self.max_attention_size - q.shape[1]) // frame_seqlen

            x = attention(
                causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=query_start_frame
                ).type_as(v),
                causal_rope_apply(
                    k_segment, grid_sizes_kv, freqs, start_frame=0
                ).type_as(v),
                v_segment,
            )

        if not overwrite_cache:
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock3Sink(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 long_sink_size=0,
                 mid_sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 long_compression_alpha=1.0,
                 mid_compression_alpha=0.999):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention3Sink(
            dim, num_heads, local_attn_size,
            long_sink_size, mid_sink_size,
            qk_norm, eps,
            long_compression_alpha, mid_compression_alpha
        )
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                       num_heads,
                                                                       (-1, -1),
                                                                       qk_norm,
                                                                       eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start)

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel3Sink(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    3-sink variant with long_sink + mid_sink + rolling window KV cache.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['CausalWanAttentionBlock3Sink']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 long_sink_size=0,
                 mid_sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 long_compression_alpha=1.0,
                 mid_compression_alpha=0.999):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            long_sink_size (`int`, *optional*, defaults to 0):
                Number of long-sink frames (frozen / slowly updated via cascade EMA)
            mid_sink_size (`int`, *optional*, defaults to 0):
                Number of mid-sink frames (EMA-updated from evicted rolling tokens)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            long_compression_alpha (`float`, *optional*, defaults to 1.0):
                EMA alpha for long_sink cascade (1.0 = frozen after first write)
            mid_compression_alpha (`float`, *optional*, defaults to 0.999):
                EMA alpha for mid_sink updates from evicted rolling tokens
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock3Sink(
                cross_attn_type, dim, ffn_dim, num_heads,
                local_attn_size, long_sink_size, mid_sink_size,
                qk_norm, cross_attn_norm, eps,
                long_compression_alpha, mid_compression_alpha
            )
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
