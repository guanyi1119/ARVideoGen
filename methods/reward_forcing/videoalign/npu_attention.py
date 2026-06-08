import os
import math

import torch

_PATCHED = False


def _npu_fusion_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask=None,
    dropout=0.0,
    scaling=None,
    is_causal=None,
    **kwargs,
):
    """Drop-in attention forward that calls torch_npu.npu_fusion_attention.

    transformers calls this with query/key/value shaped [B, num_heads, L, head_dim]
    (post-GQA in Qwen2-VL). npu_fusion_attention with input_layout="BNSD" accepts
    that layout directly, so no transposes are needed for the LLM path. After the
    op we transpose to [B, L, num_heads, head_dim] to match the contract the rest
    of transformers expects (sdpa_attention_forward also returns the transposed
    layout).
    """
    from torch_npu import npu_fusion_attention

    # GQA: expand kv heads to match q heads when needed (npu_fusion_attention does
    # not natively support GQA in BNSD layout for this transformers version).
    if hasattr(module, "num_key_value_groups") and module.num_key_value_groups > 1:
        n_rep = module.num_key_value_groups
        b, kv_heads, slen, head_dim = key.shape
        key = key[:, :, None, :, :].expand(b, kv_heads, n_rep, slen, head_dim)
        key = key.reshape(b, kv_heads * n_rep, slen, head_dim)
        value = value[:, :, None, :, :].expand(b, kv_heads, n_rep, slen, head_dim)
        value = value.reshape(b, kv_heads * n_rep, slen, head_dim)

    if scaling is None:
        scaling = 1.0 / math.sqrt(query.shape[-1])

    keep_prob = 1.0 - (dropout if dropout is not None else 0.0)
    head_num = query.shape[1]

    # is_causal handling mirrors sdpa_attention_forward semantics: only treat as
    # causal when query has more than 1 token and no explicit mask was provided.
    effective_is_causal = (
        is_causal
        if is_causal is not None
        else getattr(module, "is_causal", False)
    )
    effective_is_causal = bool(
        query.shape[2] > 1 and attention_mask is None and effective_is_causal
    )

    atten_mask = None
    sparse_mode = 0
    if effective_is_causal:
        # Down-right aligned causal mask via cached upper-triangular mask.
        atten_mask = torch.triu(
            torch.ones([2048, 2048], device=query.device), diagonal=1
        ).bool()
        sparse_mode = int(os.getenv("NPU_FA2_SPARSE_MODE", "3"))
    elif attention_mask is not None:
        # npu_fusion_attention expects a boolean mask where True = masked-out.
        if attention_mask.dtype == torch.bool:
            atten_mask = attention_mask
        else:
            # transformers commonly passes additive masks (0/-inf). Convert to bool.
            atten_mask = (attention_mask < 0)

    output = npu_fusion_attention(
        query,
        key,
        value,
        head_num,
        "BNSD",
        atten_mask=atten_mask,
        scale=scaling,
        keep_prob=keep_prob,
        sparse_mode=sparse_mode,
    )[0]

    # transformers downstream expects [B, L, num_heads, head_dim].
    attn_output = output.transpose(1, 2).contiguous()
    return attn_output, None


# ---------------------------------------------------------------------------
# Qwen2VLAttention (LLM text attention) NPU patch
# ---------------------------------------------------------------------------

def _make_npu_forward(original_forward):
    """Build a replacement forward() for Qwen2VLAttention that uses NPU fusion
    attention instead of the ALL_ATTENTION_FUNCTIONS dispatch.

    We replicate the projection + RoPE + KV-cache logic from the original
    forward, then call _npu_fusion_attention_forward instead of going through
    ALL_ATTENTION_FUNCTIONS.get_interface().
    """
    from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb

    def npu_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.config.rope_parameters["mrope_section"]
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = _npu_fusion_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return npu_forward


# ---------------------------------------------------------------------------
# VisionAttention (visual encoder attention) NPU patch
# ---------------------------------------------------------------------------

def _make_vision_npu_forward(original_forward):
    """Build a replacement forward() for VisionAttention that uses NPU fusion
    attention instead of the ALL_ATTENTION_FUNCTIONS dispatch.

    VisionAttention differs from Qwen2VLAttention:
    - Uses self.qkv (single fused projection) instead of q_proj/k_proj/v_proj
    - Uses apply_rotary_pos_emb_vision instead of apply_multimodal_rotary_pos_emb
    - Has cu_seqlens for variable-length packed sequences
    - No KV-cache
    - Output projection is self.proj instead of self.o_proj
    """
    from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb_vision

    def vision_npu_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # Always use chunk processing (non-flash path), replacing
        # ALL_ATTENTION_FUNCTIONS.get_interface() with _npu_fusion_attention_forward.
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
        ]

        attn_outputs = [
            _npu_fusion_attention_forward(
                self,
                q,
                k,
                v,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )[0]
            for q, k, v in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output

    return vision_npu_forward


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def patch_qwen2vl_for_npu(model):
    """Monkey-patch all Qwen2VLAttention AND VisionAttention modules in *model*
    to use NPU fusion attention instead of the default SDPA/flash-attn dispatch.

    Call this AFTER model = Qwen2VLRewardModelBT.from_pretrained(...) with
    attn_implementation="sdpa" (which passes transformers validation).

    Safe to call multiple times. No-op when not on NPU.
    """
    global _PATCHED
    if _PATCHED:
        return
    if os.environ.get("DEVICE_TYPE", "cuda") != "npu":
        return

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return

    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention, Qwen2VLSdpaAttention, VisionAttention, VisionSdpaAttention

    patched_text = 0
    patched_vision = 0

    for module in model.modules():
        if isinstance(module, Qwen2VLAttention) or isinstance(module, Qwen2VLSdpaAttention):
            module.forward = _make_npu_forward(module.forward).__get__(module, Qwen2VLAttention)
            patched_text += 1
        elif isinstance(module, VisionAttention) or isinstance(module, VisionSdpaAttention):
            module.forward = _make_vision_npu_forward(module.forward).__get__(module, VisionAttention)
            patched_vision += 1

    if patched_text > 0:
        print(f"[NPU] Patched {patched_text} Qwen2VLAttention modules to use npu_fusion_attention")
    if patched_vision > 0:
        print(f"[NPU] Patched {patched_vision} VisionAttention modules to use npu_fusion_attention")

    _PATCHED = True
