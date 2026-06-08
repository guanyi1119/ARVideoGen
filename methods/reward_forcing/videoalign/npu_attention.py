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


def patch_qwen2vl_for_npu(model):
    """Monkey-patch all Qwen2VLAttention modules in *model* to use NPU fusion
    attention instead of the default SDPA/flash-attn dispatch.

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

    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention

    patched_count = 0
    for module in model.modules():
        if isinstance(module, Qwen2VLAttention):
            module.forward = _make_npu_forward(module.forward).__get__(module, Qwen2VLAttention)
            patched_count += 1

    if patched_count > 0:
        print(f"[NPU] Patched {patched_count} Qwen2VLAttention modules to use npu_fusion_attention")

    _PATCHED = True
