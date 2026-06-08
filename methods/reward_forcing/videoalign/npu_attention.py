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
    that layout directly. After the op we transpose to [B, L, num_heads, head_dim]
    to match the contract the rest of transformers expects.
    """
    from torch_npu import npu_fusion_attention

    # GQA: expand kv heads to match q heads when needed.
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

    # is_causal handling mirrors sdpa_attention_forward semantics.
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
    """Build a replacement forward() for Qwen2VLAttention.

    Copies the original forward logic from transformers 4.50.0 Qwen2VLAttention,
    but replaces the manual matmul+softmax attention with _npu_fusion_attention_forward.
    """
    from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb

    def npu_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value=None,
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
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Slice attention_mask to match key length (same as original eager path).
        causal_mask = None
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # Replace manual matmul+softmax with NPU fusion attention.
        attn_output, attn_weights = _npu_fusion_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=causal_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=1.0 / math.sqrt(self.head_dim),
            **kwargs,
        )

        # _npu_fusion_attention_forward returns [B, L, num_heads, head_dim].
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return npu_forward


# ---------------------------------------------------------------------------
# VisionAttention (visual encoder attention) NPU patch
# ---------------------------------------------------------------------------

def _make_vision_npu_forward(original_forward):
    """Build a replacement forward() for VisionAttention.

    Copies the original forward logic from transformers 4.50.0 VisionAttention,
    but replaces F.scaled_dot_product_attention with _npu_fusion_attention_forward.
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
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            print(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Build block-diagonal mask from cu_seqlens.
        # F.scaled_dot_product_attention: True=keep. npu_fusion_attention: True=masked.
        # So we create a mask where True=masked (within-chunk = False).
        attention_mask = torch.ones([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = False

        # Reshape to [B, num_heads, L, head_dim] for _npu_fusion_attention_forward.
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        # Replace F.scaled_dot_product_attention with NPU fusion attention.
        attn_output, _ = _npu_fusion_attention_forward(
            self,
            q,
            k,
            v,
            attention_mask=attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=False,
            **kwargs,
        )

        # _npu_fusion_attention_forward returns [1, seq_length, num_heads, head_dim].
        attn_output = attn_output.squeeze(0).reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output

    return vision_npu_forward


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def patch_qwen2vl_for_npu(model):
    """Monkey-patch all Qwen2VLAttention AND VisionAttention modules in *model*
    to use NPU fusion attention instead of the default SDPA/matmul dispatch.

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

    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention, VisionAttention

    patched_text = 0
    patched_vision = 0

    for module in model.modules():
        if isinstance(module, Qwen2VLAttention):
            module.forward = _make_npu_forward(module.forward).__get__(module, Qwen2VLAttention)
            patched_text += 1
        elif isinstance(module, VisionAttention):
            module.forward = _make_vision_npu_forward(module.forward).__get__(module, VisionAttention)
            patched_vision += 1

    if patched_text > 0:
        print(f"[NPU] Patched {patched_text} Qwen2VLAttention modules to use npu_fusion_attention")
    if patched_vision > 0:
        print(f"[NPU] Patched {patched_vision} VisionAttention modules to use npu_fusion_attention")

    _PATCHED = True
