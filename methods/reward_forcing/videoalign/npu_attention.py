import os
import math

import torch

_REGISTERED = False


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


def register_npu_fusion_attention():
    """Register the `npu_fusion` attention implementation with transformers.

    Safe to call multiple times. No-op when not running on NPU (so CUDA setups
    are unaffected).
    """
    global _REGISTERED
    if _REGISTERED:
        return
    if os.environ.get("DEVICE_TYPE", "cuda") != "npu":
        return

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return

    from transformers.modeling_utils import AttentionInterface

    AttentionInterface.register("npu_fusion", _npu_fusion_attention_forward)
    _REGISTERED = True
