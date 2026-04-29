
import os
import math

import torch
import torch.nn.functional as F

_IS_NPU = os.environ.get('DEVICE_TYPE', 'cuda') == 'npu'
if _IS_NPU:
    import torch_npu


def chunked_flex_attention(query, key, value, block_mask=None):
    """
    Drop-in replacement for flex_attention(query, key, value, block_mask).
    Same [B, H, L, D] input/output format.

    Processes attention per Q-block using block_mask.mask_mod to determine
    allowed KV blocks (via corner-checking) and fine-grained boolean masks.
    This avoids materializing the full N*N attention matrix, reducing peak
    memory from O(N^2) to O(block_size * N).

    NOTE: We pre-compute block sparsity ourselves by evaluating mask_mod at
    block corners, because block_mask.kv_num_blocks/kv_indices may be
    unreliable when mask_mod uses tensor closures with advanced indexing
    (e.g. context_ends[q_idx]) that don't compose well under vmap.
    """
    B, H, L, _ = query.shape
    Q_BLOCK_SIZE, KV_BLOCK_SIZE = block_mask.BLOCK_SIZE
    mask_mod = block_mask.mask_mod
    num_q_blocks = L // Q_BLOCK_SIZE + (1 if L % Q_BLOCK_SIZE else 0)

    # Pre-compute block-level sparsity by evaluating mask_mod at block corners.
    # Checking all 4 corners ensures we don't miss blocks where the mask
    # transitions within the block.
    q_starts = torch.arange(0, L, Q_BLOCK_SIZE, device=query.device)
    kv_starts = torch.arange(0, L, KV_BLOCK_SIZE, device=query.device)
    q_ends = torch.minimum(q_starts + Q_BLOCK_SIZE - 1,
                           torch.tensor(L - 1, device=query.device))
    kv_ends = torch.minimum(kv_starts + KV_BLOCK_SIZE - 1,
                            torch.tensor(L - 1, device=query.device))

    # Evaluate at 4 corners with broadcasting: [num_q, 1] vs [1, num_kv]
    qs = q_starts.unsqueeze(1)
    qe = q_ends.unsqueeze(1)
    kss = kv_starts.unsqueeze(0)
    ke = kv_ends.unsqueeze(0)

    block_allowed = (
        mask_mod(0, 0, qs, kss) |
        mask_mod(0, 0, qs, ke) |
        mask_mod(0, 0, qe, kss) |
        mask_mod(0, 0, qe, ke)
    )

    output = torch.zeros_like(query)

    for q_bi in range(num_q_blocks):
        q_start = q_bi * Q_BLOCK_SIZE
        q_end = min(q_start + Q_BLOCK_SIZE, L)
        if q_start >= L:
            break

        allowed_kv = block_allowed[q_bi].nonzero().squeeze(-1)
        if len(allowed_kv) == 0:
            continue

        kv_slices = []
        kv_positions = []
        for kv_bi in allowed_kv.tolist():
            ks = kv_bi * KV_BLOCK_SIZE
            ke = min(ks + KV_BLOCK_SIZE, L)
            kv_slices.append((ks, ke))
            kv_positions.append(torch.arange(ks, ke, device=query.device))

        k_gathered = torch.cat([key[:, :, s:e] for s, e in kv_slices], dim=2)
        v_gathered = torch.cat([value[:, :, s:e] for s, e in kv_slices], dim=2)
        kv_indices = torch.cat(kv_positions)

        q_indices = torch.arange(q_start, q_end, device=query.device)
        q_2d = q_indices.unsqueeze(1)
        kv_2d = kv_indices.unsqueeze(0)
        fine_mask = mask_mod(0, 0, q_2d, kv_2d)

        # Bool attn_mask: True=attend, False=masked
        q_chunk = query[:, :, q_start:q_end]
        attn_mask = fine_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)

        output[:, :, q_start:q_end] = F.scaled_dot_product_attention(
            q_chunk, k_gathered, v_gathered, attn_mask=attn_mask)

    return output


def npu_flex_attention(query, key, value, block_mask=None):
    """
    Drop-in replacement for flex_attention(query, key, value, block_mask).
    Same [B, H, L, D] input/output format.

    Generates a token-level atten_mask from block_mask.mask_mod and calls
    npu_fusion_attention with sparse_mode=1 (allMask).

    atten_mask semantics:
        - mask_mod / flex_attention: True = attend (allowed)
        - npu_fusion_attention: 1 = masked (skip), 0 = attend (compute)
    So we invert: npu_atten_mask = (~token_mask).to(uint8)
    """
    B, H, L, D = query.shape
    mask_mod = block_mask.mask_mod

    # Generate token-level mask from mask_mod: [L, L] bool, True=attend
    q_idx = torch.arange(L, device=query.device).unsqueeze(1)
    kv_idx = torch.arange(L, device=query.device).unsqueeze(0)
    token_mask = mask_mod(0, 0, q_idx, kv_idx)

    # Invert to npu_fusion_attention semantics: 1=masked, 0=attend
    npu_atten_mask = (~token_mask).to(torch.uint8)

    # Pad Q, K, V and mask to alignment if needed
    NPU_ALIGN = 16
    pad_len = (NPU_ALIGN - L % NPU_ALIGN) % NPU_ALIGN

    if pad_len > 0:
        query = F.pad(query, (0, 0, 0, pad_len))
        key = F.pad(key, (0, 0, 0, pad_len))
        value = F.pad(value, (0, 0, 0, pad_len))
        L_padded = L + pad_len
        # Extend mask: padded Q rows are all masked (1), padded KV cols are all masked (1)
        mask_kv_pad = torch.ones(L, pad_len, device=query.device, dtype=torch.uint8)
        mask_q_pad = torch.ones(pad_len, L_padded, device=query.device, dtype=torch.uint8)
        npu_atten_mask = torch.cat([npu_atten_mask, mask_kv_pad], dim=1)
        npu_atten_mask = torch.cat([npu_atten_mask, mask_q_pad], dim=0)
    else:
        L_padded = L

    scale_val = 1.0 / math.sqrt(D)

    attn_out, _, _, _, _, _, _ = torch_npu.npu_fusion_attention(
        query, key, value,
        head_num=H,
        input_layout="BNSD",
        atten_mask=npu_atten_mask,
        scale=scale_val,
        sparse_mode=1,
    )

    # Trim padding from output
    if pad_len > 0:
        attn_out = attn_out[:, :, :L, :]

    return attn_out


def npu_flex_attention_v2(query, key, value, block_mask=None):
    """
    Drop-in replacement for flex_attention using TND layout + varlen sparse mode.

    When block_mask has _tf_mask_meta (set by _prepare_teacher_forcing_mask),
    uses TND layout + sparse_mode=7 to avoid materializing the full [L, L]
    dense atten_mask. Otherwise falls back to chunked_flex_attention.
    """
    if not hasattr(block_mask, '_tf_mask_meta'):
        return chunked_flex_attention(query, key, value, block_mask)

    num_frames, frame_seqlen, num_frame_per_block = block_mask._tf_mask_meta

    B, H, L, D = query.shape
    assert B == 1, "TND layout only supports batch_size=1 currently"

    block_size = frame_seqlen * num_frame_per_block
    num_blocks = num_frames * frame_seqlen // block_size
    N = num_frames * frame_seqlen  # boundary between clean and noisy tokens
    real_L = 2 * N  # actual token count (without 128-alignment padding)

    # Strip 128-alignment padding — TND decomposition only covers real tokens
    q_real = query[:, :, :real_L]
    k_real = key[:, :, :real_L]
    v_real = value[:, :, :real_L]

    # Convert BNSD [1, H, real_L, D] to TND-sliced [real_L, H, D]
    q_tnd = q_real[0].permute(1, 0, 2)
    k_tnd = k_real[0].permute(1, 0, 2)
    v_tnd = v_real[0].permute(1, 0, 2)

    # Build TND-format Q, K, V and metadata
    tnd_q_list = []
    tnd_k_list = []
    tnd_v_list = []
    q_seq_lengths = []
    kv_seq_lengths = []

    for i in range(num_blocks):
        # --- Clean block i ---
        q_start = i * block_size
        q_end = (i + 1) * block_size
        kv_end = (i + 1) * block_size  # KV includes context blocks 0..i

        tnd_q_list.append(q_tnd[q_start:q_end])
        tnd_k_list.append(k_tnd[:kv_end])
        tnd_v_list.append(v_tnd[:kv_end])

        q_seq_lengths.append(block_size)
        kv_seq_lengths.append(kv_end)

    for i in range(num_blocks):
        # --- Noisy block i ---
        q_start = N + i * block_size
        q_end = N + (i + 1) * block_size
        context_len = i * block_size

        if context_len > 0:
            tnd_k_list.append(torch.cat([k_tnd[:context_len], k_tnd[q_start:q_end]], dim=0))
            tnd_v_list.append(torch.cat([v_tnd[:context_len], v_tnd[q_start:q_end]], dim=0))
        else:
            tnd_k_list.append(k_tnd[q_start:q_end])
            tnd_v_list.append(v_tnd[q_start:q_end])

        tnd_q_list.append(q_tnd[q_start:q_end])
        q_seq_lengths.append(block_size)
        kv_seq_lengths.append(context_len + block_size)

    # Concatenate all sub-sequences into TND format: [total_len, H, D]
    tnd_q = torch.cat(tnd_q_list, dim=0)
    tnd_k = torch.cat(tnd_k_list, dim=0)
    tnd_v = torch.cat(tnd_v_list, dim=0)

    # Cumulative sequence lengths for npu_fusion_attention
    actual_seq_qlen = torch.cumsum(torch.tensor(q_seq_lengths, dtype=torch.int64), dim=0).tolist()
    actual_seq_kvlen = torch.cumsum(torch.tensor(kv_seq_lengths, dtype=torch.int64), dim=0).tolist()

    scale_val = 1.0 / math.sqrt(D)

    # sparse_mode=7 (varlen外切): optimized for variable-length sequences
    # Each sub-sequence computes full bidirectional attention, no atten_mask needed
    attn_out_tnd, _, _, _, _, _, _ = torch_npu.npu_fusion_attention(
        tnd_q, tnd_k, tnd_v,
        head_num=H,
        input_layout="TND",
        scale=scale_val,
        sparse_mode=7,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )

    # TND output segments are already in sequential order:
    #   clean_0, clean_1, ..., clean_{n-1}, noisy_0, noisy_1, ..., noisy_{n-1}
    # which maps to positions [0, real_L) directly.
    # [total_q_len, H, D] -> [1, real_L, H, D] -> [1, H, real_L, D]
    attn_out = attn_out_tnd.reshape(1, real_L, H, D).permute(0, 2, 1, 3)

    # Restore 128-alignment padding (caller will slice it off with [:, :, :-padded_length])
    if L > real_L:
        attn_out = F.pad(attn_out, (0, 0, 0, L - real_L))

    return attn_out
