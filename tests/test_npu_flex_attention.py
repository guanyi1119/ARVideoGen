"""
Compare npu_flex_attention vs flex_attention + BlockMask.

npu_flex_attention (v1) is a drop-in replacement for flex_attention that uses
torch_npu.npu_fusion_attention internally. It generates a token-level
atten_mask from block_mask.mask_mod and calls npu_fusion_attention once
with sparse_mode=1 (allMask).

npu_flex_attention_v2 uses TND layout + prefix sparse mode (sparse_mode=5/7)
to avoid materializing the full [L, L] dense atten_mask.

Usage (CUDA — tests mask generation + SDPA fallback):
    python tests/test_npu_flex_attention.py --num_frames 9 --frame_seqlen 256 --num_frame_per_block 3

    # Float32 to check precision
    python tests/test_npu_flex_attention.py --num_frames 9 --frame_seqlen 256 --num_frame_per_block 3 --dtype float32

    # Production scale
    python tests/test_npu_flex_attention.py --num_frames 21 --frame_seqlen 1560 --num_frame_per_block 3

Usage (NPU — tests with actual npu_fusion_attention):
    python tests/test_npu_flex_attention.py --num_frames 9 --frame_seqlen 256 --num_frame_per_block 3 --device npu

Usage (v2 — TND + prefix sparse mode):
    python tests/test_npu_flex_attention.py --test_v2 --num_frames 9 --frame_seqlen 256 --num_frame_per_block 3

    # v2 on NPU
    python tests/test_npu_flex_attention.py --test_v2 --num_frames 9 --frame_seqlen 256 --num_frame_per_block 3 --device npu
"""

import argparse
import math
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention as _flex_attention
from wan.modules.causal_model import npu_flex_attention_v2, chunked_flex_attention


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
    import torch_npu

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


def prepare_teacher_forcing_mask(device, num_frames, frame_seqlen, num_frame_per_block):
    """Recreate _prepare_teacher_forcing_mask from causal_model.py"""
    total_length = num_frames * frame_seqlen * 2
    padded_length = math.ceil(total_length / 128) * 128 - total_length

    clean_ends = num_frames * frame_seqlen
    context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
    noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
    noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
    noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
    noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

    attention_block_size = frame_seqlen * num_frame_per_block
    frame_indices = torch.arange(
        start=0, end=num_frames * frame_seqlen,
        step=attention_block_size, device=device, dtype=torch.long)

    for start in frame_indices:
        context_ends[start:start + attention_block_size] = start + attention_block_size

    noisy_image_start_list = torch.arange(
        num_frames * frame_seqlen, total_length,
        step=attention_block_size, device=device, dtype=torch.long)
    noisy_image_end_list = noisy_image_start_list + attention_block_size

    for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
        noise_noise_starts[start:end] = start
        noise_noise_ends[start:end] = end
        noise_context_ends[start:end] = block_index * attention_block_size

    def attention_mask(b, h, q_idx, kv_idx):
        clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
        C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
        C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
        noise_mask = (q_idx >= clean_ends) & (C1 | C2)
        eye_mask = q_idx == kv_idx
        return eye_mask | clean_mask | noise_mask

    block_mask = create_block_mask(
        attention_mask, B=None, H=None,
        Q_LEN=total_length + padded_length,
        KV_LEN=total_length + padded_length,
        _compile=False, device=device)
    block_mask._tf_mask_meta = (num_frames, frame_seqlen, num_frame_per_block)
    return block_mask


def prepare_blockwise_causal_mask(device, num_frames, frame_seqlen, num_frame_per_block):
    """Recreate _prepare_blockwise_causal_attn_mask from causal_model.py"""
    total_length = num_frames * frame_seqlen
    padded_length = math.ceil(total_length / 128) * 128 - total_length

    ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

    attention_block_size = frame_seqlen * num_frame_per_block
    frame_indices = torch.arange(
        start=0, end=total_length,
        step=attention_block_size, device=device)

    for tmp in frame_indices:
        ends[tmp:tmp + attention_block_size] = tmp + attention_block_size

    def attention_mask(b, h, q_idx, kv_idx):
        return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

    block_mask = create_block_mask(
        attention_mask, B=None, H=None,
        Q_LEN=total_length + padded_length,
        KV_LEN=total_length + padded_length,
        _compile=False, device=device)
    return block_mask


def test_mask_generation(block_mask, L, device):
    """Verify that mask generated from mask_mod matches BlockMask.to_dense() expansion."""
    mask_mod = block_mask.mask_mod

    # Token-level mask from mask_mod
    q_idx = torch.arange(L, device=device).unsqueeze(1)
    kv_idx = torch.arange(L, device=device).unsqueeze(0)
    token_mask = mask_mod(0, 0, q_idx, kv_idx)

    # Block-level mask from to_dense, expanded
    block_dense = block_mask.to_dense()
    Q_BS, KV_BS = block_mask.BLOCK_SIZE
    expanded = block_dense[0, 0].repeat_interleave(Q_BS, dim=0).repeat_interleave(KV_BS, dim=1)[:L, :L]

    match = (expanded.bool() == token_mask).all().item()
    true_count = token_mask.sum().item()
    total = token_mask.numel()
    sparsity = 1.0 - true_count / total

    print(f"  mask shape: {token_mask.shape}, dtype: {token_mask.dtype}")
    print(f"  True(attend): {true_count} / {total}, sparsity: {sparsity:.2%}")
    print(f"  matches BlockMask expansion: {match}")

    # Verify inversion for npu_fusion_attention
    npu_mask = (~token_mask).to(torch.uint8)
    print(f"  npu_atten_mask: 0(attend)={int((npu_mask == 0).sum())}, 1(masked)={int((npu_mask == 1).sum())}")

    return match


def test_npu_flex_attention_vs_flex(args):
    """Main comparison test: npu_flex_attention vs flex_attention."""
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    total_len = args.num_frames * args.frame_seqlen * 2
    N = args.num_frames * args.frame_seqlen

    print(f"=== Config: {args.num_frames} frames, {args.frame_seqlen} tok/frame, "
          f"{args.num_frame_per_block} frames/block, {args.dtype}, {args.device} ===\n")

    # --- Prepare inputs ---
    q = torch.randn(1, total_len, args.num_heads, args.head_dim, device=device, dtype=dtype)
    k = torch.randn(1, total_len, args.num_heads, args.head_dim, device=device, dtype=dtype)
    v = torch.randn(1, total_len, args.num_heads, args.head_dim, device=device, dtype=dtype)

    # Pad to 128 alignment
    pad = math.ceil(total_len / 128) * 128 - total_len
    p_q = torch.cat([q, torch.zeros(1, pad, args.num_heads, args.head_dim, device=device, dtype=dtype)], dim=1)
    p_k = torch.cat([k, torch.zeros(1, pad, args.num_heads, args.head_dim, device=device, dtype=dtype)], dim=1)
    p_v = torch.cat([v, torch.zeros(1, pad, args.num_heads, args.head_dim, device=device, dtype=dtype)], dim=1)

    # BHLD format
    bhld_q = p_q.transpose(1, 2)
    bhld_k = p_k.transpose(1, 2)
    bhld_v = p_v.transpose(1, 2)

    # BlockMask
    block_mask = prepare_teacher_forcing_mask(device, args.num_frames, args.frame_seqlen, args.num_frame_per_block)

    # --- Test mask generation ---
    print("--- Mask Generation ---")
    full_len = total_len + pad
    mask_ok = test_mask_generation(block_mask, full_len, device)
    print()

    # --- Run flex_attention ---
    print("--- flex_attention ---")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            out_flex = _flex_attention(bhld_q, bhld_k, bhld_v, block_mask=block_mask)
        flex_ok = True
        flex_mem = torch.cuda.max_memory_allocated() / 1024**3 if device.type == 'cuda' else 0
        print(f"  shape={out_flex.shape}, peak_mem={flex_mem:.2f} GB")
    except Exception as e:
        flex_ok = False
        flex_mem = 0
        print(f"  FAILED: {e}")

    # --- Run npu_flex_attention ---
    print("\n--- npu_flex_attention ---")
    try:
        import torch_npu
        has_npu = True
    except ModuleNotFoundError:
        has_npu = False
        print("  torch_npu not available, skipping npu_fusion_attention test")

    npu_ok = False
    npu_mem = 0
    out_npu = None

    if has_npu:
        if device.type == 'npu':
            torch.npu.reset_peak_memory_stats()
        elif device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        try:
            with torch.no_grad():
                out_npu = npu_flex_attention(bhld_q, bhld_k, bhld_v, block_mask=block_mask)
            npu_ok = True
            if device.type == 'npu':
                npu_mem = torch.npu.max_memory_allocated() / 1024**3
            elif device.type == 'cuda':
                npu_mem = torch.cuda.max_memory_allocated() / 1024**3
            else:
                npu_mem = 0
            print(f"  shape={out_npu.shape}, peak_mem={npu_mem:.2f} GB")
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()

    # --- Fallback: validate mask correctness via SDPA when torch_npu unavailable ---
    if not npu_ok:
        print("\n--- SDPA Fallback (validate mask generation) ---")
        print("  torch_npu unavailable — using generated mask with SDPA to verify correctness")
        try:
            mask_mod = block_mask.mask_mod
            q_idx = torch.arange(full_len, device=device).unsqueeze(1)
            kv_idx = torch.arange(full_len, device=device).unsqueeze(0)
            token_mask = mask_mod(0, 0, q_idx, kv_idx)
            # SDPA bool mask: True=attend (same as mask_mod), expand to [B, H, L, L]
            attn_mask = token_mask.unsqueeze(0).unsqueeze(0).expand(bhld_q.shape[0], bhld_q.shape[1], -1, -1)
            with torch.no_grad():
                out_sdpa = F.scaled_dot_product_attention(bhld_q, bhld_k, bhld_v, attn_mask=attn_mask)
            if flex_ok:
                diff = (out_flex[:, :, :total_len].float() - out_sdpa[:, :, :total_len].float()).abs()
                print(f"  SDPA vs flex_attention: max_abs_diff={diff.max().item():.2e}, mean={diff.mean().item():.2e}")
                # bf16 accumulation between flex_attention and SDPA can differ up to ~5e-3
                threshold = 1e-5 if dtype == torch.float32 else 5e-3
                if diff.max().item() < 1e-5:
                    print("  VERDICT: Mask generation is CORRECT (SDPA matches flex_attention)")
                elif diff.max().item() < threshold:
                    print(f"  VERDICT: Mask generation is CORRECT (within {args.dtype} tolerance)")
                else:
                    print("  VERDICT: MISMATCH — mask generation may have issues (try --dtype float32)")
        except Exception as e:
            import traceback
            print(f"  SDPA fallback FAILED: {e}")
            traceback.print_exc()

    # --- Compare ---
    if flex_ok and npu_ok:
        a = out_flex[:, :, :total_len].transpose(1, 2).float()
        b = out_npu[:, :, :total_len].transpose(1, 2).float()
        diff = (a - b).abs()

        print(f"\n--- Comparison ---")
        print(f"  max_abs_diff:  {diff.max().item():.2e}")
        print(f"  mean_abs_diff: {diff.mean().item():.2e}")

        cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0)
        print(f"  cosine_sim:    {cos.item():.8f}")

        if npu_mem > 0 and flex_mem > 0:
            print(f"  memory saved:  {flex_mem - npu_mem:.2f} GB ({(1 - npu_mem/flex_mem)*100:.1f}%)")

        if diff.max().item() < 1e-5:
            print(f"\n  VERDICT: EXACT MATCH")
        elif diff.max().item() < 1e-3:
            print(f"\n  VERDICT: PASS")
        elif diff.max().item() < 5e-3:
            print(f"\n  VERDICT: ACCEPTABLE (bf16 accumulation)")
        else:
            print(f"\n  VERDICT: MISMATCH (try --dtype float32 to diagnose)")
    elif npu_ok and not flex_ok:
        print("\n  flex_attention OOM'd, npu_flex_attention succeeded!")
    elif flex_ok and not npu_ok:
        if not has_npu:
            print("\n  npu_flex_attention: skipped (torch_npu not available on this device)")
        else:
            print("\n  npu_flex_attention failed!")

    # --- Also test blockwise causal mask (non-TF) ---
    if args.test_non_tf:
        print(f"\n{'='*60}")
        print("--- Non-TF (blockwise causal) mask test ---\n")
        total_len_ntf = args.num_frames * args.frame_seqlen
        pad_ntf = math.ceil(total_len_ntf / 128) * 128 - total_len_ntf

        q_ntf = torch.randn(1, total_len_ntf, args.num_heads, args.head_dim, device=device, dtype=dtype)
        k_ntf = torch.randn(1, total_len_ntf, args.num_heads, args.head_dim, device=device, dtype=dtype)
        v_ntf = torch.randn(1, total_len_ntf, args.num_heads, args.head_dim, device=device, dtype=dtype)

        pq_ntf = torch.cat([q_ntf, torch.zeros(1, pad_ntf, args.num_heads, args.head_dim, device=device, dtype=dtype)], dim=1)
        pk_ntf = torch.cat([k_ntf, torch.zeros(1, pad_ntf, args.num_heads, args.head_dim, device=device, dtype=dtype)], dim=1)
        pv_ntf = torch.cat([v_ntf, torch.zeros(1, pad_ntf, args.num_heads, args.head_dim, device=device, dtype=dtype)], dim=1)

        bhld_q_ntf = pq_ntf.transpose(1, 2)
        bhld_k_ntf = pk_ntf.transpose(1, 2)
        bhld_v_ntf = pv_ntf.transpose(1, 2)

        block_mask_ntf = prepare_blockwise_causal_mask(device, args.num_frames, args.frame_seqlen, args.num_frame_per_block)

        test_mask_generation(block_mask_ntf, total_len_ntf + pad_ntf, device)
        print()

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        try:
            with torch.no_grad():
                out_flex_ntf = _flex_attention(bhld_q_ntf, bhld_k_ntf, bhld_v_ntf, block_mask=block_mask_ntf)
            print(f"  flex_attention: shape={out_flex_ntf.shape}")
        except Exception as e:
            print(f"  flex_attention FAILED: {e}")
            out_flex_ntf = None

        if has_npu and out_flex_ntf is not None:
            try:
                with torch.no_grad():
                    out_npu_ntf = npu_flex_attention(bhld_q_ntf, bhld_k_ntf, bhld_v_ntf, block_mask=block_mask_ntf)
                diff_ntf = (out_flex_ntf[:, :, :total_len_ntf].float() - out_npu_ntf[:, :, :total_len_ntf].float()).abs()
                print(f"  npu_flex_attention: shape={out_npu_ntf.shape}")
                print(f"  max_abs_diff: {diff_ntf.max().item():.2e}, mean: {diff_ntf.mean().item():.2e}")
            except Exception as e:
                print(f"  npu_flex_attention FAILED: {e}")


def npu_flex_attention_v2_sdpa_fallback(query, key, value, block_mask=None):
    """
    CUDA/CPU fallback for npu_flex_attention_v2 using SDPA + manual TND decomposition.

    This verifies the TND decomposition logic is correct by computing each sub-sequence
    independently with SDPA. Each sub-sequence uses full bidirectional attention (no mask).
    Reads TF metadata from block_mask._tf_mask_meta (same as npu_flex_attention_v2).
    """
    num_frames, frame_seqlen, num_frame_per_block = block_mask._tf_mask_meta

    B, H, L, D = query.shape
    block_size = frame_seqlen * num_frame_per_block
    num_blocks = num_frames * frame_seqlen // block_size
    N = num_frames * frame_seqlen

    # Convert BNSD [1, H, L, D] to TND [L, H, D] for slicing
    q_tnd = query[0].permute(1, 0, 2)  # [L, H, D]
    k_tnd = key[0].permute(1, 0, 2)
    v_tnd = value[0].permute(1, 0, 2)

    attn_out = torch.zeros(1, L, H, D, device=query.device, dtype=query.dtype)

    for i in range(num_blocks):
        # --- Clean block i ---
        q_start = i * block_size
        q_end = (i + 1) * block_size
        kv_end = (i + 1) * block_size

        seg_q = q_tnd[q_start:q_end].unsqueeze(0)          # [1, block_size, H, D]
        seg_k = k_tnd[:kv_end].unsqueeze(0)                  # [1, kv_len, H, D]
        seg_v = v_tnd[:kv_end].unsqueeze(0)

        # BNSD format for SDPA: [1, H, q_len, D]
        seg_q_bn = seg_q.transpose(1, 2)
        seg_k_bn = seg_k.transpose(1, 2)
        seg_v_bn = seg_v.transpose(1, 2)

        # Full bidirectional attention within this sub-sequence — no mask needed
        with torch.no_grad():
            seg_out = F.scaled_dot_product_attention(seg_q_bn, seg_k_bn, seg_v_bn)
        # seg_out: [1, H, q_len, D] -> [1, q_len, H, D]
        attn_out[0, q_start:q_end] = seg_out[0].permute(1, 0, 2)

    for i in range(num_blocks):
        # --- Noisy block i ---
        q_start = N + i * block_size
        q_end = N + (i + 1) * block_size
        context_len = i * block_size

        seg_q = q_tnd[q_start:q_end].unsqueeze(0)
        if context_len > 0:
            seg_k = torch.cat([k_tnd[:context_len], k_tnd[q_start:q_end]], dim=0).unsqueeze(0)
            seg_v = torch.cat([v_tnd[:context_len], v_tnd[q_start:q_end]], dim=0).unsqueeze(0)
        else:
            seg_k = k_tnd[q_start:q_end].unsqueeze(0)
            seg_v = v_tnd[q_start:q_end].unsqueeze(0)

        seg_q_bn = seg_q.transpose(1, 2)
        seg_k_bn = seg_k.transpose(1, 2)
        seg_v_bn = seg_v.transpose(1, 2)

        # Full bidirectional attention — no mask needed
        with torch.no_grad():
            seg_out = F.scaled_dot_product_attention(seg_q_bn, seg_k_bn, seg_v_bn)
        attn_out[0, q_start:q_end] = seg_out[0].permute(1, 0, 2)

    # Convert [1, L, H, D] -> [1, H, L, D]
    attn_out = attn_out.permute(0, 2, 1, 3)
    return attn_out


def test_npu_flex_attention_v2(args):
    """Test npu_flex_attention_v2 vs flex_attention."""
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    total_len = args.num_frames * args.frame_seqlen * 2
    N = args.num_frames * args.frame_seqlen
    block_size = args.frame_seqlen * args.num_frame_per_block
    num_blocks = N // block_size

    print(f"=== V2 Config: {args.num_frames} frames, {args.frame_seqlen} tok/frame, "
          f"{args.num_frame_per_block} frames/block, {args.dtype}, {args.device} ===")
    print(f"  total_len={total_len}, N={N}, block_size={block_size}, num_blocks={num_blocks}\n")

    # --- Prepare inputs ---
    q = torch.randn(1, total_len, args.num_heads, args.head_dim, device=device, dtype=dtype)
    k = torch.randn(1, total_len, args.num_heads, args.head_dim, device=device, dtype=dtype)
    v = torch.randn(1, total_len, args.num_heads, args.head_dim, device=device, dtype=dtype)

    # Pad to 128 alignment
    pad = math.ceil(total_len / 128) * 128 - total_len
    p_q = torch.cat([q, torch.zeros(1, pad, args.num_heads, args.head_dim, device=device, dtype=dtype)], dim=1)
    p_k = torch.cat([k, torch.zeros(1, pad, args.num_heads, args.head_dim, device=device, dtype=dtype)], dim=1)
    p_v = torch.cat([v, torch.zeros(1, pad, args.num_heads, args.head_dim, device=device, dtype=dtype)], dim=1)

    bhld_q = p_q.transpose(1, 2)
    bhld_k = p_k.transpose(1, 2)
    bhld_v = p_v.transpose(1, 2)

    # BlockMask
    block_mask = prepare_teacher_forcing_mask(device, args.num_frames, args.frame_seqlen, args.num_frame_per_block)

    # --- Run flex_attention (reference) ---
    print("--- flex_attention ---")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            out_flex = _flex_attention(bhld_q, bhld_k, bhld_v, block_mask=block_mask)
        flex_ok = True
        flex_mem = torch.cuda.max_memory_allocated() / 1024**3 if device.type == 'cuda' else 0
        print(f"  shape={out_flex.shape}, peak_mem={flex_mem:.2f} GB")
    except Exception as e:
        flex_ok = False
        flex_mem = 0
        print(f"  FAILED: {e}")

    # --- Run v2 SDPA fallback (validate decomposition) ---
    print("\n--- npu_flex_attention_v2 SDPA fallback ---")
    try:
        with torch.no_grad():
            out_v2_sdpa = npu_flex_attention_v2_sdpa_fallback(
                bhld_q, bhld_k, bhld_v,
                block_mask=block_mask
            )
        v2_sdpa_ok = True
        print(f"  shape={out_v2_sdpa.shape}")
    except Exception as e:
        v2_sdpa_ok = False
        import traceback
        print(f"  FAILED: {e}")
        traceback.print_exc()

    if flex_ok and v2_sdpa_ok:
        a = out_flex[:, :, :total_len].float()
        b = out_v2_sdpa[:, :, :total_len].float()
        diff = (a - b).abs()
        print(f"  max_abs_diff:  {diff.max().item():.2e}")
        print(f"  mean_abs_diff: {diff.mean().item():.2e}")
        cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0)
        print(f"  cosine_sim:    {cos.item():.8f}")

        threshold = 1e-5 if dtype == torch.float32 else 5e-3
        if diff.max().item() < 1e-5:
            print("  VERDICT: EXACT MATCH")
        elif diff.max().item() < threshold:
            print(f"  VERDICT: PASS (within {args.dtype} tolerance)")
        else:
            print("  VERDICT: MISMATCH")

    # --- Run v2 on NPU ---
    print("\n--- npu_flex_attention_v2 (NPU) ---")
    try:
        import torch_npu
        has_npu = True
    except ModuleNotFoundError:
        has_npu = False

    npu_v2_ok = False
    out_v2_npu = None

    if has_npu:
        if device.type == 'npu':
            torch.npu.reset_peak_memory_stats()
        elif device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        try:
            with torch.no_grad():
                out_v2_npu = npu_flex_attention_v2(
                    bhld_q, bhld_k, bhld_v,
                    block_mask=block_mask
                )
            npu_v2_ok = True
            if device.type == 'npu':
                npu_mem = torch.npu.max_memory_allocated() / 1024**3
            elif device.type == 'cuda':
                npu_mem = torch.cuda.max_memory_allocated() / 1024**3
            else:
                npu_mem = 0
            print(f"  shape={out_v2_npu.shape}, peak_mem={npu_mem:.2f} GB")
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
    else:
        print("  torch_npu not available, skipping NPU test")

    # --- Compare v2 NPU vs flex ---
    if flex_ok and npu_v2_ok:
        a = out_flex[:, :, :total_len].float()
        b = out_v2_npu[:, :, :total_len].float()
        diff = (a - b).abs()
        print(f"\n--- Comparison (v2 NPU vs flex) ---")
        print(f"  max_abs_diff:  {diff.max().item():.2e}")
        print(f"  mean_abs_diff: {diff.mean().item():.2e}")
        cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0)
        print(f"  cosine_sim:    {cos.item():.8f}")

        threshold = 1e-5 if dtype == torch.float32 else 5e-3
        if diff.max().item() < 1e-5:
            print("\n  VERDICT: EXACT MATCH")
        elif diff.max().item() < 1e-3:
            print("\n  VERDICT: PASS")
        elif diff.max().item() < threshold:
            print(f"\n  VERDICT: ACCEPTABLE ({args.dtype} accumulation)")
        else:
            print(f"\n  VERDICT: MISMATCH (try --dtype float32)")

    # --- Memory comparison v1 vs v2 ---
    if flex_ok and npu_v2_ok:
        print(f"\n  V2 avoids full [L,L] dense mask (~{total_len**2 / 1024**3:.2f} GB)")
        print(f"  V2 KV duplication overhead: ~{sum((i+1)*2 for i in range(num_blocks)) * block_size * args.num_heads * args.head_dim * 2 / 1024**3:.2f} GB")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_frames', type=int, default=9)
    parser.add_argument('--frame_seqlen', type=int, default=256)
    parser.add_argument('--num_frame_per_block', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--head_dim', type=int, default=128)
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_non_tf', action='store_true',
                        help='Also test with blockwise causal (non-TF) mask')
    parser.add_argument('--test_v2', action='store_true',
                        help='Test npu_flex_attention_v2 (TND + prefix sparse mode)')
    args = parser.parse_args()

    if args.test_v2:
        test_npu_flex_attention_v2(args)
    else:
        test_npu_flex_attention_vs_flex(args)


if __name__ == '__main__':
    main()