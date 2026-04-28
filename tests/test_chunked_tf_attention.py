"""
Standalone test script for comparing chunked_flex_attention vs flex_attention + BlockMask.

Usage:
    # Quick test (small shapes, bf16)
    python tests/test_chunked_tf_attention.py --num_frames 9 --frame_seqlen 256 --num_frame_per_block 3

    # Float32 mode to check if differences are precision-related
    python tests/test_chunked_tf_attention.py --num_frames 9 --frame_seqlen 256 --num_frame_per_block 3 --dtype float32

    # Production scale
    python tests/test_chunked_tf_attention.py --num_frames 21 --frame_seqlen 1560 --num_frame_per_block 3
"""

import argparse
import math
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention as _flex_attention


def rope_apply_ref(x, grid_sizes, freqs):
    """Reference rope_apply matching wan/modules/model.py:rope_apply"""
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)


def prepare_teacher_forcing_mask(device, num_frames, frame_seqlen, num_frame_per_block):
    """Recreate _prepare_teacher_forcing_mask from causal_model.py:578-664"""
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
    return block_mask


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
    B, H, L, D = query.shape
    Q_BLOCK_SIZE, KV_BLOCK_SIZE = block_mask.BLOCK_SIZE
    mask_mod = block_mask.mask_mod
    num_q_blocks = L // Q_BLOCK_SIZE + (1 if L % Q_BLOCK_SIZE else 0)
    num_kv_blocks = L // KV_BLOCK_SIZE + (1 if L % KV_BLOCK_SIZE else 0)

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
    qs = q_starts.unsqueeze(1)   # [nq, 1]
    qe = q_ends.unsqueeze(1)     # [nq, 1]
    kss = kv_starts.unsqueeze(0) # [1, nkv]
    ke = kv_ends.unsqueeze(0)    # [1, nkv]

    block_allowed = (
        mask_mod(0, 0, qs, kss) |
        mask_mod(0, 0, qs, ke) |
        mask_mod(0, 0, qe, kss) |
        mask_mod(0, 0, qe, ke)
    )  # [num_q_blocks, num_kv_blocks]

    output = torch.zeros_like(query)

    for q_bi in range(num_q_blocks):
        q_start = q_bi * Q_BLOCK_SIZE
        q_end = min(q_start + Q_BLOCK_SIZE, L)
        if q_start >= L:
            break

        # Get allowed KV blocks for this Q block
        allowed_kv = block_allowed[q_bi].nonzero().squeeze(-1)
        if len(allowed_kv) == 0:
            continue

        # Gather KV for all allowed blocks
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

        # Build fine-grained mask via mask_mod with broadcasting
        q_indices = torch.arange(q_start, q_end, device=query.device)
        q_2d = q_indices.unsqueeze(1)    # [q_size, 1]
        kv_2d = kv_indices.unsqueeze(0)  # [1, kv_gathered_size]
        fine_mask = mask_mod(0, 0, q_2d, kv_2d)  # [q_size, kv_gathered_size]

        # Invert mask: mask_mod uses True=allowed, SDPA uses True=masked_out
        q_chunk = query[:, :, q_start:q_end]
        attn_mask = (~fine_mask).unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)

        output[:, :, q_start:q_end] = F.scaled_dot_product_attention(
            q_chunk, k_gathered, v_gathered, attn_mask=attn_mask)

    return output


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
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    total_len = args.num_frames * args.frame_seqlen * 2
    N = args.num_frames * args.frame_seqlen

    print(f"=== Config: {args.num_frames} frames, {args.frame_seqlen} tok/frame, "
          f"{args.num_frame_per_block} frames/block, {args.dtype} ===\n")

    # --- Prepare common inputs ---
    q = torch.randn(1, total_len, args.num_heads, args.head_dim, device=device, dtype=dtype)
    k = torch.randn(1, total_len, args.num_heads, args.head_dim, device=device, dtype=dtype)
    v = torch.randn(1, total_len, args.num_heads, args.head_dim, device=device, dtype=dtype)

    h = int(math.sqrt(args.frame_seqlen))
    while args.frame_seqlen % h != 0:
        h -= 1
    w = args.frame_seqlen // h
    grid_sizes = torch.tensor([[args.num_frames, h, w]], dtype=torch.long)

    max_seq_len = max(args.num_frames, h, w) + 100
    dim = args.head_dim // 2
    t_dim, hw_dim = dim - 2 * (dim // 3), dim // 3
    freqs = torch.cat([
        torch.randn(max_seq_len, t_dim, device=device),
        torch.randn(max_seq_len, hw_dim, device=device),
        torch.randn(max_seq_len, hw_dim, device=device),
    ], dim=1)

    # Apply RoPE (same as causal_model.py:136-148)
    q_c, q_n = torch.chunk(q, 2, dim=1)
    k_c, k_n = torch.chunk(k, 2, dim=1)
    roped_q = torch.cat([rope_apply_ref(q_c, grid_sizes, freqs).type_as(v),
                         rope_apply_ref(q_n, grid_sizes, freqs).type_as(v)], dim=1)
    roped_k = torch.cat([rope_apply_ref(k_c, grid_sizes, freqs).type_as(v),
                         rope_apply_ref(k_n, grid_sizes, freqs).type_as(v)], dim=1)

    # Pad to 128 alignment (same as causal_model.py:150-168)
    pad = math.ceil(total_len / 128) * 128 - total_len
    p_q = torch.cat([roped_q, torch.zeros(1, pad, args.num_heads, args.head_dim, device=device, dtype=dtype)], dim=1)
    p_k = torch.cat([roped_k, torch.zeros(1, pad, args.num_heads, args.head_dim, device=device, dtype=dtype)], dim=1)
    p_v = torch.cat([v, torch.zeros(1, pad, args.num_heads, args.head_dim, device=device, dtype=dtype)], dim=1)

    # BHLD format for both functions
    bhld_q = p_q.transpose(1, 2)
    bhld_k = p_k.transpose(1, 2)
    bhld_v = p_v.transpose(1, 2)

    # BlockMask
    block_mask = prepare_teacher_forcing_mask(device, args.num_frames, args.frame_seqlen, args.num_frame_per_block)

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
        print(f"  FAILED: {e}")

    # --- Run chunked_flex_attention ---
    print("--- chunked_flex_attention ---")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            out_chunked = chunked_flex_attention(bhld_q, bhld_k, bhld_v, block_mask=block_mask)
        chunked_ok = True
        chunked_mem = torch.cuda.max_memory_allocated() / 1024**3 if device.type == 'cuda' else 0
        print(f"  shape={out_chunked.shape}, peak_mem={chunked_mem:.2f} GB")
    except Exception as e:
        chunked_ok = False
        import traceback
        print(f"  FAILED: {e}")
        traceback.print_exc()

    # --- Compare ---
    if flex_ok and chunked_ok:
        a = out_flex[:, :, :total_len].transpose(1, 2).float()
        b = out_chunked[:, :, :total_len].transpose(1, 2).float()
        diff = (a - b).abs()

        print(f"\n--- Comparison ---")
        print(f"  max_abs_diff:  {diff.max().item():.2e}")
        print(f"  mean_abs_diff: {diff.mean().item():.2e}")

        cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0)
        print(f"  cosine_sim:    {cos.item():.8f}")

        if device.type == 'cuda':
            print(f"  memory saved:  {flex_mem - chunked_mem:.2f} GB ({(1 - chunked_mem/flex_mem)*100:.1f}%)")

        if diff.max().item() < 1e-5:
            print(f"\n  VERDICT: EXACT MATCH")
        elif diff.max().item() < 1e-3:
            print(f"\n  VERDICT: PASS")
        elif diff.max().item() < 5e-3:
            print(f"\n  VERDICT: ACCEPTABLE (bf16 accumulation)")
        else:
            print(f"\n  VERDICT: MISMATCH (try --dtype float32 to diagnose)")
    elif chunked_ok and not flex_ok:
        print("\n  flex_attention OOM'd, chunked_flex_attention succeeded!")
    elif flex_ok and not chunked_ok:
        print("\n  chunked_flex_attention failed!")


if __name__ == '__main__':
    main()
