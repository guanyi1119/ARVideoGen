"""
Standalone test script for comparing _chunked_tf_attention vs flex_attention + BlockMask.

This script:
1. Creates a CausalWanSelfAttention-like setup with the TF mask
2. Runs both flex_attention (with BlockMask) and _chunked_tf_attention
3. Compares numerical results and GPU memory usage

Usage:
    python tests/test_chunked_tf_attention.py [--num_frames 9] [--frame_seqlen 256] [--num_frame_per_block 3]
    # Use small values for quick local testing
    # Use production values (21, 1560, 3) on GPU with enough memory
"""

import argparse
import math
import os
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention as _flex_attention, BlockMask


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
    """
    Recreate _prepare_teacher_forcing_mask from causal_model.py:578-664
    """
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
        step=attention_block_size, device=device, dtype=torch.long
    )

    for start in frame_indices:
        context_ends[start:start + attention_block_size] = start + attention_block_size

    noisy_image_start_list = torch.arange(
        num_frames * frame_seqlen, total_length,
        step=attention_block_size, device=device, dtype=torch.long
    )
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
        _compile=False, device=device
    )
    return block_mask


def chunked_tf_attention(q, k, v, grid_sizes, freqs, num_frame_per_block):
    """
    Chunk-based attention for teacher forcing.
    Mathematically equivalent to flex_attention with BlockMask,
    but processes one query chunk at a time to reduce peak memory.

    Key design: RoPE is applied to the full clean/noisy halves first (matching
    the original code which applies rope to each half separately), then we slice
    the already-roped tensors into chunks for per-chunk sdpa.

    Args:
        q, k, v: [B, total_len, num_heads, head_dim] where total_len = 2 * N
        grid_sizes: [B, 3] tensor with (F, H, W) per sample
        freqs: RoPE frequencies
        num_frame_per_block: number of frames per causal block
    Returns:
        [B, total_len, num_heads, head_dim]
    """
    frame_seqlen = grid_sizes[0, 1].item() * grid_sizes[0, 2].item()
    chunk_size = frame_seqlen * num_frame_per_block
    N = q.shape[1] // 2  # clean part length
    num_chunks = N // chunk_size

    # Split into clean and noisy halves
    q_clean = q[:, :N]
    q_noisy = q[:, N:]
    k_clean = k[:, :N]
    k_noisy = k[:, N:]
    v_clean = v[:, :N]
    v_noisy = v[:, N:]

    # Apply RoPE to full halves first (matching original code's split-then-rope pattern)
    # This avoids the shape mismatch from applying rope to short chunks
    roped_q_clean = rope_apply_ref(q_clean, grid_sizes, freqs).type_as(v)
    roped_k_clean = rope_apply_ref(k_clean, grid_sizes, freqs).type_as(v)
    roped_q_noisy = rope_apply_ref(q_noisy, grid_sizes, freqs).type_as(v)
    roped_k_noisy = rope_apply_ref(k_noisy, grid_sizes, freqs).type_as(v)

    outputs = []

    # Process clean chunks
    for chunk_idx in range(num_chunks):
        q_start = chunk_idx * chunk_size
        q_end = q_start + chunk_size

        roped_q_chunk = roped_q_clean[:, q_start:q_end]
        # KV: clean tokens [0, q_end) - block causal within clean part
        roped_k_slice = roped_k_clean[:, :q_end]
        v_slice = v_clean[:, :q_end]

        # Standard causal pattern (lower triangular) within [0, q_end)
        attn_out = F.scaled_dot_product_attention(
            roped_q_chunk.transpose(1, 2), roped_k_slice.transpose(1, 2), v_slice.transpose(1, 2),
            attn_mask=None, is_causal=True
        ).transpose(1, 2)
        outputs.append(attn_out)

    # Process noisy chunks
    for chunk_idx in range(num_chunks):
        q_start = chunk_idx * chunk_size
        q_end = q_start + chunk_size

        roped_q_chunk = roped_q_noisy[:, q_start:q_end]

        # KV range for noisy chunk k:
        # - clean tokens [0, chunk_idx * chunk_size)
        # - noisy tokens in same block [N + chunk_idx * chunk_size, N + (chunk_idx+1) * chunk_size)
        clean_kv_end = chunk_idx * chunk_size
        noisy_kv_start_in_noisy = chunk_idx * chunk_size
        noisy_kv_end_in_noisy = (chunk_idx + 1) * chunk_size

        roped_k_clean_slice = roped_k_clean[:, :clean_kv_end]
        v_clean_slice = v_clean[:, :clean_kv_end]
        roped_k_noisy_slice = roped_k_noisy[:, noisy_kv_start_in_noisy:noisy_kv_end_in_noisy]
        v_noisy_slice = v_noisy[:, noisy_kv_start_in_noisy:noisy_kv_end_in_noisy]

        roped_k_combined = torch.cat([roped_k_clean_slice, roped_k_noisy_slice], dim=1)
        v_combined = torch.cat([v_clean_slice, v_noisy_slice], dim=1)

        # No causal mask needed - we've already selected only allowed KV tokens
        attn_out = F.scaled_dot_product_attention(
            roped_q_chunk.transpose(1, 2), roped_k_combined.transpose(1, 2), v_combined.transpose(1, 2),
            attn_mask=None, is_causal=False
        ).transpose(1, 2)
        outputs.append(attn_out)

    return torch.cat(outputs, dim=1)


def flex_attention_baseline(q, k, v, grid_sizes, freqs, block_mask):
    """
    Baseline using flex_attention with BlockMask (matching causal_model.py:132-175)
    """
    N = q.shape[1] // 2

    q_chunk = torch.chunk(q, 2, dim=1)
    k_chunk = torch.chunk(k, 2, dim=1)
    roped_query = []
    roped_key = []
    for ii in range(2):
        rq = rope_apply_ref(q_chunk[ii], grid_sizes, freqs).type_as(v)
        rk = rope_apply_ref(k_chunk[ii], grid_sizes, freqs).type_as(v)
        roped_query.append(rq)
        roped_key.append(rk)

    roped_query = torch.cat(roped_query, dim=1)
    roped_key = torch.cat(roped_key, dim=1)

    # Padding to 128 alignment
    padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
    padded_roped_query = torch.cat([
        roped_query,
        torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                     device=q.device, dtype=v.dtype)
    ], dim=1)
    padded_roped_key = torch.cat([
        roped_key,
        torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                     device=k.device, dtype=v.dtype)
    ], dim=1)
    padded_v = torch.cat([
        v,
        torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                     device=v.device, dtype=v.dtype)
    ], dim=1)

    x = _flex_attention(
        query=padded_roped_query.transpose(2, 1),
        key=padded_roped_key.transpose(2, 1),
        value=padded_v.transpose(2, 1),
        block_mask=block_mask
    )[:, :, :-padded_length].transpose(2, 1)

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_frames', type=int, default=9,
                        help='Number of frames (use 9 for quick test, 21 for production)')
    parser.add_argument('--frame_seqlen', type=int, default=256,
                        help='Tokens per frame (use 256 for quick test, 1560 for production)')
    parser.add_argument('--num_frame_per_block', type=int, default=3,
                        help='Frames per causal block')
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

    print(f"=== Configuration ===")
    print(f"  num_frames={args.num_frames}, frame_seqlen={args.frame_seqlen}")
    print(f"  num_frame_per_block={args.num_frame_per_block}")
    print(f"  total_len={total_len} (clean={N}, noisy={N})")
    print(f"  num_heads={args.num_heads}, head_dim={args.head_dim}")
    print(f"  dtype={args.dtype}, device={args.device}")
    print()

    # Create input tensors
    q = torch.randn(1, total_len, args.num_heads, args.head_dim, device=device, dtype=dtype)
    k = torch.randn(1, total_len, args.num_heads, args.head_dim, device=device, dtype=dtype)
    v = torch.randn(1, total_len, args.num_heads, args.head_dim, device=device, dtype=dtype)

    # grid_sizes: [B, 3] = (F, H, W)
    h = int(math.sqrt(args.frame_seqlen))
    while args.frame_seqlen % h != 0:
        h -= 1
    w = args.frame_seqlen // h
    grid_sizes = torch.tensor([[args.num_frames, h, w]], dtype=torch.long)

    # Create RoPE frequencies on the same device as q/k/v
    max_seq_len = max(args.num_frames, h, w) + 100
    head_dim = args.head_dim
    dim = head_dim // 2
    t_dim = dim - 2 * (dim // 3)
    hw_dim = dim // 3

    freqs_t = torch.randn(max_seq_len, t_dim, dtype=torch.float32, device=device)
    freqs_h = torch.randn(max_seq_len, hw_dim, dtype=torch.float32, device=device)
    freqs_w = torch.randn(max_seq_len, hw_dim, dtype=torch.float32, device=device)
    freqs = torch.cat([freqs_t, freqs_h, freqs_w], dim=1)

    # Create BlockMask
    print("Creating BlockMask...")
    block_mask = prepare_teacher_forcing_mask(
        device, args.num_frames, args.frame_seqlen, args.num_frame_per_block
    )
    print(f"  BlockMask: {block_mask}")
    print()

    # --- Test flex_attention (baseline) ---
    print("--- Running flex_attention (baseline) ---")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    try:
        with torch.no_grad():
            out_flex = flex_attention_baseline(q, k, v, grid_sizes, freqs, block_mask)
        flex_success = True
        if device.type == 'cuda':
            flex_peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  Output shape: {out_flex.shape}")
            print(f"  Peak GPU memory: {flex_peak_mem:.2f} GB")
        else:
            print(f"  Output shape: {out_flex.shape}")
            print(f"  Peak GPU memory: N/A (non-CUDA device)")
    except Exception as e:
        flex_success = False
        print(f"  FAILED: {e}")
    print()

    # --- Test chunked_tf_attention ---
    print("--- Running chunked_tf_attention ---")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    try:
        with torch.no_grad():
            out_chunked = chunked_tf_attention(
                q, k, v, grid_sizes, freqs, args.num_frame_per_block
            )
        chunked_success = True
        if device.type == 'cuda':
            chunked_peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  Output shape: {out_chunked.shape}")
            print(f"  Peak GPU memory: {chunked_peak_mem:.2f} GB")
        else:
            print(f"  Output shape: {out_chunked.shape}")
            print(f"  Peak GPU memory: N/A (non-CUDA device)")
    except Exception as e:
        chunked_success = False
        print(f"  FAILED: {e}")
    print()

    # --- Compare results ---
    if flex_success and chunked_success:
        print("--- Numerical Comparison ---")
        out_flex_f = out_flex.float()
        out_chunked_f = out_chunked.float()

        abs_diff = (out_flex_f - out_chunked_f).abs()
        rel_diff = abs_diff / (out_flex_f.abs().clamp(min=1e-6))

        print(f"  Max absolute difference: {abs_diff.max().item():.2e}")
        print(f"  Mean absolute difference: {abs_diff.mean().item():.2e}")
        print(f"  Max relative difference: {rel_diff.max().item():.2e}")
        print(f"  Mean relative difference: {rel_diff.mean().item():.2e}")

        # Per-chunk comparison
        chunk_size = args.frame_seqlen * args.num_frame_per_block
        num_chunks = N // chunk_size

        print(f"\n  Per-chunk analysis:")
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = start + chunk_size
            diff_clean = (out_flex_f[:, start:end] - out_chunked_f[:, start:end]).abs()
            print(f"    Clean chunk {chunk_idx}: max_diff={diff_clean.max().item():.2e}, mean_diff={diff_clean.mean().item():.2e}")

        for chunk_idx in range(num_chunks):
            start = N + chunk_idx * chunk_size
            end = N + (chunk_idx + 1) * chunk_size
            diff_noisy = (out_flex_f[:, start:end] - out_chunked_f[:, start:end]).abs()
            print(f"    Noisy chunk {chunk_idx}: max_diff={diff_noisy.max().item():.2e}, mean_diff={diff_noisy.mean().item():.2e}")

        # Cosine similarity
        cos_sim = F.cosine_similarity(
            out_flex_f.flatten(), out_chunked_f.flatten(), dim=0
        )
        print(f"\n  Cosine similarity: {cos_sim.item():.8f}")

        # Verdict
        if abs_diff.max().item() < 1e-3:
            print(f"\n  VERDICT: PASS (max abs diff < 1e-3)")
        elif abs_diff.max().item() < 1e-2:
            print(f"\n  VERDICT: ACCEPTABLE (max abs diff < 1e-2, likely due to bf16 precision)")
        else:
            print(f"\n  VERDICT: MISMATCH (max abs diff >= 1e-2, needs investigation)")

        if device.type == 'cuda':
            print(f"\n  Memory savings: {flex_peak_mem - chunked_peak_mem:.2f} GB ({(1 - chunked_peak_mem/flex_peak_mem)*100:.1f}% reduction)")

    elif chunked_success and not flex_success:
        print("  flex_attention failed but chunked_tf_attention succeeded!")
        print("  This suggests chunked_tf_attention is more memory-efficient.")
    elif flex_success and not chunked_success:
        print("  chunked_tf_attention failed but flex_attention succeeded!")
        print("  This suggests a bug in the chunked implementation.")
    else:
        print("  Both methods failed!")


if __name__ == '__main__':
    main()
