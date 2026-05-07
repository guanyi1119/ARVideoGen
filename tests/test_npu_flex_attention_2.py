"""
精度测试脚本：验证 FlexAttentionNPU 与 flex_attention 行为一致。

策略：
    1. 在 NPU 上运行 FlexAttentionNPU，与「NPU 上的 reference SDPA」对比
    2. 在 CUDA 上运行 flex_attention（如果有 GPU），与「CUDA 上的 reference SDPA」对比
    3. 两边的 reference SDPA 用同样的代码 + 同一份 mask_mod，等价 → 间接证明两个实现一致
    4. 同时打印命中的 sparse_mode，校验派发器选择正确

运行方式：
    # 仅 NPU 端测试
    python test_flex_attention_npu.py --device npu
    # 仅 GPU 端测试（用于核对 reference 与 flex_attention 一致）
    python test_flex_attention_npu.py --device cuda
"""

import argparse
import math
from typing import Callable, Optional, Tuple

import torch

# 假设上一轮的实现保存在 flex_attention_npu.py 中
# from flex_attention_npu import FlexAttentionNPU


# ============================================================
# 1. 参考实现：两端通用的朴素 SDPA + 显式 bool mask
# ============================================================

def reference_attention(
    query: torch.Tensor,         # (B, H, S_q, D)
    key:   torch.Tensor,         # (B, H_kv, S_kv, D)
    value: torch.Tensor,         # (B, H_kv, S_kv, D)
    mask_mod: Optional[Callable],
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    朴素 SDPA 实现。在 fp32 中累加以减小数值误差，最后再转回输入 dtype。
    输出与 flex_attention / FlexAttentionNPU 严格对齐。
    """
    B, H, S_q, D = query.shape
    H_kv, S_kv = key.shape[1], key.shape[2]
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    in_dtype = query.dtype

    # GQA 自动 broadcast
    if H_kv != H:
        assert H % H_kv == 0
        repeat = H // H_kv
        key   = key.repeat_interleave(repeat,   dim=1)
        value = value.repeat_interleave(repeat, dim=1)

    q_f = query.float()
    k_f = key.float()
    v_f = value.float()

    scores = torch.matmul(q_f, k_f.transpose(-1, -2)) * scale   # (B, H, S_q, S_kv)

    if mask_mod is not None:
        device = query.device
        b_idx = torch.tensor(0, device=device)
        h_idx = torch.tensor(0, device=device)
        q_idx  = torch.arange(S_q,  device=device).view(-1, 1)
        kv_idx = torch.arange(S_kv, device=device).view(1, -1)
        keep = mask_mod(b_idx, h_idx, q_idx, kv_idx)            # (S_q, S_kv) bool
        scores = scores.masked_fill(~keep, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    # 整行 -inf 的位置 softmax 会得到 nan，置 0
    probs = torch.nan_to_num(probs, nan=0.0)
    out = torch.matmul(probs, v_f).to(in_dtype)
    return out


# ============================================================
# 2. 各种 mask_mod 测试用例
# ============================================================

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def make_sliding_window(window_size: int):
    def fn(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= window_size)
    fn.__name__ = f"sliding_window_{window_size}"
    return fn

def make_prefix_lm(prefix_len: int):
    def fn(b, h, q_idx, kv_idx):
        return (kv_idx < prefix_len) | (q_idx >= kv_idx)
    fn.__name__ = f"prefix_lm_{prefix_len}"
    return fn

def make_chunked_causal_with_sink(chunk_size: int, sink_size: int):
    def fn(b, h, q_idx, kv_idx):
        is_sink = kv_idx < sink_size
        q_chunk  = (q_idx  - sink_size).clamp(min=0) // chunk_size
        kv_chunk = (kv_idx - sink_size).clamp(min=0) // chunk_size
        non_sink = (q_idx >= sink_size) & (kv_idx >= sink_size)
        return is_sink | (non_sink & (kv_chunk <= q_chunk))
    fn.__name__ = f"chunked_sink_{chunk_size}_{sink_size}"
    return fn

def no_mask(b, h, q_idx, kv_idx):
    return torch.ones_like(q_idx + kv_idx, dtype=torch.bool)


TEST_CASES = [
    # (name, mask_mod, expected_sparse_mode)  expected=None 表示不强制
    ("no_mask",            None,                                  0),
    ("causal",             causal_mask,                           3),
    ("sliding_window_512", make_sliding_window(512),              4),
    ("prefix_lm_256",      make_prefix_lm(256),                   5),
    ("chunked_sink_64_4",  make_chunked_causal_with_sink(64, 4),  1),  # 兜底
]


# ============================================================
# 3. NPU 端测试
# ============================================================

def run_npu_tests(B, H, S, D, dtype, atol, rtol, seed=42):
    import torch_npu  # noqa
    from flex_attention_npu import FlexAttentionNPU

    print("=" * 70)
    print(f"[NPU] B={B} H={H} S={S} D={D} dtype={dtype}")
    print("=" * 70)

    g = torch.Generator(device="cpu").manual_seed(seed)
    q_cpu = torch.randn(B, H, S, D, generator=g, dtype=torch.float32)
    k_cpu = torch.randn(B, H, S, D, generator=g, dtype=torch.float32)
    v_cpu = torch.randn(B, H, S, D, generator=g, dtype=torch.float32)

    q = q_cpu.to("npu").to(dtype)
    k = k_cpu.to("npu").to(dtype)
    v = v_cpu.to("npu").to(dtype)

    pass_count = 0
    for name, mask_mod, expected_sparse in TEST_CASES:
        attn = FlexAttentionNPU(mask_mod=mask_mod, verbose=False)
        out_npu = attn(q, k, v)
        plan = attn._get_plan(S, S, q.device)
        out_ref = reference_attention(q, k, v, mask_mod)

        diff = (out_npu.float() - out_ref.float()).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        denom = out_ref.float().abs().clamp(min=1e-6)
        max_rel = (diff / denom).max().item()

        sparse_ok = (expected_sparse is None) or (plan.sparse_mode == expected_sparse)
        prec_ok   = (max_abs <= atol) and (max_rel <= rtol)
        ok = sparse_ok and prec_ok
        pass_count += int(ok)

        flag = "✓" if ok else "✗"
        print(f"  {flag} {name:24s}  "
              f"sparse_mode={plan.sparse_mode}(expect {expected_sparse})  "
              f"max_abs={max_abs:.2e}  mean_abs={mean_abs:.2e}  "
              f"max_rel={max_rel:.2e}")
        if not sparse_ok:
            print(f"    └─ 派发器选错了 sparse_mode！pattern={plan.pattern_name}")
        if not prec_ok:
            print(f"    └─ 精度超标：atol={atol}, rtol={rtol}")

    print(f"\n[NPU] {pass_count}/{len(TEST_CASES)} 用例通过")
    return pass_count == len(TEST_CASES)


# ============================================================
# 4. CUDA 端测试（验证 reference 与 flex_attention 等价）
# ============================================================

def run_cuda_tests(B, H, S, D, dtype, atol, rtol, seed=42):
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    print("=" * 70)
    print(f"[CUDA] B={B} H={H} S={S} D={D} dtype={dtype}  "
          f"(用于校验 reference_attention 与 flex_attention 等价)")
    print("=" * 70)

    flex_compiled = torch.compile(flex_attention, dynamic=False)

    g = torch.Generator(device="cpu").manual_seed(seed)
    q_cpu = torch.randn(B, H, S, D, generator=g, dtype=torch.float32)
    k_cpu = torch.randn(B, H, S, D, generator=g, dtype=torch.float32)
    v_cpu = torch.randn(B, H, S, D, generator=g, dtype=torch.float32)

    q = q_cpu.to("cuda").to(dtype)
    k = k_cpu.to("cuda").to(dtype)
    v = v_cpu.to("cuda").to(dtype)

    pass_count = 0
    for name, mask_mod, _ in TEST_CASES:
        out_ref = reference_attention(q, k, v, mask_mod)

        if mask_mod is None:
            # flex_attention 的无 mask 场景：传 None 即可
            out_flex = flex_compiled(q, k, v)
        else:
            block_mask = create_block_mask(
                mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S,
                device="cuda",
            )
            out_flex = flex_compiled(q, k, v, block_mask=block_mask)

        diff = (out_flex.float() - out_ref.float()).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        denom = out_ref.float().abs().clamp(min=1e-6)
        max_rel = (diff / denom).max().item()
        ok = (max_abs <= atol) and (max_rel <= rtol)
        pass_count += int(ok)

        flag = "✓" if ok else "✗"
        print(f"  {flag} {name:24s}  "
              f"max_abs={max_abs:.2e}  mean_abs={mean_abs:.2e}  "
              f"max_rel={max_rel:.2e}")

    print(f"\n[CUDA] {pass_count}/{len(TEST_CASES)} 用例通过")
    return pass_count == len(TEST_CASES)


# ============================================================
# 5. 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["npu", "cuda", "both"], default="npu")
    parser.add_argument("--B",      type=int, default=2)
    parser.add_argument("--H",      type=int, default=8)
    parser.add_argument("--S",      type=int, default=2048)
    parser.add_argument("--D",      type=int, default=128)
    parser.add_argument("--dtype",  choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--atol",   type=float, default=5e-3)
    parser.add_argument("--rtol",   type=float, default=5e-2)
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    all_ok = True
    if args.device in ("npu", "both"):
        all_ok &= run_npu_tests(args.B, args.H, args.S, args.D, dtype,
                                args.atol, args.rtol)
    if args.device in ("cuda", "both"):
        all_ok &= run_cuda_tests(args.B, args.H, args.S, args.D, dtype,
                                 args.atol, args.rtol)

    print("\n" + ("✅ 全部通过" if all_ok else "❌ 有用例失败"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
