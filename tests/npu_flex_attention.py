import torch
try:
    import torch_npu
except:
    pass
from typing import Callable, Optional, Tuple, Union, List
from dataclasses import dataclass, field


@dataclass
class DispatchPlan:
    """描述一次 npu_fusion_attention 调用所需的全部参数。"""
    sparse_mode: int
    atten_mask: Optional[torch.Tensor] = None
    pre_tockens: int = 2147483647
    next_tockens: int = 2147483647
    prefix: Optional[List[int]] = None
    pattern_name: str = "custom"     # 仅用于调试日志


# ---------- Step 1：把 mask_mod 物化为粗粒度块级矩阵 ----------

def _materialize_block_mask(
    mask_mod: Callable,
    S_q: int, S_kv: int,
    block_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    模仿 flex_attention 的 create_block_mask：
        在 (S_q/block_size) x (S_kv/block_size) 的粗网格上对每个 block 的四个角采样，
        判断它是 full / partial / empty。

    返回:
        block_status: (Bq, Bkv) int tensor，0=empty, 1=partial, 2=full
        full_mask:    (S_q, S_kv) bool，True=屏蔽（备用，用于 sparse_mode=1 兜底）
    """
    Bq, Bkv = (S_q + block_size - 1) // block_size, (S_kv + block_size - 1) // block_size

    q_idx = torch.arange(S_q, device=device).view(-1, 1)
    kv_idx = torch.arange(S_kv, device=device).view(1, -1)
    b0 = torch.tensor(0, device=device)
    h0 = torch.tensor(0, device=device)

    # 物化完整 mask（True=屏蔽，符合 npu 约定）
    keep = mask_mod(b0, h0, q_idx, kv_idx)
    full_mask = ~keep

    # block 状态：每个 block 内 keep 的最小/最大值
    keep_blocks = keep[: Bq * block_size, : Bkv * block_size] \
        .reshape(Bq, block_size, Bkv, block_size).permute(0, 2, 1, 3) \
        .reshape(Bq, Bkv, -1)
    has_kept    = keep_blocks.any(dim=-1)        # 块内有 True
    all_kept    = keep_blocks.all(dim=-1)        # 块内全 True
    block_status = has_kept.int() + all_kept.int()   # 0/1/2

    return block_status, full_mask


# ---------- Step 2：在块级矩阵上做模板识别 ----------

def _detect_pattern(
    block_status: torch.Tensor,
    full_mask: torch.Tensor,
    S_q: int, S_kv: int,
) -> DispatchPlan:
    """
    依次尝试把 block_status 匹配到 npu_fusion_attention 支持的几种模板。
    匹配失败则回退到 sparse_mode=1。
    """
    Bq, Bkv = block_status.shape
    device = full_mask.device
    is_kept_block  = (block_status == 2)   # 整块保留
    is_empty_block = (block_status == 0)   # 整块屏蔽
    is_partial     = (block_status == 1)   # 部分保留

    # ---- 模板 0：无 mask（所有块都 full） ----
    if is_kept_block.all():
        return DispatchPlan(sparse_mode=0, atten_mask=None,
                            pattern_name="no_mask")

    # ---- 模板 3：rightDownCausal ----
    # 判定：右下对齐的下三角（含对角线主块）保留，对角线之上全屏蔽
    offset_b = (S_kv - S_q) // (S_q // Bq if Bq else 1)   # 块级 offset
    rd_ref_full   = torch.zeros(Bq, Bkv, dtype=torch.bool, device=device)
    rd_ref_empty  = torch.zeros(Bq, Bkv, dtype=torch.bool, device=device)
    for i in range(Bq):
        # block 行 i 对应的最后一个允许的 kv block 索引
        last_kept_block = i + (Bkv - Bq)
        if last_kept_block >= 0:
            rd_ref_full[i, : last_kept_block]    = True   # 严格小于：full
            rd_ref_empty[i, last_kept_block + 1:] = True  # 严格大于：empty
            # 对角线那一块允许是 partial
    rd_match = (
        (is_kept_block  | ~rd_ref_full).all() and
        (is_empty_block | ~rd_ref_empty).all()
    )
    if rd_match:
        compact = torch.triu(
            torch.ones(2048, 2048, dtype=torch.bool, device=device),
            diagonal=1,
        )
        return DispatchPlan(sparse_mode=3, atten_mask=compact,
                            pattern_name="rightDownCausal")

    # ---- 模板 4：band（滑窗） ----
    # 判定：每个 Q block 行 i 的保留区间 [L_i, R_i] 满足
    #   L_i = i + offset - pre_blocks,  R_i = i + offset + next_blocks
    # 对所有 i 都成立（同一对常数 pre_blocks / next_blocks）
    pre_blocks, next_blocks, ok = None, None, True
    offset_b = Bkv - Bq
    for i in range(Bq):
        row = is_kept_block[i] | is_partial[i]
        if not row.any():
            continue
        L = int(row.float().argmax().item())              # 第一个保留块
        R = int(Bkv - 1 - row.flip(0).float().argmax().item())  # 最后一个保留块
        cur_pre  = (i + offset_b) - L
        cur_next = R - (i + offset_b)
        if pre_blocks is None:
            pre_blocks, next_blocks = cur_pre, cur_next
        elif (cur_pre, cur_next) != (pre_blocks, next_blocks):
            ok = False
            break
    if ok and pre_blocks is not None:
        block_size = (S_q + Bq - 1) // Bq
        compact = torch.triu(
            torch.ones(2048, 2048, dtype=torch.bool, device=device),
            diagonal=1,
        )
        return DispatchPlan(
            sparse_mode=4, atten_mask=compact,
            pre_tockens=pre_blocks * block_size,
            next_tockens=next_blocks * block_size,
            pattern_name="band",
        )

    # ---- 模板 5：prefix LM（causal + 左侧矩形） ----
    # 判定：存在 prefix_len，使得 mask = rightDownCausal AND (kv_idx >= prefix_len)
    # 即第 0 行（q_idx=0）保留 [0, prefix_len) ∪ {0..offset}（如有）
    # 简化检查：第 0 行的保留区间起点是 0，终点超过对角线位置
    if is_kept_block[0, 0]:
        # 找 q=0 这一行最右的保留块
        row0 = is_kept_block[0] | is_partial[0]
        R0 = int(Bkv - 1 - row0.flip(0).float().argmax().item())
        prefix_blocks_candidate = R0 + 1
        # 验证：所有 i 行都保留 [0, prefix_blocks_candidate) 且其余按 causal
        valid = True
        for i in range(Bq):
            last_kept_block = i + (Bkv - Bq)
            expected_R = max(last_kept_block, prefix_blocks_candidate - 1)
            row = is_kept_block[i] | is_partial[i]
            R = int(Bkv - 1 - row.flip(0).float().argmax().item()) if row.any() else -1
            if R != expected_R or not row[: prefix_blocks_candidate].all():
                valid = False
                break
        if valid:
            block_size = (S_q + Bq - 1) // Bq
            prefix_len = prefix_blocks_candidate * block_size
            return DispatchPlan(
                sparse_mode=5,
                atten_mask=full_mask.unsqueeze(0).unsqueeze(0),  # 5 需要 BNSS/B1SS
                prefix=[prefix_len],
                pattern_name="prefixLM",
            )

    # ---- 兜底：sparse_mode=1 ----
    return DispatchPlan(
        sparse_mode=1,
        atten_mask=full_mask.unsqueeze(0).unsqueeze(0),    # (1,1,S_q,S_kv)
        pattern_name="custom_allMask",
    )


# ---------- Step 3：缓存 + 派发 ----------

class FlexAttentionNPU:
    """
    模仿 flex_attention 的接口：传入一个 mask_mod，内部缓存 DispatchPlan，
    每次前向自动选最优的 sparse_mode 调 npu_fusion_attention。
    """
    def __init__(self,
                 mask_mod: Optional[Callable] = None,
                 block_size: int = 128,
                 verbose: bool = False):
        self.mask_mod = mask_mod
        self.block_size = block_size
        self.verbose = verbose
        self._plan_cache: dict = {}

    def _get_plan(self, S_q: int, S_kv: int, device: torch.device) -> DispatchPlan:
        if self.mask_mod is None:
            return DispatchPlan(sparse_mode=0, pattern_name="no_mask")
        key = (S_q, S_kv, str(device))
        if key not in self._plan_cache:
            block_status, full_mask = _materialize_block_mask(
                self.mask_mod, S_q, S_kv, self.block_size, device
            )
            plan = _detect_pattern(block_status, full_mask, S_q, S_kv)
            if self.verbose:
                print(f"[FlexAttentionNPU] S_q={S_q} S_kv={S_kv} "
                      f"→ sparse_mode={plan.sparse_mode} ({plan.pattern_name})")
            self._plan_cache[key] = plan
        return self._plan_cache[key]

    def __call__(self,
                 query: torch.Tensor,
                 key: torch.Tensor,
                 value: torch.Tensor,
                 scale: Optional[float] = None,
                 return_lse: bool = False,
                 ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, H_q, S_q, D = query.shape
        S_kv = key.shape[2]
        scale_value = 1.0 / (D ** 0.5) if scale is None else float(scale)

        plan = self._get_plan(S_q, S_kv, query.device)

        result = torch_npu.npu_fusion_attention(
            query, key, value,
            head_num=H_q,
            input_layout="BNSD",
            atten_mask=plan.atten_mask,
            scale=scale_value,
            keep_prob=1.0,
            pre_tockens=plan.pre_tockens,
            next_tockens=plan.next_tockens,
            sparse_mode=plan.sparse_mode,
            prefix=plan.prefix,
        )
        out = result[0]
        if return_lse:
            lse = result[1][..., 0].to(torch.float32) + \
                  torch.log(result[2][..., 0].to(torch.float32))
            return out, lse
        return out
