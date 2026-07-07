# Generate the AE training latent corpus from prompts for LongLive-RAG.
#
# Multi-node sharding model (torchrun):
#   - torchrun sets RANK, WORLD_SIZE, LOCAL_RANK env vars.
#   - All ranks independently sample the SAME set of prompts (seeded by
#     config.seed) and then deterministically slice it: rank r takes
#     positions {r, r + W, r + 2W, ...} of the sorted sampled indices.
#   - Each prompt's position in the sorted sampled list IS its filename
#     (latent_{global_pos:06d}.pt), so the union of all ranks' outputs is a
#     flat, gap-free, collision-free dataset directory regardless of NGPU.
#
# Pipeline selection:
#   - Default: CausalInferencePipeline (for pure RF / reward-forcing base).
#   - If config.use_streaming_pipeline is True: StreamingCausalInferencePipeline
#     (for RF base + LongLive LoRA streaming inference).
#
# Usage (single node, 8 GPUs):
#   torchrun --nproc_per_node=8 scripts/generate_longlive_rag_latent.py \
#       --config_path configs/longlive_rag/generate_latent_reward_forcing.yaml
#
# Usage (multi-node):
#   NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
#   MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
#   torchrun --nnodes=$NNODES --node_rank=$NODE_RANK --nproc_per_node=$NPROC_PER_NODE \
#       --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
#       scripts/generate_longlive_rag_latent.py \
#       --config_path configs/longlive_rag/generate_latent_longlive.yaml
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import random
import time
from datetime import datetime

# NPU support (mirrors inference_<method>.py / train_<method>.py convention).
DEVICE_TYPE = os.environ.get('DEVICE_TYPE', 'cuda')
if DEVICE_TYPE == "npu":
    from torch_npu.contrib import transfer_to_npu

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange
from torch.utils.data import DataLoader, SequentialSampler, Subset

# OBS-aware IO: lets `output_folder` be either a local path or `obs://bucket/...`.
# moxing is imported lazily inside io_utils so this import is safe on CPU-only test envs.
from methods.longlive_rag.io_utils import (
    safe_exists, safe_makedirs, safe_save_torch,
)

# Module-level placeholder: set to None to avoid triggering core.misc.memory
# (which calls torch.cuda at module level and crashes on CPU-only builds).
# Lazily imported inside compute_shard_pairs if not monkeypatched by tests.
TextDataset = None

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--reverse", action="store_true",
                        help="Process this rank's shard in reverse order")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip global positions whose latent_{pos:06d}.pt already exists")
    parser.add_argument("--sample_ratio", type=float, default=None,
                        help="Override sample_ratio from yaml/merge (0.0-1.0). "
                             "When None, falls back to config.sample_ratio (default 0.1).")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Print per-rank throughput every N successfully generated "
                             "prompts (matches the scripts/process_custom_data.py pace). "
                             "Set 0 to disable periodic throughput logging; the end-of-run "
                             "overall line still prints.")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline construction
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(config, device, local_rank):
    """Construct the inference pipeline and load generator weights + optional LoRA.

    Pipeline selection:
      - Default: CausalInferencePipeline (pure RF / reward-forcing base).
      - If config.use_streaming_pipeline is True: StreamingCausalInferencePipeline
        (for RF base + LongLive LoRA streaming inference).
    """
    # Lazy imports: these modules depend on torch CUDA and may not be
    # importable in CPU-only test environments.
    from methods.reward_forcing.pipelines import (
        CausalInferencePipeline,
        StreamingCausalInferencePipeline,
    )
    from core.misc.lora_utils import configure_lora_for_model
    from core.misc.memory import get_cuda_free_memory_gb, DynamicSwapInstaller
    import peft

    # Always have denoising_step_list for latent generation, so we use
    # CausalInferencePipeline unless config explicitly opts into the
    # streaming variant via use_streaming_pipeline: true.
    if getattr(config, "use_streaming_pipeline", False):
        pipeline = StreamingCausalInferencePipeline(config, device=device)
    else:
        pipeline = CausalInferencePipeline(config, device=device)

    # ── Load generator checkpoint ──────────────────────────────────────────
    if getattr(config, "generator_ckpt", None):
        state_dict = torch.load(config.generator_ckpt, map_location="cpu")
        if "generator" in state_dict or "generator_ema" in state_dict:
            raw_gen_state_dict = state_dict["generator_ema" if getattr(config, "use_ema", False) else "generator"]
        elif "model" in state_dict:
            raw_gen_state_dict = state_dict["model"]
        else:
            raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")

        if getattr(config, "use_ema", False):
            cleaned = {k.replace("_fsdp_wrapped_module.", ""): v for k, v in raw_gen_state_dict.items()}
            missing, unexpected = pipeline.generator.load_state_dict(cleaned, strict=False)
            if local_rank == 0:
                if missing:
                    print(f"[rank 0] {len(missing)} missing params, e.g. {missing[:4]}")
                if unexpected:
                    print(f"[rank 0] {len(unexpected)} unexpected params, e.g. {unexpected[:4]}")
        else:
            try:
                pipeline.generator.load_state_dict(raw_gen_state_dict)
            except RuntimeError:
                fixed = {}
                for k, v in raw_gen_state_dict.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                pipeline.generator.load_state_dict(fixed, strict=False)

    # ── Optional LoRA ──────────────────────────────────────────────────────
    pipeline.is_lora_enabled = False
    if getattr(config, "adapter", None) is not None and configure_lora_for_model is not None:
        if local_rank == 0:
            print(f"[rank 0] LoRA enabled with config: {config.adapter}")
        pipeline.generator.model = configure_lora_for_model(
            pipeline.generator.model,
            model_name="generator",
            lora_config=config.adapter,
            is_main_process=(local_rank == 0),
        )
        lora_ckpt_path = getattr(config, "lora_ckpt", None)
        if lora_ckpt_path:
            if local_rank == 0:
                print(f"[rank 0] Loading LoRA checkpoint from {lora_ckpt_path}")
            lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
            if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
            else:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
        pipeline.is_lora_enabled = True

    pipeline = pipeline.to(dtype=torch.bfloat16)
    low_memory = get_cuda_free_memory_gb(device) < 40
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic sampling + sharding
# ─────────────────────────────────────────────────────────────────────────────

def compute_shard_pairs(config, rank, world_size, reverse):
    """Sample prompts deterministically and stripe them across ranks.

    Args:
        config: Duck-typed config object with ``data_path`` and ``seed`` attributes.
        rank: This process's rank (0..world_size-1).
        world_size: Total number of processes.
        reverse: If True, invert this rank's iteration order.

    Returns:
        full_dataset: The underlying TextDataset.
        my_pairs:     list of (global_pos, prompt_idx) handled by this rank.
                      global_pos doubles as the output file id, so the union of
                      all ranks is a flat, gap-free dataset.
    """
    # Uses module-level TextDataset (set in main() for real runs, or
    # monkeypatched by tests for unit testing).
    full_dataset = TextDataset(prompt_path=config.data_path, extended_prompt_path=config.data_path)
    num_total = len(full_dataset)
    # Default 0.1 matches README and the shipped yaml configs in
    # configs/longlive_rag/generate_latent_*.yaml. When main() applies a
    # CLI --sample_ratio override it overwrites config.sample_ratio before
    # calling this function.
    sample_ratio = float(getattr(config, "sample_ratio", 0.1))
    num_sample = max(1, int(num_total * sample_ratio))

    # Re-seed Python's ``random`` with config.seed (NOT seed+rank) so every rank
    # draws the exact same sample. Independent of torch RNG (offset by rank),
    # which only governs the per-prompt diffusion noise.
    random.seed(config.seed)
    sampled_indices = sorted(random.sample(range(num_total), num_sample))

    # ``g`` (position in the sorted list) doubles as the global file id.
    my_pairs = [(g, idx) for g, idx in enumerate(sampled_indices) if g % world_size == rank]
    if reverse:
        my_pairs = my_pairs[::-1]

    print(f"[rank {rank}/{world_size}] total prompts={num_total}  "
          f"sampled {sample_ratio*100:.0f}%={len(sampled_indices)}  "
          f"this rank={len(my_pairs)}")
    return full_dataset, my_pairs


def prompt_id(prompt: str) -> str:
    """Deterministic 16-char hex ID from prompt string.

    Same prompt always yields same ID, regardless of sample_ratio / world_size /
    rank. This makes ``--skip_existing`` safe across configuration changes: a
    latent file for prompt X is always named ``latent_<id(X)>.pt``, so re-running
    with different sample_ratio won't accidentally skip a different prompt that
    happened to share the same ``global_pos`` slot.

    Uses sha1 truncated to 16 hex chars (64-bit collision space) — sufficient
    for prompt pools up to ~10^9 entries before collision probability rises
    above 1% (birthday bound). ARVideoGen prompt pools are ≤10^5.
    """
    import hashlib
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]


def latent_path_for(latent_folder: str, num_samples: int, pid: str, seed_idx: int) -> str:
    """Flat output path ``latent_{pid}[_s{seed_idx}].pt``.

    ``pid`` is the :func:`prompt_id` (16-char hex). When ``num_samples == 1``,
    no seed suffix is appended.
    """
    if num_samples == 1:
        return os.path.join(latent_folder, f"latent_{pid}.pt")
    return os.path.join(latent_folder, f"latent_{pid}_s{seed_idx}.pt")


# ─────────────────────────────────────────────────────────────────────────────
# Generation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_generation(pipeline, dataloader, my_pairs, config, device, args,
                   latent_folder, video_folder, num_samples, max_video_saves,
                   low_memory, local_rank):
    """Iterate this rank's prompts, generating and saving latents (+ a few videos).

    Returns:
        meta_entries: list of dicts, one per successfully generated prompt, to be
            written by main() as ``prompts_rank{R:04d}.jsonl``. Each entry has:
            uuid, prompt, extended_prompt, prompt_idx, global_pos, rank,
            world_size, seed, sample_ratio, num_output_frames.
            This preserves the prompt→latent mapping for debugging, cross-corpus
            alignment, and future prompt-aware retrieval extensions. AE training
            itself does NOT need prompts (loss is purely on latent/embedding
            space), but saving the mapping is cheap and useful.

    Throughput logging (every args.log_interval successfully-generated prompts
    from this rank) follows the same shape as ``scripts/process_custom_data.py``
    so logs are greppable for "DI_throughput" across the data-prep + latent-gen
    pipeline.
    """
    # Lazy import: write_video may not be available in all torchvision builds
    try:
        from torchvision.io import write_video as _write_video
    except ImportError:
        _write_video = None
    videos_saved = 0

    rank_label = (f"rank{local_rank}" if not dist.is_initialized()
                  else f"rank{dist.get_rank()}")
    world_size_or_one = dist.get_world_size() if dist.is_initialized() else 1
    backend_label = os.environ.get("DEVICE_TYPE", "cuda")  # 'cuda' or 'npu'

    # Throughput counters — accumulate since the last periodic print.
    throughput_start_time = time.time()
    throughput_start_count = 0  # number of prompts completed since last print

    total_this_rank = len(my_pairs)
    meta_entries = []  # prompt→latent metadata, returned to main for jsonl write

    for i, batch_data in tqdm(enumerate(dataloader), total=len(dataloader),
                              desc=f"rank{local_rank}", position=local_rank):
        global_pos, prompt_idx = my_pairs[i]

        batch = batch_data[0] if isinstance(batch_data, list) else batch_data
        prompt = batch["prompts"][0]
        extended_prompt = batch.get("extended_prompts", [None])[0]
        prompts_for_inference = [extended_prompt if extended_prompt is not None else prompt] * num_samples

        # ── Deterministic filename: prompt_id(prompt) ───────────────────────
        # Use the (possibly extended) prompt that was actually fed to the
        # generator as the identity. This way the same prompt→latent file
        # mapping is stable across sample_ratio / world_size changes.
        canonical_prompt = extended_prompt if extended_prompt is not None else prompt
        pid = prompt_id(canonical_prompt)

        if args.skip_existing:
            # If every seed's latent already exists for this prompt, skip.
            # `safe_exists` transparently handles obs:// paths via mox.file.exists.
            # Because pid is deterministic on the prompt text, this is safe
            # across re-runs with different sample_ratio / world_size.
            if all(safe_exists(latent_path_for(latent_folder, num_samples, pid, s))
                   for s in range(num_samples)):
                continue

        sampled_noise = torch.randn(
            [num_samples, config.num_output_frames, 16, 60, 104],
            device=device, dtype=torch.bfloat16,
        )

        # Only rank 0 saves a small number of verification videos to keep the
        # output count == max_video_saves rather than max_video_saves * world_size.
        save_video = (local_rank == 0 and videos_saved < max_video_saves)

        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts_for_inference,
            return_latents=True,
            low_memory=low_memory,
            profile=False,
            skip_vae_decode=(not save_video),
        )
        if save_video:
            pipeline.vae.model.clear_cache()

        for seed_idx in range(num_samples):
            # `safe_save_torch` streams the bytes through mox.file.File when the
            # latent_folder is on OBS — no local disk spilling required.
            safe_save_torch(
                latents[seed_idx].cpu(),
                latent_path_for(latent_folder, num_samples, pid, seed_idx),
            )

            if save_video and video is not None:
                current_video = rearrange(video, "b t c h w -> b t h w c").cpu()
                vid_tensor = 255.0 * current_video
                video_path = os.path.join(video_folder, f"latent_{pid}_s{seed_idx}.mp4")
                if _write_video is not None:
                    _write_video(video_path, vid_tensor[seed_idx], fps=16)
                videos_saved += 1
                print(f"[rank 0] saved verification video {video_path}")

        # ── Record prompt→latent metadata for the jsonl sidecar ─────────────
        meta_entries.append({
            "uuid": pid,
            "prompt": prompt,
            "extended_prompt": extended_prompt,
            "prompt_idx": prompt_idx,
            "global_pos": global_pos,  # kept for backward compat / debugging
            "rank": dist.get_rank() if dist.is_initialized() else 0,
            "world_size": world_size_or_one,
            "seed": config.seed,
            "sample_ratio": float(getattr(config, "sample_ratio", 0.1)),
            "num_output_frames": config.num_output_frames,
            "num_samples": num_samples,
        })

        # ── Throughput accounting ────────────────────────────────────────────
        throughput_start_count += 1
        cur_total_successful = throughput_start_count

        if args.log_interval > 0 and cur_total_successful % args.log_interval == 0:
            end_time = time.time()
            time_diff = end_time - throughput_start_time
            interval_count = throughput_start_count
            rate = interval_count / time_diff if time_diff > 0 else 0.0
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"{timestamp}: [{rank_label}/{world_size_or_one}/{backend_label}] "
                f"processed {cur_total_successful}/{total_this_rank}  "
                f"DI_throughput: {rate:.2f} prompts/s/{backend_label}"
            )
            throughput_start_time = end_time
            throughput_start_count = 0

    # ── End-of-run tail throughput per rank ─────────────────────────────────
    overall_elapsed = time.time() - throughput_start_time
    overall_count = throughput_start_count
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if overall_count > 0 and overall_elapsed > 0:
        rate = overall_count / overall_elapsed
        print(
            f"{timestamp}: [{rank_label}/{world_size_or_one}/{backend_label}] "
            f"done. {overall_count} prompts in {overall_elapsed:.1f}s, "
            f"tail: {rate:.2f} prompts/s/{backend_label}"
        )
    else:
        print(
            f"{timestamp}: [{rank_label}/{world_size_or_one}/{backend_label}] "
            f"done. (throughput accounting fell back to main() overall line)"
        )

    return meta_entries


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    main_start_time = time.time()

    # ── Config load: merge default_config.yaml + user yaml (user wins) ──
    # The user yaml may carry a ``default_config_path`` field that points at
    # the method's defaults (e.g. configs/reward_forcing/default_config.yaml).
    # We honour it here so default fields are populated before sample_ratio/seed/...
    # are read below. This matches the pattern used by inference_reward_forcing.py
    # and inference_longlive_rag.py.
    user_yaml = OmegaConf.load(args.config_path)
    default_cfg_path = (
        getattr(user_yaml, "default_config_path", None)
        or os.path.join(os.path.dirname(args.config_path), "default_config.yaml")
    )
    from core.config import load_config
    config = load_config(args.config_path, default_config_path=default_cfg_path)

    # ── CLI override ──
    # `sample_ratio` is the most likely field to want to tweak at run time
    # (e.g. quick smoke run with --sample_ratio 0.005). Override the merged
    # config in-place so compute_shard_pairs sees the final value.
    if args.sample_ratio is not None:
        config.sample_ratio = float(args.sample_ratio)

    # ── Distributed init (torchrun supplies RANK, WORLD_SIZE, LOCAL_RANK) ──
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    max_video_saves = getattr(config, "max_video_saves", 2)
    num_samples = getattr(config, "num_samples", 1)
    low_memory = True

    # Log the effective sample ratio on rank 0 — this is the number actually
    # consumed by compute_shard_pairs. Previously this print only existed
    # inside compute_shard_pairs and was easy to miss; surfacing it here makes
    # the CLI-override -> function-call pipeline explicit.
    if rank == 0:
        eff_sample_ratio = float(getattr(config, "sample_ratio", 0.1))
        src = "yaml" if args.sample_ratio is None else f"CLI --sample_ratio={args.sample_ratio}"
        print(f"[rank 0] effective sample_ratio={eff_sample_ratio:.4f} (source: {src})")

    # Same seed across ranks for Python ``random`` (governs prompt sampling).
    # Offset by rank for torch RNG so each rank's torch.randn is independent.
    # Import set_seed directly from misc module to avoid triggering core.misc.memory
    # (which calls torch.cuda at module level and crashes on CPU-only builds).
    from core.misc.misc import set_seed
    set_seed(config.seed + rank)
    torch.set_grad_enabled(False)

    # Set module-level TextDataset for real runs (tests monkeypatch it instead).
    from core.data.dataset import TextDataset as _TextDataset
    global TextDataset
    TextDataset = _TextDataset

    pipeline = build_pipeline(config, device, local_rank)

    full_dataset, my_pairs = compute_shard_pairs(config, rank, world_size, args.reverse)

    # Build a Subset over the prompt indices in shard order.
    shard_prompt_idxs = [idx for _, idx in my_pairs]
    dataset = Subset(full_dataset, shard_prompt_idxs)
    dataloader = DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset),
                            num_workers=0, drop_last=False)

    # ── Output layout — flat dataset of latent_{global_pos:06d}.pt ──────────
    latent_folder = config.output_folder
    video_folder = os.path.join(config.output_folder, "_verification_videos")

    if rank == 0:
        # `safe_makedirs` transparently handles obs:// uris via mox.file.mkdir
        # (no-op on OBS because the bucket-key hierarchy is implicit on first
        # write). Local paths fall through to os.makedirs unchanged.
        safe_makedirs(latent_folder, exist_ok=True)
        safe_makedirs(video_folder, exist_ok=True)
    dist.barrier()

    if rank == 0:
        print(f"[rank 0] latents -> {latent_folder}/latent_<prompt_id>.pt")
        if max_video_saves > 0:
            print(f"[rank 0]   verification videos (<= {max_video_saves}, rank 0 only) -> {video_folder}/")

    meta_entries = run_generation(pipeline, dataloader, my_pairs, config, device, args,
                                  latent_folder, video_folder, num_samples, max_video_saves,
                                  low_memory, local_rank)

    # ── Write per-rank prompts metadata sidecar ────────────────────────────
    # Each rank writes its own prompts_rank{R:04d}.jsonl to avoid concurrent
    # appends to a single file. The jsonl preserves the prompt→latent mapping
    # (uuid == filename stem) for debugging, cross-corpus alignment, and future
    # prompt-aware retrieval. AE training itself does NOT read this file —
    # LatentFrameDataset only globs latent_*.pt tensors — but having the
    # mapping available is cheap insurance.
    if meta_entries:
        import json
        meta_path = os.path.join(latent_folder, f"prompts_rank{rank:04d}.jsonl")
        if latent_folder.startswith("obs://"):
            # OBS: stream text via mox.file.File
            try:
                mox = __import__("mox")
                lines = "\n".join(json.dumps(e, ensure_ascii=False) for e in meta_entries) + "\n"
                with mox.file.File(meta_path, "wb") as f:
                    f.write(lines.encode("utf-8"))
            except Exception as e:
                print(f"[rank {rank}] WARN: failed to write {meta_path} to OBS: {e}")
        else:
            with open(meta_path, "w", encoding="utf-8") as f:
                for e in meta_entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
        if rank == 0 or (dist.is_initialized() and dist.get_rank() == 0):
            print(f"[rank {rank}] wrote {len(meta_entries)} entries -> {meta_path}")

    # ── Canonical overall throughput line ──────────────────────────────────
    # Ranks other than 0 wait for rank 0's progress / final barrier; the
    # canonical "overall prompts/s" is computed from main_start_time so it
    # covers the full main() body (init + dataloader + generation loop). For
    # a tighter loop-only rate, see the per-rank tail line in run_generation.
    main_total_time = time.time() - main_start_time
    total_prompts = len(my_pairs)
    backend_label = os.environ.get("DEVICE_TYPE", "cuda")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if total_prompts > 0 and main_total_time > 0:
        overall = total_prompts / main_total_time
        print(
            f"{timestamp}: [rank{rank}] "
            f"done. processed {total_prompts} prompts -> {latent_folder}  "
            f"overall: {overall:.2f} prompts/s/{backend_label}"
        )
    else:
        print(f"[rank {rank}] done. processed {total_prompts} prompts -> {latent_folder}")

    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
              f"[rank0] all ranks finished, destroying process group.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
