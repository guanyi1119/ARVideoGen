# ARVideoGen 方法指南

## 目录结构概览

```
ARVideoGen/
  core/                    # 共享核心模块
    data/                  #   数据集、LMDB工具
    distributed/           #   分布式训练工具
    scheduler/             #   FlowMatchScheduler
    loss/                  #   去噪损失函数
    config/                #   OmegaConf配置加载
    misc/                  #   种子、调试、LoRA、内存工具
    wan_wrapper/           #   WanDiffusionWrapper（3个变体）
    demo_utils/            #   Demo工具
  wan/                     # Wan2.1 基础模型代码
    modules/               #   注意力、T5、VAE、4种CausalModel
  methods/                 # 各方法实现
    base/                  #   BaseModel继承体系
    causal_forcing/        #   Causal-Forcing
    self_forcing/          #   Self-Forcing
    causvid/               #   CausVid
    longlive/              #   LongLive
    deep_forcing/          #   DeepForcing
  configs/                 # 各方法配置文件
  scripts/                 # 数据处理脚本
  tests/                   # 测试脚本
  train_<method>.py        # 训练入口
  inference_<method>.py    # 推理入口
  archive/                 # 原始项目归档
```

## 四种方法概览

| | Causal-Forcing | Self-Forcing | CausVid | LongLive | DeepForcing |
|---|---|---|---|---|---|
| **核心理念** | 自回归+teacher forcing训练 | 自回归+self-forcing训练（无teacher signal） | 双向扩散初始化→因果蒸馏 | 流式训练+延迟cache更新 | 自回归+DeepForcing机制（增强上下文建模） |
| **DMD继承** | `SelfForcingModel(BaseModel)` | `SelfForcingModel(BaseModel)` | `nn.Module`（自包含） | `SelfForcingModel(BaseModel)` | `SelfForcingModel(BaseModel)` |
| **WanWrapper** | `wan_wrapper.py` | `wan_wrapper.py` | `wan_wrapper_causvid.py` | `wan_wrapper_longlive.py` | `wan_wrapper_deepforcing.py` |
| **CausalModel** | `causal_model.py` | `causal_model.py` | `causal_model_causvid.py` | `causal_model_longlive.py` / `causal_model_infinity.py` | `causal_model.py` + optional `causal_model_DS.py` |
| **Config加载** | `load_config(path, default)` | `load_config(path, default)` | `load_config(path)` | `load_config(path, default)` | `load_config(path, default)` |
| **训练脚本** | `train_causal_forcing.py` | `train_self_forcing.py` | `train_causvid.py` | `train_longlive.py` | `train_deep_forcing.py`（待添加） |
| **推理脚本** | `inference_causal_forcing.py` | `inference_self_forcing.py` | `inference_causvid.py` | `inference_longlive.py` | `inference_deep_forcing.py` |

---

## 1. Causal-Forcing

### 核心思想

自回归视频生成框架，使用 teacher forcing 训练 generator：训练时每帧以 ground-truth 前帧为条件，推理时以自身生成的前帧为条件。支持 DMD 蒸馏、ODE 回归、GAN、Consistency Distillation 等多种训练模式。

### 关键特性

- **Teacher Forcing / Bidirectional 训练**：[DIFF-CausalForcing] 除了 SelfForcingTrainingPipeline，还提供 `TeacherForcingTrainingPipeline` 和 `BidirectionalTrainingPipeline`
- **`_prepare_generator_input()`**：[DIFF-CausalForcing] 从 ODE 轨迹中选择 timestep 作为 generator 输入，包含 `ts_schedule` / `ts_schedule_max` 参数控制 timestep 采样策略
- **`fake_guidance_scale`**：[DIFF-CausalForcing] 支持 CFG（Classifier-Free Guidance）scale
- **clean latent 支持**：[DIFF-CausalForcing] `clean_x` / `aug_t` 参数，用于 teacher forcing 和 bidirectional 训练
- **硬编码帧数**：[DIFF-CausalForcing] `_run_generator` 中 slice=21 硬编码，最小帧数固定为 20/21
- **Denoising step list 在 CPU**：[DIFF-CausalForcing] `denoising_step_list` 存储在 CPU tensor 上
- **GAN 支持**：[DIFF-CausalForcing] 支持 `classify_mode` 和 `concat_time_embeddings` 参数用于 GAN 训练
- **返回值**：`forward()` 返回 `(pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to)` 四元组

### 配置示例

```yaml
# configs/causal_forcing/causal_forcing_dmd_chunkwise.yaml
denoising_step_list: [1000, 750, 500, 250]
model_kwargs:
  timestep_shift: 5.0
num_frame_per_block: 3
trainer: score_distillation
```

### 使用方法

```bash
# 训练
python train_causal_forcing.py --config configs/causal_forcing/causal_forcing_dmd_chunkwise.yaml

# 推理
torchrun --nproc_per_node=8 inference_causal_forcing.py \
  --config configs/causal_forcing/causal_forcing_dmd_chunkwise.yaml \
  --data_path prompts.txt --output_folder outputs/
```

### 注意事项

- Causal-Forcing 的 `default_config.yaml` 提供了所有默认值，运行时会自动合并
- 支持 Image-to-Video (I2V) 模式，推理时加 `--i2v` 参数
- `--tf` 参数可启用 tensor flow 记录

---

## 2. Self-Forcing

### 核心思想

自回归视频生成框架，使用 self-forcing 训练：训练时 generator 以自身生成的前帧为条件（非 ground-truth），通过可微模拟反向传播梯度。相比 Causal-Forcing 消除了 train-test mismatch。

### 关键差异（与 Causal-Forcing 对比）

- **无 Teacher Forcing / Bidirectional**：[DIFF-SelfForcing] 不含 `TeacherForcingTrainingPipeline` 和 `BidirectionalTrainingPipeline`
- **无 `_prepare_generator_input()`**：[DIFF-SelfForcing] 不从 ODE 轨迹选择 timestep
- **可配置帧切片**：[DIFF-SelfForcing] `slice_last_frames` 参数可配置（非硬编码），支持动态截取最后 N 帧作为下一 chunk 的起始条件
- **`min_num_training_frames`**：[DIFF-SelfForcing] 替代 Causal-Forcing 中硬编码的最小帧数
- **`local_attn_size` 追踪**：[DIFF-SelfForcing] 从 `model_kwargs` 获取并传递给 pipeline
- **GAN 训练**：[DIFF-SelfForcing] 支持 GAN trainer（Causal-Forcing 的 GAN 逻辑写在模型代码中而非独立 trainer）
- **Denoising step list 在 CPU**：与 Causal-Forcing 一致

### 配置示例

```yaml
# configs/self_forcing/self_forcing_dmd.yaml
denoising_step_list: [1000, 750, 500, 250]
model_kwargs:
  timestep_shift: 5.0
  local_attn_size: 12      # [DIFF-SelfForcing] 局部注意力窗口大小
  sink_size: 3              # [DIFF-SelfForcing] sink token 数量
num_frame_per_block: 3
trainer: score_distillation
```

### 使用方法

```bash
# 训练
python train_self_forcing.py --config configs/self_forcing/self_forcing_dmd.yaml

# 推理
torchrun --nproc_per_node=8 inference_self_forcing.py \
  --config configs/self_forcing/self_forcing_dmd.yaml \
  --data_path prompts.txt --output_folder outputs/ --num_samples 4
```

### 注意事项

- `local_attn_size` 和 `sink_size` 是 Self-Forcing 特有参数，Causal-Forcing 不使用
- 推理时支持 `--extended_prompt_path` 和 `--num_samples` 参数

---

## 3. CausVid

### 核心思想

通过双向扩散模型初始化，然后蒸馏为因果（自回归）模型。独特之处：先用 bidirectional teacher 训练，再用 DMD 蒸馏为 causal generator。代码结构与其他方法差异最大——DMD 不继承 BaseModel，而是自包含的 `nn.Module`。

### 关键差异

- **DMD 自包含**：[DIFF-CausVid] `DMD(nn.Module)` 而非继承 `SelfForcingModel(BaseModel)`，自行管理 generator / real_score / fake_score / text_encoder / vae 的初始化
- **Task Type 系统**：[DIFF-CausVid] `generator_task` / `real_task` / `fake_task` 参数，支持 `image`、`bidirectional_video`、`causal_video` 三种任务类型
- **工厂模式**：[DIFF-CausVid] 通过 `get_inference_pipeline_wrapper()` 创建推理 pipeline，而非直接实例化
- **`_process_timestep()`**：[DIFF-CausVid] 根据 task type 处理 timestep，与其他方法的 `_run_generator()` 逻辑不同
- **`_consistency_backward_simulation()`**：[DIFF-CausVid] 直接调用 pipeline 而非通过 BaseModel
- **返回值不同**：[DIFF-CausVid] `forward()` 返回 `(pred_image, gradient_mask)` 二元组（无 timestep from/to）
- **Score model 返回值**：[DIFF-CausVid] score model 只返回 `pred_x0`（其他方法返回 `(noise_pred, x0_pred)` 元组）
- **WanWrapper 变体**：[DIFF-CausVid] 使用 `wan_wrapper_causvid.py`（`current_end` 替代 `cache_start`，返回 `pred_x0` 而非元组）
- **CausalModel 变体**：[DIFF-CausVid] 使用 `causal_model_causvid.py`（`window_size` 替代 `local_attn_size` / `sink_size`，简化 KV cache）
- **无 default_config.yaml**：[DIFF-CausVid] 配置直接加载，不与 default 合并
- **数据路径**：[DIFF-CausVid] 默认使用 `mixkit_latents_lmdb`

### 配置示例

```yaml
# configs/causvid/wan_causal_dmd.yaml
generator_task: causal_video        # [DIFF-CausVid] task type
denoising_step_list: [1000, 757, 522, 0]  # [DIFF-CausVid] 不同的step列表
num_frame_per_block: 3
data_path: mixkit_latents_lmdb      # [DIFF-CausVid] 不同的数据源
```

### 使用方法

```bash
# 训练
python train_causvid.py --config configs/causvid/wan_causal_dmd.yaml

# 推理（无需 torchrun）
python inference_causvid.py \
  --config_path configs/causvid/wan_causal_dmd.yaml \
  --checkpoint_path checkpoints/causvid/
```

### 注意事项

- CausVid 推理脚本不需要分布式启动，单 GPU 即可
- `--checkpoint_path` 是必需参数
- 训练只支持 `distillation` 和 `ode` 两种 trainer

---

## 4. LongLive

### 核心思想

面向长视频生成的流式训练框架。核心创新：延迟 cache 更新（deferred cache update）——在自回归生成时，sink token 和 recent token 的 KV cache 不立即更新，而是在切换 chunk 时才更新，大幅减少计算量。支持 Infinity 注意力（Block-Relativistic RoPE）实现无限长度视频生成。

### 关键差异

- **延迟 cache 更新**：[DIFF-LongLive] `sink_recache_after_switch` 参数，chunk 切换时才更新 sink token 的 KV cache
- **`is_causal` 可配置**：[DIFF-LongLive] `args.causal` 标志控制 generator 是否使用因果注意力（其他方法始终 `is_causal=True`）
- **Denoising step list 在 GPU**：[DIFF-LongLive] `denoising_step_list` 存储在 GPU tensor 上（其他方法在 CPU）
- **调试/性能追踪**：[DIFF-LongLive] 内置 `DEBUG`、`LOG_GPU_MEMORY`、计时统计（`gen_time`、`loss_time`）
- **动态序列长度**：[DIFF-LongLive] `seq_len` 根据 `local_attn_size` 动态计算（`1560 * local_attn_size`），其他方法固定 32760
- **Chunked VAE 解码**：[DIFF-LongLive] `decode_to_pixel_chunk()` 方法，防止长视频 OOM
- **Infinity 注意力**：[DIFF-LongLive] `causal_model_infinity.py`，使用 Block-Relativistic RoPE 实现无限长度注意力
- **LoRA 支持**：[DIFF-LongLive] 推理脚本内置 LoRA 合并功能
- **DMD Switch**：[DIFF-LongLive] 独有的 `dmd_switch.py`，支持 switch 模式的蒸馏训练
- **流式训练**：[DIFF-LongLive] `streaming_training.py` 模型和 pipeline，专为流式长视频训练设计
- **WanWrapper 变体**：[DIFF-LongLive] 使用 `wan_wrapper_longlive.py`（`sink_recache_after_switch`、动态 seq_len、chunked decode、GPU text encoder）
- **CausalModel 变体**：[DIFF-LongLive] `causal_model_longlive.py`（延迟 cache 更新、list/tuple `local_attn_size`）和 `causal_model_infinity.py`（Block-Relativistic RoPE）

### 配置示例

```yaml
# configs/longlive/longlive_train_init.yaml
generator_task: video                # [DIFF-LongLive] video (非 causal_video)
denoising_step_list: [1000, 750, 500, 250]
data_path: prompts/vidprom_filtered_extended.txt
batch_size: 1
total_batch_size: 64
```

### 使用方法

```bash
# 初始训练
python train_longlive.py --config configs/longlive/longlive_train_init.yaml

# 长视频训练
python train_longlive.py --config configs/longlive/longlive_train_long.yaml

# 推理
torchrun --nproc_per_node=8 inference_longlive.py \
  --config configs/longlive/longlive_inference.yaml
```

### 注意事项

- LongLive 训练有 `--no-auto-resume` 和 `--no-one-logger` 参数
- 推理配置文件中包含 checkpoint 路径、LoRA 设置等，无需额外 CLI 参数
- 长视频生成建议使用 `longlive_inference_infinity.yaml` 配置（启 Infinity 注意力）

---

## 5. DeepForcing

### 核心思想

自回归视频生成框架，使用 DeepForcing 机制增强上下文建模能力。通过 Prompt Context（PC）机制实现灵活的长视频生成，支持动态的最近帧上下文和固定的历史上下文，在生成长视频时保持连贯性和质量。支持 DS-Only 模式，针对特定场景优化。

### 关键差异（与 Causal-Forcing/Self-Forcing 对比）

- **DeepForcing 特有参数**：[DIFF-DeepForcing] 新增 `is_ds_only`、`budget`、`recent` 三个核心参数，通过 `model_kwargs` 传递
  - `is_ds_only`：启用 DS-Only 模式，使用 `CausalWanModelDS`
  - `budget`：设置 Prompt Context 的预算大小（帧数量）
  - `recent`：设置最近帧窗口大小
- **PC 容量计算**：[DIFF-DeepForcing] PC 容量 = `1560 * budget`，最近窗口 = `1560 * recent`（与其他方法不同，以 token 数计算）
- **CausalWanModelDS**：[DIFF-DeepForcing] 新增 `wan.modules.causal_model_DS.CausalWanModelDS`，支持 DS 模式的注意力机制
- **推理过程打印**：[DIFF-DeepForcing] 在推理过程中打印 `current_timestep`，便于调试和监控
- **扩展的 CLI 参数**：[DIFF-DeepForcing] 推理脚本支持 `--extended_prompt_path`、`--num_samples`、`--save_with_index` 等
- **Denoising step list 在 CPU**：与 Causal-Forcing/Self-Forcing 一致

### 配置示例

```yaml
# configs/deep_forcing/deep_forcing_dmd.yaml
denoising_step_list: [1000, 750, 500, 250]
model_kwargs:
  timestep_shift: 5.0
  is_ds_only: false          # [DIFF-DeepForcing] 是否启用 DS-Only 模式
  budget: 16                  # [DIFF-DeepForcing] PC 预算（帧数）
  recent: 4                   # [DIFF-DeepForcing] 最近帧窗口大小
num_frame_per_block: 3
trainer: score_distillation
```

### 使用方法

```bash
# 推理（DeepForcing 特有参数）
python inference_deep_forcing.py \
  --config_path configs/deep_forcing/deep_forcing_dmd.yaml \
  --checkpoint_path path/to/checkpoint \
  --data_path prompts.txt \
  --output_folder outputs/ \
  --num_output_frames 126 \
  --Budget 16 \
  --Recent 4 \
  --num_samples 4

# DS-Only 模式
python inference_deep_forcing.py \
  --config_path configs/deep_forcing/deep_forcing_ds.yaml \
  --checkpoint_path path/to/checkpoint \
  --data_path prompts.txt \
  --output_folder outputs/ \
  --is_ds_only 1
```

### 注意事项

- DeepForcing 的 `default_config.yaml` 会与指定配置合并（与 CausVid 不同）
- `CausalWanModelDS` 是可选的，默认使用标准的 `CausalWanModel` 并应用 PC 机制
- `budget` 和 `recent` 参数通过 `model_kwargs` 传递，需要在 config 或 CLI 中指定
- 原始 DeepForcing 代码位于 `archive/DeepForcing/` 目录，作为参考
- 训练脚本 `train_deep_forcing.py` 待添加（目前可使用其他方法的训练脚本）

---

## WanDiffusionWrapper 选择指南

| 方法 | Wrapper 模块 | 选择方式 |
|------|-------------|---------|
| Causal-Forcing / Self-Forcing | `core.wan_wrapper.wan_wrapper` | `get_wan_wrapper_classes('default')` |
| CausVid | `core.wan_wrapper.wan_wrapper_causvid` | `get_wan_wrapper_classes('causvid')` |
| LongLive | `core.wan_wrapper.wan_wrapper_longlive` | `get_wan_wrapper_classes('longlive')` |
| DeepForcing | `core.wan_wrapper.wan_wrapper_deepforcing` | `get_wan_wrapper_classes('deepforcing')` |

关键差异：

| 参数 | Base (CF/SF) | CausVid | LongLive | DeepForcing |
|------|-------------|---------|----------|----------|
| `cache_start` | 有 | 无 | 有 | 有 |
| `current_end` | 无 | 有 | 无 | 无 |
| `sink_recache_after_switch` | 无 | 无 | 有 | 无 |
| `classify_mode` | 有 | 无 | 有 | 有 |
| `concat_time_embeddings` | 有 | 无 | 有 | 有 |
| `clean_x` / `aug_t` | 有 | 无 | 有 | 有 |
| `is_ds_only` | 无 | 无 | 无 | 有 |
| `budget` | 无 | 无 | 无 | 有 |
| `recent` | 无 | 无 | 无 | 有 |
| 返回值 | `(flow_pred, pred_x0)` | `pred_x0` | `(flow_pred, pred_x0)` | `(flow_pred, pred_x0)` |
| `decode_to_pixel_chunk()` | 无 | 无 | 有 | 无 |

## CausalModel 选择指南

| 方法 | CausalModel 模块 | 选择方式 |
|------|-----------------|---------|
| Causal-Forcing / Self-Forcing | `wan.modules.causal_model` | `get_causal_model_class('default')` |
| CausVid | `wan.modules.causal_model_causvid` | `get_causal_model_class('causvid')` |
| LongLive | `wan.modules.causal_model_longlive` | `get_causal_model_class('longlive')` |
| LongLive (Infinity) | `wan.modules.causal_model_infinity` | `get_causal_model_class('infinity')` |
| DeepForcing | `wan.modules.causal_model` 或 `causal_model_DS` | `get_causal_model_class('default')` + 配置参数 |

关键差异：

| 特性 | Default (CF/SF) | CausVid | LongLive | Infinity | DeepForcing |
|------|----------------|---------|----------|----------|----------|
| 局部注意力参数 | `local_attn_size`, `sink_size` | `window_size` | `local_attn_size`, `sink_size` | `local_attn_size`, `sink_size` | `local_attn_size`, `sink_size` |
| cache 控制 | `cache_start` | `current_end` | `cache_start` + 延迟更新 | `cache_start` + 延迟更新 | `cache_start` |
| Teacher Forcing | 有 | 无 | 有 | 未实现 | 无（待验证） |
| 延迟 cache 更新 | 无 | 无 | `_apply_cache_updates()` | `_apply_cache_updates()` | 无 |
| RoPE | `causal_rope_apply()` | `causal_rope_apply()` | `causal_rope_apply()` | `block_relativistic_rope()` | `causal_rope_apply()` |
| Sink recache | 无 | 无 | `sink_recache_after_switch` | `sink_recache_after_switch` | 无 |
| Prompt Context (PC) | 无 | 无 | 无 | 无 | `PC_capacity`, `PC_window` |
| DS 模式支持 | 无 | 无 | 无 | 无 | `CausalWanModelDS` |

## BaseModel 继承体系

```
BaseModel (methods/base/base_causal_forcing.py)
├── SelfForcingModel       → 被各方法的 DMD 继承
├── TeacherForcingModel    → [DIFF-CausalForcing] 仅 Causal-Forcing 使用
└── BidirectionalModel     → [DIFF-CausalForcing] 仅 Causal-Forcing 使用

BaseModel (methods/base/base_self_forcing.py)
└── SelfForcingModel       → 可配置 slice_last_frames, min_num_training_frames

BaseModel (methods/base/base_longlive.py)
└── SelfForcingModel       → args.causal 标志, GPU denoising_step_list, 调试支持
```

共享方法（所有 BaseModel 版本都有）：
- `_run_generator()`：运行 generator 生成预测
- `_consistency_backward_simulation()`：一致性反向模拟
- `_initialize_inference_pipeline()`：初始化推理 pipeline

## 训练模式对比

| 训练模式 | Causal-Forcing | Self-Forcing | CausVid | LongLive |
|---------|---------------|-------------|---------|----------|
| Diffusion (ODE) | `DiffusionTrainer` | `DiffusionTrainer` | `ODETrainer` | - |
| DMD 蒸馏 | `ScoreDistillationTrainer` | `ScoreDistillationTrainer` | `DistillationTrainer` | `ScoreDistillationTrainer` |
| GAN | (模型内 GAN 逻辑) | `GANTrainer` | - | - |
| Consistency Distillation | `ConsistencyDistillationTrainer` | - | - | - |
| Teacher Forcing | `TeacherForcingTrainingPipeline` | - | - | - |
| Bidirectional | `BidirectionalTrainingPipeline` | - | - | - |
| Streaming | - | - | - | `streaming_training.py` |

## 数据处理脚本

| 脚本 | 适用方法 | 说明 |
|------|---------|------|
| `scripts/create_lmdb_iterative.py` | 通用 (CF/SF) | 从 .pt 文件聚合 ODE 对到 LMDB |
| `scripts/merge_lmdb.py` | 通用 | 合并多个 LMDB 分片 |
| `scripts/merge_and_get_clean.py` | 通用 | 合并分片并提取 clean latent |
| `scripts/create_lmdb_14b_shards.py` | Self-Forcing | 处理 14B 模型数据的分片 LMDB |
| `scripts/generate_ode_pairs.py` | Self-Forcing | 生成 ODE 轨迹配对数据 |
| `scripts/compute_vae_latent.py` | CausVid | 计算 VAE latent 表示 |
| `scripts/download_mixkit.py` | CausVid | 下载 MixKit 数据集 |
| `scripts/process_mixkit.py` | CausVid | 处理 MixKit 数据集 |

---

## 如何添加新方法

### 1. 创建方法目录

```
methods/<new_method>/
  __init__.py
  dmd.py                    # DMD 模型（参考现有方法选择继承方式）
  pipelines/
    __init__.py
    <pipeline_name>.py
  trainers/
    __init__.py
    <trainer_name>.py
```

### 2. 选择 BaseModel 继承方式

- **继承 `BaseModel`**：如果新方法与 Causal-Forcing/Self-Forcing/LongLive 类似（共享 `_run_generator`、`_consistency_backward_simulation`），从 `methods/base/` 中选择最接近的基类继承
- **独立 `nn.Module`**：如果新方法的训练流程完全不同（如 CausVid），可以直接继承 `nn.Module` 自包含实现

### 3. 选择 WanWrapper

- **默认** (`core.wan_wrapper.wan_wrapper`)：需要 `cache_start`、`classify_mode`、`clean_x` 等
- **CausVid 风格** (`core.wan_wrapper.wan_wrapper_causvid`)：需要 `current_end`、简化接口
- **LongLive 风格** (`core.wan_wrapper.wan_wrapper_longlive`)：需要延迟 cache 更新、chunked decode
- **新建变体**：如果都不满足，在 `core/wan_wrapper/` 下新建 `wan_wrapper_<method>.py`，并在 `__init__.py` 的 `get_wan_wrapper_classes()` 中注册

### 4. 选择/创建 CausalModel

- 使用 `get_causal_model_class(model_type)` 选择现有版本
- 如果需要新的注意力机制，在 `wan/modules/` 下新建 `causal_model_<method>.py`，并在 `__init__.py` 的 `get_causal_model_class()` 中注册

### 5. 创建配置文件

```
configs/<new_method>/
  default_config.yaml       # [可选] 默认配置
  <method>_<task>.yaml      # 各训练/推理配置
```

### 6. 创建入口脚本

```
train_<new_method>.py       # 参考 train_causal_forcing.py
inference_<new_method>.py   # 参考 inference_causal_forcing.py
```

### 7. 在代码中标记差异

在各方法的独有逻辑处标注 `[DIFF-<MethodName>]` 注释，方便后续维护者理解差异来源。

---

## 常见陷阱

1. **WanWrapper 返回值不一致**：CausVid 的 wrapper 返回 `pred_x0`（单个 tensor），其他方法返回 `(flow_pred, pred_x0)` 元组。跨方法调用时注意解包。

2. **Denoising step list 设备**：Causal-Forcing/Self-Forcing 的 `denoising_step_list` 在 CPU 上，LongLive 在 GPU 上。直接 `.to(device)` 可能导致隐式设备转移。

3. **CausalModel 参数不兼容**：`cache_start` (CF/SF/LongLive) vs `current_end` (CausVid) vs `sink_recache_after_switch` (LongLive)。不能混用。

4. **CausVid 配置无 default_config.yaml**：CausVid 的配置文件直接加载，不与 `default_config.yaml` 合并。添加新配置时需要写全所有字段。

5. **Score model 返回值**：CausVid 的 score model 只返回 `pred_x0`，其他方法返回 `(noise_pred, x0_pred)` 元组。

6. **`local_attn_size` 语义差异**：Self-Forcing/LongLive 中 `local_attn_size` 是帧数，CausVid 的 `window_size` 是 token 数。确保 config 中的值与方法匹配。

7. **LongLive `is_causal` 标志**：LongLive 的 `args.causal` 可以为 False（使用双向注意力），其他方法始终为 True。训练和推理时确保一致。
