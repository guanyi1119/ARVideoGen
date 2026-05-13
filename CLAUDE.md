# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

ARVideoGen是一个包含多种自回归视频生成方法的研究项目集合，基于Wan2.1基础模型。项目整合了五种主要方法：
- **Causal-Forcing**：使用teacher forcing训练自回归模型
- **Self-Forcing**：使用自强迫训练消除train-test mismatch
- **CausVid**：双向扩散初始化后蒸馏为因果模型
- **LongLive**：面向长视频生成的流式训练框架
- **DeepForcing**：使用Prompt Context机制增强长视频上下文建模

## 常用命令

### 训练命令

```bash
# Causal-Forcing 训练
python train_causal_forcing.py --config configs/causal_forcing/causal_forcing_dmd_chunkwise.yaml

# Self-Forcing 训练
python train_self_forcing.py --config configs/self_forcing/self_forcing_dmd.yaml

# CausVid 训练
python train_causvid.py --config configs/causvid/wan_causal_dmd.yaml

# LongLive 训练
python train_longlive.py --config configs/longlive/longlive_train_init.yaml
python train_longlive.py --config configs/longlive/longlive_train_long.yaml

# DeepForcing 训练（待添加）
# python train_deep_forcing.py --config configs/deep_forcing/deep_forcing_dmd.yaml
```

### 推理命令

```bash
# Causal-Forcing 推理
torchrun --nproc_per_node=8 inference_causal_forcing.py \
  --config configs/causal_forcing/causal_forcing_dmd_chunkwise.yaml \
  --data_path prompts.txt --output_folder outputs/

# Self-Forcing 推理
torchrun --nproc_per_node=8 inference_self_forcing.py \
  --config configs/self_forcing/self_forcing_dmd.yaml \
  --data_path prompts.txt --output_folder outputs/ --num_samples 4

# CausVid 推理（单GPU）
python inference_causvid.py \
  --config_path configs/causvid/wan_causal_dmd.yaml \
  --checkpoint_path checkpoints/causvid/

# LongLive 推理
torchrun --nproc_per_node=8 inference_longlive.py \
  --config configs/longlive/longlive_inference.yaml

# DeepForcing 推理
python inference_deep_forcing.py \
  --config_path configs/deep_forcing/deep_forcing_dmd.yaml \
  --checkpoint_path checkpoints/deep_forcing/ \
  --data_path prompts.txt --output_folder outputs/ \
  --num_output_frames 126 --Budget 16 --Recent 4

# DeepForcing 推理（DS-Only模式）
python inference_deep_forcing.py \
  --config_path configs/deep_forcing/deep_forcing_dmd.yaml \
  --checkpoint_path checkpoints/deep_forcing/ \
  --data_path prompts.txt --output_folder outputs/ \
  --num_output_frames 126 --is_ds_only 1
```

### 数据处理脚本

```bash
# 生成ODE配对数据
python scripts/generate_ode_pairs.py

# 创建LMDB数据集
python scripts/create_lmdb_iterative.py --data_path XXX --lmdb_path XXX

# CausVid数据处理
python scripts/download_mixkit.py --local_dir XXX
python scripts/compute_vae_latent.py
```

## 代码架构概览

### 目录结构

```
ARVideoGen/
├── core/                      # 共享核心模块
│   ├── data/                 # 数据集、LMDB工具
│   ├── distributed/          # 分布式训练工具
│   ├── scheduler/            # FlowMatchScheduler
│   ├── loss/                 # 去噪损失函数
│   ├── config/               # OmegaConf配置加载
│   ├── misc/                 # 种子、调试、LoRA、内存工具
│   ├── wan_wrapper/          # WanDiffusionWrapper（3个变体）
│   └── demo_utils/           # Demo工具
├── wan/                      # Wan2.1基础模型代码（来自Wan-Video/Wan2.1）
│   └── modules/              # 注意力、T5、VAE、4种CausalModel
├── methods/                  # 各方法实现
│   ├── base/                 # BaseModel继承体系
│   ├── causal_forcing/       # Causal-Forcing方法
│   ├── self_forcing/         # Self-Forcing方法
│   ├── causvid/              # CausVid方法
│   ├── longlive/             # LongLive方法
│   └── deep_forcing/         # DeepForcing方法
├── configs/                  # 各方法配置文件
├── scripts/                  # 数据处理脚本
├── train_<method>.py         # 训练入口脚本
├── inference_<method>.py     # 推理入口脚本
└── archive/                  # 原始项目归档
```

### 核心组件选择

#### WanDiffusionWrapper变体

| 方法 | Wrapper模块 | 选择方式 |
|------|-------------|---------|
| Causal-Forcing / Self-Forcing | `core.wan_wrapper.wan_wrapper` | `get_wan_wrapper_classes('default')` |
| CausVid | `core.wan_wrapper.wan_wrapper_causvid` | `get_wan_wrapper_classes('causvid')` |
| LongLive | `core.wan_wrapper.wan_wrapper_longlive` | `get_wan_wrapper_classes('longlive')` |
| DeepForcing | `core.wan_wrapper.wan_wrapper_deepforcing` | `get_wan_wrapper_classes('deepforcing')` |

关键差异：
- Base (CF/SF): `cache_start`, `classify_mode`, `clean_x`/`aug_t`, 返回 `(flow_pred, pred_x0)`
- CausVid: `current_end`, 返回 `pred_x0`
- LongLive: `sink_recache_after_switch`, `decode_to_pixel_chunk()`
- DeepForcing: `is_ds_only`, `budget`, `recent`, 支持 CausalWanModelDS

#### CausalModel变体

| 方法 | CausalModel模块 | 选择方式 |
|------|-----------------|---------|
| Causal-Forcing / Self-Forcing | `wan.modules.causal_model` | `get_causal_model_class('default')` |
| CausVid | `wan.modules.causal_model_causvid` | `get_causal_model_class('causvid')` |
| LongLive | `wan.modules.causal_model_longlive` | `get_causal_model_class('longlive')` |
| LongLive (Infinity) | `wan.modules.causal_model_infinity` | `get_causal_model_class('infinity')` |
| DeepForcing | `wan.modules.causal_model` / `wan.modules.causal_model_DS` | `get_causal_model_class('default')` 或 `CausalWanModelDS.from_pretrained()` |

### BaseModel继承体系

```
BaseModel (methods/base/base_causal_forcing.py)
├── SelfForcingModel       → 被各方法的DMD继承
├── TeacherForcingModel    → 仅Causal-Forcing使用
└── BidirectionalModel     → 仅Causal-Forcing使用

BaseModel (methods/base/base_self_forcing.py)
└── SelfForcingModel       → 可配置slice_last_frames, min_num_training_frames

BaseModel (methods/base/base_longlive.py)
└── SelfForcingModel       → args.causal标志, GPU denoising_step_list, 调试支持
```

## 各方法关键差异

### Causal-Forcing
- Teacher Forcing/Bidirectional训练
- `_prepare_generator_input()`从ODE轨迹选择timestep
- 支持CFG（`fake_guidance_scale`）
- 硬编码帧数slice=21
- `denoising_step_list`存储在CPU

### Self-Forcing
- 无Teacher Forcing/Bidirectional
- 可配置`slice_last_frames`和`min_num_training_frames`
- 追踪`local_attn_size`
- GAN训练在独立trainer中
- `denoising_step_list`存储在CPU

### CausVid
- DMD自包含`nn.Module`（不继承BaseModel）
- Task Type系统（`generator_task`等）
- 工厂模式创建pipeline
- 无default_config.yaml合并
- Score model只返回`pred_x0`

### LongLive
- 延迟cache更新（`sink_recache_after_switch`）
- `is_causal`可配置
- `denoising_step_list`存储在GPU
- 内置调试/性能追踪
- 动态序列长度计算
- Chunked VAE解码防止OOM
- Infinity注意力（Block-Relativistic RoPE）

### DeepForcing
- **DeepForcing机制**：通过Prompt Context（PC）机制实现灵活的长视频生成
- **核心参数**：新增 `is_ds_only`（启用DS模式）、`budget`（PC容量）、`recent`（最近帧窗口）
- **PC计算**：PC容量 = 1560 * budget，最近窗口 = 1560 * recent（以token数计算）
- **CausalWanModelDS**：新增DS模式专用模型，优化特定场景
- **推理过程**：打印`current_timestep`，便于调试
- **CLI参数**：支持`--extended_prompt_path`、`--num_samples`、`--save_with_index`
- **WanWrapper**：`wan_wrapper_deepforcing.py`提供完整支持

## 常见陷阱

1. **WanWrapper返回值不一致**：CausVid返回单个tensor，其他返回元组
2. **Denoising step list设备**：CF/SF在CPU，LongLive在GPU
3. **CausalModel参数不兼容**：`cache_start` vs `current_end` vs `sink_recache_after_switch`
4. **CausVid无default_config**：需要写全所有配置字段
5. **Score model返回值**：CausVid只返回`pred_x0`，其他返回`(noise_pred, x0_pred)`
6. **`local_attn_size`语义**：Self-Forcing/LongLive是帧数，CausVid的`window_size`是token数
7. **LongLive `is_causal`标志**：可以为False，其他方法始终为True
8. **DeepForcing参数传递**：`is_ds_only`, `budget`, `recent`通过`model_kwargs`传递，需在配置中正确设置
9. **CausalWanModelDS可用性**：如果没有该类，代码会自动回退到CausalWanModel，但PC功能会受限

## 配置文件说明

配置文件使用YAML格式，通过OmegaConf加载：
- Causal-Forcing/Self-Forcing/LongLive：会与同目录下的`default_config.yaml`合并
- CausVid：直接加载配置，不合并

主要配置项：
- `denoising_step_list`：去噪步骤列表
- `num_frame_per_block`：每块帧数
- `model_kwargs.timestep_shift`：时间步偏移
- `trainer`：训练器类型（`score_distillation`, `diffusion`, `gan`等）

## 依赖

主要依赖（见archive/CausVid/requirements.txt）：
- torch>=2.4.0
- torchvision>=0.19.0
- diffusers>=0.31.0
- transformers>=4.49.0
- flash_attn
- omegaconf
- lmdb
- wandb
- 其他...

## 方法文档

详细的方法说明和差异对比请参考[METHODS.md](METHODS.md)。
