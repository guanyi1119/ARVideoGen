# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

ARVideoGen是一个包含多种自回归视频生成方法的研究项目集合，基于Wan2.1基础模型。项目整合了七种主要方法：
- **Causal-Forcing**：使用teacher forcing训练自回归模型
- **Self-Forcing**：使用自强迫训练消除train-test mismatch
- **CausVid**：双向扩散初始化后蒸馏为因果模型
- **LongLive**：面向长视频生成的流式训练框架
- **DeepForcing**：使用Prompt Context机制增强长视频上下文建模
- **Rolling Forcing**：使用滚动窗口训练和随机选择推理策略的长视频生成方法
- **AnyFlow**：基于Flow Map Distillation的任意步数视频生成方法

## 常用命令

### 训练命令

```bash
# Causal-Forcing / Self-Forcing / CausVid / LongLive / Rolling Forcing
python train_<method>.py --config configs/<method>/<config>.yaml

# DeepForcing（待添加）
# python train_deep_forcing.py --config configs/deep_forcing/deep_forcing_dmd.yaml

# AnyFlow（使用torchrun + --config_path，配置为yml格式）
torchrun --nproc_per_node=8 train_anyflow.py --config_path configs/anyflow/train/farwan_causal/pretrain/train_farwan1b_student_shift5_81f_480p_lr5e-5_6k_b32.yml
```

### 推理命令

```bash
# 通用格式
torchrun --nproc_per_node=8 inference_<method>.py --config configs/<method>/<config>.yaml --data_path prompts.txt --output_folder outputs/

# CausVid（单GPU）
python inference_causvid.py --config_path configs/causvid/wan_causal_dmd.yaml --checkpoint_path checkpoints/causvid/

# AnyFlow（单样本推理）
python inference_anyflow.py --model_path checkpoints/AnyFlow-FAR-Wan2.1-1.3B-Diffusers --task_type t2v --save_dir outputs/

# AnyFlow（批量评估）
torchrun --nproc_per_node=8 inference_anyflow.py --config_path configs/anyflow/test/test_AnyFlow-FAR-Wan2.1-1.3B-Diffusers.yml
```

### 数据处理脚本

```bash
python scripts/generate_ode_pairs.py                          # Self-Forcing ODE配对
python scripts/create_lmdb_iterative.py --data_path XXX --lmdb_path XXX  # LMDB数据集
python scripts/download_mixkit.py --local_dir XXX              # CausVid MixKit下载
python scripts/compute_vae_latent.py                           # CausVid VAE latent
python scripts/convert_anyflow_to_diffusers.py                 # AnyFlow模型转换
python scripts/extract_negative_embedding.py                   # AnyFlow负嵌入提取
```

## 代码架构概览

```
ARVideoGen/
├── core/                      # 共享核心模块（data, distributed, scheduler, loss, config, misc, wan_wrapper）
├── wan/                       # Wan2.1基础模型代码（modules: 注意力, T5, VAE, CausalModel变体）
├── methods/                   # 各方法实现
│   ├── base/                 #   BaseModel继承体系
│   ├── causal_forcing/       #   Causal-Forcing
│   ├── self_forcing/         #   Self-Forcing
│   ├── causvid/              #   CausVid
│   ├── longlive/             #   LongLive
│   ├── deep_forcing/         #   DeepForcing
│   ├── rolling_forcing/      #   Rolling Forcing
│   └── anyflow/              #   AnyFlow（自包含far包 + assets）
├── configs/                  # 各方法配置文件
├── scripts/                  # 数据处理脚本
├── train_<method>.py         # 训练入口
├── inference_<method>.py     # 推理入口
└── archive/                  # 原始项目归档
```

## 核心组件选择

详细的方法差异对比请参考[METHODS.md](METHODS.md)。

| 组件 | 选择方式 |
|------|---------|
| WanDiffusionWrapper | `get_wan_wrapper_classes('default'\|'causvid'\|'longlive'\|'deepforcing'\|'rollingforcing')` |
| CausalModel | `get_causal_model_class('default'\|'causvid'\|'longlive'\|'infinity'\|'rolling_forcing')` |
| AnyFlow | 不使用WanWrapper/CausalModel，使用diffusers原生WanTransformer3DModel |

## 常见陷阱

1. **WanWrapper返回值不一致**：CausVid返回单个tensor，其他返回元组
2. **Denoising step list设备**：CF/SF在CPU，LongLive在GPU
3. **CausalModel参数不兼容**：`cache_start` vs `current_end` vs `sink_recache_after_switch`
4. **`local_attn_size`语义**：Self-Forcing/LongLive是帧数，CausVid的`window_size`是token数
5. **配置加载差异**：CF/SF/LongLive/Rolling Forcing合并`default_config.yaml`；CausVid/AnyFlow直接加载不合并
6. **AnyFlow不依赖core/和wan/**：使用自包含`far`包，入口脚本通过`sys.path.insert(0, 'methods/anyflow/')`加载，不能直接`import far`
7. **AnyFlow的`ANYFLOW_ROOT`环境变量**：入口脚本设置此变量，`far/metrics/`中使用它解析assets路径

## 开发规范

- 验证编码正确性时使用 `causvid_cpu` conda 虚拟环境
- 尽量避免为验证代码结构或框架而创建临时测试脚本，若必须创建，需在测试后删除

主要依赖：torch>=2.4.0, torchvision, diffusers>=0.31.0, transformers>=4.49.0, flash_attn, omegaconf, lmdb, wandb

AnyFlow额外依赖：peft, decord, vbench, imageio-ffmpeg（见archive/AnyFlow/requirements.txt）
