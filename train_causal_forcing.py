"""
Causal Forcing Training Script

Usage:
    # Basic usage with config file
    python train_causal_forcing.py --config_path configs/causal_forcing/xxx.yaml

    # Override config parameters from command line
    python train_causal_forcing.py --config_path configs/causal_forcing/xxx.yaml \
        learning_rate=1e-4 \
        batch_size=32 \
        trainer.diffusion_steps=1000 \
        model.num_layers=12

    # With other flags
    python train_causal_forcing.py --config_path configs/causal_forcing/xxx.yaml \
        --logdir my_exp \
        --no_save \
        learning_rate=1e-4
"""
import argparse
import os
from omegaconf import OmegaConf

import torch
DEVICE_TYPE = os.environ.get('DEVICE_TYPE', 'cuda')
if DEVICE_TYPE == "npu":
    from torch_npu.contrib import transfer_to_npu

from core.config import load_config
from methods.causal_forcing.trainers import DiffusionTrainer, ODETrainer, ScoreDistillationTrainer, ConsistencyDistillationTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument("--wandb-save-dir", type=str, default="")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--tf", action="store_true")
    # Accept arbitrary overrides in the format key=value
    parser.add_argument("overrides", nargs="*", help="Override config parameters, e.g. learning_rate=1e-4 batch_size=32")

    args = parser.parse_args()

    default_config_path = os.path.join(os.path.dirname(args.config_path), "default_config.yaml")
    config = load_config(args.config_path, default_config_path=default_config_path)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize
    config.tf = args.tf
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    output_root = os.environ.get('OUTPUT_URL', '.')
    config.logdir = os.path.join(output_root, args.logdir)
    config.wandb_save_dir = os.path.join(output_root, args.wandb_save_dir)
    config.disable_logging = args.disable_wandb

    # Apply command line overrides
    for override in args.overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        try:
            # Try to parse as number first, fallback to string
            if "." in value or "e" in value.lower():
                parsed_value = float(value)
            elif value.lower() in ("true", "false"):
                parsed_value = value.lower() == "true"
            else:
                parsed_value = int(value)
        except ValueError:
            # Keep as string
            parsed_value = value
        OmegaConf.update(config, key, parsed_value, force_add=True)

    if config.trainer == "diffusion":
        trainer = DiffusionTrainer(config)
    elif config.trainer == "ode":
        trainer = ODETrainer(config)
    elif config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    elif config.trainer == "consistency_distillation":
        trainer = ConsistencyDistillationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
