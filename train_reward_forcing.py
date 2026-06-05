import argparse
import os
from omegaconf import OmegaConf

import torch
DEVICE_TYPE = os.environ.get('DEVICE_TYPE', 'cuda')
if DEVICE_TYPE == "npu":
    from torch_npu.contrib import transfer_to_npu

from core.config import load_config
from methods.reward_forcing.trainers import DiffusionTrainer, GANTrainer, ODETrainer, ScoreDistillationTrainer, RewardedDistillationTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument("--disable-logging", action="store_true")

    args = parser.parse_args()

    default_config_path = os.path.join(os.path.dirname(args.config_path), "default_config.yaml")
    config = load_config(args.config_path, default_config_path=default_config_path)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize
    config.disable_logging = args.disable_logging
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    output_root = os.environ.get('OUTPUT_URL', '.')
    config.logdir = os.path.join(output_root, args.logdir)

    if config.trainer == "diffusion":
        trainer = DiffusionTrainer(config)
    elif config.trainer == "gan":
        trainer = GANTrainer(config)
    elif config.trainer == "ode":
        trainer = ODETrainer(config)
    elif config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    elif config.trainer == "rewarded_distillation":
        trainer = RewardedDistillationTrainer(config)
    else:
        raise ValueError(f"Unknown trainer type: {config.trainer}")
    trainer.train()


if __name__ == "__main__":
    main()
