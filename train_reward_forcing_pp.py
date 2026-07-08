#!/usr/bin/env python
"""Self-Forcing++ post-training entry point.

Usage:
  torchrun --nproc_per_node=8 train_reward_forcing_pp.py \
      --config_path configs/reward_forcing/post_training_sfpp.yaml

SPDX-License-Identifier: Apache-2.0
"""
import argparse
import os

DEVICE_TYPE = os.environ.get('DEVICE_TYPE', 'cuda')
if DEVICE_TYPE == "npu":
    from torch_npu.contrib import transfer_to_npu

from core.config import load_config
from methods.reward_forcing.trainers.distillation_pp import PPTrainer


def main():
    parser = argparse.ArgumentParser(description="Self-Forcing++ post-training")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to SF++ post-training YAML config")
    parser.add_argument("--no_save", action="store_true",
                        help="Disable checkpoint saving")
    parser.add_argument("--no_visualize", action="store_true",
                        help="Disable visualization")
    parser.add_argument("--logdir", type=str, default="",
                        help="Output log directory")
    parser.add_argument("--disable-logging", action="store_true",
                        help="Disable wandb/tensorboard logging")
    args = parser.parse_args()

    # Load config with default merge (same pattern as train_reward_forcing.py)
    default_config_path = os.path.join(
        os.path.dirname(args.config_path), "default_config.yaml"
    )
    config = load_config(args.config_path, default_config_path=default_config_path)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize
    config.disable_logging = args.disable_logging
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    output_root = os.environ.get('OUTPUT_URL', '.')
    config.logdir = os.path.join(output_root, args.logdir)

    # Create trainer and start training
    trainer = PPTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
