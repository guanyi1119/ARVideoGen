import argparse
import os
from omegaconf import OmegaConf

from core.config import load_config
from methods.longlive.trainers import ScoreDistillationTrainer

DEVICE_TYPE = os.environ.get('DEVICE_TYPE', 'cuda')
if DEVICE_TYPE == "npu":
    from torch_npu.contrib import transfer_to_npu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument("--wandb-save-dir", type=str, default="")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--no-auto-resume", action="store_true")
    parser.add_argument("--no-one-logger", action="store_true")

    args = parser.parse_args()

    default_config_path = os.path.join(os.path.dirname(args.config_path), "default_config.yaml")
    config = load_config(args.config_path, default_config_path=default_config_path)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize
    config_name = os.path.dirname(args.config_path).split("/")[-1]
    config.config_name = config_name
    output_root = os.environ.get('OUTPUT_URL', '.')
    config.logdir = os.path.join(output_root, args.logdir)
    config.wandb_save_dir = os.path.join(output_root, args.wandb_save_dir)
    config.disable_logging = args.disable_wandb
    config.auto_resume = not args.no_auto_resume
    config.use_one_logger = not args.no_one_logger

    if config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    else:
        raise ValueError(f"Unknown trainer: {config.trainer}")
    trainer.train()


if __name__ == "__main__":
    main()
