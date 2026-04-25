import argparse
import os
from omegaconf import OmegaConf

from core.config import load_config
from methods.causvid.trainers import DistillationTrainer, ODETrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")

    args = parser.parse_args()

    config = load_config(args.config_path)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    output_root = os.environ.get('OUTPUT_URL', '.')
    config.output_path = os.path.join(output_root, config.output_path)

    if config.trainer == "distillation":
        trainer = DistillationTrainer(config)
    elif config.trainer == "ode":
        trainer = ODETrainer(config)
    else:
        raise ValueError(f"Unknown trainer: {config.trainer}")
    trainer.train()


if __name__ == "__main__":
    main()
