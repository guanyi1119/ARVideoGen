import argparse
import os
import sys

_ANYFLOW_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'methods', 'anyflow')
sys.path.insert(0, _ANYFLOW_ROOT)
os.environ['ANYFLOW_ROOT'] = _ANYFLOW_ROOT

import torch
DEVICE_TYPE = os.environ.get('DEVICE_TYPE', 'cuda')
if DEVICE_TYPE == "npu":
    from torch_npu.contrib import transfer_to_npu

from omegaconf import OmegaConf

from far.main import BaseTrainer


def resolve_paths(cfg, anyflow_root):
    """Resolve relative asset paths in config to absolute paths under anyflow_root."""
    def _resolve(obj):
        if isinstance(obj, str) and obj.startswith('assets/'):
            return os.path.join(anyflow_root, obj)
        elif isinstance(obj, dict):
            return {k: _resolve(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_resolve(v) for v in obj]
        return obj
    return _resolve(cfg)


def main():
    parser = argparse.ArgumentParser(description='AnyFlow Training & Evaluation')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to config YAML file')
    args, extra_args = parser.parse_known_args()

    cfg = OmegaConf.merge(
        OmegaConf.load(args.config_path),
        OmegaConf.from_cli(extra_args)
    )
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg['config_path'] = args.config_path

    cfg = resolve_paths(cfg, _ANYFLOW_ROOT)

    if cfg['mode'] == 'train':
        BaseTrainer(cfg).train()
    else:
        BaseTrainer(cfg).evaluate()


if __name__ == '__main__':
    main()
