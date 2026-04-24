from omegaconf import OmegaConf, DictConfig
import os


def load_config(config_path: str, default_config_path: str = None) -> DictConfig:
    """Load a config file, optionally merged with a method default config.

    Each method has its own default_config.yaml in configs/<method>/.
    The method config overrides/adds values to the default.

    Args:
        config_path: Path to the method-specific config.
        default_config_path: Path to the method's default config.
            If None, no default merging is performed.

    Returns:
        Merged OmegaConf DictConfig.
    """
    cfg = OmegaConf.load(config_path)
    if default_config_path is not None and os.path.exists(default_config_path):
        default_cfg = OmegaConf.load(default_config_path)
        cfg = OmegaConf.merge(default_cfg, cfg)
    return cfg


def save_config(cfg: DictConfig, save_path: str):
    """Save config to a YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        OmegaConf.save(cfg, f)
