"""Test that the SF++ config loads and merges with default_config.yaml."""
import os
import pytest
from omegaconf import OmegaConf


class TestConfigLoading:
    def test_config_loads_with_defaults(self):
        config_path = "configs/reward_forcing/post_training_sfpp.yaml"
        if not os.path.exists(config_path):
            pytest.skip("Config file not found")

        # Load user yaml to get default_config_path
        user_yaml = OmegaConf.load(config_path)
        default_path = (
            getattr(user_yaml, "default_config_path", None)
            or os.path.join(os.path.dirname(config_path), "default_config.yaml")
        )

        # Merge default + user (user wins), same as core.config.load_config
        config = OmegaConf.merge(
            OmegaConf.load(default_path),
            OmegaConf.load(config_path),
        )

        # SF++ specific fields
        assert config.sfpp_rollout_length == 150
        assert config.sfpp_window_size == 21
        assert config.sfpp_beta == 0.0

        # Merged from default_config.yaml
        assert config.causal == True

        # Trainer field
        assert config.trainer == "streaming_distillation_pp"
        assert config.distribution_loss == "dmd"

    def test_entry_point_importable(self):
        """Entry point script should be importable (syntax check).
        Full import will fail without GPU deps, so we only check the file
        parses as valid Python."""
        import ast
        entry_path = "train_reward_forcing_pp.py"
        if not os.path.exists(entry_path):
            pytest.skip("Entry point not found")
        with open(entry_path, "r", encoding="utf-8") as f:
            source = f.read()
        # Should not raise SyntaxError
        ast.parse(source)
