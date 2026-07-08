"""Pytest configuration: mock torch.cuda and core.misc to avoid GPU/tensorboard deps."""
import sys
import types
import torch

# 1. Patch torch.cuda.current_device to avoid AssertionError on CPU-only envs
torch.cuda.current_device = lambda: 0  # noqa: E731

# 2. Pre-seed core.misc modules so importing StreamingTrainingModelPP doesn't
#    trigger tensorboard import (which isn't installed in CPU-only env).
_mock_debug = types.ModuleType("core.misc.debug_option")
_mock_debug.DEBUG = False
_mock_debug.LOG_GPU_MEMORY = False

_mock_memory = types.ModuleType("core.misc.memory")
_mock_memory.log_gpu_memory = lambda *a, **kw: None

sys.modules["core.misc.debug_option"] = _mock_debug
sys.modules["core.misc.memory"] = _mock_memory
