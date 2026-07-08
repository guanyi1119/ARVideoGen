"""Pytest configuration: mock torch.cuda and core.misc to avoid GPU/tensorboard deps.

The repo's `core.misc.__init__` and `wan.modules.t5` call torch.cuda at
module-import time, which fails on CPU-only torch. We patch torch.cuda and
pre-seed sys.modules with lightweight stubs so importing the modules under
test works without a CUDA build.
"""
import sys
import types
import torch

# 1. Patch torch.cuda.current_device to avoid AssertionError on CPU-only envs
torch.cuda.current_device = lambda: 0  # noqa: E731

# 2. Pre-seed core.misc modules so importing code that does
#    `from core.misc.debug_option import DEBUG, ...` doesn't trigger
#    tensorboard/CUDA imports. Stub every name the real modules export.
_mock_debug = types.ModuleType("core.misc.debug_option")
_mock_debug.DEBUG = False
_mock_debug.LOG_GPU_MEMORY = False
_mock_debug.DEBUG_GRADIENT = False

_mock_memory = types.ModuleType("core.misc.memory")
_mock_memory.log_gpu_memory = lambda *a, **kw: None
_mock_memory.gpu = lambda *a, **kw: None
_mock_memory.get_cuda_free_memory_gb = lambda *a, **kw: 80
_mock_memory.DynamicSwapInstaller = type("DynamicSwapInstaller", (), {"install_model": staticmethod(lambda *a, **kw: None)})

sys.modules["core.misc.debug_option"] = _mock_debug
sys.modules["core.misc.memory"] = _mock_memory
