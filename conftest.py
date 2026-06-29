"""Pytest configuration.

Pre-populates ``sys.modules`` with lightweight stubs for packages whose
``__init__.py`` triggers ``torch.cuda`` calls at import time, so that
pytest collection works on CPU-only machines.
"""

import os as _os
import sys
import types

# ---------------------------------------------------------------------------
# Only stub when the real module hasn't been imported yet (i.e. on CPU-only
# machines where the CUDA-dependent chain would fail).  On GPU machines the
# real modules load fine and we leave them alone.
# ---------------------------------------------------------------------------
_STUB_PACKAGES = {
    "methods.reward_forcing": "methods/reward_forcing",
    "methods.reward_forcing.pipelines": None,
}

for _name, _fs_path in _STUB_PACKAGES.items():
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        if _fs_path is not None and _os.path.isdir(_fs_path):
            _m.__path__ = [_os.path.abspath(_fs_path)]
        else:
            _m.__path__ = []
        _m.__all__ = []
        sys.modules[_name] = _m
