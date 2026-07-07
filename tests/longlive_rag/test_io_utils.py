"""TDD tests for LongLive-RAG io_utils (mox.file-aware IO layer).

These tests exercise the contract that P5-mox must implement in
`methods/longlive_rag/io_utils.py`:

  - is_obs_path(path) detects `obs://` or `obsxxx://` prefixes
  - safe_exists(path) checks existence locally or via mox.file.exists
  - safe_makedirs(path, exist_ok) creates dirs locally or via mox.file
  - safe_save_torch(obj, path) writes bytes via mox.file.File (streaming)
  - safe_load_torch(path) reads bytes via mox.file.File (streaming)

cpu-only env: `mox` is NOT installed in causvid_cpu; we mock the import
globally so tests can exercise the remote branch without OBS access.
"""
import io
import os
import sys
import types
import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# is_obs_path — pure string check, never needs mox
# ---------------------------------------------------------------------------

def test_is_obs_path_detects_obs_prefix():
    from methods.longlive_rag.io_utils import is_obs_path
    assert is_obs_path("obs://bucket/path/to/file.pt")
    assert is_obs_path("obsS3://bucket/key")
    assert is_obs_path("obsfs://xxx")


def test_is_obs_path_rejects_local_paths():
    from methods.longlive_rag.io_utils import is_obs_path
    assert not is_obs_path("/tmp/foo.pt")
    assert not is_obs_path("datasets/longlive_rag/latent_000000.pt")
    assert not is_obs_path("relative/path")
    assert not is_obs_path("oss://other-provider/path")  # not obs
    assert not is_obs_path("")


# ---------------------------------------------------------------------------
# mox mocket — install a fake `mox` module before tests run
# ---------------------------------------------------------------------------

class _FakeMoxFile:
    """Simulates mox.file.File for test purposes — a binary file-like wrapper
    over an in-memory BytesIO when the path is obs://.

    Real mox.file.File offers mode 'wb' / 'rb' streaming access to OBS.
    """
    _store = {}   # class-level "remote storage": path -> bytes

    def __init__(self, path, mode="rb"):
        self._path = path
        self._mode = mode
        if "w" in mode:
            self._buf = io.BytesIO()
        elif "r" in mode:
            if path not in self._store:
                raise FileNotFoundError(f"obs path not found: {path}")
            self._buf = io.BytesIO(self._store[path])
        else:
            raise ValueError(f"unsupported mode: {mode}")

    def write(self, data):
        return self._buf.write(data)

    def read(self, *args):
        return self._buf.read(*args)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if "w" in self._mode:
            self._store[self._path] = self._buf.getvalue()
        self._buf.close()
        return False


class _FakeMoxModule:
    """Minimal stand-in for `import mox`."""
    class _file_ns:
        File = _FakeMoxFile
        @staticmethod
        def exists(path):
            return path in _FakeMoxFile._store
        @staticmethod
        def is_directory(path):
            # Treat any path that has a written file as a directory ancestor.
            return any(k.startswith(path.rstrip("/") + "/") for k in _FakeMoxFile._store)
        @staticmethod
        def mkdir(path, exist_ok=False):
            # No-op: the fake store has no real dirs.
            return None
    file = _file_ns


@pytest.fixture(autouse=True)
def _install_fake_mox(monkeypatch):
    """Install the fake mox module into sys.modules before any test imports io_utils.

    Tests that want to mox** on the local branch simply reset the path to local.
    """
    fake = types.ModuleType("mox")
    fake.file = _FakeMoxModule.file
    monkeypatch.setitem(sys.modules, "mox", fake)
    # Reset the in-memory store between tests
    _FakeMoxFile._store.clear()
    yield
    _FakeMoxFile._store.clear()


# ---------------------------------------------------------------------------
# safe_exists
# ---------------------------------------------------------------------------

def test_safe_exists_local_true(tmp_path):
    from methods.longlive_rag.io_utils import safe_exists
    p = tmp_path / "foo.txt"
    p.write_text("x")
    assert safe_exists(str(p)) is True


def test_safe_exists_local_false(tmp_path):
    from methods.longlive_rag.io_utils import safe_exists
    assert safe_exists(str(tmp_path / "nope.txt")) is False


def test_safe_exists_obs_true():
    # Pre-populate the fake store
    _FakeMoxFile._store["obs://bkt/foo.pt"] = b"\x00"
    from methods.longlive_rag.io_utils import safe_exists
    assert safe_exists("obs://bkt/foo.pt") is True


def test_safe_exists_obs_false():
    from methods.longlive_rag.io_utils import safe_exists
    assert safe_exists("obs://bkt/missing.pt") is False


# ---------------------------------------------------------------------------
# safe_save_torch / safe_load_torch round-trip
# ---------------------------------------------------------------------------

def test_safe_save_torch_local_roundtrip(tmp_path):
    from methods.longlive_rag.io_utils import safe_save_torch, safe_load_torch
    obj = {"a": torch.tensor([1.0, 2.0, 3.0]), "b": "hello"}
    p = tmp_path / "x.pt"
    safe_save_torch(obj, str(p))
    assert p.exists()
    out = safe_load_torch(str(p))
    assert torch.allclose(out["a"], obj["a"])
    assert out["b"] == obj["b"]


def test_safe_save_torch_obs_roundtrip():
    from methods.longlive_rag.io_utils import safe_save_torch, safe_load_torch
    obj = {"a": torch.tensor([4.5, 5.5]), "b": "remote"}
    p = "obs://bucket/test/roundtrip.pt"
    safe_save_torch(obj, p)
    # Should be present in fake store
    assert p in _FakeMoxFile._store
    out = safe_load_torch(p)
    assert torch.allclose(out["a"], obj["a"])
    assert out["b"] == obj["b"]


def test_safe_save_torch_obs_uses_mox_file():
    """Saving to obs:// should go through mox.file.File streaming, NOT torch.save(path)."""
    from methods.longlive_rag.io_utils import safe_save_torch
    safe_save_torch({"t": torch.zeros(2)}, "obs://bucket/x.pt")
    # Verify the fake mox store actually received the bytes
    assert "obs://bucket/x.pt" in _FakeMoxFile._store
    assert len(_FakeMoxFile._store["obs://bucket/x.pt"]) > 0


# ---------------------------------------------------------------------------
# safe_makedirs
# ---------------------------------------------------------------------------

def test_safe_makedirs_local(tmp_path):
    from methods.longlive_rag.io_utils import safe_makedirs
    p = tmp_path / "a" / "b" / "c"
    safe_makedirs(str(p), exist_ok=True)
    assert p.is_dir()


def test_safe_makedirs_obs_no_crash():
    """safe_makedirs on obs:// must not raise — mox.file.mkdir is called."""
    from methods.longlive_rag.io_utils import safe_makedirs
    safe_makedirs("obs://bucket/new/dir", exist_ok=True)  # fake store no-ops


# ---------------------------------------------------------------------------
# Contracts on the module's public API surface
# ---------------------------------------------------------------------------

def test_io_utils_public_api():
    from methods.longlive_rag import io_utils
    for fn in ("is_obs_path", "safe_exists", "safe_makedirs",
                "safe_save_torch", "safe_load_torch"):
        assert hasattr(io_utils, fn), f"io_utils missing {fn}"


def test_io_utils_mox_is_lazy_imported_inside_call():
    """The mox import must be inside the function (lazy) so that importing
    io_utils on a machine without mox installed doesn't crash. Verified by
    uninstalling the fake mox and confirming a LOCAL operation still works.
    """
    import importlib
    saved_mox = sys.modules.pop("mox", None)
    try:
        # Force re-import so any module-level `import mox` would have re-triggered.
        import methods.longlive_rag.io_utils as iou
        importlib.reload(iou)
        # Local path operations must still work
        assert iou.is_obs_path("local.pt") is False
        assert iou.safe_exists("local.pt") is False
    finally:
        if saved_mox is not None:
            sys.modules["mox"] = saved_mox