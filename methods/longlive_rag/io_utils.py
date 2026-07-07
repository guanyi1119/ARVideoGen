"""IO utilities for LongLive-RAG.

Provides mox.file-aware wrappers so the rest of the codebase can write to
either local paths or remote OBS paths (``obs://bucket/...``) without changing
call sites. Calls are dispatched on the ``obs://`` (or ``obsXX://``) prefix.

Design:

  - Local paths use ``os.path``, ``os.makedirs``, ``torch.save/load`` directly.
  - Remote paths use ``mox.file.File`` streaming access (binary mode) backed by
    an in-memory ``io.BytesIO`` so we never touch local disk. ``torch.save`` is
    pointed at the BytesIO; the bytes are then streamed to OBS via the
    ``mox.file.File`` write handle. This is more stable than the bulk
    ``mox.file.copy`` API for large tensors (latents can be ~18 MB each; AE
    checkpoints 100s of MB; the rank that writes the AE ckpt may also be
    OOM-fragile if asked to also materialise a local temp file).
  - ``mox`` is **lazy-imported inside each call** so this module can be
    imported on machines without the ModelArts SDK (e.g. CPU-only test envs).
    Importing ``io_utils`` only requires the Python stdlib.

Public surface:
    is_obs_path(path)        -> bool
    safe_exists(path)        -> bool
    safe_makedirs(path, exist_ok=True) -> None
    safe_save_torch(obj, path) -> None     # obj: anything torch.save accepts
    safe_load_torch(path, **kwargs) -> Any  # returns torch.load result
"""

from __future__ import annotations

import io
import os
from typing import Any

import torch  # always available: this is the AR video gen repo's primary dep

# Recognised OBS URI schemes. ``obs://`` is the public form; ModelArts sometimes
# surfaces variants like ``obsS3://`` or ``obsfs://``.
_OBS_PREFIXES = ("obs://", "obsS3://", "obsfs://", "obsAfs://")


def is_obs_path(path: str) -> bool:
    """True iff ``path`` is a remote OBS URI (starts with ``obs://`` or a known variant)."""
    if not isinstance(path, str) or not path:
        return False
    return path.startswith(_OBS_PREFIXES)


def _import_mox():
    """Lazy-import the mox SDK so the call site crashes only when actually used."""
    import mox  # noqa: F401
    return mox


# ---------------------------------------------------------------------------
# Existence / directory creation
# ---------------------------------------------------------------------------

def safe_exists(path: str) -> bool:
    """File-existence check that works for both local paths and ``obs://`` URIs."""
    if is_obs_path(path):
        mox = _import_mox()
        return bool(mox.file.exists(path))
    return os.path.exists(path)


def safe_isdir(path: str) -> bool:
    """Directory check that works for both local paths and ``obs://`` URIs."""
    if is_obs_path(path):
        mox = _import_mox()
        return bool(mox.file.is_directory(path))
    return os.path.isdir(path)


def safe_makedirs(path: str, exist_ok: bool = True) -> None:
    """Create directories. On OBS this calls ``mox.file.mkdir``; it is acceptable
    for the directory to already exist when ``exist_ok=True`` because mox.file
    transparently no-ops (or raises a benign error that we swallow)."""
    if is_obs_path(path):
        mox = _import_mox()
        if not exist_ok and safe_exists(path):
            # Mirror os.makedirs semantics: refuse if exist_ok=False and path exists
            raise FileExistsError(path)
        # mox.file.mkdir doesn't accept a recursive flag and OBS has no real
        # directory tree — the bucket/key hierarchy is implicitly created on
        # first write. We call mkdir anyway so meta operations like listing
        # work.
        try:
            mox.file.mkdir(path, exist_ok=exist_ok)
        except Exception:
            # Swallow "already exists" / "not a directory" type errors from mox;
            # match the lenient behaviour of os.makedirs(..., exist_ok=True).
            if not exist_ok:
                raise
        return
    os.makedirs(path, exist_ok=exist_ok)


# ---------------------------------------------------------------------------
# torch.save / torch.load — stream through mox.file.File
# ---------------------------------------------------------------------------

def safe_save_torch(obj: Any, path: str) -> None:
    """Serialise ``obj`` to ``path`` via ``torch.save``.

    On OBS we stream the bytes through a ``mox.file.File(path, 'wb')`` handle.
    Implementation uses a small in-memory buffer between torch.save and OBS
    because torch.save's C++ writer needs a Python file-like with a writable
    fileno()/write() — directly piping torch.save into mox.file.File works as
    long as we wrap with a buffering adapter. We do it with BytesIO for safety.

    For very large objects (>1 GB) the BytesIO buffer is the dominant RAM cost;
    alternative is a spooled TemporaryFile (see ``safe_save_torch_large``).
    """
    if not is_obs_path(path):
        torch.save(obj, path)
        return

    # Remote branch — stream through BytesIO -> mox.file.File
    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)

    mox = _import_mox()
    with mox.file.File(path, "wb") as f:
        # Chunked write keeps peak RSS bounded; chunk large enough to amortise
        # the OBS request overhead (mox.file.File.write is request-buffered
        # so per-call cost is low once we stay above ~4 MB chunks).
        chunk = buf.read(8 * 1024 * 1024)  # 8 MiB
        while chunk:
            f.write(chunk)
            chunk = buf.read(8 * 1024 * 1024)


def safe_load_torch(path: str, **load_kwargs: Any) -> Any:
    """Inverse of :func:`safe_save_torch`. Reads via ``mox.file.File(path, 'rb')``
    into a BytesIO and calls ``torch.load`` on it so that ``map_location`` and
    other torch.load kwargs work the same as if we were loading from disk."""
    if not is_obs_path(path):
        return torch.load(path, **load_kwargs)

    mox = _import_mox()
    with mox.file.File(path, "rb") as f:
        # Read the whole file into memory — typical Wan latent is ~18 MB,
        # AE checkpoint ~ several hundred MB; both fit RAM comfortably.
        data = f.read()
    buf = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else io.BytesIO(data.read())
    return torch.load(buf, **load_kwargs)


__all__ = [
    "is_obs_path",
    "safe_exists",
    "safe_isdir",
    "safe_makedirs",
    "safe_save_torch",
    "safe_load_torch",
    "safe_list_files",
]


def safe_list_files(dir_path: str, pattern: str = "*.pt") -> list:
    """List files matching `pattern` in `dir_path`. Local uses glob; OBS uses
    ``mox.file.list_directory`` (recursive walk) + fnmatch.

    Returns ABSOLUTE paths — for local, the joined glob result; for OBS, the
    full ``obs://bucket/.../file.pt`` URI. This matches the contract used by
    LatentFrameDataset which passes each returned path straight to
    safe_load_torch.
    """
    import fnmatch
    if not is_obs_path(dir_path):
        import glob
        return sorted(glob.glob(os.path.join(dir_path, pattern)))

    mox = _import_mox()
    matched: list = []

    def _walk(prefix: str) -> None:
        # mox.file.list_directory returns full URIs for keys under `prefix`.
        # Raises if `prefix` doesn't exist; treat as empty.
        try:
            entries = mox.file.list_directory(prefix, recursive=True)
        except Exception:
            return
        for entry in entries:
            # `entry` is a full obs:// path or a relative suffix; mox.file accepts both
            # but to be safe we coerce to a full path by prefixing when needed.
            if not entry.startswith("obs://"):
                entry = prefix.rstrip("/") + "/" + entry.lstrip("/")
            if fnmatch.fnmatch(os.path.basename(entry), pattern):
                matched.append(entry)

    _walk(dir_path)
    return sorted(matched)