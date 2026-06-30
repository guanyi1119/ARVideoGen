"""Debug-print helper for teleport hook.

Activates when EITHER:
- ``core.misc.debug_option.DEBUG`` is True (source-level), OR
- env var ``TELEPORT_DEBUG`` is set to a truthy value (runtime)
"""
from __future__ import annotations

import os


def is_teleport_debug() -> bool:
    """Return True when teleport-debug logging should be emitted."""
    env_val = os.environ.get("TELEPORT_DEBUG", "0").strip().lower()
    if env_val in ("1", "true", "yes", "on"):
        return True
    try:
        from core.misc.debug_option import DEBUG

        return bool(DEBUG)
    except Exception:
        return False


def tdprint(*args, **kwargs) -> None:
    """Print only when teleport-debug is active.  Prefix [TeleportDebug]."""
    if is_teleport_debug():
        print("[TeleportDebug]", *args, **kwargs, flush=True)
