"""Teleport mode normalization — zero-dependency helper.

OmegaConf parses unquoted ``off`` as boolean ``False``, which would
otherwise crash with ``ValueError``.  This module provides the single
canonical normalizer used by both the real ``ReDMD`` class and the
isolated test harness.
"""

from __future__ import annotations

import typing as _t

_VALID_TELEPORT_MODES: _t.FrozenSet[str] = frozenset({"off", "reweight", "aux_loss"})


def normalize_teleport_mode(mode: object) -> str:
    """Normalize *mode* to a canonical lowercase string.

    Handles the OmegaConf bool trap:
    - ``None`` / ``False`` → ``"off"``
    - strings are stripped and lowercased
    - everything else is stringified, stripped and lowercased
    """
    if mode is None or mode is False:
        return "off"
    if isinstance(mode, str):
        return mode.strip().lower()
    return str(mode).strip().lower()
