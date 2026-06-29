"""
PromptRewriter: rule-based prompt rewriting for teleport mitigation.

Takes the original next-segment prompt and the three-way visibility
classification from ``TeleportVLM.check_visibility``, and appends
templated suffix notes to guide the video generation model away from
teleport artifacts.

Pure rule-based — no LLM / VLM calls, no natural-language parsing.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

RewriteMode = Literal["off", "conservative", "aggressive"]

_VISIBILITY_T = Dict[str, List[str]]

# ---------------------------------------------------------------------------
# Template constants (module-level, can be injected for testing)
#
# Hook point: in the future these could be loaded from a YAML file, but
# for now they live as code constants to keep the module zero-dependency.
# ---------------------------------------------------------------------------

_REWRITE_TEMPLATES: Dict[str, Dict[str, str]] = {
    "conservative": {
        # Light-touch notes — minimally invasive, just flags entities that
        # are no longer fully visible.  Visible entities are NOT altered.
        "absent_suffix": "Note: {entities} are no longer in the scene.",
        "partial_suffix": "Note: {entities} are only partially visible.",
    },
    "aggressive": {
        # Emphasises re-entry rules to guide the diffusion model toward
        # edge-aware entrance (Stage 2 edge-aware regex will exploit this).
        "absent_suffix": (
            "{entities} have completely left the scene and must not reappear "
            "suddenly. If they return, they must enter naturally from the "
            "edge of the frame."
        ),
        "partial_suffix": (
            "{entities} are partially out of frame; describe their re-entry "
            "from the edge if applicable."
        ),
    },
}

# Canonical set of mode values (used for validation)
_VALID_MODES = frozenset({"off", "conservative", "aggressive"})


# ---------------------------------------------------------------------------
# PromptRewriter
# ---------------------------------------------------------------------------


class PromptRewriter:
    """Rule-based prompt suffix appender driven by VLM visibility results.

    Parameters:
        mode:
            One of ``"off"``, ``"conservative"``, ``"aggressive"``.
            ``"off"`` short-circuits ``rewrite()`` to return the original
            prompt unchanged.
        templates:
            Optional override for the default ``_REWRITE_TEMPLATES``.
            Must contain ``"conservative"`` and ``"aggressive"`` keys
            with ``"absent_suffix"`` / ``"partial_suffix"`` sub-keys.
            Missing sub-keys are back-filled from the defaults (deep-copy
            merge — the caller's dict is never mutated).

    Raises:
        ValueError: If *mode* is not one of the recognised values.
    """

    def __init__(
        self,
        mode: str = "conservative",
        templates: Optional[dict] = None,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Unknown rewrite mode {mode!r}. "
                f"Supported: {sorted(_VALID_MODES)}"
            )
        self._mode: str = mode

        # Merge caller-provided templates with defaults.
        # We deep-copy the caller dict so we never mutate it.
        if templates is None:
            self._templates = deepcopy(_REWRITE_TEMPLATES)
        else:
            merged = deepcopy(_REWRITE_TEMPLATES)
            for mode_key in ("conservative", "aggressive"):
                if mode_key in templates:
                    merged.setdefault(mode_key, {})
                    for slot_key in ("absent_suffix", "partial_suffix"):
                        if slot_key in templates[mode_key]:
                            merged[mode_key][slot_key] = templates[
                                mode_key
                            ][slot_key]
            self._templates = merged

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rewrite(
        self,
        original_prompt: str,
        visibility: _VISIBILITY_T,
    ) -> str:
        """Return the (possibly rewritten) prompt string.

        Parameters:
            original_prompt:
                The original next-segment prompt **before** VLM rewriting.
            visibility:
                Three-way classification dict from
                ``TeleportVLM.check_visibility``:
                ``{"visible": [...], "absent": [...], "partial": [...]}``.

        Returns:
            Rewritten prompt string.

        Rules (applied in order):

        1. ``mode == "off"`` → return *original_prompt* as-is.
        2. All three lists empty (fail-safe) → return *original_prompt*.
        3. ``absent`` non-empty → append ``absent_suffix`` (entities joined
           with ``", "``).
        4. ``partial`` non-empty → append ``partial_suffix`` (entities
           joined with ``", "``).
        5. ``visible`` entities are **never** mentioned in the suffix
           (the original prompt already describes them).
        6. Suffix order: absent first, then partial (one space between).
        """
        # --- Rule 1: off mode ---
        if self._mode == "off":
            return original_prompt

        # --- Rule 2: fail-safe (empty visibility) ---
        absent = visibility.get("absent", [])
        partial = visibility.get("partial", [])
        # visible is deliberately ignored
        if not absent and not partial:
            return original_prompt

        # --- Pick the right template set ---
        tmpl = self._templates[self._mode]

        # --- Build suffix parts ---
        parts: List[str] = [original_prompt]

        if absent:
            entities_str = ", ".join(absent)
            parts.append(tmpl["absent_suffix"].format(entities=entities_str))

        if partial:
            entities_str = ", ".join(partial)
            parts.append(tmpl["partial_suffix"].format(entities=entities_str))

        return " ".join(parts)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_prompt_rewriter(
    cfg: Optional[dict],
) -> Optional[PromptRewriter]:
    """Factory: create a ``PromptRewriter`` from a config dict.

    Parameters:
        cfg:
            Optional config dict with keys:
            - ``mode`` (str): ``"off"`` / ``"conservative"`` / ``"aggressive"``.
            - ``templates`` (dict, optional): override default templates.

    Returns:
        ``None`` when *cfg* is ``None``, ``{}``, or ``cfg["mode"] == "off"``
        (disabled path — no rewriting overhead).
        Otherwise a ``PromptRewriter`` instance.

    Raises:
        ValueError: If ``cfg["mode"]`` is not recognised.
    """
    if cfg is None or cfg == {}:
        return None

    mode = cfg.get("mode", "off")
    if mode == "off":
        return None

    templates = cfg.get("templates", None)
    return PromptRewriter(mode=mode, templates=templates)
