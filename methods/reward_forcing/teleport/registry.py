"""
EntityRegistry: cross-chunk persistent entity visibility state machine.

State per entity: 'never_seen' | 'visible' | 'absent'
Lock: visible → absent is permanent (never reverts).
Only tracks entities that have appeared in prompt_hint at some point.
"""

from __future__ import annotations

from typing import Dict, List


class EntityRegistry:
    """Cross-chunk persistent entity visibility state machine.

    State per entity: ``'never_seen'`` | ``'visible'`` | ``'absent'``.

    Lock: visible → absent is permanent (never reverts).
    Only tracks entities that have appeared in *prompt_hint* at some point.
    """

    def __init__(self) -> None:
        self._state: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Wipe all state.  Called at the start of every new inference call."""
        self._state.clear()

    def register_prompt_entities(self, prompt_hint: List[str]) -> None:
        """Add prompt-hint entities to the registry as NEVER_SEEN if not already
        present.  Idempotent and order-preserving."""
        for name in prompt_hint:
            if name not in self._state:
                self._state[name] = "never_seen"

    def update_from_observation(
        self,
        visible: List[str],
        absent: List[str],
        partial: List[str],
    ) -> None:
        """Apply state transitions based on a single chunk's VLM observation.

        Only entities currently tracked in the registry (i.e. previously
        added via ``register_prompt_entities``) are considered.  Out-of-registry
        names are ignored (open-vocab ``in_frame`` items that were never in
        any prompt hint).

        Transition rules:
            NEVER_SEEN + (visible | partial) → VISIBLE
            NEVER_SEEN + absent             → NEVER_SEEN (don't lock unseen)
            VISIBLE    + (visible | partial) → VISIBLE
            VISIBLE    + absent             → ABSENT (lock)
            ABSENT     + anything           → ABSENT (permanent lock)
        """
        # Build a lookup set for fast membership checks
        visible_set = set(visible)
        absent_set = set(absent)
        partial_set = set(partial)

        for name in self._state:
            current = self._state[name]
            if current == "absent":
                # Permanent lock — never revert
                continue

            in_visible = name in visible_set
            in_partial = name in partial_set
            in_absent = name in absent_set

            if current == "never_seen":
                if in_visible or in_partial:
                    self._state[name] = "visible"
                # in_absent stays never_seen (don't lock what you haven't seen)
                continue

            if current == "visible":
                if in_absent:
                    self._state[name] = "absent"
                # visible or partial keeps it visible
                continue

    def get_absent(self) -> List[str]:
        """Return the persistent ABSENT set (sorted for determinism)."""
        return sorted(
            name for name, st in self._state.items() if st == "absent"
        )

    def get_visible(self) -> List[str]:
        """Return the current VISIBLE set (sorted for determinism)."""
        return sorted(
            name for name, st in self._state.items() if st == "visible"
        )

    def snapshot(self) -> Dict[str, str]:
        """Return ordered dict ``{name: state}`` for debug logging.

        Returns a copy so callers cannot mutate internal state.
        """
        return dict(self._state)

    def __len__(self) -> int:
        return len(self._state)
