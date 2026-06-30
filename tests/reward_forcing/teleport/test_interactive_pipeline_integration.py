"""Integration sanity checks for hook injection in interactive_causal_inference.py.

Uses ``ast`` to parse the source file and verify structural invariants —
**never imports** ``InteractiveCausalInferencePipeline`` (would trigger CUDA).
"""

import ast
import os


_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_SRC = os.path.join(
    _PROJECT_ROOT,
    "methods",
    "reward_forcing",
    "pipelines",
    "interactive_causal_inference.py",
)


def _parse() -> ast.Module:
    with open(_SRC, encoding="utf-8") as f:
        return ast.parse(f.read(), filename=_SRC)


# ---------------------------------------------------------------------------
# 1. Hook attribute referenced (inherited from parent class)
# ---------------------------------------------------------------------------


def test_hook_attribute_referenced():
    """Source must reference ``self._teleport_hook`` (inherited from parent)."""
    source = open(_SRC, encoding="utf-8").read()
    assert "self._teleport_hook" in source, (
        "Expected 'self._teleport_hook' in interactive_causal_inference.py"
    )


# ---------------------------------------------------------------------------
# 2. torch.no_grad guard present before maybe_rewrite
# ---------------------------------------------------------------------------


def test_torch_no_grad_guard():
    """The maybe_rewrite call must be wrapped in ``with torch.no_grad():``."""
    source = open(_SRC, encoding="utf-8").read()

    assert "with torch.no_grad():" in source, (
        "Expected 'with torch.no_grad():' guard around hook call"
    )
    assert "self._teleport_hook.maybe_rewrite" in source, (
        "Expected 'self._teleport_hook.maybe_rewrite' call"
    )

    # Structural check: verify they appear in the right order
    nograd_pos = source.index("with torch.no_grad():")
    maybe_rewrite_pos = source.index("self._teleport_hook.maybe_rewrite")
    assert nograd_pos < maybe_rewrite_pos, (
        "torch.no_grad() must appear before maybe_rewrite call"
    )


# ---------------------------------------------------------------------------
# 3. maybe_rewrite is called
# ---------------------------------------------------------------------------


def test_maybe_rewrite_called():
    """Source must contain ``self._teleport_hook.maybe_rewrite``."""
    source = open(_SRC, encoding="utf-8").read()
    assert "self._teleport_hook.maybe_rewrite" in source, (
        "Expected 'self._teleport_hook.maybe_rewrite' in interactive_causal_inference.py"
    )


# ---------------------------------------------------------------------------
# 4. cond_list deferred encoding (None placeholders)
# ---------------------------------------------------------------------------


def test_cond_list_deferred_encoding():
    """cond_list[1:] must use None placeholders for deferred encoding."""
    source = open(_SRC, encoding="utf-8").read()

    # Must contain the None extension pattern
    assert "None" in source, "Expected 'None' placeholder for deferred encoding"
    assert "[None]" in source or "[None]" in source, (
        "Expected '[None] *' pattern for deferred cond_list encoding"
    )
    # Must encode segment 0 immediately
    assert "text_prompts_list[0]" in source, (
        "Expected segment 0 immediate encoding: text_prompts_list[0]"
    )


# ---------------------------------------------------------------------------
# 5. cond_list fallback encode between maybe_rewrite and _recache_after_switch
# ---------------------------------------------------------------------------


def test_cond_list_fallback_encode():
    """Between maybe_rewrite and _recache_after_switch there must be a
    cond_list[segment_idx] assignment (fallback path)."""
    source = open(_SRC, encoding="utf-8").read()

    recache_pos = source.index("self._recache_after_switch")
    hook_pos = source.index("self._teleport_hook.maybe_rewrite")

    between = source[hook_pos:recache_pos]
    assert "cond_list[segment_idx]" in between, (
        "cond_list[segment_idx] must be assigned between hook call and _recache_after_switch"
    )


# ---------------------------------------------------------------------------
# 6. No redundant OmegaConf import (hook inherited from parent class)
# ---------------------------------------------------------------------------


def test_no_omegaconf_redundant_import():
    """Source must NOT have a redundant ``from omegaconf import OmegaConf``
    (the hook is inherited from parent, child doesn't read config directly)."""
    tree = _parse()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "omegaconf":
                names = [alias.name for alias in node.names]
                assert "OmegaConf" not in names, (
                    "interactive_causal_inference.py should NOT import OmegaConf — "
                    "hook is inherited from SwitchCausalInferencePipeline parent class"
                )


# ---------------------------------------------------------------------------
# 7. per_chunk gate present and guarded by segment_idx >= 1
# ---------------------------------------------------------------------------


def test_per_chunk_gate_with_segment_guard():
    """Interactive pipeline's else branch must check
    ``self._teleport_hook.per_chunk`` AND ``segment_idx >= 1`` so the
    first segment never triggers per-chunk rewriting."""
    source = open(_SRC, encoding="utf-8").read()
    assert "self._teleport_hook.per_chunk" in source, (
        "Expected 'self._teleport_hook.per_chunk' in interactive_causal_inference.py"
    )
    assert "segment_idx >= 1" in source, (
        "Expected 'segment_idx >= 1' guard so per-chunk rewriting only "
        "happens after the first prompt switch"
    )
