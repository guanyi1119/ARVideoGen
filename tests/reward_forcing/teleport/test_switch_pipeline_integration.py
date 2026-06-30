"""Integration sanity checks for hook injection in switch_causal_inference.py.

Uses ``ast`` to parse the source file and verify structural invariants —
**never imports** ``SwitchCausalInferencePipeline`` (would trigger CUDA).
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
    "switch_causal_inference.py",
)


def _parse() -> ast.Module:
    with open(_SRC, encoding="utf-8") as f:
        return ast.parse(f.read(), filename=_SRC)


# ---------------------------------------------------------------------------
# 1. Hook attribute present
# ---------------------------------------------------------------------------


def test_hook_attribute_exists():
    """Source must reference ``self._teleport_hook``."""
    source = open(_SRC, encoding="utf-8").read()
    assert "self._teleport_hook" in source, (
        "Expected 'self._teleport_hook' in switch_causal_inference.py"
    )


# ---------------------------------------------------------------------------
# 2. torch.no_grad guard present
# ---------------------------------------------------------------------------


def test_torch_no_grad_guard():
    """The maybe_rewrite call must be wrapped in ``with torch.no_grad():``."""
    source = open(_SRC, encoding="utf-8").read()

    # Verify the source contains both strings
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
# 3. cond_second assigned before _recache_after_switch
# ---------------------------------------------------------------------------


def test_cond_second_assigned_before_recache():
    """cond_second must be assigned (by hook or fallback) before _recache_after_switch."""
    source = open(_SRC, encoding="utf-8").read()

    # After the hook block, cond_second must be assigned before _recache
    recache_pos = source.index("self._recache_after_switch")
    hook_pos = source.index("self._teleport_hook.maybe_rewrite")

    # Between maybe_rewrite and _recache_after_switch, there should be a
    # fallback cond_second assignment
    between = source[hook_pos:recache_pos]
    assert "cond_second" in between, (
        "cond_second must be assigned between hook call and _recache_after_switch"
    )


# ---------------------------------------------------------------------------
# 4. OmegaConf import present
# ---------------------------------------------------------------------------


def test_omegaconf_import():
    """Source must import OmegaConf for config access."""
    tree = _parse()
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    found = False
    for imp in imports:
        if isinstance(imp, ast.ImportFrom):
            if imp.module == "omegaconf":
                names = [alias.name for alias in imp.names]
                if "OmegaConf" in names:
                    found = True
                    break
    assert found, "Expected 'from omegaconf import OmegaConf' in source"


# ---------------------------------------------------------------------------
# 5. OmegaConf.select usage for config access
# ---------------------------------------------------------------------------


def test_omegaconf_select_usage():
    """Config access must use OmegaConf.select for DictConfig compatibility."""
    source = open(_SRC, encoding="utf-8").read()
    assert "OmegaConf.select" in source, (
        "Expected OmegaConf.select for config access (DictConfig-safe)"
    )


# ---------------------------------------------------------------------------
# 6. cond_second = None (deferred encoding)
# ---------------------------------------------------------------------------


def test_cond_second_deferred():
    """cond_second must be initialised as None (lazy encode)."""
    source = open(_SRC, encoding="utf-8").read()
    assert "cond_second = None" in source, (
        "Expected 'cond_second = None' for deferred encoding"
    )


# ---------------------------------------------------------------------------
# 7. build_teleport_switch_hook import
# ---------------------------------------------------------------------------


def test_build_teleport_switch_hook_import():
    """Source must import build_teleport_switch_hook (possibly lazy)."""
    source = open(_SRC, encoding="utf-8").read()
    assert "build_teleport_switch_hook" in source, (
        "Expected 'build_teleport_switch_hook' reference in source"
    )


# ---------------------------------------------------------------------------
# 8. _frame_decode must align latent device with VAE device
# ---------------------------------------------------------------------------


def test_frame_decode_aligns_latent_device():
    """Regression test for the cpu/npu device mismatch bug.

    The ``_frame_decode`` closure runs ``self.vae.decode_to_pixel(latent)``.
    When ``low_memory=True`` the ``output`` tensor lives on CPU while the VAE
    lives on GPU/NPU, so the closure MUST move ``latent`` to the VAE's device
    before calling ``decode_to_pixel`` — otherwise PyTorch raises
    "Expected all tensors to be on the same device, but found at least two
    devices, cpu and npu:0".

    We check this structurally by reading the source and asserting that
    ``decode_to_pixel`` is preceded (in the closure body) by a ``.to(`` call
    that references the VAE device.
    """
    source = open(_SRC, encoding="utf-8").read()

    # Locate the _frame_decode closure body.
    closure_start = source.index("def _frame_decode")
    closure_end = source.index("self._teleport_hook = build_teleport_switch_hook")
    closure_body = source[closure_start:closure_end]

    # Must contain a .to(...) call before decode_to_pixel.
    decode_pos = closure_body.index("decode_to_pixel")
    pre_decode = closure_body[:decode_pos]

    assert ".to(" in pre_decode, (
        "_frame_decode must call .to(vae_device) before "
        "self.vae.decode_to_pixel — otherwise CPU latent + GPU/NPU VAE will "
        "raise a device mismatch error. Current closure body:\n"
        + closure_body
    )

    # Must reference vae.parameters or vae.device to derive target device.
    assert "self.vae.parameters" in closure_body or "self.vae.device" in closure_body, (
        "_frame_decode must derive the target device from self.vae "
        "(via parameters() or .device) so it adapts to wherever the VAE "
        "actually lives. Current closure body:\n"
        + closure_body
    )


# ---------------------------------------------------------------------------
# 9. per_chunk gate present in else branch
# ---------------------------------------------------------------------------


def test_per_chunk_gate_in_else_branch():
    """The else branch must reference ``self._teleport_hook.per_chunk`` to
    trigger per-chunk rewriting when configured."""
    source = open(_SRC, encoding="utf-8").read()
    assert "self._teleport_hook.per_chunk" in source, (
        "Expected 'self._teleport_hook.per_chunk' reference for per-chunk "
        "rewrite gate in switch_causal_inference.py"
    )

    # The per-chunk hook call must NOT trigger _recache_after_switch
    # (recache is only for prompt-identity switching, not per-chunk refresh).
    # Verify by checking the per_chunk reference is OUTSIDE the
    # _recache_after_switch invocation context.
    per_chunk_pos = source.index("self._teleport_hook.per_chunk")
    # Find the next _recache_after_switch occurrence AFTER the per_chunk gate.
    next_recache = source.find("self._recache_after_switch", per_chunk_pos)
    # If there is one, it must be a fresh switch entry (i.e. surrounded by
    # the using_second/segment_idx switch trigger block), NOT inside the
    # per-chunk branch.  A weak structural check: between the per_chunk
    # reference and the next recache there must be the closing of the
    # per-chunk block (we expect at least one "if (not using_second)" or
    # equivalent OR end-of-loop iteration).  Pragmatic: assert no recache
    # within 800 chars after per_chunk reference.
    if next_recache != -1:
        distance = next_recache - per_chunk_pos
        assert distance > 200 or distance < 0, (
            "_recache_after_switch appears too close to per_chunk reference — "
            "per-chunk rewrite must NOT trigger recache."
        )


# ---------------------------------------------------------------------------
# 10. inference() resets hook registry
# ---------------------------------------------------------------------------


def test_inference_resets_hook_registry():
    """inference() must call self._teleport_hook.reset() at startup so
    registry state doesn't bleed across separate inference calls."""
    source = open(_SRC, encoding="utf-8").read()
    assert "self._teleport_hook.reset()" in source, (
        "Expected 'self._teleport_hook.reset()' in switch_causal_inference.py "
        "to wipe registry state at the start of every inference call"
    )
