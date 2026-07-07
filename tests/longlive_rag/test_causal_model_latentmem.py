"""TDD tests for the latentmem CausalModel variant + wan_wrapper registration (RED phase).

These tests validate the contract that P3 must implement. They use AST inspection
rather than direct module import, because:
  - `wan/modules/t5.py` has `device=torch.cuda.current_device()` as a class-body
    default argument, evaluated at import time. That crashes on CPU-only torch.
  - Importing `wan.modules.*` triggers transitive import of `t5.py` on CPU envs.

So the contract is validated by parsing the file with `ast` and checking
signatures / function defaults without executing module-level code.

For the dynamic-runtime tests (actually instantiating classes), we skip when
`torch.cuda.is_available()` is False.
"""
import ast
import os
import sys
import inspect
import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CUDA_AVAILABLE = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA-enabled torch")


# ---------------------------------------------------------------------------
# Helpers — AST-based static contract checking
# ---------------------------------------------------------------------------

def _ast_module(file_path: str) -> ast.Module:
    """Parse a .py file to an AST."""
    with open(file_path, "r", encoding="utf-8") as f:
        return ast.parse(f.read(), filename=file_path)


def _find_class(module_ast: ast.Module, class_name: str) -> ast.ClassDef:
    """Return the ast.ClassDef of `class_name` from a parsed module, or raise."""
    for node in module_ast.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"class {class_name!r} not found in module")


def _find_method(class_ast: ast.ClassDef, method_name: str) -> ast.FunctionDef:
    """Return the ast.FunctionDef of a method from a class AST, or raise."""
    for node in class_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"method {method_name!r} not found in class {class_ast.name}")


def _has_param(func_ast: ast.FunctionDef, name: str, default=None) -> bool:
    """True if `func_ast`'s args list has a parameter named `name` with
    (optional) default equal to `default`. A None `default` means just check existence."""
    args = func_ast.args
    all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    # Defaults: posonly + positional pack the trailing defaults
    pos_defaults = list(args.defaults)
    n_pos = len(args.posonlyargs) + len(args.args)
    # Map parameter positions to their defaults (only the last len(pos_defaults) have one)
    has_default_at = {}
    for i, d in enumerate(pos_defaults):
        pos = n_pos - len(pos_defaults) + i
        has_default_at[pos] = d
    for idx, a in enumerate(all_args):
        if a.arg == name:
            if default is None:
                return True
            # Check whether this position has a default matching `default`
            d = has_default_at.get(idx)
            if d is None:
                return False
            # Try to evaluate the literal (covers None/0/False/-1)
            try:
                val = ast.literal_eval(d)
            except Exception:
                return False  # non-literal default like function call
            return val == default
    return False


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------

def test_latentmem_module_file_exists():
    p = os.path.join(REPO_ROOT, "wan", "modules", "causal_model_latentmem.py")
    assert os.path.isfile(p), f"missing file: {p}"


def test_wan_wrapper_longlive_rag_file_exists():
    p = os.path.join(REPO_ROOT, "core", "wan_wrapper", "wan_wrapper_longlive_rag.py")
    assert os.path.isfile(p), f"missing file: {p}"


# ---------------------------------------------------------------------------
# Latentmem CausalModel contract (via AST — works on CPU)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def latentmem_ast():
    return _ast_module(os.path.join(REPO_ROOT, "wan", "modules", "causal_model_latentmem.py"))


@pytest.fixture(scope="module")
def wrapper_ast():
    return _ast_module(os.path.join(REPO_ROOT, "core", "wan_wrapper", "wan_wrapper_longlive_rag.py"))


def test_attention_class_accepts_memory_size(latentmem_ast):
    """CausalWanSelfAttention.__init__ accepts memory_size=0 (default)."""
    cls = _find_class(latentmem_ast, "CausalWanSelfAttention")
    init_ = _find_method(cls, "__init__")
    assert _has_param(init_, "memory_size", default=0), (
        "CausalWanSelfAttention.__init__ must have memory_size=0 param"
    )


def test_attention_class_inherits_compression_alpha_from_rf(latentmem_ast):
    """Latentmem must keep RF's compression_alpha parameter (EMA sink)."""
    cls = _find_class(latentmem_ast, "CausalWanSelfAttention")
    init_ = _find_method(cls, "__init__")
    assert _has_param(init_, "compression_alpha"), (
        "CausalWanSelfAttention.__init__ must have compression_alpha param "
        "(inherited from causal_model_reward_forcing.py)"
    )


def test_attention_forward_accepts_memory_indices(latentmem_ast):
    """CausalWanSelfAttention.forward accepts memory_indices=None keyword."""
    cls = _find_class(latentmem_ast, "CausalWanSelfAttention")
    fwd = _find_method(cls, "forward")
    assert _has_param(fwd, "memory_indices", default=None), (
        "CausalWanSelfAttention.forward must have memory_indices=None"
    )


def test_attention_block_forward_accepts_memory_indices(latentmem_ast):
    """CausalWanAttentionBlock.forward must accept memory_indices."""
    cls = _find_class(latentmem_ast, "CausalWanAttentionBlock")
    fwd = _find_method(cls, "forward")
    assert _has_param(fwd, "memory_indices", default=None)


def test_model_forward_or_inference_accepts_memory_indices(latentmem_ast):
    """CausalWanModel must surface memory_indices somewhere (forward or _forward_inference)."""
    cls = _find_class(latentmem_ast, "CausalWanModel")
    found_in_fwd = _has_param(_find_method(cls, "forward"), "memory_indices")
    found_in_inference = False
    try:
        found_in_inference = _has_param(_find_method(cls, "_forward_inference"), "memory_indices")
    except AssertionError:
        pass
    assert found_in_fwd or found_in_inference, (
        "CausalWanModel must accept memory_indices in forward or _forward_inference"
    )


def test_model_init_forwards_memory_size_to_attention(latentmem_ast):
    """CausalWanModel.__init__ (or its builder) must accept memory_size, and either
    hold it as instance attribute or pass it to attention blocks. We check the simplest
    contract: __init__ accepts memory_size as a parameter."""
    cls = _find_class(latentmem_ast, "CausalWanModel")
    init_ = _find_method(cls, "__init__")
    assert _has_param(init_, "memory_size", default=0), (
        "CausalWanModel.__init__ must accept memory_size=0"
    )


# ---------------------------------------------------------------------------
# WanDiffusionWrapper forward passthrough (via AST)
# ---------------------------------------------------------------------------

def test_wan_wrapper_forward_accepts_memory_indices(wrapper_ast):
    cls = _find_class(wrapper_ast, "WanDiffusionWrapper")
    fwd = _find_method(cls, "forward")
    assert _has_param(fwd, "memory_indices", default=None)


def test_wan_wrapper_init_accepts_memory_size(wrapper_ast):
    cls = _find_class(wrapper_ast, "WanDiffusionWrapper")
    init_ = _find_method(cls, "__init__")
    assert _has_param(init_, "memory_size", default=0)


def test_wan_wrapper_imports_latentmem_model(wrapper_ast):
    """WanDiffusionWrapper must import CausalWanModel from causal_model_latentmem."""
    found = False
    for node in ast.walk(wrapper_ast):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            for n in node.names:
                if n.name == "CausalWanModel" and mod and "causal_model_latentmem" in mod:
                    found = True
                # Also accept `import package.causal_model_latentmem as ...`
                if isinstance(node, ast.Import):
                    for n2 in node.names:
                        if "causal_model_latentmem" in (n2.name or ""):
                            found = True
    assert found, "wan_wrapper_longlive_rag.py must import CausalWanModel from causal_model_latentmem"


# ---------------------------------------------------------------------------
# Factory registration (via AST)
# ---------------------------------------------------------------------------

def test_causal_model_factory_registered_latentmem():
    """wan/modules/__init__.py's get_causal_model_class must branch on 'latentmem'."""
    init_path = os.path.join(REPO_ROOT, "wan", "modules", "__init__.py")
    src = open(init_path, encoding="utf-8").read()
    assert "'latentmem'" in src or '"latentmem"' in src, (
        "get_causal_model_class must handle 'latentmem'"
    )
    assert "causal_model_latentmem" in src, (
        "the latentmem branch must import wan.modules.causal_model_latentmem"
    )


def test_wan_wrapper_factory_registered_longlive_rag():
    """core/wan_wrapper/__init__.py's get_wan_wrapper_classes must branch on 'longlive_rag'."""
    init_path = os.path.join(REPO_ROOT, "core", "wan_wrapper", "__init__.py")
    src = open(init_path, encoding="utf-8").read()
    assert "'longlive_rag'" in src or '"longlive_rag"' in src, (
        "get_wan_wrapper_classes must handle 'longlive_rag'"
    )
    assert "wan_wrapper_longlive_rag" in src, (
        "the longlive_rag branch must import core.wan_wrapper.wan_wrapper_longlive_rag"
    )


# ---------------------------------------------------------------------------
# Regression contract — existing method factories still registered
# ---------------------------------------------------------------------------

def test_existing_causal_model_methods_still_registered():
    init_path = os.path.join(REPO_ROOT, "wan", "modules", "__init__.py")
    src = open(init_path, encoding="utf-8").read()
    for name in ('default', 'causvid', 'longlive', 'infinity', 'rolling_forcing'):
        assert f"'{name}'" in src or f'"{name}"' in src, f"method '{name}' branch missing"


def test_existing_wan_wrapper_methods_still_registered():
    init_path = os.path.join(REPO_ROOT, "core", "wan_wrapper", "__init__.py")
    src = open(init_path, encoding="utf-8").read()
    for name in ('default', 'causvid', 'longlive', 'deepforcing', 'rolling_forcing',
                 'reward_forcing', 'reward_forcing_3sink'):
        assert f"'{name}'" in src or f'"{name}"' in src, f"wrapper method '{name}' branch missing"


# ---------------------------------------------------------------------------
# Dynamic runtime tests — skip on CPU
# ---------------------------------------------------------------------------

@skip_no_cuda
def test_causal_model_factory_runtime():
    from wan.modules import get_causal_model_class
    cls = get_causal_model_class('latentmem')
    assert cls.__name__ == "CausalWanModel"


@skip_no_cuda
def test_wan_wrapper_factory_runtime():
    from core.wan_wrapper import get_wan_wrapper_classes
    result = get_wan_wrapper_classes('longlive_rag')
    assert isinstance(result, tuple) and len(result) >= 3