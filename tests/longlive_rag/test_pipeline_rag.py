"""TDD tests for the LongLive-RAG inference pipeline + inference entry (RED phase).

These tests exercise the contract that P4 must implement. As with P3, they use
AST inspection rather than direct module import, because importing
`methods.reward_forcing.pipelines.*` transitively imports `wan/modules/t5.py`
which evaluates `torch.cuda.current_device()` at class-body load time — fails
on CPU-only torch.

For dynamic-runtime tests, we skip when `torch.cuda.is_available()` is False.
"""
import ast
import os
import sys
import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CUDA_AVAILABLE = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA-enabled torch")


def _ast_module(file_path: str) -> ast.Module:
    with open(file_path, "r", encoding="utf-8") as f:
        return ast.parse(f.read(), filename=file_path)


def _find_class(module_ast: ast.Module, class_name: str) -> ast.ClassDef:
    for node in module_ast.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"class {class_name!r} not found in module")


def _find_method(class_ast: ast.ClassDef, method_name: str) -> ast.FunctionDef:
    for node in class_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"method {method_name!r} not found in class {class_ast.name}")


def _has_param(func_ast: ast.FunctionDef, name: str, default=None) -> bool:
    args = func_ast.args
    all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    pos_defaults = list(args.defaults)
    n_pos = len(args.posonlyargs) + len(args.args)
    has_default_at = {}
    for i, d in enumerate(pos_defaults):
        pos = n_pos - len(pos_defaults) + i
        has_default_at[pos] = d
    kw_defaults = list(args.kw_defaults) if hasattr(args, "kw_defaults") else []
    kw_only = list(args.kwonlyargs)
    for i, d in enumerate(kw_defaults):
        idx = len(all_args) - len(kw_only) + i
        if d is not None:
            has_default_at[idx] = d
    for idx, a in enumerate(all_args):
        if a.arg == name:
            if default is None:
                return True
            d = has_default_at.get(idx)
            if d is None:
                return False
            try:
                val = ast.literal_eval(d)
            except Exception:
                return False
            return val == default
    return False


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------

def test_pipelines_file_exists():
    p = os.path.join(REPO_ROOT, "methods", "longlive_rag", "pipelines.py")
    assert os.path.isfile(p), f"missing: {p}"


def test_inference_entry_exists():
    p = os.path.join(REPO_ROOT, "inference_longlive_rag.py")
    assert os.path.isfile(p), f"missing: {p}"


def test_default_config_exists():
    p = os.path.join(REPO_ROOT, "configs", "longlive_rag", "default_config.yaml")
    assert os.path.isfile(p), f"missing: {p}"


def test_inference_yaml_reward_forcing_exists():
    p = os.path.join(REPO_ROOT, "configs", "longlive_rag", "reward_forcing_latentmem.yaml")
    assert os.path.isfile(p), f"missing: {p}"


def test_inference_yaml_longlive_exists():
    p = os.path.join(REPO_ROOT, "configs", "longlive_rag", "longlive_latentmem.yaml")
    assert os.path.isfile(p), f"missing: {p}"


# ---------------------------------------------------------------------------
# LatentMemCausalInferencePipeline class contract (AST)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipelines_ast():
    return _ast_module(os.path.join(REPO_ROOT, "methods", "longlive_rag", "pipelines.py"))


def test_pipeline_class_defined(pipelines_ast):
    cls = _find_class(pipelines_ast, "LatentMemCausalInferencePipeline")
    assert cls is not None


def test_pipeline_class_extends_causal_inference_pipeline(pipelines_ast):
    cls = _find_class(pipelines_ast, "LatentMemCausalInferencePipeline")
    bases = [b for b in cls.bases if isinstance(b, ast.Name)]
    base_names = [b.id for b in bases]
    assert "CausalInferencePipeline" in base_names, (
        f"LatentMemCausalInferencePipeline must inherit CausalInferencePipeline, got bases={base_names}"
    )


def test_pipeline_imports_causal_inference_pipeline(pipelines_ast):
    """pipelines.py must ImportFrom methods.reward_forcing.pipelines importing CausalInferencePipeline."""
    found = False
    for node in ast.walk(pipelines_ast):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if "methods.reward_forcing.pipelines" in mod:
                for n in node.names:
                    if n.name == "CausalInferencePipeline":
                        found = True
    assert found, "pipelines.py must import CausalInferencePipeline from methods.reward_forcing.pipelines"


def test_pipeline_imports_latentae(pipelines_ast):
    """pipelines.py must import LatentAE from methods.longlive_rag.ae.model."""
    found = False
    for node in ast.walk(pipelines_ast):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if "methods.longlive_rag.ae.model" in mod:
                for n in node.names:
                    if n.name == "LatentAE":
                        found = True
    assert found, "pipelines.py must import LatentAE from methods.longlive_rag.ae.model"


def test_pipeline_init_accepts_device_and_args(pipelines_ast):
    """LatentMemCausalInferencePipeline.__init__ must accept args (config) and device.

    It should mirror CausalInferencePipeline.__init__ signature shape
    (self, args, device, generator=None, text_encoder=None, vae=None).
    """
    cls = _find_class(pipelines_ast, "LatentMemCausalInferencePipeline")
    init_ = _find_method(cls, "__init__")
    arg_names = list(args_.arg for args_ in list(init_.args.posonlyargs) + list(init_.args.args))
    assert "args" in arg_names, f"__init__ must accept 'args' parameter, got {arg_names}"
    assert "device" in arg_names, f"__init__ must accept 'device', got {arg_names}"


def test_pipeline_inference_accepts_memory_indices_passing(pipelines_ast):
    """inference() method must accept the standard inference kwargs
    (noise, text_prompts, return_latents, etc) — used by the entry script."""
    cls = _find_class(pipelines_ast, "LatentMemCausalInferencePipeline")
    fwd = _find_method(cls, "inference")
    arg_names = list(args_.arg for args_ in list(fwd.args.posonlyargs) + list(fwd.args.args))
    assert "noise" in arg_names
    assert "text_prompts" in arg_names
    assert "return_latents" in arg_names or any(a.arg == "return_latents" for a in fwd.args.kwonlyargs), (
        "inference() must accept return_latents kwarg"
    )


def test_pipeline_source_contains_topk_retrieval(pipelines_ast):
    """The pipeline source must contain some top-K retrieval logic over latent_descriptors."""
    src = ast.unparse(pipelines_ast)
    # Look for either torch.Tensor.topk call or topk method on a similarity tensor
    assert "topk" in src.lower(), (
        "RAG pipeline must compute top-k similarity over latent_descriptors"
    )


def test_pipeline_source_contains_memory_indices_passing(pipelines_ast):
    """The pipeline source must call self.generator with memory_indices."""
    src = ast.unparse(pipelines_ast)
    assert "memory_indices" in src, (
        "RAG pipeline must pass memory_indices to self.generator"
    )


def test_pipeline_source_contains_latent_descriptors(pipelines_ast):
    """The pipeline must maintain a latent_descriptors list."""
    src = ast.unparse(pipelines_ast)
    assert "latent_descriptors" in src, (
        "RAG pipeline must keep self.latent_descriptors"
    )


def test_pipeline_source_contains_recent_exclude(pipelines_ast):
    """The pipeline must respect recent_exclude config (skip recent blocks)."""
    src = ast.unparse(pipelines_ast)
    assert "recent_exclude" in src, (
        "RAG pipeline must query config.recent_exclude"
    )


def test_pipeline_source_contains_ae_model_load(pipelines_ast):
    """The pipeline must instantiate LatentAE from a config-supplied checkpoint."""
    src = ast.unparse(pipelines_ast)
    assert "ae_ckpt" in src, "RAG pipeline must read ae_ckpt from config"
    assert "LatentAE" in src, "RAG pipeline must construct a LatentAE"


# ---------------------------------------------------------------------------
# Inference entry script contract (AST)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def inference_entry_ast():
    return _ast_module(os.path.join(REPO_ROOT, "inference_longlive_rag.py"))


def test_inference_entry_imports_latentmem_pipeline(inference_entry_ast):
    """inference_longlive_rag.py must import LatentMemCausalInferencePipeline."""
    found = False
    for node in ast.walk(inference_entry_ast):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if "methods.longlive_rag.pipelines" in mod:
                for n in node.names:
                    if n.name == "LatentMemCausalInferencePipeline":
                        found = True
    assert found, "inference_longlive_rag.py must import LatentMemCausalInferencePipeline"


# ---------------------------------------------------------------------------
# Inference yaml contracts
# ---------------------------------------------------------------------------

def test_yaml_reward_forcing_latentmem_has_rag_fields():
    import yaml
    p = os.path.join(REPO_ROOT, "configs", "longlive_rag", "reward_forcing_latentmem.yaml")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    assert "model_kwargs" in cfg
    mk = cfg["model_kwargs"]
    for field in ("memory_size", "use_latentmem", "compression_method", "ae_ckpt"):
        assert field in mk, f"reward_forcing_latentmem.yaml missing model_kwargs.{field}"
    assert mk["compression_method"] == "ae", "compression_method must be 'ae'"
    assert mk["memory_size"] >= 1, "memory_size must be >= 1 for RAG inference"


def test_yaml_longlive_latentmem_has_lora_and_rag_fields():
    import yaml
    p = os.path.join(REPO_ROOT, "configs", "longlive_rag", "longlive_latentmem.yaml")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    assert "lora_ckpt" in cfg, "longlive_latentmem.yaml must have lora_ckpt"
    assert "adapter" in cfg, "longlive_latentmem.yaml must have adapter block"
    assert "model_kwargs" in cfg
    for field in ("memory_size", "use_latentmem", "compression_method", "ae_ckpt"):
        assert field in cfg["model_kwargs"], f"longlive_latentmem.yaml missing model_kwargs.{field}"


def test_yaml_reward_forcing_has_no_lora():
    import yaml
    p = os.path.join(REPO_ROOT, "configs", "longlive_rag", "reward_forcing_latentmem.yaml")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    assert cfg.get("lora_ckpt") is None, "reward_forcing_latentmem.yaml must NOT have lora_ckpt"


def test_yaml_both_use_method_longlive_rag():
    """Both inference yamls must select 'longlive_rag' wrapper/method (if present)."""
    import yaml
    for name in ("reward_forcing_latentmem.yaml", "longlive_latentmem.yaml"):
        p = os.path.join(REPO_ROOT, "configs", "longlive_rag", name)
        with open(p) as f:
            cfg = yaml.safe_load(f)
        # Check either model_kwargs.method or top-level method or absence (default_config.yaml controls)
        # We accept use_latentmem flag as the contract.
        assert cfg["model_kwargs"].get("use_latentmem") is True


# ---------------------------------------------------------------------------
# Default config — optional RAG fields
# ---------------------------------------------------------------------------

def test_yaml_default_config_has_optional_rag_fields():
    """default_config.yaml must accept memory_size, recent_exclude, etc without breaking."""
    import yaml
    p = os.path.join(REPO_ROOT, "configs", "longlive_rag", "default_config.yaml")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    # Either explicit keys or absence is fine — but the file must exist and be parseable
    assert cfg is not None
    # Smoke: existing model_kwargs must not break
    if "model_kwargs" in cfg:
        assert isinstance(cfg["model_kwargs"], dict)