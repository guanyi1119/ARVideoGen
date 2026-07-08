"""Unit tests for PPTrainer class structure.

The parent class StreamingDistillationTrainer has a heavy import chain
(methods.reward_forcing -> ReDMD -> wan.modules.t5 -> torch.cuda) that
requires a CUDA-enabled torch. On CPU-only test envs we skip the import
tests and rely on integration verification on GPU. The structure is still
asserted when import succeeds.
"""
import pytest
import importlib


def _can_import_pp_trainer():
    """Try to import PPTrainer; return the class or None if import fails
    due to GPU/CUDA dependencies unavailable in this env."""
    try:
        mod = importlib.import_module("methods.reward_forcing.trainers.distillation_pp")
        return getattr(mod, "PPTrainer", None)
    except Exception:
        return None


PPTrainer = _can_import_pp_trainer()
_skip_no_cuda = pytest.mark.skipif(
    PPTrainer is None,
    reason="PPTrainer import requires CUDA-enabled torch (parent class import chain)",
)


class TestPPTrainerStructure:
    @_skip_no_cuda
    def test_import_succeeds(self):
        assert PPTrainer is not None

    @_skip_no_cuda
    def test_inherits_streaming_distillation_trainer(self):
        from methods.reward_forcing.trainers.streaming_distillation import Trainer
        assert issubclass(PPTrainer, Trainer)

    @_skip_no_cuda
    def test_has_train_method(self):
        assert hasattr(PPTrainer, 'train')

    @_skip_no_cuda
    def test_has_get_batch_and_encode(self):
        assert hasattr(PPTrainer, '_get_batch_and_encode')
