from .detector import (
    OpticalFlowTeleportDetector,
    TeleportDetector,
    build_teleport_detector,
)
from .hook import TeleportSwitchHook, build_teleport_switch_hook
from .registry import EntityRegistry
from .rewriter import PromptRewriter, build_prompt_rewriter
from .vlm import TeleportVLM, QwenTeleportVLM, build_teleport_vlm

from .reweighter import (
    AbsoluteReweighter,
    TeacherRelativeReweighter,
    TeleportReweighter,
    ThresholdedReweighter,
    build_reweighter,
)

# === T9: aux_loss (Family 2) ===
from methods.reward_forcing.teleport.aux_loss import (
    TeleportAuxLoss,
    MaskedMeanAuxLoss,
    build_aux_loss,
)

__all__ = [
    "EntityRegistry",
    "OpticalFlowTeleportDetector",
    "PromptRewriter",
    "QwenTeleportVLM",
    "TeleportDetector",
    "TeleportSwitchHook",
    "TeleportVLM",
    "build_prompt_rewriter",
    "build_teleport_detector",
    "build_teleport_switch_hook",
    "build_teleport_vlm",
    # T8
    "AbsoluteReweighter",
    "TeacherRelativeReweighter",
    "TeleportReweighter",
    "ThresholdedReweighter",
    "build_reweighter",
    # T9
    "TeleportAuxLoss",
    "MaskedMeanAuxLoss",
    "build_aux_loss",
]
