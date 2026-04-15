"""Phase B intervention module — prompt-side hidden-state patching."""

from stage1.intervention.patcher import (
    METHODOLOGY_TAG,
    PatchConfig,
    RESTORATION_PATCHES,
    REVERSE_CORRUPTION_PATCHES,
    extract_all_layer_hidden_states,
    forward_with_patches,
    get_all_patch_configs,
    run_patched_inference,
    run_patched_inference_single,
)

__all__ = [
    "METHODOLOGY_TAG",
    "PatchConfig",
    "RESTORATION_PATCHES",
    "REVERSE_CORRUPTION_PATCHES",
    "extract_all_layer_hidden_states",
    "forward_with_patches",
    "get_all_patch_configs",
    "run_patched_inference",
    "run_patched_inference_single",
]
