"""Run logging: directory creation, results saving, manifest writing."""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

from stage1.utils.manifest_parity import extract_parity_block
from stage1.utils.provenance import build_runtime_provenance


def create_run_dir(base_dir: str = "stage1/outputs") -> str:
    """Create a timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_results(
    run_dir: str,
    condition: str,
    results: List[Dict],
):
    """Save inference results as JSONL (hidden_states excluded; saved separately).

    UTF-8 explicit (Stage 1 hardening): MGSM zh + Windows cp949 ambient locale
    means an unspecified encoding can corrupt non-ASCII output_text.
    """
    path = os.path.join(run_dir, f"results_{condition}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            row = {k: v for k, v in r.items() if k != "hidden_states"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_hidden_states(
    run_dir: str,
    condition: str,
    results: List[Dict],
):
    """Save hidden states as a .pt file keyed by sample_id."""
    hs_dict = {r["sample_id"]: r["hidden_states"] for r in results}
    path = os.path.join(run_dir, f"hidden_states_{condition}.pt")
    torch.save(hs_dict, path)


def save_bds_results(
    run_dir: str,
    condition: str,
    bds_results: Dict,
):
    """Save BDS analysis results as JSON."""
    path = os.path.join(run_dir, f"bds_{condition}.json")
    serializable = {
        "aggregate":  bds_results["aggregate"],
        "n_samples":  bds_results["n_samples"],
        "b":          bds_results["b"],
        "t":          bds_results["t"],
        "per_sample": bds_results["per_sample"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def save_evaluation(
    run_dir: str,
    evaluation: Dict,
):
    """Save evaluation results as JSON."""
    path = os.path.join(run_dir, "evaluation.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)


def save_manifest(
    run_dir: str,
    config: Any,
    conditions: List[str],
    sample_ids: List[str],
    hidden_state_info: Optional[Dict] = None,
    random_donor_source_start: Optional[Dict[str, int]] = None,
    random_donor_condition_seed: Optional[Dict[str, int]] = None,
    config_path: Optional[str] = None,
):
    """
    Save run manifest with full metadata.

    Args:
        run_dir: Path to the run directory.
        config: Stage1Config object.
        conditions: List of condition names that were run.
        sample_ids: Ordered list of sample IDs.
        hidden_state_info: Dict with shape, dtype, layer_count info.
        random_donor_source_start: Actual source layer offset used for random_donor.
        random_donor_condition_seed: Condition-specific random_donor seeds
            after applying the deterministic seed formula.
        config_path: Optional path to the YAML, surfaced in runtime_provenance.
    """
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "models": {
                "recipient": config.models.recipient,
                "donor":     config.models.donor,
                # RED LIGHT Fix B: include revision fields for manifest parity.
                "recipient_revision": getattr(config.models, "recipient_revision", None),
                "donor_revision":     getattr(config.models, "donor_revision", None),
            },
            "boundary_grid": config.boundary_grid,
            "t_fixed":       config.t_fixed,
            "reference": {
                "b_ref": config.reference.b_ref,
                "t_ref": config.reference.t_ref,
            },
            "hidden_state": {"pooling": config.hidden_state.pooling},
            "dataset": {
                "name":    config.dataset.name,
                "lang":    config.dataset.lang,
                "split":   config.dataset.split,
                "debug_n": config.dataset.debug_n,
                # Stage 1 hardening (2026-04-25): pinned dataset provenance.
                "revision": getattr(config.dataset, "revision", None),
                "expected_sha256": getattr(config.dataset, "expected_sha256", None),
            },
            "generation": {
                "do_sample":      config.generation.do_sample,
                "temperature":    config.generation.temperature,
                "max_new_tokens": config.generation.max_new_tokens,
            },
        },
        "conditions":            conditions,
        "sample_ordering":       sample_ids,
        "hidden_state_pooling":  config.hidden_state.pooling,
        "reference_note": (
            f"Anchor point is hard_swap_b{config.reference.b_ref} "
            f"(b_ref={config.reference.b_ref}, t_ref={config.reference.t_ref}), selected a priori before sweep. "
            "No separate reference condition is run."
        ),
        # P2: random donor reproducibility
        "random_donor_seed":           config.random_donor.seed,
        "random_donor_source_start":   random_donor_source_start,
        "random_donor_condition_seed": random_donor_condition_seed,
        "t_fixed_justification": (
            "t=20 fixes the upper boundary at layer 20 of 28, preserving the top 8 layers as recipient (Instruct). "
            "Upper layers handle task-specific output formatting and vocabulary projection (Geva et al., 2022, arXiv:2203.14680). "
            "Bandarkar et al. (2025) use a similar upper-layer preservation structure. "
            "Sensitivity to t is deferred to Stage 2."
        ),
    }

    if hidden_state_info:
        manifest["hidden_state_layer_count"] = hidden_state_info.get("layer_count")
        manifest["hidden_state_shape"]       = hidden_state_info.get("shape")
        manifest["hidden_state_dtype"]       = hidden_state_info.get("dtype")

    # v4 P2: embed canonical parity block (with sample_regime) so Stage 1
    # manifests are interchangeable with Phase A/B's parity contract. The
    # legacy ``config`` block is kept for backward compatibility with older
    # post-analysis code that reads it directly.
    manifest["parity"] = extract_parity_block(config, sample_ids=sample_ids)

    # Stage 1 hardening (2026-04-25): full runtime provenance — git SHA,
    # torch / transformers / numpy versions, exact CLI command, dataset
    # revision + realised SHA-256, hostname, working directory.
    manifest["runtime_provenance"] = build_runtime_provenance(
        config=config,
        config_path=config_path,
    )

    path = os.path.join(run_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
