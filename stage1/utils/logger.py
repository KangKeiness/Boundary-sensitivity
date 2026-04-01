"""Run logging: directory creation, results saving, manifest writing."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


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
    """
    Save inference results as JSONL (without hidden_states, which are saved separately).
    """
    path = os.path.join(run_dir, f"results_{condition}.jsonl")
    with open(path, "w") as f:
        for r in results:
            row = {k: v for k, v in r.items() if k != "hidden_states"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_hidden_states(
    run_dir: str,
    condition: str,
    results: List[Dict],
):
    """Save hidden states as a .pt file keyed by sample_id."""
    hs_dict = {}
    for r in results:
        hs_dict[r["sample_id"]] = r["hidden_states"]

    path = os.path.join(run_dir, f"hidden_states_{condition}.pt")
    torch.save(hs_dict, path)


def save_bds_results(
    run_dir: str,
    condition: str,
    bds_results: Dict,
):
    """Save BDS analysis results as JSON."""
    path = os.path.join(run_dir, f"bds_{condition}.json")

    # Make serializable
    serializable = {
        "aggregate": bds_results["aggregate"],
        "n_samples": bds_results["n_samples"],
        "b": bds_results["b"],
        "t": bds_results["t"],
        "per_sample": bds_results["per_sample"],
    }

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def save_evaluation(
    run_dir: str,
    evaluation: Dict,
):
    """Save evaluation results as JSON."""
    path = os.path.join(run_dir, "evaluation.json")
    with open(path, "w") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)


def save_manifest(
    run_dir: str,
    config: Any,
    conditions: List[str],
    sample_ids: List[str],
    hidden_state_info: Optional[Dict] = None,
):
    """
    Save run manifest with metadata.

    Args:
        run_dir: Path to the run directory.
        config: The Stage1Config object.
        conditions: List of condition names that were run.
        sample_ids: Ordered list of sample IDs.
        hidden_state_info: Dict with shape, dtype, layer_count info.
    """
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "models": {
                "recipient": config.models.recipient,
                "donor": config.models.donor,
            },
            "boundary_grid": config.boundary_grid,
            "t_fixed": config.t_fixed,
            "reference": {
                "b_ref": config.reference.b_ref,
                "t_ref": config.reference.t_ref,
            },
            "hidden_state": {
                "pooling": config.hidden_state.pooling,
            },
            "dataset": {
                "name": config.dataset.name,
                "lang": config.dataset.lang,
                "split": config.dataset.split,
                "debug_n": config.dataset.debug_n,
            },
            "generation": {
                "do_sample": config.generation.do_sample,
                "temperature": config.generation.temperature,
                "max_new_tokens": config.generation.max_new_tokens,
            },
        },
        "conditions": conditions,
        "sample_ordering": sample_ids,
        "hidden_state_pooling": config.hidden_state.pooling,
    }

    if hidden_state_info:
        manifest["hidden_state_layer_count"] = hidden_state_info.get("layer_count")
        manifest["hidden_state_shape"] = hidden_state_info.get("shape")
        manifest["hidden_state_dtype"] = hidden_state_info.get("dtype")

    path = os.path.join(run_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
