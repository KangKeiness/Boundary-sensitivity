"""Stage 1 pipeline: MGSM Telugu evaluation with Qwen2.5-1.5B boundary sweep."""

import argparse
import gc
import sys
import os

import torch

from stage1.utils.config import load_config
from stage1.data.loader import load_mgsm
from stage1.models.composer import load_models, get_condition_model
from stage1.inference.runner import run_inference
from stage1.inference.parser import parse_answer
from stage1.analysis.bds import compute_bds
from stage1.analysis.evaluator import evaluate_experiment
from stage1.utils.logger import (
    create_run_dir,
    save_results,
    save_hidden_states,
    save_bds_results,
    save_evaluation,
    save_manifest,
)


def build_conditions(config):
    """Build the list of (condition_name, b, t) tuples to run."""
    conditions = []

    # 1. no_swap
    conditions.append(("no_swap", None, None))

    # 2. hard_swap for each boundary
    for b in config.boundary_grid:
        conditions.append((f"hard_swap_b{b}", b, config.t_fixed))

    # 3. reference
    conditions.append(("reference", config.reference.b_ref, config.reference.t_ref))

    # 4. random_donor (use first boundary width as representative)
    if config.boundary_grid:
        b_rand = config.boundary_grid[0]
        conditions.append(("random_donor", b_rand, config.t_fixed))

    return conditions


def run_condition(
    condition_name,
    b,
    t,
    recipient,
    donor,
    tokenizer,
    samples,
    config,
):
    """Run a single experimental condition and return results."""
    print(f"\n{'='*60}")
    print(f"Running condition: {condition_name} (b={b}, t={t})")
    print(f"{'='*60}")

    # Get the model for this condition
    model = get_condition_model(
        recipient=recipient,
        donor=donor,
        condition=condition_name.split("_b")[0] if condition_name.startswith("hard_swap") else condition_name,
        b=b,
        t=t,
        b_ref=config.reference.b_ref,
        t_ref=config.reference.t_ref,
    )

    # Run inference
    gen_config = {
        "do_sample": config.generation.do_sample,
        "temperature": config.generation.temperature,
        "max_new_tokens": config.generation.max_new_tokens,
    }
    inference_results = run_inference(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        generation_config=gen_config,
        pooling=config.hidden_state.pooling,
    )

    # Parse answers
    parsed_results = []
    for r in inference_results:
        parsed = parse_answer(r["output_text"])
        parsed_results.append({
            "sample_id": r["sample_id"],
            "output_text": r["output_text"],
            **parsed,
        })

    # Free composed model if it's not the recipient
    if condition_name != "no_swap":
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return inference_results, parsed_results


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Boundary sensitivity sweep")
    parser.add_argument(
        "--config",
        type=str,
        default="stage1/configs/stage1_main.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(f"  Models: {config.models.recipient} / {config.models.donor}")
    print(f"  Boundary grid: {config.boundary_grid}")
    print(f"  Dataset: {config.dataset.name} ({config.dataset.lang}, {config.dataset.split})")
    if config.dataset.debug_n:
        print(f"  Debug mode: {config.dataset.debug_n} samples")

    # Create run directory
    run_dir = create_run_dir()
    print(f"\nRun directory: {run_dir}")

    # Load dataset
    print("\nLoading dataset...")
    samples = load_mgsm(
        dataset_name=config.dataset.name,
        lang=config.dataset.lang,
        split=config.dataset.split,
        debug_n=config.dataset.debug_n,
    )
    print(f"  Loaded {len(samples)} samples")

    # Load models
    print("\nLoading models...")
    recipient, donor, tokenizer = load_models(
        recipient_name=config.models.recipient,
        donor_name=config.models.donor,
    )
    print("  Models loaded successfully")

    # Build conditions
    conditions = build_conditions(config)
    print(f"\nConditions to run: {[c[0] for c in conditions]}")

    # Run all conditions
    all_inference = {}
    all_parsed = {}

    for cond_name, b, t in conditions:
        inference_results, parsed_results = run_condition(
            condition_name=cond_name,
            b=b,
            t=t,
            recipient=recipient,
            donor=donor,
            tokenizer=tokenizer,
            samples=samples,
            config=config,
        )

        all_inference[cond_name] = inference_results
        all_parsed[cond_name] = parsed_results

        # Save results incrementally
        save_results(run_dir, cond_name, parsed_results)
        save_hidden_states(run_dir, cond_name, inference_results)
        print(f"  Saved results for {cond_name}")

    # Compute BDS for each non-baseline condition
    print("\n" + "=" * 60)
    print("Computing BDS scores...")
    print("=" * 60)

    recipient_hs = [
        {"sample_id": r["sample_id"], "hidden_states": r["hidden_states"]}
        for r in all_inference["no_swap"]
    ]

    bds_all = {}
    for cond_name, b, t in conditions:
        if cond_name == "no_swap" or b is None:
            continue

        composed_hs = [
            {"sample_id": r["sample_id"], "hidden_states": r["hidden_states"]}
            for r in all_inference[cond_name]
        ]

        bds = compute_bds(recipient_hs, composed_hs, b, t)
        bds_all[cond_name] = bds
        save_bds_results(run_dir, cond_name, bds)
        print(f"  BDS for {cond_name}: total={bds['aggregate']['mean_bds_total']:.4f}")

    # Evaluation
    print("\n" + "=" * 60)
    print("Running evaluation...")
    print("=" * 60)

    eval_config = {
        "bootstrap_n": config.evaluation.bootstrap_n,
        "bootstrap_ci": config.evaluation.bootstrap_ci,
        "criteria_threshold": config.evaluation.criteria_threshold,
    }

    evaluation = evaluate_experiment(
        samples=samples,
        condition_results=all_parsed,
        bds_results=bds_all,
        boundary_grid=config.boundary_grid,
        config=eval_config,
    )

    save_evaluation(run_dir, evaluation)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nBaseline accuracy (no_swap): {evaluation['baseline_accuracy']:.4f}")
    print("\nBoundary table:")
    print(f"  {'Boundary':>10} {'Accuracy':>10} {'Delta':>10} {'BDS_total':>10}")
    for row in evaluation["boundary_table"]:
        print(
            f"  {row['boundary']:>10} "
            f"{row.get('accuracy', 'N/A'):>10.4f} "
            f"{row.get('delta_vs_baseline', 'N/A'):>10.4f} "
            f"{row.get('bds_total', 'N/A'):>10.4f}"
        )

    if evaluation["bds_delta_rank_correlation"] is not None:
        print(f"\nBDS-delta rank correlation: {evaluation['bds_delta_rank_correlation']:.4f}")

    print(f"\nOrdering consistent: {evaluation['ordering_consistent']}")
    print(f"\nSystematic criteria: {evaluation['criteria']}")

    # Save manifest
    hidden_state_info = None
    if all_inference.get("no_swap"):
        hs0 = all_inference["no_swap"][0]["hidden_states"]
        hidden_state_info = {
            "layer_count": hs0.shape[0],
            "shape": list(hs0.shape),
            "dtype": str(hs0.dtype),
        }

    save_manifest(
        run_dir=run_dir,
        config=config,
        conditions=[c[0] for c in conditions],
        sample_ids=[s["sample_id"] for s in samples],
        hidden_state_info=hidden_state_info,
    )

    print(f"\nAll results saved to: {run_dir}")
    print("Stage 1 pipeline complete.")


if __name__ == "__main__":
    main()
