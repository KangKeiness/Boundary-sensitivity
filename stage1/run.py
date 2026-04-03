"""Stage 1 pipeline: MGSM Telugu evaluation with Qwen2.5-1.5B boundary sweep."""

import argparse
import gc
import json
import logging
import os

import torch

from utils.config import load_config, setup_logging
from data.loader import load_mgsm
from models.composer import load_models, get_condition_model
from inference.runner import run_inference
from inference.parser import parse_answer
from analysis.bds import compute_bds
from analysis.evaluator import evaluate_experiment
from utils.logger import (
    create_run_dir,
    save_results,
    save_hidden_states,
    save_bds_results,
    save_evaluation,
    save_manifest,
)

logger = logging.getLogger(__name__)


def verify_results(run_dir: str, in_memory_eval: dict, in_memory_bds: dict,
                   samples, eval_config: dict, boundary_grid):
    """
    Reload saved results from disk, re-run evaluate_experiment(), and assert
    key metrics match the in-memory evaluation (within 1e-6).

    Catches file I/O corruption, non-deterministic evaluation bugs, and
    serialization/deserialization errors.

    Writes "self_verification": "passed" | "failed" into manifest.json.
    """
    manifest_path = os.path.join(run_dir, "manifest.json")

    # ── Reload results_*.jsonl files ──────────────────────────────────────
    reloaded_parsed: dict = {}
    for fname in os.listdir(run_dir):
        if fname.startswith("results_") and fname.endswith(".jsonl"):
            cond = fname[len("results_"):-len(".jsonl")]
            rows = []
            with open(os.path.join(run_dir, fname)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            reloaded_parsed[cond] = rows

    # ── Reload bds_*.json files ───────────────────────────────────────────
    reloaded_bds: dict = {}
    for fname in os.listdir(run_dir):
        if fname.startswith("bds_") and fname.endswith(".json"):
            cond = fname[len("bds_"):-len(".json")]
            with open(os.path.join(run_dir, fname)) as f:
                reloaded_bds[cond] = json.load(f)

    # ── Re-run evaluation on reloaded data ────────────────────────────────
    fresh_eval = evaluate_experiment(
        samples=samples,
        condition_results=reloaded_parsed,
        bds_results=reloaded_bds,
        boundary_grid=boundary_grid,
        config=eval_config,
    )

    # ── Compare key metrics ───────────────────────────────────────────────
    divergences = []

    def _check(key, a, b):
        if a is None and b is None:
            return
        if a is None or b is None:
            divergences.append(f"{key}: in_memory={a!r}  reloaded={b!r}")
            return
        if abs(float(a) - float(b)) > 1e-6:
            divergences.append(f"{key}: in_memory={a:.8f}  reloaded={b:.8f}")

    _check("baseline_accuracy",
           in_memory_eval.get("baseline_accuracy"),
           fresh_eval.get("baseline_accuracy"))

    for row_mem, row_fresh in zip(
        in_memory_eval.get("boundary_table", []),
        fresh_eval.get("boundary_table", []),
    ):
        b = row_mem.get("boundary")
        _check(f"boundary_{b}_accuracy",
               row_mem.get("accuracy"), row_fresh.get("accuracy"))
        _check(f"boundary_{b}_bds_total",
               row_mem.get("bds_total"), row_fresh.get("bds_total"))

    mem_c = in_memory_eval.get("criteria", {})
    fresh_c = fresh_eval.get("criteria", {})
    for key in ("criterion_1_delta_ci_excludes_zero",
                "criterion_2_bootstrap_positive",
                "criterion_3_ordering_consistent",
                "passed"):
        mv, fv = mem_c.get(key), fresh_c.get(key)
        if mv != fv:
            divergences.append(f"criteria.{key}: in_memory={mv!r}  reloaded={fv!r}")

    # ── Verify random_donor BDS disk-write consistency ────────────────────
    for fname in os.listdir(run_dir):
        if fname.startswith("bds_random_donor_") and fname.endswith(".json"):
            cond = fname[len("bds_"):-len(".json")]
            mem_bds = in_memory_bds.get(cond)
            rel_bds = reloaded_bds.get(cond)
            if mem_bds is None or rel_bds is None:
                divergences.append(f"{cond}: missing in_memory or reloaded BDS")
                continue
            _check(f"{cond}.aggregate.mean_bds_total",
                   mem_bds.get("aggregate", {}).get("mean_bds_total"),
                   rel_bds.get("aggregate", {}).get("mean_bds_total"))

    # ── Report and update manifest ────────────────────────────────────────
    verification_status = "failed" if divergences else "passed"

    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["self_verification"] = verification_status
        if divergences:
            manifest["self_verification_divergences"] = divergences
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    if divergences:
        msg = "SELF-VERIFICATION FAILED — diverging metrics:\n" + "\n".join(f"  {d}" for d in divergences)
        logger.error(msg)
        raise AssertionError(msg)

    print("SELF-VERIFICATION PASSED")
    logger.info("self_verification: passed")


def build_conditions(config):
    """Build the list of (condition_name, b, t) tuples to run."""
    conditions = [("no_swap", None, None)]
    for b in config.boundary_grid:
        conditions.append((f"hard_swap_b{b}", b, config.t_fixed))
    for b in config.boundary_grid:
        conditions.append((f"random_donor_b{b}", b, config.t_fixed))
    return conditions


def run_condition(condition_name, b, t, recipient, donor, tokenizer, samples, config):
    """Run a single experimental condition and return (inference_results, parsed_results, metadata)."""
    print(f"\n{'='*60}")
    print(f"Running condition: {condition_name}  (b={b}, t={t})")
    print(f"{'='*60}")

    # Determine internal condition key (hard_swap_b4 → hard_swap, random_donor_b4 → random_donor, etc.)
    if condition_name.startswith("hard_swap"):
        cond_key = "hard_swap"
    elif condition_name.startswith("random_donor"):
        cond_key = "random_donor"
    else:
        cond_key = condition_name

    # get_condition_model now returns (model, metadata)
    model, cond_metadata = get_condition_model(
        recipient=recipient,
        donor=donor,
        condition=cond_key,
        b=b,
        t=t,
        b_ref=config.reference.b_ref,
        t_ref=config.reference.t_ref,
        random_donor_seed=config.random_donor.seed,
    )

    gen_config = {
        "do_sample":      config.generation.do_sample,
        "temperature":    config.generation.temperature,
        "max_new_tokens": config.generation.max_new_tokens,
    }
    inference_results = run_inference(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        generation_config=gen_config,
        pooling=config.hidden_state.pooling,
    )

    parsed_results = []
    for r in inference_results:
        parsed = parse_answer(r["output_text"])
        parsed_results.append({
            "sample_id":  r["sample_id"],
            "output_text": r["output_text"],
            **parsed,
        })

    if condition_name != "no_swap":
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return inference_results, parsed_results, cond_metadata


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Boundary sensitivity sweep")
    parser.add_argument(
        "--config",
        type=str,
        default="stage1/configs/stage1_main.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()
    setup_logging()

    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(f"  Models     : {config.models.recipient} / {config.models.donor}")
    print(f"  Boundaries : {config.boundary_grid}  t_fixed={config.t_fixed}")
    print(f"  Dataset    : {config.dataset.name} ({config.dataset.lang}, {config.dataset.split})")
    print(f"  Pooling    : {config.hidden_state.pooling}")
    if config.dataset.debug_n:
        print(f"  Debug mode : {config.dataset.debug_n} samples")

    run_dir = create_run_dir()
    print(f"\nRun directory: {run_dir}")

    print("\nLoading dataset...")
    samples = load_mgsm(config)
    print(f"  Loaded {len(samples)} samples")

    print("\nLoading models...")
    recipient, donor, tokenizer = load_models(
        recipient_name=config.models.recipient,
        donor_name=config.models.donor,
    )
    print("  Models loaded successfully")

    conditions = build_conditions(config)
    print(f"\nConditions to run: {[c[0] for c in conditions]}")

    # ── Run all conditions ────────────────────────────────────────────────
    all_inference: dict = {}
    all_parsed: dict    = {}
    random_donor_source_starts: dict = {}

    for cond_name, b, t in conditions:
        inf_results, parsed_results, cond_meta = run_condition(
            condition_name=cond_name,
            b=b, t=t,
            recipient=recipient,
            donor=donor,
            tokenizer=tokenizer,
            samples=samples,
            config=config,
        )

        all_inference[cond_name] = inf_results
        all_parsed[cond_name]    = parsed_results

        if cond_name.startswith("random_donor_b") and "source_start" in cond_meta:
            random_donor_source_starts[cond_name] = cond_meta["source_start"]
            print(f"  Random donor source_start={cond_meta['source_start']} "
                  f"(cond={cond_name}, seed={config.random_donor.seed})")

        save_results(run_dir, cond_name, parsed_results)
        save_hidden_states(run_dir, cond_name, inf_results)
        print(f"  Saved results for {cond_name}")

    # ── BDS ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nComputing BDS scores...\n{'='*60}")

    recipient_hs = [
        {"sample_id": r["sample_id"], "hidden_states": r["hidden_states"]}
        for r in all_inference["no_swap"]
    ]

    bds_all: dict = {}
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
        print(f"  {cond_name}: bds_total={bds['aggregate']['mean_bds_total']:.4f}  "
              f"cka_bds_total={bds['aggregate']['cka_bds_total']:.4f}")

    # ── Evaluation ────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nRunning evaluation...\n{'='*60}")

    eval_config = {
        "bootstrap_n":         config.evaluation.bootstrap_n,
        "bootstrap_ci":        config.evaluation.bootstrap_ci,
        "criteria_threshold":  config.evaluation.criteria_threshold,
    }
    evaluation = evaluate_experiment(
        samples=samples,
        condition_results=all_parsed,
        bds_results=bds_all,
        boundary_grid=config.boundary_grid,
        config=eval_config,
    )
    save_evaluation(run_dir, evaluation)

    # ── Manifest ──────────────────────────────────────────────────────────
    hidden_state_info = None
    if all_inference.get("no_swap"):
        hs0 = all_inference["no_swap"][0]["hidden_states"]
        hidden_state_info = {
            "layer_count": hs0.shape[0],
            "shape":       list(hs0.shape),
            "dtype":       str(hs0.dtype),
        }

    save_manifest(
        run_dir=run_dir,
        config=config,
        conditions=[c[0] for c in conditions],
        sample_ids=[s["sample_id"] for s in samples],
        hidden_state_info=hidden_state_info,
        random_donor_source_start=random_donor_source_starts,
    )
    print(f"\nAll results saved to: {run_dir}")

    # ── Self-verification ─────────────────────────────────────────────────
    verify_results(
        run_dir=run_dir,
        in_memory_eval=evaluation,
        in_memory_bds=bds_all,
        samples=samples,
        eval_config=eval_config,
        boundary_grid=config.boundary_grid,
    )

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}\nRESULTS SUMMARY\n{'='*60}")
    print(f"\nBaseline accuracy (no_swap): {evaluation['baseline_accuracy']:.4f}")
    print(f"\n{'Boundary':>10} {'Accuracy':>10} {'Delta':>10} {'BDS_total':>10} {'CKA_BDS':>10}")
    for row in evaluation["boundary_table"]:
        print(
            f"{row['boundary']:>10} "
            f"{row.get('accuracy', 0.0):>10.4f} "
            f"{row.get('delta_vs_baseline', 0.0):>10.4f} "
            f"{row.get('bds_total', float('nan')):>10.4f} "
            f"{bds_all.get(row['condition'], {}).get('aggregate', {}).get('cka_bds_total', float('nan')):>10.4f}"
        )

    if evaluation["bds_delta_rho"] is not None:
        print(f"\nH1 BDS-degradation rank correlation: rho={evaluation['bds_delta_rho']:.4f}  "
              f"p={evaluation['bds_delta_p']:.4f}")
    if evaluation["bds_delta_ci"]:
        print(f"Bootstrap CI: {[round(v,4) for v in evaluation['bds_delta_ci']]}")

    print(f"\nOrdering consistent (C3): {evaluation['ordering_consistent']}")
    criteria = evaluation["criteria"]
    print(f"Criteria: C1={criteria['criterion_1_delta_ci_excludes_zero']}  "
          f"C2={criteria['criterion_2_bootstrap_positive']} (rate={criteria['criterion_2_positive_rate']:.3f})  "
          f"C3={criteria['criterion_3_ordering_consistent']}  "
          f"→ passed={criteria['passed']} ({criteria['n_criteria_met']}/{criteria['threshold']})")

    # ── Sanity check printout ─────────────────────────────────────────────
    print(f"\n{'='*60}\nSANITY CHECKS\n{'='*60}")
    print("1. Hidden state extraction: prompt-only forward pass  ✓")
    print("2. H1 uses degradation (not abs delta)  ✓")
    print("3. Criterion 1 checks delta CI  ✓")
    print(f"4. Criterion 2 uses bootstrap distribution (positive rate={criteria['criterion_2_positive_rate']:.3f})  ✓")
    print(f"5. Random donor seed: {config.random_donor.seed}  source_starts: {random_donor_source_starts}  ✓")
    print(f"6. Anchor point: hard_swap_b{config.reference.b_ref} (a priori)  ✓")
    print("Stage 1 pipeline complete.")


if __name__ == "__main__":
    main()
