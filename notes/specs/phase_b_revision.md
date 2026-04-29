# phase_b_revision

Scope note: This spec is a *revision* of the Phase B prompt-side restoration intervention. It is layered on top of `notes/specs/phase_b_rewrite.md` (which remains the structural baseline for the patcher and run driver) and is driven by the Phase B revision brief at `notes/phase_b_revision_claude_prompt.md`. The previous Phase B full clean run (`stage1/outputs/phase_b/run_20260428_071043_694696`) completed but failed the cross-check gate because Phase B's patched-generation path is not semantically equivalent to Stage 1's `model.generate()` path — patched runs append a long `Human:` continuation and diverge in normalized answer for >55% of samples. The revision (a) fixes generation parity (P0 — code correctness), and (b) re-frames Phase B as a *conditional* diagnostic on broken-by-swap and answer-flipped sample subsets while preserving every existing global metric (research scope; additive only).

Methodological caveat to echo in every summary artifact: "Patching applies only to prompt-side hidden-state processing. Clean hidden states are available for prompt tokens only. This is prompt-side restoration intervention, not full-sequence causal intervention." (Unchanged from `phase_b_rewrite.md` §0; reused verbatim.)

Layering rule: where this spec is silent, `notes/specs/phase_b_rewrite.md` remains authoritative. Where the two disagree, this spec wins (it is the newer document). The do-not-modify list, the FORBIDDEN_PHRASES gate, the cross-phase tolerance `PHASE_A_CROSS_CHECK_TOL = 0.008`, the comparative-sentence epsilon `EPSILON_DELTA = 0.02`, the architecture assertions, and the sanity-mode condition set are all unchanged from `phase_b_rewrite.md` and MUST NOT be relaxed.

## 1. Goal

Make Phase B a *valid* prompt-side restoration diagnostic by (a) fixing the patched-generation path so that `clean_no_patch` reproduces Stage 1 `no_swap` and `restoration_no_patch` reproduces Stage 1 `hard_swap_b8` within `|Δacc| ≤ 0.008` and the `Human:` continuation count for the no-patch conditions drops to within `stage1_count + 2` of the Stage 1 reference (i.e., ≤ 2/250 since Stage 1 is 0/250 per brief §A), and (b) emitting subset-conditional restoration metrics (recovery on broken-by-swap, answer-restoration on answer-flipped, preservation on stable-correct) alongside the existing global tables, with subset-size warnings and conservative wording enforced by the existing `utils/wording.py::FORBIDDEN_PHRASES` gate (extended only as listed in §8) over the explicit user-facing artifact filename set in §11.11.

Testable: `python -m pytest stage1/tests -q -rs` exits 0 with no new skips; one full Phase B clean rerun (`python -m stage1.run_phase_b --config stage1/configs/stage2_confound.yaml`) writes `phase_b_summary.json.run_status == "passed"` and `RUN_STATUS.txt == "PASSED"` to a fresh `stage1/outputs/phase_b/run_<NEW_TIMESTAMP>/` directory whose contents satisfy every numeric criterion in §11. Failure of any single §11 criterion fails the revision.

## 2. Hypothesis and falsification

Hypothesis H1 (engineering — generation parity): the revised patched-generation path used by `run_patched_inference_single` is semantically equivalent to Stage 1's `runner.run_inference` greedy generate-call path (`stage1/inference/runner.py:118`) when `patch_layers=[]`. Equivalence is measured on the full 250-sample MGSM-zh subset under the same seed and the same `generation_config` (`do_sample=False`, `temperature=0.0`, `max_new_tokens=512`).

Falsification H1: any of:
- `|PhaseB clean_no_patch acc − Stage1 no_swap anchor acc| > 0.008`,
- `|PhaseB restoration_no_patch acc − Stage1 hard_swap_b8 anchor acc| > 0.008`,
- count of samples whose Phase B `clean_no_patch` or `restoration_no_patch` output contains `"Human:"` exceeds the matched Stage 1 condition's count by more than 2/250,
- mean Phase B output length on `clean_no_patch` (vs Stage 1 `no_swap`) or on `restoration_no_patch` (vs Stage 1 `hard_swap_b8`) exceeds 1.25× the Stage 1 reference. Patched (non no-patch) conditions are NOT length-gated as a hard fail; an optional warning fires when their length ratio falls below 0.75× (see §11.6).

Hypothesis H2 (research — conditional diagnostic): even when the global hard_swap degradation is small, the broken-by-swap subset `S_broken` and the answer-flipped subset `S_flipped` carry signal that prompt-side hidden-state patching can selectively address. We do NOT claim restoration is causal; we claim it is *measurable and conditionally interpretable*. Per the brief §G, all reporting uses the "prompt-side restoration intervention evidence" / "conditional restoration on broken-by-swap samples" / "answer-flip localization" / "behaviorally consequential vs behaviorally silent drift" / "diagnostic probe" vocabulary only.

Falsification H2: subset construction is well-defined and computable from artifact-level data (Stage 1 anchor JSONLs + Phase B per-condition JSONLs); subset-size warnings (`n_broken < 20` or `n_flipped < 20`) are emitted in the summary when triggered and any patched-recovery claim is downgraded to neutral wording. If the subset construction code disagrees with `analysis/evaluator.py::exact_match` on any sample (mismatch between locally recomputed `correct` and the anchor-JSONL `correct` field where present), the run hard-fails with a parser/evaluator-drift error.

H1 and H2 are engineering and reporting-discipline hypotheses. Neither asserts a scientific conclusion about the recipient model. Restoration claims remain conservative per `phase_b_rewrite.md` §11 and the brief §G.

## 3. Prior art and delta

Reuse the two primary references already cited in `phase_b_rewrite.md` §3 (verified arxiv IDs; not re-fetched per the role rulebook):

- **Meng et al. 2022, arXiv:2202.05262** — activation-patching primitive (clean states injected into a corrupted forward). Delta vs. ROME is unchanged from `phase_b_rewrite.md` §3 (residual-stream patching at the decoder-block boundary, not MLP-only; donor/target are recipient vs. composed-model variants of the same Qwen2.5-1.5B-Instruct, not noised embeddings). Additional delta in this revision: we now condition the restoration interpretation on *behavioral subsets* (`S_broken`, `S_flipped`) rather than reporting only global accuracy recovery.

- **Vig et al. 2020, arXiv:2004.12265** — causal mediation framing (NDE/NIE via activation substitution). Delta vs. CMA is unchanged: Phase B does NOT estimate NDE/NIE and MUST NOT use that vocabulary; "natural direct effect", "natural indirect effect", "NIE", "NDE", "formal mediation", and "causal mediation" are all already in the Phase C `FORBIDDEN_PHRASES_PHASE_C` list (`stage1/utils/wording.py:42-57`). Phase B's `FORBIDDEN_PHRASES_PHASE_B` (`stage1/utils/wording.py:29-39`) reserves the additional Phase C vocabulary ("restoration effect", "residual effect", "restoration proportion"); this revision extends Phase B's list with the brief §G additions only as listed in §8.

Delta of THIS revision vs. `phase_b_rewrite.md`:
1. (P0) Fix the patched-generation path so it is bytewise / semantically equivalent to Stage 1's `runner.py` greedy generate path on `clean_no_patch` (recipient, no patches) and `restoration_no_patch` (composed, no patches). The current `_greedy_continue_with_cache` (`stage1/intervention/patcher.py:506-566`) does not honor a per-sample finished mask in the same way `model.generate(do_sample=False)` does (it stops only when the very last emitted token equals `eos_token_id`), and it does not slice / decode the continuation in a way that suppresses the long `Human:` re-prompt that a stop-token-aware `model.generate` would have terminated.
2. (Research) Add subset construction (`S_stable_correct`, `S_broken`, `S_repaired`, `S_stable_wrong`, `S_flipped`, optional `S_correct_flipped`) computed from the *Stage 1 anchor* JSONLs (not from Phase B itself), with subset counts, rates, and per-subset warnings persisted to `phase_b_subsets.json` and `phase_b_subsets.csv`.
3. (Research) Add per-patch-condition conditional metrics (broken-subset recovery, flipped-subset answer restoration, stable-subset preservation, parse-behavior, output-behavior) persisted to `phase_b_conditional_summary.json` and `phase_b_conditional_summary.csv`. Existing global tables (`restoration_table.csv`, `corruption_table.csv`, `phase_b_summary.json`) are *not* removed; the new artifacts are additive.
4. (Optional) Add a representation-drift-vs-behavior diagnostic computed from `hidden_states_no_swap.pt` / `hidden_states_hard_swap_b8.pt` if both are available under the chosen Stage 1 anchor, persisted to `phase_b_drift_behavior_summary.json` and `phase_b_drift_behavior.csv`. Skipped silently with a `summary["drift_diagnostic"] = "skipped: artifacts unavailable"` annotation if either file is missing or shape-incompatible.
5. Extend the wording gate vocabulary (`stage1/utils/wording.py::FORBIDDEN_PHRASES_PHASE_B`) only with brief §G additions that are not already covered (see §8 for the exhaustive diff).
6. Wire the wording gate over the new artifacts (`phase_b_subsets.{json,csv}`, `phase_b_conditional_summary.{json,csv}`, optional `phase_b_drift_behavior_*`).

Phase A (`stage1/run_phase_a.py`) is DO-NOT-MODIFY. Stage 1 (`stage1/inference/runner.py`, `stage1/inference/parser.py`, `stage1/analysis/evaluator.py`) are DO-NOT-MODIFY. Phase C (`stage1/run_phase_c.py`, `notes/specs/phase_c_mediation.md`) MUST NOT be touched until Phase B passes. Composer (`stage1/models/composer.py`) is read-only.

## 4. Datasets

Identical to `phase_b_rewrite.md` §4. No data regeneration:
- name: MGSM
- language: Chinese (zh)
- split: test
- n_samples: 250 (full); 5 under `--sanity`
- loader: `stage1/data/loader.py::load_mgsm(config)` (DO-NOT-MODIFY; this spec only *reads* from it via `stage1/run_phase_b.py`).
- hash/version pointer: `notes/data_changelog.md` MGSM-zh entry, mirrored verbatim into `phase_b_summary.json.dataset.phase_a_dataset_manifest` (already wired at `stage1/run_phase_b.py:561`).
- license: MGSM = GSM8K (MIT) + Google translations (Apache-2.0); inherited unchanged from Phase A.

No new data source. No translation. No filtering changes.

## 5. Models and tokenizers

Identical to `phase_b_rewrite.md` §5. Pinned via `stage1/configs/stage2_confound.yaml`:
- recipient: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- donor: `Qwen/Qwen2.5-1.5B`, revision `8faed761d45a263340a0528343f099c05c9a4323`
- tokenizer: from recipient, same revision
- composed model: `compose_model(recipient, donor, b=8, t=20, condition="hard_swap")` (treatment `hard_swap_b8`).
- Architecture assertions (already enforced at `stage1/run_phase_b.py:329-333`): `num_hidden_layers == 28`, `hidden_size == 1536`. Fail-fast preserved.
- dtype: weights loaded in `float16`; comparisons / hashing cast to `float32` per `phase_b_rewrite.md` §5.
- `pad_token_id` policy: identical to Stage 1 (`runner.py` does not set `pad_token_id` explicitly and does not pass `attention_mask` to `model.generate`; this is the parity contract for `clean_no_patch` and `restoration_no_patch`). The revised greedy loop MUST NOT introduce a different `pad_token_id` or `attention_mask` policy on the no-patch path. For non-empty patch sets, the loop MAY pass an extended attention mask if and only if it is provably equivalent on no-patch by §10 test #6 (greedy parity on the 250-sample full subset).

No new models. No fine-tuning. No tokenizer revision change. No new revisions.

## 6. Training config

n/a — inference-only. Determinism, seed, and precision policy are unchanged from `phase_b_rewrite.md` §6 and `stage1/run_phase_b.py:134-155`:
- `seed = 42` wired into `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `transformers.set_seed`
- `torch.use_deterministic_algorithms(True, warn_only=True)`
- `os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")` set BEFORE any CUDA op (already at `stage1/run_phase_b.py:141`)
- generation: `do_sample=False`, `temperature=0.0`, `max_new_tokens=512` (from `stage1/configs/stage2_confound.yaml`)
- precision: fp16 weights, fp32 for analysis reductions and SHA-256 hashing
- single-seed per phase (Phase C may multi-seed; Phase B does not)

`phase_b_summary.json.environment` block unchanged; `runtime_provenance` (`stage1/run_phase_b.py:533-535`) re-used verbatim.

## 7. Evaluation protocol

Two layers of metrics, both emitted in the same run.

### 7.1 Global metrics (preserved verbatim)

All metrics defined in `phase_b_rewrite.md` §7 remain. No condition is renamed; no condition is removed. The 11 conditions enumerated at `phase_b_rewrite.md` §7 are unchanged:

- Restoration on composed (hard_swap_b8): `no_patch`, `patch_boundary_local`, `patch_recovery_early`, `patch_recovery_full`, `patch_final_only`, `patch_all_downstream`
- Clean baseline on recipient: `clean_no_patch`
- Reverse corruption on recipient: `corrupt_boundary_local`, `corrupt_recovery_early`, `corrupt_recovery_full`, `corrupt_final_only`

Per-condition `accuracy` is computed via the unchanged `analysis/evaluator.py::exact_match` and `inference/parser.py::parse_answer`. Sanity-mode condition set is unchanged: `{no_patch, patch_recovery_full, clean_no_patch, corrupt_recovery_full} × 5 samples`.

Cross-phase parity check is unchanged from the existing implementation at `stage1/run_phase_b.py:480-516` (delegated to `stage1/utils/anchor_gate.py::evaluate_phase_b_anchor_gate`) with tolerance `PHASE_A_CROSS_CHECK_TOL = 0.008`. The revision MUST NOT alter the gate, the tolerance, or the "BOTH anchors required for full mode" rule.

### 7.2 Conditional metrics (NEW — additive)

#### 7.2.1 Subset definitions (computed once per run from Stage 1 anchor JSONLs)

Source data:
- `clean` = Stage 1 `no_swap` per-sample results from the parity-selected anchor (`stage1/utils/anchor_gate.py::evaluate_phase_b_anchor_gate.anchor_no_swap_source`).
- `corrupted` = Stage 1 `hard_swap_b8` per-sample results from the parity-selected anchor (`anchor_hard_swap_source`).

For each sample `i` (matched by `sample_id`, sample order parity already enforced at `manifest_parity.py`):
- `ans_clean[i]` = clean condition's `normalized_answer` (may be `None`)
- `ans_corrupt[i]` = corrupted condition's `normalized_answer` (may be `None`)
- `correct_clean[i]`:
  - If the JSONL row has a `correct` field, recompute via `analysis/evaluator.py::exact_match(samples[i].gold_answer, ans_clean[i])` and compare. On disagreement on ANY sample, the run hard-fails with `RuntimeError("evaluator drift: anchor JSONL correct field disagrees with recomputed exact_match for sample_id=<id>")`. This is a strict-fail; warning-only is NOT acceptable. The JSONL value is then used as the canonical `correct_clean[i]`.
  - If the JSONL row does NOT have a `correct` field, recompute via the same call and use the recomputed value. Set the per-sample annotation `recomputed_correct=true` in `phase_b_subsets.csv` (boolean column) AND record `summary["correct_field_source"] = {"clean": "recomputed", "corrupt": "recomputed"|"jsonl"}` so downstream review can audit which path was taken.
- `correct_corrupt[i]` = analogous.

Subsets (per brief §D):

| Symbol | Definition |
|--------|------------|
| `S_stable_correct` | `correct_clean == True AND correct_corrupt == True` |
| `S_broken` | `correct_clean == True AND correct_corrupt == False` |
| `S_repaired` | `correct_clean == False AND correct_corrupt == True` |
| `S_stable_wrong` | `correct_clean == False AND correct_corrupt == False` |
| `S_flipped` | `ans_clean != ans_corrupt`, with `None vs None` NOT flipped, `None vs non-None` flipped, `non-None vs None` flipped |
| `S_correct_flipped` (optional) | `correct_clean == correct_corrupt == True AND ans_clean != ans_corrupt` |

Subset rates emitted into `phase_b_subsets.json`:
- `n_total`, `n_stable_correct`, `n_broken`, `n_repaired`, `n_stable_wrong`, `n_flipped`, `n_correct_flipped`
- `broken_rate = n_broken / n_total`
- `repaired_rate = n_repaired / n_total`
- `answer_flip_rate = n_flipped / n_total`

`phase_b_subsets.csv`: one row per `sample_id` with columns `sample_id, ans_clean, ans_corrupt, correct_clean, correct_corrupt, recomputed_correct_clean, recomputed_correct_corrupt, in_S_stable_correct, in_S_broken, in_S_repaired, in_S_stable_wrong, in_S_flipped, in_S_correct_flipped` (boolean cells written as `true` / `false`). The two `recomputed_correct_*` columns are `true` only when the corresponding anchor JSONL row had no `correct` field and the value was computed by the §7.2.1 fallback.

#### 7.2.2 Per-patch-condition conditional metrics (per brief §E)

For each restoration patch condition `pc ∈ {restoration_no_patch, patch_boundary_local, patch_recovery_early, patch_recovery_full, patch_final_only, patch_all_downstream}` (and analogously for the corruption-direction conditions, treated symmetrically — see §7.2.5 for the symmetry rule), compute and persist:

(1) Global patched accuracy (already exists in `restoration_table.csv`): `acc_patched = mean(correct_pc)`.
(2) Global delta vs `restoration_no_patch`: `acc_patched - acc_restoration_no_patch` (already exists as `delta_from_no_patch` in `restoration_table.csv`; preserved).
(3) **Broken-subset recovery on `S_broken`**:
    - `recovery_to_correct_rate = mean(correct_pc[i] for i in S_broken)`
    - `recovery_to_clean_answer_rate = mean(ans_pc[i] == ans_clean[i] for i in S_broken)` (uses normalized answers; `None == None` counted as match only when both sides parsed to `None`, mirroring `S_flipped` semantics)
    - `recovery_gain_vs_no_patch = recovery_to_correct_rate - mean(correct_restoration_no_patch[i] for i in S_broken)`
(4) **Answer-flipped subset restoration on `S_flipped`**:
    - `answer_restoration_rate = mean(ans_pc[i] == ans_clean[i] for i in S_flipped)`
    - `answer_corrupt_retention_rate = mean(ans_pc[i] == ans_corrupt[i] for i in S_flipped)`
    - `other_answer_rate = 1 - answer_restoration_rate - answer_corrupt_retention_rate + overlap_rate`, where `overlap_rate = mean(ans_clean[i] == ans_corrupt[i] for i in S_flipped)` (which is 0 by construction of `S_flipped` — included in the formula for self-documenting correctness).
(5) **Stable-correct disruption on `S_stable_correct`**:
    - `stable_correct_preservation_rate = mean(correct_pc[i] for i in S_stable_correct)`
    - `clean_answer_preservation_rate = mean(ans_pc[i] == ans_clean[i] for i in S_stable_correct)`
(6) **Repaired-subset reversal on `S_repaired`** (exploratory; flagged as `"exploratory": true` in JSON):
    - `patched_correct_rate_on_S_repaired = mean(correct_pc[i] for i in S_repaired)`
    - Reported with explicit caveat in summary text: "exploratory; subset is small by construction; do not overinterpret".
(7) **Parse behavior** per condition:
    - `parse_success_rate_global = mean(parse_success_pc)`
    - `parse_success_rate_per_subset` (dict keyed by subset name)
    - `parse_failure_increase_vs_no_patch = parse_failure_rate_pc - parse_failure_rate_restoration_no_patch`
(8) **Output behavior** per condition:
    - `avg_output_length_chars = mean(len(output_text_pc))`
    - `human_continuation_count = sum(1 for i if "Human:" in output_text_pc[i])`
    - `human_continuation_rate = human_continuation_count / n_total`
    - `answer_extraction_failure_count = sum(1 for i if parse_success_pc[i] is False)` (redundant with parse-behavior block but persisted under `output_behavior` for symmetry with brief §E.8).

Persisted as `phase_b_conditional_summary.json` (one top-level dict keyed by patch condition, each value is the eight blocks above) and `phase_b_conditional_summary.csv` (long format, one row per `(condition, metric_name, subset_name|"global", value)` 4-tuple — keeps the CSV grep-friendly).

#### 7.2.3 Subset-size warnings

Emitted into `phase_b_summary.json.subset_warnings` (a list of strings) and into the `phase_b_summary.txt` body before the comparative-claim line:

- If `n_broken < 20`: append `"warning: n_broken=<n> < 20 — broken-subset recovery metrics are noisy; do not claim strong restoration"`.
- If `n_flipped < 20`: append `"warning: n_flipped=<n> < 20 — answer-restoration metrics are noisy"`.
- If `n_correct_flipped > 0 AND n_correct_flipped < 10`: append `"warning: n_correct_flipped=<n> < 10 — exploratory only"`.

When any of these warnings is active, the comparative-sentence gate at `stage1/run_phase_b.py:589-655` MUST NOT be relaxed (epsilon stays at 0.02; CI rule unchanged); the warnings are *additional* output, not a substitute for the gate.

#### 7.2.4 Sanity-mode and anchor-unavailability handling

The conditional metrics layer (§7.2) depends on Stage 1 anchor JSONLs being available and parity-selectable. Behavior on missing / unavailable anchors is mode-dependent and MUST NOT be silent fall-through:

- **Full mode** (no `--sanity`): If `evaluate_phase_b_anchor_gate` returns `gate.passed=False` because the canonical Stage 1 / Phase A anchors at the spec-§9-pinned paths are absent or fail parity, the run hard-fails with the existing `RuntimeError("missing anchor(s)")` from `stage1/utils/anchor_gate.py`. NO subset / conditional artifacts are emitted in this case (their precondition is unsatisfied). The run is `RUN_STATUS_FAILED`. Do NOT auto-fall-back to older or auto-latest anchors.
- **Sanity mode** (`--sanity`): The existing sanity-mode anchor-gate skip at `stage1/run_phase_b.py:792-797` is preserved. When the gate is in sanity skip mode AND anchors are unavailable, `_emit_conditional_artifacts` MUST still write `phase_b_subsets.json` and `phase_b_conditional_summary.json`, but each as a single explicit-skipped object:
  ```json
  {"status": "skipped", "reason": "sanity mode: stage1 anchors unavailable for subset construction", "n_total": 5}
  ```
  The CSV files are written with header row only and zero data rows, plus a sidecar `phase_b_subsets.SKIPPED.txt` containing the same `reason` string. `summary["subset_summary"] = {"status": "skipped", "reason": "..."}` and `summary["subset_warnings"] = ["sanity mode: subset construction skipped"]`. The wording-gate scan still runs over these explicit-skipped files. This guarantees that no full-vs-sanity behavioral divergence in artifact presence can be mistaken for a bug; the skipped-status is always observable.
- **Sanity mode with anchors AVAILABLE**: Subset construction runs on the 5-sample subset; warnings (`n_broken < 20` etc.) WILL fire and that is expected and correct.

#### 7.2.5 Symmetry rule for corruption direction

The conditional metrics in §7.2.2 are defined with respect to the restoration direction (composed-model patched with clean states). For the corruption-direction conditions (recipient patched with corrupt states), the *same* metric formulas apply with `restoration_no_patch` replaced by `clean_no_patch` and the role of `ans_clean` / `ans_corrupt` swapped (the "clean answer" for a corruption-direction patched run is `ans_clean[i]` — the recipient's own no-patch answer; the "corrupt answer" is `ans_corrupt[i]` — the composed model's hard_swap answer). Persisted under `phase_b_conditional_summary.json["corruption_<patch_name>"]` keys to keep restoration and corruption flat-namespaced and CSV-friendly.

### 7.3 Optional drift-vs-behavior diagnostic (per brief §F)

Computed only if BOTH `hidden_states_no_swap.pt` AND `hidden_states_hard_swap_b8.pt` exist under the parity-selected Stage 1 anchor and load successfully via `torch.load(..., map_location="cpu")` with shape assertion `[N_samples, n_layers, hidden_dim]` (Stage 1's prompt-pooled format per `stage1/inference/runner.py:14-46`).

Per-sample drift metrics:
- `cosine_distance[i, layer] = 1 - cos(h_clean[i, layer], h_corrupt[i, layer])`
- `l2_distance[i, layer] = ||h_clean[i, layer] - h_corrupt[i, layer]||_2`
- `boundary_layer_drift[i] = cosine_distance[i, 8]` — **PRIMARY** drift variable (boundary layer for hard_swap_b8). This is the variable used as the AUROC score in the predictive block below.
- Secondary aggregates (persisted alongside the primary, never used as the AUROC score by default):
  - `mean_layer_drift[i] = mean(cosine_distance[i, :])` over all 28 layers
  - `max_layer_drift[i] = max(cosine_distance[i, :])` over all 28 layers
  - `downstream_mean_drift[i] = mean(cosine_distance[i, 9:28])` (layers strictly after the boundary)
  - L2 analogues: `boundary_layer_l2[i]`, `mean_layer_l2[i]`, `max_layer_l2[i]`, `downstream_mean_l2[i]`

Aggregated per subset (`mean`, `median`, `std` of each of the four cosine drift variables and the four L2 variables above) — 8 variable types × 3 reductions × 6 subsets = 144 numbers in `phase_b_drift_behavior_summary.json` under `subset_aggregates`.

Predictive metrics (subset-membership prediction from drift) — **the AUROC score variable is `boundary_layer_drift` (cosine at layer 8) by default**; the secondary aggregates are persisted but NOT used as the default AUROC score:
- `auroc_boundary_drift_to_answer_flip` (label = `i in S_flipped`, score = `boundary_layer_drift[i]`) — primary
- `auroc_boundary_drift_to_broken` (label = `i in S_broken`, score = `boundary_layer_drift[i]`) — primary
- `point_biserial_boundary_drift_vs_flip` (Pearson); `spearman_boundary_drift_vs_flip` — primary
- For audit only, ALSO persist the same four metrics computed against `mean_layer_drift` and `downstream_mean_drift` under keys `auroc_mean_drift_*`, `auroc_downstream_drift_*`, etc. The summary's TXT block reports only the boundary-drift variants; the JSON exposes all three for downstream review.

Persisted as `phase_b_drift_behavior_summary.json` (subset-level aggregates + AUROC + correlations, with `primary_drift_variable: "boundary_layer_drift"` recorded at the top level so the choice is self-documenting) and `phase_b_drift_behavior.csv` (one row per `sample_id`: `sample_id, boundary_drift, mean_cos_drift, max_cos_drift, downstream_mean_cos_drift, boundary_l2, mean_l2, max_l2, downstream_mean_l2, in_S_flipped, in_S_broken, in_S_repaired`).

If the diagnostic is skipped (artifacts missing or shape-incompatible), `phase_b_summary.json.drift_diagnostic` MUST be set to one of `"skipped: hidden_states_no_swap.pt missing"`, `"skipped: hidden_states_hard_swap_b8.pt missing"`, or `"skipped: shape mismatch (expected [N, n_layers, hidden_dim])"`. No partial drift artifact is written when skipped.

### 7.4 Statistical test

The single existing comparative claim (best restoration vs. `patch_boundary_local`, paired bootstrap 1000 resamples seed=0 95% CI, gated by `EPSILON_DELTA = 0.02` AND `ci_lo > 0`) is unchanged — see `stage1/run_phase_b.py:217-238` and `:589-655`. No new comparative sentences are added by this revision; the conditional metrics in §7.2 are reported as numbers in artifacts only and are NOT promoted to natural-language claims in `phase_b_summary.txt` beyond a single new neutral block titled "Conditional metrics (subset-level)".

## 8. Interfaces to add/change

All paths absolute under `C:\Users\system1\Boundary-sensitivity\`. Function signatures Python 3.12 type-hinted.

### 8.1 `stage1/intervention/patcher.py` — generation parity fix (P0)

Modify `_greedy_continue_with_cache` (`stage1/intervention/patcher.py:506-566`) and `run_patched_inference_single` (`stage1/intervention/patcher.py:569-683`) so the no-patch and patched paths share a generation contract that matches Stage 1's `runner.py:118` `model.generate(input_ids, do_sample=False, max_new_tokens=...)`:

```python
def _greedy_continue_with_cache(
    model: "AutoModelForCausalLM",
    first_token_id: torch.Tensor,
    cache: "DynamicCache",
    prompt_len: int,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    pad_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Manual greedy decode reusing a prompt-seeded ``DynamicCache``.

    Parity contract with ``stage1/inference/runner.py::run_inference``:
    - greedy (do_sample=False, temperature=0.0)
    - stops the *batch* when every sample in the batch has emitted ``eos_token_id``
      (here batch=1, so equivalent to: stop the moment ``eos_token_id`` is emitted)
    - does NOT append any tokens after ``eos_token_id`` for that sample
    - tokens emitted at or after the per-sample EOS step MUST NOT influence the
      tokenizer.decode output; either truncate the returned tensor at the EOS
      index (preferred) OR replace post-EOS positions with ``pad_token_id`` and
      rely on ``skip_special_tokens=True`` to drop them
    - returns LongTensor [K] of generated token IDs on CPU; K is the count BEFORE
      truncation if truncation is the chosen mechanism (i.e., the returned
      tensor is the bytewise output that ``model.generate(..., return_dict_in_generate=False)[0, prompt_len:]``
      would have produced under the same kwargs).

    Implementation requirements:
    - Maintain a per-sample ``finished`` boolean (here scalar, batch=1).
      The current loop condition ``int(current.item()) == int(eos_token_id)``
      checks AFTER feeding ``current`` through the model — i.e., one extra
      forward / sample step happens at the EOS token. The fix is: check
      ``finished`` at the TOP of the loop and break BEFORE the next forward,
      which mirrors HF generate's stopping criterion.
    - The cache and ``cache_position`` indexing MUST remain consistent with the
      existing fix at line 545 (``pos_idx = prompt_len + step``); no changes
      to RoPE / cache indexing.
    - For the ``no_patch`` path (``patch_layers=[]``), the manual loop's output
      MUST be bytewise identical to ``model.generate(input_ids, do_sample=False,
      max_new_tokens=max_new_tokens)[0, prompt_len:]`` on every fixture prompt
      in §10 test #6.
    """
```

```python
def run_patched_inference_single(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    prompt: str,
    patch_config: PatchConfig,
    clean_layer_states: Optional[List[torch.Tensor]] = None,
    corrupt_layer_states: Optional[List[torch.Tensor]] = None,
    generation_config: Optional[dict] = None,
    device=None,
) -> Dict:
    """Unchanged signature.

    Internals diff vs current implementation:
    - Pass tokenizer.pad_token_id (or eos_token_id when pad is None) to
      ``_greedy_continue_with_cache`` so the truncation mechanism has the same
      pad fallback HF generate uses internally.
    - After ``_greedy_continue_with_cache`` returns, do tokenizer.decode with
      ``skip_special_tokens=True`` (already done at line 675) — this remains the
      single decode site so any pad tokens introduced by the truncation
      mechanism are dropped.
    - Add a defensive assertion: if ``patch_config.patch_layers == []`` and the
      env var ``PHASE_B_PARITY_DEBUG=1`` is set, additionally call
      ``model.generate(input_ids, do_sample=False, max_new_tokens=...)`` and
      assert tensor-equality of the two continuations. This is a debug-only
      guard, NOT in the default code path (no inference cost in production).
    """
```

No other patcher signatures change. `forward_with_patches`, `extract_all_layer_hidden_states`, `run_patched_inference` all keep their current signatures and current internals. The patch semantics (input-side + output-side; cache consistency) documented at `stage1/intervention/patcher.py:316-372` are PRESERVED.

### 8.2 `stage1/run_phase_b.py` — subset construction + conditional metrics

Add new top-level helpers (no signature changes to `run_phase_b` itself):

```python
def _load_anchor_per_sample(
    anchor_jsonl_path: str,
    samples: List[Dict],
) -> Tuple[List[Dict], str]:
    """Load per-sample rows from a Stage 1 anchor JSONL, ordered by samples.

    Validates that ``len(rows) == len(samples)`` and that ``sample_id`` matches
    pairwise. Each returned row dict contains at least: sample_id, output_text,
    normalized_answer (may be None), parse_success (bool), correct (bool),
    recomputed_correct (bool — see below).

    Correctness-source policy (per spec §7.2.1):
    - If the JSONL has a ``correct`` field, recompute via
      ``analysis.evaluator.exact_match(samples[i].gold_answer, normalized_answer)``
      and compare. On disagreement on ANY sample, raise
      ``RuntimeError("evaluator drift: anchor JSONL correct field disagrees with
      recomputed exact_match for sample_id=<id>")``. This is strict-fail; warning
      is NOT acceptable. Use the JSONL value as ``correct``; set
      ``recomputed_correct=False``.
    - If the JSONL lacks ``correct``, recompute via the same call and set
      ``correct = recomputed_value``, ``recomputed_correct=True``.

    Returns the row list AND a single string label
    ``"recomputed" | "jsonl"`` indicating which path the majority of rows took
    (used to populate ``summary["correct_field_source"]``).
    """
```

```python
def _build_subsets(
    clean_rows: List[Dict],
    corrupt_rows: List[Dict],
) -> Dict[str, List[bool]]:
    """Return dict of subset name -> per-sample boolean membership list.

    Subsets per spec §7.2.1. Length of every list equals len(clean_rows) ==
    len(corrupt_rows). None-handling for ``S_flipped`` per spec §7.2.1
    (None==None NOT flipped; None vs non-None flipped).
    """
```

```python
def _compute_conditional_metrics_one_condition(
    pc_rows: List[Dict],            # per-sample results for this condition
    no_patch_rows: List[Dict],      # restoration_no_patch (or clean_no_patch
                                    # for corruption direction) reference
    ans_clean: List[Optional[str]],
    ans_corrupt: List[Optional[str]],
    correct_clean: List[bool],
    correct_corrupt: List[bool],
    subsets: Dict[str, List[bool]],
    direction: str,                 # "restoration" or "corruption"
) -> Dict[str, Any]:
    """Return the 8-block dict per spec §7.2.2 / §7.2.4."""
```

```python
def _emit_conditional_artifacts(
    run_dir: str,
    subsets: Dict[str, List[bool]],
    sample_ids: List[str],
    ans_clean: List[Optional[str]],
    ans_corrupt: List[Optional[str]],
    correct_clean: List[bool],
    correct_corrupt: List[bool],
    per_condition_metrics: Dict[str, Dict[str, Any]],
) -> None:
    """Write phase_b_subsets.{json,csv} and phase_b_conditional_summary.{json,csv}."""
```

```python
def _maybe_compute_drift_diagnostic(
    run_dir: str,
    anchor_dir: str,
    sample_ids: List[str],
    subsets: Dict[str, List[bool]],
) -> str:
    """Optional per spec §7.3.

    Primary drift variable used as the AUROC score is
    ``boundary_layer_drift = cosine_distance[:, 8]`` (per spec §7.3 — do NOT
    use mean-over-layers as the primary variable). Secondary aggregates
    (mean / max / downstream-mean over layers, plus L2 analogues) are persisted
    alongside but NOT used as the default AUROC score; they are exposed in the
    JSON for downstream review only. Record
    ``primary_drift_variable: "boundary_layer_drift"`` at the top of
    ``phase_b_drift_behavior_summary.json``.

    Returns one of:
        "computed"
        "skipped: hidden_states_no_swap.pt missing"
        "skipped: hidden_states_hard_swap_b8.pt missing"
        "skipped: shape mismatch (expected [N, n_layers, hidden_dim])"
    """
```

Insertion points in `run_phase_b`:
- After step (12) `evaluate_phase_b_anchor_gate` returns, gate.passed is True (or gate is in sanity skip mode), and `gate.anchor_no_swap_source` / `gate.anchor_hard_swap_source` resolve to JSONL paths: load per-sample anchor rows and build subsets.
- After step (10) accuracy tables are built and BEFORE step (15) `_persist_summary(RUN_STATUS_PENDING)`: compute per-condition conditional metrics for every condition in `restoration_results`, `clean_baseline_results`, `corruption_results`. Write `phase_b_subsets.{json,csv}` and `phase_b_conditional_summary.{json,csv}`.
- After step (18) `_persist_summary(RUN_STATUS_PENDING)` and BEFORE step (19) wording gate: extend the wording-gate artifact list to include the 4 new files (and the 2 optional drift files if computed). The wording gate (existing `check_artifacts_for_forbidden`) is the SAME function; we are only adding paths.
- Append `summary["subset_summary"] = {n_total, n_broken, n_repaired, n_stable_correct, n_stable_wrong, n_flipped, n_correct_flipped, broken_rate, repaired_rate, answer_flip_rate}`.
- Append `summary["subset_warnings"] = [...]` per §7.2.3.
- Append `summary["drift_diagnostic"] = "<one of the §7.3 strings>"`.
- Append a "Conditional metrics (subset-level)" block to `body_lines` (the human-readable TXT body) listing, per restoration patch condition: `recovery_to_correct_rate`, `answer_restoration_rate`, `stable_correct_preservation_rate`, `parse_success_rate_global`, `human_continuation_rate`. No comparative sentence in this block — numbers only. The block lives BETWEEN the existing "Reverse corruption table" and "Interpretation" sections.

### 8.3 `stage1/utils/wording.py` — extend `FORBIDDEN_PHRASES_PHASE_B`

Add to `FORBIDDEN_PHRASES_PHASE_B` (the existing tuple at `stage1/utils/wording.py:29-39`) the brief §G additions that are not already covered. Diff:

```python
FORBIDDEN_PHRASES_PHASE_B: Tuple[str, ...] = (
    # --- existing entries (unchanged, do not reorder) ---
    "proves the mechanism",
    "proves mechanism",
    "causal proof",
    "identifies the true cause",
    "fully explains",
    "demonstrates causation",
    "restoration effect",
    "residual effect",
    "restoration proportion",
    # --- new entries from brief §G (added by this revision) ---
    "formal causal proof",
    "formal mediation",
    "causal mediation",
    "we prove causality",
    "restoration fully explains degradation",
    "natural direct effect",
    "natural indirect effect",
    "NIE",
    "NDE",
)
```

`_ACRONYM_PHRASES` (at `stage1/utils/wording.py:64`) is extended to include `"nie"` and `"nde"` if not already present (it already contains both). No change to `_contains_forbidden_phrase`.

`FORBIDDEN_PHRASES_PHASE_C` is unchanged. The backward-compat alias `FORBIDDEN_PHRASES = FORBIDDEN_PHRASES_PHASE_B` (line 61) automatically propagates the new entries to all Phase B call sites without any additional edit.

### 8.4 `stage1/tests/test_phase_b_patcher.py` — add parity tests

Add (no existing test removed):

```python
def test_eos_stop_no_post_eos_emission(): ...
def test_no_patch_generate_byte_equal_full_subset_smoke(): ...
def test_no_human_continuation_in_clean_no_patch_smoke(): ...
def test_subset_construction_correctness(): ...
def test_subset_warning_emission_below_threshold(): ...
```

`test_eos_stop_no_post_eos_emission` is a unit test on `_greedy_continue_with_cache` with a stub model that emits a fixed EOS at step k; the returned tensor MUST have length `k+1` and MUST NOT contain any token at index `>k+1` other than (optionally) `pad_token_id` if the truncation-by-pad mechanism is chosen. (Preferred mechanism is hard truncation, in which case `len(returned) == k+1` exactly.)

`test_no_patch_generate_byte_equal_full_subset_smoke` is the parity smoke on 5 fixture prompts (subset of MGSM-zh): `_greedy_continue_with_cache` output for `patch_layers=[]` MUST equal `model.generate(input_ids, do_sample=False, max_new_tokens=512)[0, prompt_len:]` token-for-token.

`test_no_human_continuation_in_clean_no_patch_smoke` is a deterministic smoke that asserts on 5 fixture prompts the `clean_no_patch` decoded output contains no `"Human:"` substring. (Stage 1 baseline has 0/250 such occurrences per the brief §A diff.)

`test_subset_construction_correctness` is a pure-Python test on `_build_subsets` with hand-constructed clean/corrupt row pairs; asserts every membership matrix entry matches the §7.2.1 truth table, including all four `None` corner cases for `S_flipped`.

`test_subset_warning_emission_below_threshold` calls the (new) summary-warning helper with `n_broken=10` and asserts the warning string is present.

Existing tests are NOT removed. `test_forbidden_phrases_gate` (`stage1/tests/test_phase_b_patcher.py:91`) automatically picks up the new phrases because it iterates `FORBIDDEN_PHRASES`.

### 8.5 `stage1/tests/test_phase_b_run_status.py`

No code change needed; the existing tests assert on `summary["run_status"]` which remains `"passed"` / `"failed"` / `"pending"` per `stage1/utils/run_status.py`. The new summary keys (`subset_summary`, `subset_warnings`, `drift_diagnostic`) are additive and will not break the test that loads the JSON.

## 9. Files-to-touch (exhaustive)

- `C:\Users\system1\Boundary-sensitivity\stage1\intervention\patcher.py` — `_greedy_continue_with_cache`, `run_patched_inference_single` (only) — **modify**. Rationale: P0 generation parity fix per brief §B; finished-mask check moved to top of loop, truncation contract documented and enforced. All other functions in this file (`forward_with_patches`, `extract_all_layer_hidden_states`, `run_patched_inference`, `_get_model_components`, `_build_prompt_inputs`, `_build_causal_mask`, etc.) are NOT touched.
- `C:\Users\system1\Boundary-sensitivity\stage1\run_phase_b.py` — `run_phase_b`, plus 5 new private helpers `_load_anchor_per_sample`, `_build_subsets`, `_compute_conditional_metrics_one_condition`, `_emit_conditional_artifacts`, `_maybe_compute_drift_diagnostic` — **modify (driver) + add (helpers)**. Rationale: subset construction, conditional metrics, additive summary fields, additive TXT block, extended wording-gate artifact list. Existing global tables, anchor gate call, comparative-claim gate, run-status state machine, state_dict SHA check, sanity-mode condition set are all PRESERVED.
- `C:\Users\system1\Boundary-sensitivity\stage1\utils\wording.py` — `FORBIDDEN_PHRASES_PHASE_B` tuple — **modify (extend)**. Rationale: brief §G additions (`formal causal proof`, `formal mediation`, `causal mediation`, `we prove causality`, `restoration fully explains degradation`, plus the NDE/NIE / natural-direct-effect / natural-indirect-effect set that brief §G implies under "we prove causality"-class phrasing). `FORBIDDEN_PHRASES_PHASE_C` is NOT touched; the backward-compat alias on line 61 automatically propagates the new entries.
- `C:\Users\system1\Boundary-sensitivity\stage1\tests\test_phase_b_patcher.py` — add 5 tests as enumerated in §8.4 — **modify (additive)**. Rationale: brief §B test list + brief §D subset-correctness coverage. Existing tests untouched.
- `C:\Users\system1\Boundary-sensitivity\notes\specs\phase_b_revision.md` — this spec — **add** (the writer-target).

Do-NOT-touch (enforced by memory rules + the spec-planner role + brief priority order):
- `C:\Users\system1\Boundary-sensitivity\stage1\inference\runner.py` (Stage 1 generation reference; the parity contract reads this file)
- `C:\Users\system1\Boundary-sensitivity\stage1\inference\parser.py` (committed non-finite fix is the most recent allowed change)
- `C:\Users\system1\Boundary-sensitivity\stage1\analysis\evaluator.py` (`exact_match` is the parity reference for `correct_clean` / `correct_corrupt`)
- `C:\Users\system1\Boundary-sensitivity\stage1\analysis\bds.py`
- `C:\Users\system1\Boundary-sensitivity\stage1\data\loader.py`
- `C:\Users\system1\Boundary-sensitivity\stage1\run_phase_a.py` (Phase A is frozen)
- `C:\Users\system1\Boundary-sensitivity\stage1\run_phase_c.py` and `notes/specs/phase_c_mediation.md` (Phase C MUST NOT be touched until Phase B passes per brief §H priority order)
- `C:\Users\system1\Boundary-sensitivity\stage1\models\composer.py` (read-only)
- `C:\Users\system1\Boundary-sensitivity\stage1\configs\stage2_confound.yaml` (parity contract on generation kwargs)
- `C:\Users\system1\Boundary-sensitivity\stage1\utils\anchor_gate.py` (cross-phase parity gate)
- `C:\Users\system1\Boundary-sensitivity\stage1\utils\manifest_parity.py`
- `C:\Users\system1\Boundary-sensitivity\stage1\utils\provenance.py`
- `C:\Users\system1\Boundary-sensitivity\stage1\utils\run_status.py`
- `C:\Users\system1\Boundary-sensitivity\notes\specs\phase_b_rewrite.md` (kept as the structural baseline for fields this revision is silent on)

Anchor artifacts the writer MUST consult by path (read-only; will be opened via `_load_anchor_per_sample` at run time). These are CANONICAL — no fallback to older runs (e.g., `2026-04-25` runs) and no auto-latest selection is permitted:
- Stage 1 anchor: `stage1/outputs/run_20260427_020843_372153` (per brief §A; canonical for the active branch `hardening/boundary-sensitivity-mainrun-prep`). The writer MUST configure / verify that `evaluate_phase_b_anchor_gate` resolves to THIS exact path. If the existing gate's parity filter would otherwise pick a different (older or newer) anchor as the parity-best match, the writer MUST add an explicit pin so the canonical path wins, or hard-fail with `RuntimeError("anchor pin drift: expected stage1/outputs/run_20260427_020843_372153, gate selected <other>")`. Auto-fallback to `2026-04-25` runs is FORBIDDEN.
- Phase A anchor: `stage1/outputs/phase_a/run_20260427_104320_510816` (per brief §A; canonical). Same pin / hard-fail policy as above. No auto-latest, no auto-fallback.
- Missing-anchor handling: If either canonical path is absent on disk in full mode, the run hard-fails with the existing `RuntimeError("missing anchor(s)")`. Sanity-mode behavior is governed by §7.2.4 (explicit-skipped artifacts, never silent fall-through).
- Failed Phase B (debugging artifact only; do NOT load as source of truth): `stage1/outputs/phase_b/run_20260428_071043_694696`.
- New Phase B run output: `stage1/outputs/phase_b/run_<NEW_TIMESTAMP>` — created by `stage1/utils/logger.py::create_run_dir` at run start; do NOT invent the timestamp now.

Deletions: none. The existing artifacts produced by the failed run are debugging artifacts and stay where they are; the wording gate only scans the NEW run dir.

## 10. Test plan

Unit + smoke + eval-sanity. Each command runnable from repo root.

**Execution gate ordering (writer MUST follow strictly per brief priority order):**
1. Steps 1-6 (unit + parity-smoke pytest targets) MUST all exit 0 BEFORE step 7 (sanity-mode CLI smoke) is run.
2. Step 7 MUST exit 0 BEFORE step 8 (full clean rerun) is run.
3. The writer MUST NOT launch step 8 if any of steps 1-7 has not been run-and-passed in the same session. Skipping the gate is a spec violation that the watcher and codex-reviewer MUST catch.

1. `python -m pytest stage1/tests -q -rs` — full existing test suite plus the 5 new tests in §8.4. MUST exit 0; MUST NOT introduce new skips beyond the existing torch-availability skips already documented at `stage1/tests/test_phase_b_patcher.py:39-86`.
2. `python -m pytest stage1/tests/test_phase_b_patcher.py::test_eos_stop_no_post_eos_emission -q` — pure-Python unit test on `_greedy_continue_with_cache` with a stub model. MUST pass without GPU.
3. `python -m pytest stage1/tests/test_phase_b_patcher.py::test_no_patch_generate_byte_equal_full_subset_smoke -q` — 5 fixture prompts; auto-skip if torch unavailable. MUST exit 0 when torch is available.
4. `python -m pytest stage1/tests/test_phase_b_patcher.py::test_no_human_continuation_in_clean_no_patch_smoke -q` — 5 fixture prompts; auto-skip if torch unavailable. MUST exit 0 with assertion that no `"Human:"` substring appears in any of the 5 decoded outputs.
5. `python -m pytest stage1/tests/test_phase_b_patcher.py::test_subset_construction_correctness -q` — pure-Python; MUST pass.
6. `python -m pytest stage1/tests/test_phase_b_patcher.py::test_subset_warning_emission_below_threshold -q` — pure-Python; MUST pass.
7. `python -m stage1.run_phase_b --config stage1/configs/stage2_confound.yaml --sanity` — sanity smoke (5 samples × 4 conditions). MUST complete in ≤ 10 min on the dev GPU and exit 0. The sanity-mode anchor-gate skip remains in effect (existing `stage1/run_phase_b.py:792-797`), so the cross-check criteria in §11 do not apply in sanity mode.
8. `python -m stage1.run_phase_b --config stage1/configs/stage2_confound.yaml` — full clean rerun (250 samples × 11 conditions). MUST complete in ≤ 6 GPU-hours on the dev box and exit 0.

Eval-sanity checks emitted inline by `run_phase_b` (real, not vacuous; existing checks PRESERVED, two new ones ADDED):

PRESERVED (from `phase_b_rewrite.md` §10 and `stage1/run_phase_b.py:766-820`):
- `no_patch` results exist, no NaN in tables, `clean_baseline_accuracy` finite, `t_fixed == 20`, conservative-wording gate clean, state_dict SHA-256 stable, cross-phase tolerance both anchors within 0.008.

ADDED:
- `subset construction succeeded`: in full mode, `phase_b_subsets.json` exists and `n_total == len(samples)`. False → hard fail. In sanity mode with anchors unavailable, `phase_b_subsets.json` exists with `{"status": "skipped", ...}` per §7.2.4 — also acceptable.
- `Human:-continuation parity`: `human_continuation_count` for `clean_no_patch` is within `stage1_no_swap_human_count + 2` of the Stage 1 `no_swap` anchor's count, AND for `restoration_no_patch` within `stage1_hard_swap_b8_human_count + 2`. False → hard fail (this is the explicit symptom of the failed run per brief §A; it is the direct test that the P0 fix worked). For current Stage 1 references (0/250 per brief), this resolves to `≤ 2/250` for both conditions. Patched (non no-patch) conditions are NOT hard-gated by this check.
- `Anchor pin verification`: at run start, the writer asserts that the resolved Stage 1 and Phase A anchors equal the canonical paths in §9. Mismatch → hard fail with the §9 error message.

Any single failure → hard `RuntimeError` and `_persist_summary(RUN_STATUS_FAILED)` per the existing state machine.

## 11. Acceptance criteria

All criteria are checked against artifacts under a single run directory `RD = stage1/outputs/phase_b/run_<NEW_TIMESTAMP>` produced by §10 step 8 (the full clean rerun). All thresholds are numeric and auditable; no vibes.

1. `python -m pytest stage1/tests -q -rs` exits 0 with no new skips beyond the torch-availability skips already present.
2. `RD/RUN_STATUS.txt` first non-blank line equals literal `PASSED` (existing `stage1/utils/run_status.py` contract).
3. `RD/phase_b_summary.json` exists, is valid UTF-8 JSON, and `summary["run_status"] == "passed"` and `summary["failure_reason"] is null`.
4. **Generation parity (P0)** — `RD/phase_b_summary.json.phase_a_cross_check.passed == true`, AND each of the following holds (computed by the existing `evaluate_phase_b_anchor_gate` against the parity-selected Stage 1 anchor):
    - `|summary.clean_baseline_accuracy − Stage1.no_swap.accuracy| ≤ 0.008`
    - `|summary.no_patch_accuracy − Stage1.hard_swap_b8.accuracy| ≤ 0.008`
   These thresholds are the existing `PHASE_A_CROSS_CHECK_TOL`. They MUST NOT be relaxed.
5. **Human-continuation parity (P0)** — derived from `RD/phase_b_conditional_summary.json`. Hard-gated for the two no-patch parity conditions ONLY:
    - `phase_b_conditional_summary.json["clean_no_patch"]["output_behavior"]["human_continuation_count"] ≤ stage1_no_swap_human_count + 2`
    - `phase_b_conditional_summary.json["restoration_no_patch"]["output_behavior"]["human_continuation_count"] ≤ stage1_hard_swap_b8_human_count + 2`
    where `stage1_*_human_count` is computed by scanning the Stage 1 anchor JSONL `output_text` field for the literal substring `"Human:"`. Brief §A documents 0/250 for both Stage 1 reference conditions, so the practical bound is `≤ 2/250` for both Phase B no-patch mirrors. Patched conditions are NOT hard-gated by this check (their `human_continuation_count` is recorded for review but is not a pass/fail criterion).
6. **Output-length parity (P0)** — hard-gated for the two no-patch parity conditions ONLY:
    - `mean(len(output_text) for clean_no_patch) ≤ 1.25 × mean(len(output_text) for stage1_no_swap)`
    - `mean(len(output_text) for restoration_no_patch) ≤ 1.25 × mean(len(output_text) for stage1_hard_swap_b8)`
   The 1.25× upper bound is intentional headroom over the brief §A diff (current ratio is ≈ 2.0×, deeply outside parity). Patched (non no-patch) conditions are NOT hard-gated on output length. An OPTIONAL warning line is emitted to `summary["length_warnings"]` (and to `phase_b_summary.txt` before the Interpretation block) if any patched condition's mean length falls below `0.75 ×` the matched-direction no-patch mean (potential collapse / early-EOS signal — informational only, not a fail).
7. **Subset construction artifacts** — `RD/phase_b_subsets.json` exists, valid UTF-8 JSON, contains keys `n_total`, `n_stable_correct`, `n_broken`, `n_repaired`, `n_stable_wrong`, `n_flipped`, `n_correct_flipped`, `broken_rate`, `repaired_rate`, `answer_flip_rate`. `n_total == 250` on the full run. `n_stable_correct + n_broken + n_repaired + n_stable_wrong == n_total` (partition invariant). `RD/phase_b_subsets.csv` exists with one row per `sample_id` and the column schema in §7.2.1.
8. **Conditional metrics artifacts** — `RD/phase_b_conditional_summary.json` exists, valid UTF-8 JSON, top-level keys = `{"clean_no_patch", "restoration_no_patch", "patch_boundary_local", "patch_recovery_early", "patch_recovery_full", "patch_final_only", "patch_all_downstream", "corruption_corrupt_boundary_local", "corruption_corrupt_recovery_early", "corruption_corrupt_recovery_full", "corruption_corrupt_final_only"}` (11 conditions on the full run). Each value is the 8-block dict from §7.2.2 with all numeric fields populated as JSON `number` (not `null`) when the relevant subset is non-empty; when a subset is empty, the corresponding rate fields are `null` and the `summary["subset_warnings"]` entry MUST cite that subset.
9. **Conditional metrics CSV** — `RD/phase_b_conditional_summary.csv` exists, valid UTF-8, long-format with columns `condition, metric_name, subset, value` and ≥ 11 conditions × ≥ 14 metric_name × (subsets where applicable) rows total. Header row is exactly these 4 columns.
10. **Subset-size warnings emission** — if `n_broken < 20`, `summary["subset_warnings"]` contains a string starting with `"warning: n_broken="`. Same for `n_flipped < 20`. Same for `0 < n_correct_flipped < 10`. The warning strings also appear verbatim in `RD/phase_b_summary.txt` BEFORE the `"Interpretation"` block.
11. **Conservative-wording gate clean** — `check_artifacts_for_forbidden(WORDING_GATE_PATHS)` returns `[]` over the extended `FORBIDDEN_PHRASES_PHASE_B` per §8.3, where `WORDING_GATE_PATHS` is the EXPLICIT, FIXED list of user-facing report/summary artifact filenames produced by THIS run (no glob, no legacy-filename heuristic):
    - `phase_b_summary.txt`
    - `phase_b_summary.json`
    - `restoration_table.csv`
    - `corruption_table.csv`
    - `phase_b_subsets.json`
    - `phase_b_subsets.csv`
    - `phase_b_conditional_summary.json`
    - `phase_b_conditional_summary.csv`
    - `phase_b_drift_behavior_summary.json` (only if `summary["drift_diagnostic"] == "computed"`)
    - `phase_b_drift_behavior.csv` (only if `summary["drift_diagnostic"] == "computed"`)
   The gate is intentionally scoped to user-facing report content; it is NOT applied to legacy filenames or to per-sample JSONLs (which contain raw model output text and would brittle-fire on benign quoted material). If any of the listed files is missing from `RD/`, that is a separate failure under criteria (3), (7), (8), or (15) — not a wording-gate failure.
12. **Composed state_dict immutability** — preserved from `phase_b_rewrite.md` §11.8: `compose_meta.state_dict_sha256_before == compose_meta.state_dict_sha256_after`.
13. **Comparative-sentence gate behavior preserved** — exactly one sentence in `RD/phase_b_summary.txt` begins with either `"Recovery-side intervention"` (gate fires) or `"Restoration deltas do not meet the effect-size threshold for a directional claim"` (gate does not fire). The gate's epsilon (`EPSILON_DELTA = 0.02`) is unchanged.
14. **No spurious-recovery line preserved** — `phase_b_rewrite.md` §11.9: literal substring `"of performance (rough estimate)"` does NOT appear anywhere in `RD/`.
15. **Optional drift diagnostic accounting** — `summary["drift_diagnostic"]` is one of the four strings in §7.3. If `"computed"`, then `RD/phase_b_drift_behavior_summary.json` and `RD/phase_b_drift_behavior.csv` exist and pass the wording gate. If `"skipped: ..."`, neither drift artifact exists in `RD/`.
16. **TXT header preserved** — `RD/phase_b_summary.txt` contains the literal line `"recovery-zone layers [20..27] defined at fixed t=20"` (from `phase_b_rewrite.md` §11.11).
17. **Methodology column preserved** — every row of `RD/restoration_table.csv` and `RD/corruption_table.csv` has the unchanged `methodology` cell value `"prompt-side patching; patch at prompt tokens only; continuation via DynamicCache"` (`phase_b_rewrite.md` §11.3).
18. **Evaluator-drift safety** — `_load_anchor_per_sample` did not raise: by virtue of (3) the run reached `passed`, the anchor-vs-recompute correctness check in §7.2.1 passed for all 250 samples (strict-fail policy: any single mismatch between anchor JSONL `correct` and recomputed `exact_match` would have raised, blocking `passed`). When the JSONL lacked `correct` for any sample, `phase_b_subsets.csv` records `recomputed_correct_clean=true` / `recomputed_correct_corrupt=true` for that row, AND `summary["correct_field_source"]` is populated with `{"clean": "recomputed"|"jsonl", "corrupt": "recomputed"|"jsonl"}`.

A single failure of any of (1)-(18) flips the run to `RUN_STATUS_FAILED` and the spec is unsatisfied.

## 12. Risks and ablations

Risks layered on top of `phase_b_rewrite.md` §12 (risks R1-R9 from that spec all carry forward unchanged):

- R10 — **`_greedy_continue_with_cache` semantic drift**: HF `model.generate(do_sample=False)` uses a `StoppingCriteriaList` and a per-batch finished mask. Our manual loop's correctness depends on emulating that semantics exactly (stop on EOS BEFORE the next forward, truncate the returned tensor at the EOS index). Mitigation: §10 test #6 is a full-subset bytewise parity test that catches any semantic drift; the §11.5 / §11.6 `Human:`-count and length parity criteria are the production-time check. Failure is hard, not silent.
- R11 — **Anchor JSONL `correct` field schema drift**: older Stage 1 JSONL outputs may omit `correct`. The `_load_anchor_per_sample` recompute path covers this (it computes `correct` via `analysis/evaluator.exact_match`). The drift safety check (§7.2.1) catches the case where the field is present but disagrees — e.g., a parser-version mismatch between when the anchor was generated and the current parser.
- R12 — **Subset-size ceiling**: if `S_broken` and `S_flipped` are both small (per the brief §C, hard_swap_b8 average degradation is small), the conditional restoration rates have high variance. Mitigation: (a) §7.2.3 warnings, (b) wording-gate-enforced conservative phrasing, (c) the new "Conditional metrics (subset-level)" TXT block reports raw numbers WITHOUT comparative claims.
- R13 — **Hidden-state-pooling shape mismatch for drift diagnostic**: `runner.py:14-46` saves prompt-pooled hidden states `[n_layers, hidden_dim]`. If the Stage 1 anchor saved per-token states instead `[seq_len, n_layers, hidden_dim]` or the file is missing entirely, the diagnostic skips silently with the `summary["drift_diagnostic"]` annotation. NEVER infer subset membership from a partial drift load.
- R14 — **Wording-gate over-trigger from new vocabulary**: adding `"natural direct effect"`, `"natural indirect effect"`, `"NIE"`, `"NDE"` to Phase B's list could fire on a legitimate quote of the Vig 2020 abstract. Mitigation: the existing test `test_forbidden_phrases_gate` already iterates the tuple and catches both directions (does the gate fire on the literal phrase? — yes; does it false-fire on a normal Phase B summary? — no, per §11.11). The spec text and TXT body intentionally avoid these phrases; only artifacts are scanned.
- R15 — **Stage 1 anchor JSONL `output_text` re-decoding cost**: scanning 250 samples × 11 conditions for `"Human:"` substrings is O(N) per condition and runs once at gate time. Negligible.

Ablations (informational; not part of the deliverable):
- A3 — Optional drift diagnostic with multi-layer aggregation rule: the spec uses mean cosine over layers; an alternative is `boundary_layer_drift` only. The artifact records both, so a downstream review can switch without a re-run.
- A4 — Subset-size warning thresholds: 20 / 20 / 10 are conservative defaults from the brief §E; raising or lowering them requires a spec amendment, not a code patch.

## 13. Compute budget

Inference-only, additive on top of `phase_b_rewrite.md` §13. The conditional-metrics computation (§7.2) is pure Python on already-collected JSONL rows — no extra forwards. The optional drift diagnostic (§7.3) is `torch.load` + cosine/L2 over `[N=250, n_layers=28, hidden=1536]` tensors — < 5 seconds CPU.

- Per-sample cost: unchanged from `phase_b_rewrite.md` §13 (~1.5× plain generate per condition).
- Total forwards: unchanged at 11 conditions × 250 samples = 2,750 patched-or-baseline forwards + 2,500 clean-state-extraction forwards = ~5,250 prompt forwards + 2,750 generate-continuations.
- Wall-clock target on dev GPU: ≤ 6 h full run; ≤ 10 min `--sanity`.
- GPU-hours: ≤ 6 H100-equivalent hours (or ≤ 12 A6000 hours).
- Storage delta: 4 new artifacts (`phase_b_subsets.json` ≈ 5 KB, `phase_b_subsets.csv` ≈ 30 KB, `phase_b_conditional_summary.json` ≈ 50 KB, `phase_b_conditional_summary.csv` ≈ 100 KB) + optional 2 drift artifacts (≈ 20 KB + 30 KB). Total per-run additive ≤ 250 KB. Run dir total ≤ 11 MB (vs. ≤ 10 MB in `phase_b_rewrite.md` §13).
- CPU RAM peak: unchanged. Drift diagnostic loads `[250, 28, 1536]` fp32 tensors ≈ 43 MB transient.

No multi-node, no multi-GPU, no batch-size change.

## 14. Rollback

Contained in 4 source files (patcher.py, run_phase_b.py, utils/wording.py, tests/test_phase_b_patcher.py) + 1 new spec file (this document). No schema migrations, no new model checkpoints, no config changes, no Phase A or Phase C touch.

Rollback procedure:

1. `git restore stage1/intervention/patcher.py stage1/run_phase_b.py stage1/utils/wording.py stage1/tests/test_phase_b_patcher.py` to return those files to their pre-revision state (i.e., the `phase_b_rewrite.md`-era state on the `hardening/boundary-sensitivity-mainrun-prep` branch).
2. `git rm notes/specs/phase_b_revision.md` to drop this spec.
3. Delete any `stage1/outputs/phase_b/run_<NEW_TIMESTAMP>/` directories produced by the rerun (safe — no downstream job depends on them; Phase C has not started per memory, and the failed prior Phase B run `run_20260428_071043_694696` is already a debugging artifact).
4. No HF hub state, no dataset artifacts, no caches to roll back.
5. Phase A artifacts are untouched by construction (do-not-modify list); verify `git status stage1/run_phase_a.py stage1/inference/runner.py stage1/inference/parser.py stage1/analysis/evaluator.py` shows no changes.
6. Stage 1 anchor (`run_20260427_020843_372153`) and Phase A anchor (`run_20260427_104320_510816`) are read-only references; rollback does NOT touch them.

Rollback is idempotent and completes in under 30 seconds. No data loss beyond the new Phase B run outputs, all of which are reproducible from the pre-revision code if ever needed (modulo the original parity bug, which is the entire reason for this revision).
