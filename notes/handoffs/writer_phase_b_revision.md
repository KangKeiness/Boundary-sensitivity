spec_ref: notes/specs/phase_b_revision.md

## Change 1: Generation parity fix

files_changed:
  - path: stage1/intervention/patcher.py
    diff_stat: +100 -7

tests_added (parity scope):
  - test_eos_stop_no_post_eos_emission — pure-Python unit test on `_greedy_continue_with_cache` with a stub model emitting EOS at step k; asserts no post-EOS token emission and exact-length truncation contract per spec §8.1.
  - test_no_patch_generate_byte_equal_full_subset_smoke — 5 fixture-prompt parity smoke; auto-skips when CUDA + cached Qwen weights unavailable per the `_has_cuda_and_real_weights` guard at `stage1/tests/test_phase_b_patcher.py:39-86`.
  - test_no_human_continuation_in_clean_no_patch_smoke — 5 fixture-prompt deterministic smoke; auto-skips under the same CUDA/weights guard; asserts no `"Human:"` substring in any decoded `clean_no_patch` output.

intent: Fix the patched-generation path so that `_greedy_continue_with_cache` (`stage1/intervention/patcher.py:506-566` original; now lines `~506-666` post-edit) honors `eos_token_id` with the same semantics as Stage 1's `runner.py::run_inference` greedy `model.generate(do_sample=False, max_new_tokens=...)` call. The pre-fix loop checked the finished condition AFTER the next forward, allowing one extra token-step at the EOS boundary; the fix moves the check to the TOP of the loop and adds an explicit truncation-by-position contract documented in the docstring per spec §8.1. `run_patched_inference_single` (`stage1/intervention/patcher.py:569-683` original) now passes `tokenizer.pad_token_id or eos_token_id` into the greedy helper so the post-EOS truncation mechanism mirrors HF generate's pad fallback. Empty-patch (`patch_layers == []`) byte-equality vs. `model.generate` is asserted by the new parity test under a `PHASE_B_PARITY_DEBUG=1` env-guarded debug branch.

## Change 2: Conditional subset metrics + reporting

files_changed:
  - path: stage1/run_phase_b.py
    diff_stat: +1409 -6
  - path: stage1/utils/wording.py
    diff_stat: +11 -0
  - path: stage1/tests/test_phase_b_patcher.py
    diff_stat: +317 -0 (subset of total — only the 2 conditional-tests slice: `test_subset_construction_correctness` and `test_subset_warning_emission_below_threshold`; the other 3 tests in this file's diff belong to "Change 1" above per the §8.4 / §8.1 split. Total file +317 -0 covers all 5 new tests; no clean per-change line-count subdivision is reported because the tests share the same `from ... import` block at the top of the new section.)

helpers_added (5 from spec §8.2 + 1 operator-constraint variant):
  - `_load_anchor_per_sample(anchor_jsonl_path, samples) -> Tuple[List[Dict], str]` — strict-fail evaluator-drift check per spec §7.2.1.
  - `_build_subsets(clean_rows, corrupt_rows) -> Dict[str, List[bool]]` — partition + flip subsets with None-handling per spec §7.2.1 truth table.
  - `_compute_conditional_metrics_one_condition(...)` — 8-block per-condition metrics per spec §7.2.2 / §7.2.5.
  - `_emit_conditional_artifacts(run_dir, ...)` — writes `phase_b_subsets.{json,csv}` + `phase_b_conditional_summary.{json,csv}` (full path).
  - `_emit_conditional_artifacts_skipped(run_dir, ...)` — writer-added variant that emits the spec §7.2.4 explicit-skipped JSON object + header-only CSVs + sidecar `phase_b_subsets.SKIPPED.txt` and `phase_b_conditional_summary.SKIPPED.txt`. Wired into `run_phase_b.py` at two call sites (lines ~1495 sanity-skip path and ~1623 P0-parity-failed path) so the operator constraint #4 contract ("if `clean_no_patch` or `restoration_no_patch` fails the parity gate, the conditional metrics layer MUST emit explicit failed/skipped status artifacts with reasons") is preserved.
  - `_maybe_compute_drift_diagnostic(run_dir, anchor_dir, sample_ids, subsets) -> str` — optional drift artifact per spec §7.3 with `boundary_layer_drift` (cosine at layer 8) as the primary AUROC score variable.

Operator-constraint #6 fix (sanity/debug-marking) — additive edit to `_persist_summary` in `run_phase_b.py` (lines ~2000-2030 post-edit). After `write_phase_b_status_artifacts` writes `phase_b_summary.txt`, the helper additively prepends a `"[SANITY] [DEBUG] — sanity smoke; not a full Phase B validation (mode=sanity, see phase_b_summary.json[\"mode\"])"` line ABOVE the existing `RUN STATUS:` banner when `sanity=True`. This makes the FIRST non-blank line of the TXT contain both `[SANITY]` and `[DEBUG]` literals as required by operator constraint #6. Implementation done as a post-write rewrite so `stage1/utils/run_status.py` (do-not-touch list, spec §9 absolute) is not modified. The `summary["mode"] == "sanity"` JSON marker was already present at `stage1/run_phase_b.py:1745-1746` before this writer session and is preserved verbatim.

intent: Add the spec §7.2 conditional-metrics layer (subset construction from Stage 1 anchor JSONLs, per-patch-condition recovery / answer-restoration / preservation rates with subset-size warnings) and the spec §7.3 optional drift-vs-behavior diagnostic, both layered ADDITIVELY on top of the existing global `restoration_table.csv` / `corruption_table.csv` / `phase_b_summary.{json,txt}` outputs. Extend `FORBIDDEN_PHRASES_PHASE_B` with the brief §G additions ("formal causal proof", "formal mediation", "causal mediation", "we prove causality", "restoration fully explains degradation", "natural direct effect", "natural indirect effect", "NIE", "NDE"). Extend the explicit `WORDING_GATE_PATHS` list to scan the 4 new artifacts (and the 2 optional drift artifacts when computed). Sanity-mode artifacts are explicitly marked in BOTH `phase_b_summary.json["mode"] == "sanity"` AND the `phase_b_summary.txt` first non-blank line per operator constraint #6 (the latter via the additive `_persist_summary` rewrite documented above).

## Tests run (full)

pytest_full_suite:
  command: python -m pytest stage1/tests -q -rs
  status: pass
  passed: 256
  skipped: 2
  skipped_ids:
    - "stage1/tests/test_phase_b_patcher.py:1113 — Real-weights smoke test (spec §10) requires CUDA + cached Qwen/Qwen2.5-1.5B-Instruct weights. Documented torch/CUDA-unavailable skip; matches the existing skip pattern at `stage1/tests/test_phase_b_patcher.py:39-86` per spec baseline."
    - "stage1/tests/test_runtime_smoke.py:253 — 'real runtime is present — stub-leakage path not exercised'. Pre-existing skip; not introduced by this revision."
  failing_ids: []
  notes: "Run a second time after the operator-constraint #6 additive edit to `_persist_summary`; same 256 passed / 2 skipped result. Wall-clock 58.6s second run, 82.8s first run."

per_target_results:
  - target: test_eos_stop_no_post_eos_emission
    status: pass
    reason_if_skip: null
    wall_clock_s: 6.46
  - target: test_no_patch_generate_byte_equal_full_subset_smoke
    status: pass
    reason_if_skip: null
    wall_clock_s: 7.44
    note: "Did not auto-skip — CUDA + cached Qwen weights are present on this dev box."
  - target: test_no_human_continuation_in_clean_no_patch_smoke
    status: pass
    reason_if_skip: null
    wall_clock_s: 7.32
    note: "Did not auto-skip — CUDA + cached Qwen weights are present on this dev box. Asserts no `Human:` substring in 5 fixture-prompt `clean_no_patch` outputs."
  - target: test_subset_construction_correctness
    status: pass
    reason_if_skip: null
    wall_clock_s: 7.20
  - target: test_subset_warning_emission_below_threshold
    status: pass
    reason_if_skip: null
    wall_clock_s: 7.22

## Sanity smoke result

run_dir: stage1/outputs/phase_b/run_20260429_094926_429332
exit_code: 0
wall_clock_seconds: 455 (≈ 7.6 min; under the spec §13 / §10 step 7 ≤ 10 min target)

mode_marker_present: true (phase_b_summary.json["mode"] == "sanity"; phase_b_summary.json["sanity_mode"] == true)
sanity_marker_in_txt: false (in this specific run dir's existing `phase_b_summary.txt`) — see "Important caveat" below.

Important caveat about `sanity_marker_in_txt`:
  The single sanity smoke allowed by operator constraint #2 was executed BEFORE the operator-constraint #6 additive `_persist_summary` rewrite was added. Therefore the EXISTING `phase_b_summary.txt` in this run dir lacks the `[SANITY] [DEBUG]` first-line marker; its first non-blank line is `============================================================` (the `=`-separator emitted by `write_phase_b_status_artifacts`'s status banner), with `RUN STATUS: PASSED — all gates satisfied` on line 2 and `[SANITY] PHASE B — RESTORATION INTERVENTION RESULTS` on line 6 (already present pre-edit, in the body banner). The code fix is in place and verified by a `python -c` probe that applies the same prepend logic to the existing TXT and confirms the resulting first non-blank line is `"[SANITY] [DEBUG] — sanity smoke; not a full Phase B validation (mode=sanity, see phase_b_summary.json[\"mode\"])"` — both `[SANITY]` and `[DEBUG]` substring present. The operator's full clean rerun (and any future sanity smoke) will produce a TXT with the marker as the first non-blank line. The writer did NOT mutate the existing `outputs/` artifact (writer.md hard constraint forbids editing `outputs/**`).

key_metrics:
  run_status: passed
  clean_baseline_accuracy: 0.6
  no_patch_accuracy: 0.4
  conditional_metrics_status: "skipped: sanity mode: stage1 anchors unavailable for subset construction" (per spec §7.2.4 explicit-skipped path; `_emit_conditional_artifacts_skipped` was the helper invoked, producing `phase_b_subsets.json`, `phase_b_conditional_summary.json`, the two header-only CSVs, and the two `*.SKIPPED.txt` sidecars)
  drift_diagnostic: "skipped: anchor_dir unavailable" (one of the spec §7.3 strings)
  human_continuation_count_clean_no_patch: 5 (vs Stage 1 anchor: 0/250 per brief §A; sanity-mode parity gate was SKIPPED — see "Open items" #2)
  human_continuation_count_restoration_no_patch: 5 (vs Stage 1 anchor: 0/250 per brief §A; sanity-mode parity gate was SKIPPED — see "Open items" #2)
  avg_output_length_clean_no_patch: 1456.2 chars (vs Stage 1 no_swap anchor: 676.7 chars per brief §A; ratio ≈ 2.15× — exceeds the spec §11.6 / falsification-criterion 1.25× cap, but the cap is NOT applied in sanity mode because the anchor gate is skipped)
  avg_output_length_restoration_no_patch: 1688.8 chars (vs Stage 1 hard_swap_b8 anchor: 697.9 chars per brief §A; ratio ≈ 2.42× — same caveat as above)
  avg_output_length_patch_recovery_full: 1691.8 chars
  avg_output_length_corrupt_recovery_full: 1060.0 chars

subset_summary:
  status: skipped
  reason: "sanity mode: stage1 anchors unavailable for subset construction"

state_dict_sha:
  before: 41cd117fa63b300d2dbbeb69bd223e529a6105270c69e14d85753003971c0240
  after:  41cd117fa63b300d2dbbeb69bd223e529a6105270c69e14d85753003971c0240
  stable: true (preserved spec §11.12)

label: "[SANITY] [DEBUG] — NOT a full Phase B validation; the full clean rerun is the operator's next step. The sanity smoke does NOT exercise the spec §11.4 / §11.5 / §11.6 anchor-parity hard-gates because anchors are intentionally skipped under `--sanity` per `stage1/run_phase_b.py:792-797` and the sanity-mode rule in spec §7.2.4. The Phase B SANITY CHECKS block in stdout reported all 11 checks PASS, including the three skipped-with-PASS labels: `Anchor pin verification skipped (sanity mode)`, `Subset construction (sanity-skipped, artifact present)`, `Human:-continuation parity skipped (sanity, reason: sanity mode: stage1 anchors unavailable for subset construction)`, `Phase A cross-check skipped (sanity mode, no prior anchors required)`."

## Pre-existing partial sanity attempts (debug artifacts)

  - run_20260429_085930_475110: empty (no JSONL, no summary). Kept in place per dispatch.
  - run_20260429_090039_630537: contains only `results_restoration_no_patch.jsonl` (5 samples). The previous writer agent's dispatch reported these outputs as "short outputs ending in 'Final answer: The...' with no Human: continuation explosion — strong signal that the parity fix is working". Direct re-inspection in this writer session contradicts that claim: all 5 sample outputs in this partial JSONL contain `"Human:"` substrings (positions 189, 218, 467, 224, 125 respectively), and avg output length ≈ 1645 chars — the SAME failure pattern as the failed full run `run_20260428_071043_694696` (avg ≈ 1501 for restoration_no_patch). The partial run's outputs are NOT consistent with the parity fix having succeeded. See "Open items" #2.
  - run_20260429_091148_540434: empty (no JSONL, no summary). Kept in place per dispatch.

(All three kept in place; not source of truth.)

## Reproducibility audit

seed_wired: true (seed=42 wired into `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `transformers.set_seed` per `stage1/run_phase_b.py:134-155`; preserved unchanged by this revision)
config_logged: true (`environment.runtime_provenance.config_path == "stage1/configs/stage2_confound.yaml"` in summary JSON)
run_name_required: true (timestamped — `run_20260429_094926_429332`, created by `stage1/utils/logger.py::create_run_dir`)
determinism_flags: true (`environment.deterministic_algorithms_enabled == true`, `environment.cublas_workspace_config == ":4096:8"`, `environment.determinism_warnings == []`)
wording_gate_clean: true ("[PASS] Conservative-wording gate clean" in stdout; the gate scanned the 8 base files in `_WORDING_GATE_FILENAMES_BASE` per spec §11.11; drift artifacts not in scan because `drift_diagnostic == "skipped"`)
composed_state_dict_sha_stable: true (`compose_meta.state_dict_sha256_before == compose_meta.state_dict_sha256_after == 41cd117fa63b300d2dbbeb69bd223e529a6105270c69e14d85753003971c0240`)

## Deviations from spec

  - `_emit_conditional_artifacts_skipped` was added as a writer-introduced variant of `_emit_conditional_artifacts` to satisfy operator constraint #4 (explicit failed/skipped status artifacts with `{"status": "skipped", "reason": "..."}` schema per spec §7.2.4). The spec §8.2 enumerates only 5 helpers; this 6th helper is a variant of the 5th (`_emit_conditional_artifacts`) and shares its file-write contract. Wired in 2 call sites: sanity-mode + anchors-unavailable path (line ~1495), and a P0-parity-failed path (line ~1623) — the latter is the operator-constraint-#4 contract (the writer-added defensive path that fires if the parity check would fail in full mode after the metrics layer ran).
  - The `_persist_summary` post-write rewrite to add the `[SANITY] [DEBUG]` first-line marker is a writer-added implementation of operator constraint #6. Spec §8.2 does not enumerate this edit because operator constraint #6 was added in the dispatch wrapper, not the spec body. The edit is purely additive (single new branch on `if sanity:`) and does not modify `stage1/utils/run_status.py` (do-not-touch). Documented in "Change 2" above.
  - Test-file diff stats are not cleanly subdividable per "Change 1" vs "Change 2" because the 5 new tests share a single `from ... import` block in the file. Reported as a single `+317 -0` figure under Change 2 with a verbal split.

## Dependencies added

None. The revision uses only the existing `torch`, `transformers`, `numpy`, `scipy`, and `pytest` stack present in `requirements.txt`.

## Open items (operator action)

1. Run the full clean rerun: `python -m stage1.run_phase_b --config stage1/configs/stage2_confound.yaml` — ONLY after watcher PASS and codex-reviewer (gpt-5.5 xhigh) PASS. The sanity smoke does NOT exercise the spec §11.4 / §11.5 / §11.6 anchor-parity hard-gates; the full rerun is the actual P0 acceptance check. Expected wall-clock ≤ 6 GPU-hours per spec §13.
2. **CRITICAL — possible P0 parity-fix regression / incomplete fix.** The sanity smoke's per-condition `Human:`-continuation count is 5/5 for both `clean_no_patch` and `restoration_no_patch` (spec §11.5 hard-gate threshold: `≤ 2/250`, i.e., effectively 0 in a 5-sample sanity), and the avg output length ratios are ≈ 2.15× and ≈ 2.42× the Stage 1 anchor (spec §11.6 hard-gate threshold: ≤ 1.25×). These checks are SKIPPED in sanity mode, so the run reports PASSED, but the underlying generation behavior is the SAME failure mode the spec §8.1 P0 fix was supposed to eliminate. Direct inspection of one sample shows the model emits `"The answer is 18."` then a `Human:` continuation introducing a NEW unrelated problem — i.e., the EOS stop is not firing as `model.generate` would. The 5/5 partial-run JSONL (`run_20260429_090039_630537/results_restoration_no_patch.jsonl`) shows the same pattern. The dispatch's claim that this partial run shows "no Human: continuation explosion" is CONTRADICTED by direct inspection — it shows 5/5 Human continuations at positions 189, 218, 467, 224, 125. The operator MUST treat the full clean rerun as a P0 acceptance check and inspect `phase_b_conditional_summary.json["clean_no_patch"]["output_behavior"]["human_continuation_count"]` and the ratio against `stage1_no_swap_human_count + 2` BEFORE accepting the run as valid. If the count exceeds 2/250 in the full rerun, the parity fix is insufficient and Phase B will hard-fail per spec §11.5 / §10 step 8 — operator should NOT relax the gate. Possible root causes to investigate: (a) `_greedy_continue_with_cache`'s eos_token_id is being passed but the recipient/composed model's actual EOS is different from `tokenizer.eos_token_id` (e.g., `<|im_end|>` vs `<|endoftext|>` in Qwen2.5-Instruct's chat template), (b) the `eos_token_id` argument is `None` in the no-patch path because `generation_config` doesn't propagate it, (c) the truncation contract in the docstring is documented but not enforced in the implementation. The writer did NOT modify the parity fix in this session (no spec authority to re-implement); the issue must be triaged by a follow-up writer dispatch with explicit re-implementation scope.
3. (Watcher / codex-reviewer audit) Verify the additive `_persist_summary` rewrite for the `[SANITY] [DEBUG]` marker is present and produces the correct first-non-blank line on the next run. The current run dir's existing TXT lacks the marker (it was written before the edit) — this is documented above and cannot be fixed retroactively without violating the writer.md `outputs/**` no-edit constraint.

## Open questions for watcher

1. Is the parity fix actually working? The 5/5 `Human:` count and ≈ 2.4× output-length ratio in the sanity smoke are alarming. The sanity-mode anchor-skip means this was not caught by the existing PASS/FAIL gate, but it is the same symptom the failed run `run_20260428_071043_694696` exhibited. Watcher should run a manual `python -c` probe over the new run dir's `results_*.jsonl` files and report the exact `Human:` count and output-length ratio per condition. If the watcher decides the parity fix is incomplete, the recommended next step is NOT a full clean rerun (which would burn ~6 GPU-hours to confirm a known-bad result) but rather a targeted re-implementation of `_greedy_continue_with_cache`'s EOS-stop logic with attention to (a) the actual stop-token IDs for Qwen2.5-1.5B-Instruct's chat template (likely `<|im_end|>` ID 151645 in addition to `<|endoftext|>` ID 151643), (b) propagation of `eos_token_id` from `generation_config` through `run_patched_inference_single` into `_greedy_continue_with_cache`, (c) verification via `PHASE_B_PARITY_DEBUG=1` env-guarded byte-equality assertion mentioned in spec §8.1.
2. Is the operator-constraint #6 marker rewrite acceptable as an additive `_persist_summary` post-write, or does the watcher want it implemented differently (e.g., as a request to re-spec `run_status.py` to accept a sanity-mode prefix, which would require a separate spec amendment)?
3. The writer-added `_emit_conditional_artifacts_skipped` is a 6th helper not enumerated in spec §8.2's list of 5. Watcher / codex-reviewer should confirm this is acceptable as an operator-constraint-#4-driven variant of helper 5, or flag it as a spec deviation requiring §8.2 amendment.
4. The dispatch noted the previous writer "implemented all code changes successfully but its session ended before the sanity CLI smoke completed". The 09:00 partial run JSONL inspected by this writer contradicts the dispatch's description of those outputs ("short outputs ending in 'Final answer: The...' with no Human: continuation explosion"). Watcher should verify which description is accurate by direct re-inspection of `stage1/outputs/phase_b/run_20260429_090039_630537/results_restoration_no_patch.jsonl` and reconcile.
5. The sanity smoke's `restoration_no_patch` accuracy is 0.4 and `patch_recovery_full` is 0.8 — i.e., a +0.4 delta from no_patch on a 5-sample run. Subset construction was skipped in sanity mode so no conditional metrics are available. Watcher should not interpret the 0.4 / 0.8 / 0.6 numbers as scientifically meaningful (n=5).
