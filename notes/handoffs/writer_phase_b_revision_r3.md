spec_ref: notes/specs/phase_b_revision.md (unchanged)
prior_handoff_refs:
  - notes/handoffs/writer_phase_b_revision.md (r1 — conditional metrics)
  - notes/handoffs/writer_phase_b_revision_r2.md (r2 — generation parity)
  - notes/handoffs/watcher_phase_b_revision_r2.json (watcher PASS r2)
scope: artifact-validity gate-ordering fix only (Codex BLOCK finding)

## Codex finding addressed

verbatim: "stage1/run_phase_b.py claims P0 parity includes accuracy, Human continuation, and output length. But before conditional artifact emission, parity_failure_reasons only includes Human/length/pin failures. If evaluate_phase_b_anchor_gate returns passed=False due to accuracy mismatch, conditional metrics can still be computed and emitted. The cross-check failure is applied later, after conditional artifacts have already been written. This violates the rule: conditional metrics must be skipped/invalid if no-patch parity fails."

fix_summary: Added a small upstream pre-check (`_evaluate_anchor_accuracy_parity_precheck`) inside `stage1/run_phase_b.py` that runs immediately after the existing `evaluate_phase_b_anchor_gate(...)` call and BEFORE conditional artifact emission. The pre-check inspects `gate.failed_anchors` (which is non-empty only when both anchors are present and at least one accuracy delta exceeds `PHASE_A_CROSS_CHECK_TOL`) and translates the gate's anchor-name strings into the spec-mandated `"<phase_b_condition> vs <stage1_anchor> delta=... > tol=..."` fragments + the operator-mandated `"no_patch_anchor_accuracy_parity_failed: ..."` reason string. The reason is appended to `parity_failure_reasons`; the existing branch at line ~1765 (`if parity_failure_reasons:`) then routes to `_emit_conditional_artifacts_skipped` BEFORE any computed numbers are written. Two new top-level summary fields (`anchor_accuracy_parity_status`, `anchor_accuracy_parity_metrics`) record the gate result. The late cross-check FAIL label still fires (full mode → triggers `_persist_summary(RUN_STATUS_FAILED, ...)`); a new precedence rule in that block prefers the `"no_patch anchor accuracy parity failed: ..."` reason (set on `summary["failure_reason"]` upstream) over the generic `"sanity_check_failed: ..."` prefix. `stage1/utils/anchor_gate.py`, `stage1/intervention/patcher.py`, and `wording.py` are untouched — the fix is entirely in `run_phase_b.py` plus the test file.

## Files changed

  - path: stage1/run_phase_b.py
    diff_stat: +~187 -~10 (r3 delta, on top of r1+r2 — total file diff vs main is +1760 -10)
    intent: (a) Add `_evaluate_anchor_accuracy_parity_precheck(*, cross_check_failed_anchors, no_patch_acc, clean_baseline_acc, anchor_hard_swap_acc, anchor_no_swap_acc, tolerance) -> Tuple[status, metrics, fragments, reason]` immediately above `_emit_conditional_artifacts`. The helper is pure-Python and has no torch/transformers dependency, mirroring the rest of the conditional-metrics helpers. (b) Wire the helper at line ~1640, immediately after the existing `parity_failure_reasons: List[str] = []` declaration and BEFORE `_emit_conditional_artifacts` is called (line ~1843). When the helper returns `status="failed"`, its reason string is appended to `parity_failure_reasons`, which routes through the existing skipped-emit branch at line ~1768. (c) Add two new top-level summary keys `anchor_accuracy_parity_status` ("failed" | "n/a (full mode)" | "n/a (sanity mode)") and `anchor_accuracy_parity_metrics` (per-pair accuracy + delta + tolerance dict) to the `summary` block. (d) Pre-populate `summary["failure_reason"]` with the operator-mandated `"no_patch anchor accuracy parity failed: <fragments>"` string when the pre-check fired, so the eventual `_persist_summary(RUN_STATUS_FAILED, ...)` call preserves it. (e) Extend the late-failure-branch precedence at line ~2446 with a NEW first branch that prefers the new reason format over the generic `"sanity_check_failed: ..."` prefix; the existing r2 sanity-parity precedence is preserved as the second branch.

  - path: stage1/tests/test_phase_b_patcher.py
    diff_stat: +~290 -0 (r3 delta — adds 2 new tests + a small `_import_run_phase_b_module` helper)
    intent: Two regression tests that fail cleanly on pre-r3 code (AttributeError: module 'stage1.run_phase_b' has no attribute '_evaluate_anchor_accuracy_parity_precheck') and pass on the fixed code. Both tests stub `_emit_conditional_artifacts` with a fail-loudly spy (`raise AssertionError(...)` if called) and verify (1) the helper returns the correct status / reason / metrics / fragments shape; (2) the on-disk `phase_b_subsets.json` and `phase_b_conditional_summary.json` reflect the explicit-skipped schema with `{"status": "skipped", "reason": "no_patch_anchor_accuracy_parity_failed: ..."}`; (3) the sidecar `*.SKIPPED.txt` files exist; (4) the would-be `summary["failure_reason"]` starts with `"no_patch anchor accuracy parity failed: "`. Test 2 is the EXACT Codex scenario: Human:/length parity passes (assert via the existing `_count_human_continuation` and `_mean_output_length_chars` helpers), but the accuracy delta on `no_swap` exceeds tolerance — the skipped path must still fire and the on-disk reason must NOT be the older Human:/length one. Tests use `monkeypatch` consistent with the existing test style; pure-Python (no torch / no GPU) so they run in <7s combined.

## Gate-ordering fix

chosen_approach: b
  (a = move late cross-check to pre-emit position
   b = duplicate cross-check logic in upstream pre-check
   c = reorder statements)

why: Approach (b) is the smallest, most defensive edit. The late cross-check FAIL label at line ~2415-2425 must remain to drive `_persist_summary(RUN_STATUS_FAILED, ...)` — moving it upstream (approach a) would require also relocating the entire failed-checks loop and the `render_anchor_gate_diagnostic` call, which would touch ~50 lines and create a footgun for future writers. Approach (c) (reorder statements) is incompatible with the architecture because `_emit_conditional_artifacts` is logically nested inside the anchors-loadable branch (line ~1681) which depends on both anchor JSONLs being readable; moving it AFTER the late cross-check block would require duplicating the data-loading and the `try/except`, and would introduce an unguarded read of `subsets`/`per_condition_metrics` later. Approach (b) keeps both checks in their natural positions: the new pre-check populates `parity_failure_reasons` upstream (and the existing branch routes to `_emit_conditional_artifacts_skipped`), while the late cross-check still drives the RUN_STATUS=FAILED flow. The two checks consume the same `gate.failed_anchors` source so there is no semantic divergence — the late check is an output-status sink, the upstream pre-check is an artifact-emission gate.

new_summary_fields:
  - phase_b_summary.json["anchor_accuracy_parity_status"] — "failed" | "n/a (full mode)" | "n/a (sanity mode)". Set unconditionally on every run so downstream tooling can rely on field presence.
  - phase_b_summary.json["anchor_accuracy_parity_metrics"] — dict keyed by pair name (`"restoration_no_patch_vs_hard_swap_b8"`, `"clean_no_patch_vs_no_swap"`); each value carries `no_patch_accuracy` (or `clean_baseline_accuracy`), `anchor_accuracy`, `delta`, `tolerance`. Empty dict when no failure fired.

reason_string_format:
  example_single: "no_patch_anchor_accuracy_parity_failed: clean_no_patch vs no_swap delta=0.0240 > tol=0.0080"
  example_both: "no_patch_anchor_accuracy_parity_failed: restoration_no_patch vs hard_swap_b8 delta=0.0240 > tol=0.0080; clean_no_patch vs no_swap delta=0.0120 > tol=0.0080"

(Both formats verified against the spec's example by direct `python -c` invocation of the helper before the test was written; see "Confirmation" section below.)

## Tests

pytest_full_suite:
  command: python -m pytest stage1/tests -q -rs
  status: pass
  passed: 261 (was 259 pre-r3; expected 261 post-r3 — matches)
  skipped: 1 (unchanged from r2 — test_runtime_smoke.py:253)
  skipped_ids:
    - "stage1/tests/test_runtime_smoke.py:253 — 'real runtime is present — stub-leakage path not exercised'. Pre-existing skip; not introduced by this revision."
  failing_ids: []
  notes: "Wall-clock 225.97s. Pass count up exactly +2 from r2's 259."

per_target_results:
  - target: test_conditional_metrics_skipped_on_anchor_accuracy_failure
    status: pass
    note: "Verified to FAIL on pre-r3 code via `git stash push -- stage1/run_phase_b.py` then re-running pytest: AttributeError: module 'stage1.run_phase_b' has no attribute '_evaluate_anchor_accuracy_parity_precheck'. After `git stash pop` the test passes. Wall-clock 6.5s for the pair, ~3s individually."
  - target: test_human_length_pass_but_accuracy_fail_still_skipped
    status: pass
    note: "Same pre-r3 failure mode as test 1 (AttributeError on the missing helper symbol). Asserts the EXACT Codex scenario: r1+r2 parity helpers (_count_human_continuation, _mean_output_length_chars) PASS on the input by construction, but the no_swap accuracy delta still trips the new pre-check. The on-disk reason is verified to start with `no_patch_anchor_accuracy_parity_failed:` and NOT contain the older `'P0 generation parity failed'` / `'Human:'` / `'1.25'` substrings."

## Confirmation that cross_check_passed=False blocks computed conditional artifacts

evidence: The new pre-check is at `stage1/run_phase_b.py:1640-1664` (lines `(_aa_status, anchor_accuracy_parity_metrics, anchor_accuracy_parity_failed_msgs, _aa_reason) = _evaluate_anchor_accuracy_parity_precheck(...)` followed by `if _aa_status == "failed": ... parity_failure_reasons.append(_aa_reason)`). The existing skipped-routing branch is at line ~1768 (`if parity_failure_reasons:` → `subsets_summary = _emit_conditional_artifacts_skipped(...)`), so a non-empty `parity_failure_reasons` from the new pre-check structurally prevents `_emit_conditional_artifacts` (line ~1843) from ever being called. The helper itself is at `stage1/run_phase_b.py:775-863`. Both new tests exercise this path directly — `test_conditional_metrics_skipped_on_anchor_accuracy_failure` does so with both anchors flagged in `cross_check_failed_anchors`; `test_human_length_pass_but_accuracy_fail_still_skipped` does so with no_swap-only flagged AND r1+r2 Human:/length parity helpers verified to return PASS shapes on the input. Both tests install a `monkeypatch.setattr(rpb, "_emit_conditional_artifacts", _spy_emit)` where `_spy_emit` raises `AssertionError(...)` if called — the tests pass, so the spy is NEVER invoked, proving the gate-ordering fix is structurally sound.

## Sanity smoke (optional, not run)

run_dir: n/a — sanity smoke is full-mode-specific to the bug; see dispatch step 8 ("Sanity smoke is OPTIONAL this round — run it ONLY if you want to confirm no regression in the sanity path. The fix is full-mode-specific so sanity smoke does not directly exercise it.")
exit_code: n/a
note: "Skipped for two reasons: (1) the gate-ordering fix is full-mode-specific because `gate.failed_anchors` is populated only when both anchors are PRESENT and outside tolerance — sanity mode typically lacks anchors entirely (per spec §7.2.4 the existing sanity skipped path handles that case). (2) The pure-Python regression tests already exercise both modes of the fix at unit-test granularity, and the full pytest suite (which includes the r2 sanity-related smokes for stop-token handling and Human:-continuation absence) passes 261/261. The sanity-mode codepath was inspected by reading the `_aa_status` else-branch (line ~1672: `anchor_accuracy_parity_status = 'n/a (sanity mode)'`) and confirmed to be a no-op when `cross_check_failed_anchors` is empty — which is the typical sanity case."

## Watcher / Codex re-audit recommendation

ready_for_re_audit: true

remaining_concerns:
  - The fix is structurally tied to `gate.failed_anchors` being populated correctly by `evaluate_phase_b_anchor_gate`. That contract is unchanged from r2 and was already verified by the watcher r2 PASS. We do NOT modify `anchor_gate.py`.
  - The two new summary keys (`anchor_accuracy_parity_status`, `anchor_accuracy_parity_metrics`) are additive and do not collide with any existing key. They appear unconditionally so downstream review can rely on field presence even on PASS runs (where status is `"n/a (full mode)"` or `"n/a (sanity mode)"`).
  - The full-mode parity gate's exact thresholds (spec §11.4 / §11.5 / §11.6) cannot be verified without the operator's full-mode rerun against the canonical Stage 1 anchor (run_20260427_020843_372153). The r3 fix specifically addresses §11.4 (anchor-accuracy parity) gate-ordering — §11.5 (Human:) and §11.6 (length) gate-ordering was already correct in r1/r2.
  - No do-not-touch list violation: `git diff --stat` confirms only `stage1/intervention/patcher.py`, `stage1/run_phase_b.py`, `stage1/tests/test_phase_b_patcher.py`, `stage1/utils/wording.py` are modified in stage1/. Of those, only `run_phase_b.py` and `test_phase_b_patcher.py` were touched in r3 (patcher.py and wording.py are r1+r2 only).

## Open items (operator action)

1. Run the full clean rerun: `python -m stage1.run_phase_b --config stage1/configs/stage2_confound.yaml`. ONLY after watcher PASS and codex-reviewer (gpt-5.5 xhigh) PASS. The full run is the actual P0 acceptance check; the new gate-ordering fix only matters when the full-mode anchor cross-check actually fails on accuracy (which it should NOT under the r2 generation-parity fix, but the gate-ordering fix is the safety net Codex required).
2. (Watcher / codex-reviewer audit) Verify that the new helper `_evaluate_anchor_accuracy_parity_precheck` is correctly called BEFORE `_emit_conditional_artifacts` and that the `gate.failed_anchors` strings are correctly translated to the spec-mandated `"<condition> vs <anchor> delta=... > tol=..."` format. The translation logic is at line ~824-855 in run_phase_b.py (lines starting with `if anchor_hard_swap_acc is not None and any(...)` and `if anchor_no_swap_acc is not None and any(...)`).
3. (Watcher) Confirm that `summary["anchor_accuracy_parity_status"]` does not collide with any existing summary schema key. Acceptance criteria 1-18 in spec §11 do not mention this field — purely additive, set to `"n/a (...)"` in non-failing modes.
4. (Optional) Inspect the precedence at line ~2446 where the late `_persist_summary(RUN_STATUS_FAILED, ...)` reason-string selection picks the new `"no_patch anchor accuracy parity failed: "` prefix over the generic `"sanity_check_failed: "` prefix. The new branch is checked FIRST so the most-specific signal (anchor-accuracy gate-ordering) wins; the r2 sanity-parity precedence is preserved as the second branch.

## Open questions for watcher

1. The fix uses `summary["failure_reason"] = "no_patch anchor accuracy parity failed: ..."` (with spaces, not underscores) for the human-readable reason on RUN_STATUS_FAILED, while the conditional-artifact `reason` field uses `"no_patch_anchor_accuracy_parity_failed: ..."` (with underscores) per the dispatch's example format. The space/underscore split is intentional — the underscore form is the dispatch-mandated parser-friendly format used inside artifact JSONs (consumed programmatically), while the space form is the human-readable summary `failure_reason` (consumed by an operator skimming the TXT). If the watcher prefers a single canonical format on both, the cheapest change is to align both to underscores; the test assertions check both forms separately.

2. The new pre-check is structurally a "soft duplicate" of the late cross-check failure handling (which still fires too, producing the FAIL label). The late check is intentionally NOT removed — it remains the canonical RUN_STATUS=FAILED driver. If the watcher wants the late check moved to be the primary site (approach a in the gate-ordering fix discussion), that is a 50-line refactor and beyond the narrow r3 scope; we recommend keeping the current approach (b).

3. The two regression tests are pure-Python and skip-clean when `stage1.run_phase_b` cannot be imported. They do NOT require torch/transformers/CUDA. The `test_smoke_marker` real-weights smoke (line 1354) was not re-run for r3 because the gate-ordering fix is pure control flow on Python data structures.
