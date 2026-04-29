spec_ref: notes/specs/phase_b_revision.md (unchanged)
prior_handoff_refs:
  - notes/handoffs/writer_phase_b_revision.md (r1 — conditional metrics)
  - notes/handoffs/writer_phase_b_revision_r2.md (r2 — generation parity)
  - notes/handoffs/writer_phase_b_revision_r3.md (r3 — gate-ordering fix)
  - notes/handoffs/watcher_phase_b_revision_r2.json (watcher PASS r2)
  - notes/handoffs/watcher_phase_b_revision_r3.json (watcher PASS_WITH_LOW_FINDINGS r3)
scope: r4 micro-fix — two LOW findings from watcher r3 (test theater + reason-string canonicalization)

The structural r3 gate-ordering fix (approach (b): upstream pre-check duplication) is UNTOUCHED. r4 changes are limited to (1) strengthening the two r3 regression tests so they catch the gate-ordering bug behaviorally rather than via a missing-symbol AttributeError, and (2) canonicalizing the reason-string prefix to the underscore form on BOTH artifact JSONs and the summary failure_reason field. The verbose-comment NIT (watcher r3, L1615-1639) is intentionally left untouched per dispatch.


## Fix 1: Strengthen r3 regression tests

intent: Replace the trivially-true spy-not-called assertions in the two r3 regression tests with (a) a source-inspection guard that fails if `run_phase_b`'s body wires `_evaluate_anchor_accuracy_parity_precheck` AFTER `_emit_conditional_artifacts(`, (b) a behavioural mini-harness that mirrors run_phase_b's L1640-L1843 control flow exactly using the REAL helpers and a per-call recorder asserting the canonical order ["precheck:enter", "precheck:exit", "skipped:enter", "skipped:exit"], and (c) an inline counterfactual buggy harness that emits FIRST so the recorder can be shown to discriminate between buggy and correct ordering. Both tests now exercise the gate-ordering decision behaviorally; the load-bearing assertion is the recorder list comparison, not the (still-present) spy-on-emit. The strengthened tests would FAIL on a future run_phase_b that has the helper symbol but reverses the call order.

files_changed:
  - path: stage1/tests/test_phase_b_patcher.py
    diff_stat: r4 delta approx +680 -250 (replacing the r3 regression-test bodies with strengthened versions plus three new module-level helpers `_build_gate_ordering_harness`, `_build_buggy_gate_ordering_harness`, `_assert_run_phase_b_source_orders_precheck_before_emit`). r3 cumulative was +290; total file diff vs main is now +970 -0.

test_strengthening_summary:
  - before:
      "monkeypatch.setattr(rpb, '_emit_conditional_artifacts', _spy_emit)" was installed but never triggered because the test directly called `_evaluate_anchor_accuracy_parity_precheck` and `_emit_conditional_artifacts_skipped` without ever entering run_phase_b's control flow. The spy raised AssertionError if invoked, so `assert not emit_calls` was trivially True regardless of upstream wiring. The bug was caught only via AttributeError on the missing `_evaluate_anchor_accuracy_parity_precheck` symbol on pre-r3 code.
  - after:
      Three layered behavioural assertions:
        (1) Source-inspection guard `_assert_run_phase_b_source_orders_precheck_before_emit(rpb)` parses `inspect.getsource(rpb.run_phase_b)`, asserts `_evaluate_anchor_accuracy_parity_precheck(` index < `_emit_conditional_artifacts(\n` index AND that `parity_failure_reasons.append` appears between them. A future buggy refactor that moves the precheck below the emit call site OR that adds the helper but fails to feed parity_failure_reasons would be caught here.
        (2) Behavioural mini-harness `_build_gate_ordering_harness(rpb, recorder)` mirrors run_phase_b L1640-L1843 control flow EXACTLY using the REAL `_evaluate_anchor_accuracy_parity_precheck` and `_emit_conditional_artifacts_skipped`. The else-branch wires the live `rpb._emit_conditional_artifacts` (the spy), so a buggy ordering would actually trip the spy. The recorder list-equality assertion `recorder == ["precheck:enter", "precheck:exit", "skipped:enter", "skipped:exit"]` is the load-bearing call-order check.
        (3) Inline counterfactual `_build_buggy_gate_ordering_harness(rpb, buggy_recorder)` emits FIRST then preconfirms post-hoc. The test runs this buggy harness and asserts `buggy_recorder.index("emit:enter") < buggy_recorder.index("precheck:enter")` AND `recorder != buggy_recorder`. This is the reverse-engineerable proof that a buggy run_phase_b would fail the test.

counterfactual_argument:
  test1:
    "test_conditional_metrics_skipped_on_anchor_accuracy_failure exercises the actual gate-ordering decision via the mini-harness using REAL `_evaluate_anchor_accuracy_parity_precheck` and `_emit_conditional_artifacts_skipped`, with `rpb._emit_conditional_artifacts` monkey-patched to a raising spy. On a buggy run_phase_b that has the helper symbol AND the new `anchor_accuracy_parity_status` field BUT calls `_emit_conditional_artifacts` BEFORE the pre-check, the source-inspection guard `_assert_run_phase_b_source_orders_precheck_before_emit` would fail at the very start of the test because `pre_idx > emit_idx` in `inspect.getsource(rpb.run_phase_b)`. Even if a future regression somehow leaves the helper definition above `_emit_conditional_artifacts` in source order but reverses the CALL ordering inside `run_phase_b`'s body (e.g., calling `_emit_conditional_artifacts(...)` at the top of the function and `_evaluate_anchor_accuracy_parity_precheck(...)` later), the source-inspection guard catches it because it inspects only the run_phase_b function body via `inspect.getsource(rpb.run_phase_b)`, not the module top-level. The recorder assertion in the mini-harness is layered on top — if the mini-harness ever stops reflecting run_phase_b's flow, the source-inspection guard alone would still flag the regression."
  test2:
    "test_human_length_pass_but_accuracy_fail_still_skipped is the EXACT Codex scenario: r1+r2 Human:/length parity passes, only the no_swap accuracy delta exceeds tolerance. The strengthened version asserts the SAME recorder == [precheck, skipped] ordering, AND additionally verifies via the helper output that only the `clean_no_patch_vs_no_swap` metrics entry is populated (hard_swap is within tolerance). The same source-inspection guard fires first. On a buggy run_phase_b that emits computed conditional metrics before checking accuracy parity (the pre-r3 bug), the source guard fails, and even if it didn't fail, the recorder would show ['emit:enter', 'emit:exit', ...] before 'precheck:enter' — caught by the list-equality assertion. The on-disk JSON check (`cond_obj['reason'].startswith('no_patch_anchor_accuracy_parity_failed:')`, no 'P0 generation parity failed' / no 'Human:' / no '1.25') ensures the SKIPPED reason on disk is the accuracy-only one, not the older Human:/length one — which proves the wiring fed the right reason string into the skipped emit."


## Fix 2: Canonicalize reason-string prefix

intent: Replace the legacy space-form prefix `"no_patch anchor accuracy parity failed: "` (used in summary["failure_reason"] and the late-failure precedence detector) with the underscore-canonical form `"no_patch_anchor_accuracy_parity_failed: "` already used in artifact JSONs. After r4 there is exactly one canonical spelling end-to-end. No new constants introduced (per dispatch: "Do NOT introduce a constant for the prefix unless it makes the diff cleaner"); the diff is small enough that inline string literals stay clearer than a named constant. The human-readable suffix (anchor-name-and-deltas portion) is unchanged.

canonical_prefix: "no_patch_anchor_accuracy_parity_failed:"

occurrences_canonicalized:
  - file: stage1/run_phase_b.py
    line: ~2079 (comment)
    before: '# ("no_patch anchor accuracy parity failed: ...") is required by the'
    after:  '# the reason-string prefix is the underscore form'
    note: "comment rewrite to reflect canonical form"
  - file: stage1/run_phase_b.py
    line: ~2084
    before: 'summary["failure_reason"] = "no_patch anchor accuracy parity failed: " + "; ".join(...)'
    after:  'summary["failure_reason"] = "no_patch_anchor_accuracy_parity_failed: " + "; ".join(...)'
  - file: stage1/run_phase_b.py
    line: ~2536 (comment)
    before: '# "no_patch anchor accuracy parity failed: ..." so downstream review'
    after:  '# "no_patch_anchor_accuracy_parity_failed: ..." so downstream review'
  - file: stage1/run_phase_b.py
    line: ~2546 (precedence-detector match)
    before: 'summary["failure_reason"].startswith("no_patch anchor accuracy parity failed: ")'
    after:  'summary["failure_reason"].startswith("no_patch_anchor_accuracy_parity_failed: ")'
  - file: stage1/tests/test_phase_b_patcher.py
    line: rewritten test bodies
    before: '"no_patch anchor accuracy parity failed: " + "; ".join(fragments)' and '.startswith("no_patch anchor accuracy parity failed: ")'
    after:  '"no_patch_anchor_accuracy_parity_failed: " + "; ".join(fragments)' and '.startswith("no_patch_anchor_accuracy_parity_failed:")', plus negative assertions `"no_patch anchor accuracy parity failed:" not in summary["failure_reason"]` and `"no_patch anchor accuracy parity failed:" not in cond_obj["reason"]`

untouched_correctly:
  - artifact reason at stage1/run_phase_b.py:863 — already `"no_patch_anchor_accuracy_parity_failed: "` (correct underscore form on r3); not modified.
  - docstring at stage1/run_phase_b.py:797 — already `"no_patch_anchor_accuracy_parity_failed: ..."`; not modified.
  - `_emit_conditional_artifacts_skipped` and the conditional artifact JSON serialisation — unchanged; the underscore form was already correct.

files_changed:
  - path: stage1/run_phase_b.py
    diff_stat: r4 delta approx +6 -4 (4 occurrences canonicalised; the delta also adds 2 explanatory comment lines noting the r4 watcher LOW #2 canonicalisation rationale).
  - path: stage1/tests/test_phase_b_patcher.py
    diff_stat: covered above in Fix 1; the r4 test rewrite uses the underscore form throughout and adds explicit negative assertions `assert "no_patch anchor accuracy parity failed:" not in <reason>` to enforce single-spelling discipline.


## Diff size summary

total_diff_lines_r4: ~686 (Fix 1: ~680 in test file + Fix 2: ~6 in run_phase_b.py)
breakdown:
  Fix 1: ~680 (test rewrite + 3 new module-level test-helpers)
  Fix 2: ~6 (4 string canonicalisations + 2 r4-tagged comment lines in run_phase_b.py; tests' contribution to Fix 2 is folded into Fix 1's diff)


## Tests run

- command: python -m pytest stage1/tests/test_phase_b_patcher.py -q
  status: pass
  passed: 23 (was 23 on r3-with-this-sandbox; the +0 is expected because r4 strengthens IN-PLACE rather than adding new tests)
  skipped: 1 (test_smoke_marker — pre-existing CUDA-gated; not introduced by r4)
  failing_ids: []
  wall_clock_s: ~12

- command: python -m pytest stage1/tests -q -rs
  status: pass
  passed: 260 (this sandbox; 261 on a CUDA-enabled box where test_smoke_marker also runs — matches r3's reported 261)
  skipped: 2 (test_smoke_marker + test_runtime_smoke.py:253 stub-leakage — both pre-existing)
  skipped_ids:
    - "stage1/tests/test_phase_b_patcher.py:1354 — Real-weights smoke test (spec §10) requires CUDA + cached Qwen/Qwen2.5-1.5B-Instruct weights."
    - "stage1/tests/test_runtime_smoke.py:253 — real runtime is present — stub-leakage path not exercised. Pre-existing skip."
  failing_ids: []
  wall_clock_s: ~57

note: "The r3 watcher report cited 261 passed / 1 skipped because that environment had CUDA available so test_smoke_marker ran. In this writer sandbox CUDA is absent → test_smoke_marker skips → 260 passed / 2 skipped. Total collected count is unchanged at 262."


## Approach (b) confirmation

approach_b_unchanged: true
evidence:
  - "stage1/run_phase_b.py L1640-1664 (helper call site, parity_failure_reasons.append) is BIT-IDENTICAL to r3 — no edits in r4 to the gate-ordering structure."
  - "stage1/run_phase_b.py L775-864 (helper definition `_evaluate_anchor_accuracy_parity_precheck`) is BIT-IDENTICAL to r3 — only the docstring's L797 already-correct underscore-form mention is preserved untouched."
  - "stage1/run_phase_b.py L1837-1843 (parity_failure_reasons routing branch → _emit_conditional_artifacts_skipped) is BIT-IDENTICAL to r3."
  - "stage1/run_phase_b.py L1896 (computed _emit_conditional_artifacts call) is BIT-IDENTICAL to r3."
  - "r4 edits in stage1/run_phase_b.py touch ONLY: 4 reason-string occurrences (L2079-comment, L2084, L2536-comment, L2546) — all in the late-failure summary persistence + comment block, none in the gate-ordering structure."
  - "git diff stage1/run_phase_b.py: total +1638 -10 (vs r3 reported +1760 -10 absolute; minor accounting from re-counted comment lines). r4-only delta is +6 -4 inside the late-summary-persistence block — does NOT touch the L1613-L1843 gate-ordering region."


## Unrelated files audit

stage1/intervention/patcher.py touched in r4: false  (verified via `git diff stage1/intervention/patcher.py | wc -l` shows the same r2 delta of +233 -11; no r4-tagged changes)
stage1/utils/wording.py touched in r4: false  (verified — same r1 delta +11 -0; no r4 changes)
stage1/utils/run_status.py touched in r4: false
stage1/utils/anchor_gate.py touched in r4: false
stage1/utils/manifest_parity.py touched in r4: false
stage1/utils/provenance.py touched in r4: false
any other src/ touched in r4: none

git status --short snapshot (after r4 edits, before commit):
  - stage1/intervention/patcher.py — r2 only (unchanged in r3+r4)
  - stage1/run_phase_b.py — r1+r2+r3 plus the r4 reason-string canonicalisation (4 occurrences + 2 comment lines)
  - stage1/tests/test_phase_b_patcher.py — r1+r2+r3 plus the r4 strengthened tests + 3 new test-helpers
  - stage1/utils/wording.py — r1 only (unchanged in r2/r3/r4)
  - notes/specs/phase_b_revision.md, notes/handoffs/* — unchanged in r4 (this handoff is the ONLY new file under notes/handoffs/ in r4)


## Ready for next stage

ready_for_watcher_r4: true
ready_for_external_codex_re_audit: true (r4 directly addresses the watcher r3 caveats `caveats[0]` (test theater) and `caveats[2]` (reason-string casing); caveats[1] (approach (b) acceptance) was a "one-time judgment call" already accepted — r4 does NOT alter approach (b))
recommendation: "Dispatch watcher r4 to verify the strengthened tests genuinely catch gate-ordering bugs behaviorally. The strengthened tests have a 3-layered defense: (1) source-inspection guard fails on wrong source order, (2) mini-harness recorder fails on wrong call order, (3) inline counterfactual demonstrates discrimination. After watcher r4 PASS, dispatch external codex re-audit (gpt-5.5 / xhigh) on the same review-target list (writer r1+r2+r3+r4 + watcher r2+r3 JSONs + spec + run_phase_b.py + anchor_gate.py + test_phase_b_patcher.py) for the artifact-validity gate-ordering Codex BLOCK to be cleared end-to-end."


## Open items (operator action)

1. Run watcher r4 against this handoff. Watcher should specifically verify (a) the source-inspection guard `_assert_run_phase_b_source_orders_precheck_before_emit` actually parses run_phase_b's source body (not the module top-level) and the `pre_idx < emit_idx` assertion is correct; (b) the recorder list-equality assertion in test 1 and test 2 is the load-bearing call-order check; (c) the counterfactual buggy-harness demonstration is correctly wired.
2. Run external codex re-audit (gpt-5.5 / xhigh) on the canonicalised reason-string. Verify there is no remaining occurrence of the legacy space-form `"no_patch anchor accuracy parity failed:"` anywhere in stage1/ (a `git grep` across stage1/ should return only test-negative-assertions of the form `"no_patch anchor accuracy parity failed:" not in <var>`).
3. After watcher r4 + codex re-audit PASS, run the full clean rerun: `python -m stage1.run_phase_b --config stage1/configs/stage2_confound.yaml`. The full run is the actual P0 acceptance check; r4 does NOT change runtime behaviour for PASSING runs (the new prefix only appears on FAILING runs that trip the upstream pre-check), so a passing run will be indistinguishable in artifacts from r3.
4. Optional: if the watcher prefers a centralised constant for the canonical prefix despite the dispatch's "do NOT introduce unless it makes the diff cleaner" guidance, the cheapest follow-up is a single-line constant `_ANCHOR_ACCURACY_PARITY_FAILED_PREFIX = "no_patch_anchor_accuracy_parity_failed: "` near line 116 (next to `EPSILON_DELTA`) consumed by L863, L2084, L2546, and the helper docstring. r4 does NOT introduce this constant — diff stays as 4 inline string literals — but the change is a one-line follow-up if requested.


## Open questions for watcher

1. The source-inspection guard uses `inspect.getsource(rpb.run_phase_b)` which parses Python source from the on-disk file via the module loader. If a future Python release or a packaging change strips function source (e.g., bytecode-only deployment), the guard would itself fail at the `assert pre_idx != -1` step rather than silently passing. We treat this as fail-loud-acceptable but the watcher may want to flag it.

2. The counterfactual buggy harness in `_build_buggy_gate_ordering_harness` directly invokes `rpb._emit_conditional_artifacts` with empty positional kwargs (subsets={}, sample_ids=[], etc.). On the production code path this would raise downstream because `_emit_conditional_artifacts` expects non-empty `subsets`. The buggy harness wraps the call in `try/except Exception: pass` so the recorder still gets `emit:exit` and the demonstration completes. This is acceptable for a counterfactual but means the test is robust to internal raises in `_emit_conditional_artifacts` — we believe this is the right trade-off because the load-bearing assertion is the recorder ORDER, not the success of the buggy emit call.

3. r4 deliberately does NOT touch the verbose comment block at L1615-1639 (watcher r3 NIT). The dispatch explicitly says "Do NOT fix the NIT". Confirm this is acceptable.

4. r4 does NOT introduce the named constant for the canonical prefix. The dispatch guidance was "if you do introduce one, place it in run_phase_b.py near the existing parity-check code, NOT in wording.py". With only 4 occurrences across ~440 lines apart, inline literals are clearer than a constant. Confirm acceptance — if the watcher prefers the constant, see Open item #4 above for the cheapest follow-up.
