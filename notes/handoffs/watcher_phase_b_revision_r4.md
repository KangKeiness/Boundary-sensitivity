# Watcher r4 Review

## Verdict

PASS_WITH_LOW_FINDINGS

## Executive summary

Writer r4 cleanly addresses the two LOW findings from Watcher r3 (test theater + reason-string canonicalisation) without changing approach (b) and without touching production scope outside the four reason-string sites the handoff enumerates. The strengthened tests now have real behavioural content (call-order recorder + counterfactual buggy-harness) layered on top of the source-inspection guard, so the gate-ordering bug is caught for the right reason. The +680-line test diff is on the heavy side for a LOW fix, but the three-layer defence is non-redundant in the way it discriminates buggy-vs-correct ordering, and the duplication is confined to the test file. r4 is safe to send to external Codex re-audit. The `+1 skip` change (260/2 vs 261/1) is genuinely explained by CUDA absence in the writer sandbox — `test_smoke_marker` is the only added skip and it is a pre-existing CUDA-gated test (`@pytest.mark.skipif(not _has_gpu_and_weights(), ...)`), not a new skip introduced by r4.

## Findings

| ID | Severity | File | Location | Issue | Why it matters | Required action |
|---|---:|---|---|---|---|---|
| W4-L1 | LOW | stage1/tests/test_phase_b_patcher.py | `_assert_run_phase_b_source_orders_precheck_before_emit` (L1575-1619) | The source-inspection guard relies on `inspect.getsource(rpb.run_phase_b)` and the literal substring `_emit_conditional_artifacts(\n`. This is brittle to (a) bytecode-only deployment that strips Python source (open-question #1 in writer r4), (b) a future stylistic refactor that puts the call args on the same line `_emit_conditional_artifacts(run_dir=...)` instead of newline-after-paren. Behavioural mini-harness mostly compensates. | Brittle on packaging or stylistic changes; would fail loudly (assert pre_idx != -1 / emit_idx != -1) rather than silently pass, so still safer than a no-op. | None required for r4 acceptance; if this guard is converted to a contract used in CI long-term, replace the `find('_emit_conditional_artifacts(\n')` substring match with an AST-based scan that finds the first `Call` whose `func.id == '_emit_conditional_artifacts'` inside `run_phase_b`'s body. Optional follow-up. |
| W4-L2 | LOW | stage1/tests/test_phase_b_patcher.py | `_build_buggy_gate_ordering_harness` (L1543-1558) | Counterfactual buggy harness wraps the buggy `_emit_conditional_artifacts` call in `try/except Exception: pass`. The handoff open-question #2 explicitly flags this as a deliberate trade-off. The recorder-order assertion is the load-bearing check, so this is acceptable, but a plain bare-except would fail the simplify rules. | Slight risk that a future `_emit_conditional_artifacts` raise silently masks an unrelated exception in the test path; load-bearing assertion is recorder ORDER which still works. | None required. If sharpened later, narrow the except to `(TypeError, KeyError, AssertionError)` or specifically the expected raise from invoking the helper with empty mappings. |
| W4-N1 | NIT | stage1/run_phase_b.py | L1615-1639 (verbose r3 wiring comment block, unchanged in r4) | r3 NIT carried forward. Writer r4 explicitly leaves it untouched per dispatch ("Do NOT fix the NIT"). | No correctness impact; documentation density is style preference. | None — accept the carry-forward. |

## Gate checklist

| Gate | Status | Notes |
|---|---|---|
| Test count regression explanation | PASS | Total collected unchanged at 262. The 261→260 passed / 1→2 skipped delta is fully explained by the writer sandbox lacking CUDA: `test_smoke_marker` (L1354-1403) is gated on `_has_gpu_and_weights()`, which is pre-existing and was passing on the r3 environment. The other skip (`test_runtime_smoke.py:253`, real-runtime present) is also pre-existing. No previously-passing non-CUDA test became skipped, no test was removed (`git diff` shows zero `-def test_` lines), and no new skip marker was added to hide a failure (the `+pytest.skip(...)` lines in the diff are all inside r1/r2 environment-fallback paths added previously). |
| Reason-string canonicalisation | PASS | After r4, the underscore form `no_patch_anchor_accuracy_parity_failed:` is the only spelling appearing as a positive assertion or production string. Verified via `Grep`: production sites are run_phase_b.py L797 (docstring), L863 (artifact reason), L2080-2087 (summary failure_reason — newly canonicalised), L2539-2551 (precedence detector — newly canonicalised). The legacy space-form appears ONLY in the two test negative assertions (test_phase_b_patcher.py L1780, L1945: `assert "no_patch anchor accuracy parity failed:" not in summary["failure_reason"]`), which is exactly the discipline-enforcing pattern. No machine-readable artifact emits the space form. |
| r3 test-theater concern resolved | PASS | The strengthened tests now have THREE distinct layers that each catch the gate-ordering bug for a different structural reason. (1) `_assert_run_phase_b_source_orders_precheck_before_emit` parses `inspect.getsource(rpb.run_phase_b)` and asserts `pre_idx < emit_idx` AND `parity_failure_reasons.append in src[pre_idx:emit_idx]` — verified directly via stdlib AST: `pre_idx=12555, emit_idx=23626, append_in_between=True` in the actual r4 source. (2) The mini-harness mirrors L1640-L1843 control flow with REAL helpers, monkeypatches `_emit_conditional_artifacts` to a raising spy, and asserts the recorder list equals `["precheck:enter", "precheck:exit", "skipped:enter", "skipped:exit"]`. (3) The counterfactual buggy harness emits-first and the test asserts `buggy_recorder.index("emit:enter") < buggy_recorder.index("precheck:enter")` AND `recorder != buggy_recorder` — proving the recorder discriminates buggy-vs-correct ordering. The load-bearing assertion is no longer the spy-not-called check; it is the recorder list-equality. |
| Test over-engineering acceptable | PASS (with LOW) | +680 lines is on the heavy side for fixing one LOW. However, the three layers are NOT redundant — each layer would individually catch a different class of regression (1: source/style refactor that re-orders; 2: actual control-flow break that bypasses the wiring; 3: a regression where the recorder itself fails to discriminate). The simpler 30-50 line alternative (single integration test with mocked anchor gate) would defend against (2) only. Given the Codex-flagged BLOCK history of this exact gate-ordering bug, the layered defence is justified; not LOW-blocking. |
| Approach (b) preserved | PASS | Verified via `git diff stage1/run_phase_b.py` hunk headers: r4 changes are confined to comment lines + four string-literal sites in the late-summary-persistence block (L2080-2087 + L2539-2551). The L1640-L1843 gate-ordering region (helper call + parity_failure_reasons.append + L1837 routing branch + L1896 computed-emit call) is bit-identical to r3. The helper definition `_evaluate_anchor_accuracy_parity_precheck` (L775-864) is unchanged except for the docstring's already-correct underscore-form mention. Source-inspection guard verifies the wiring is intact in the actual file. |
| Production scope discipline | PASS | r4 introduces zero new production helpers, zero new gate semantics, zero new conditional metric semantics, zero stop-token / no-patch decode semantics changes, and does not refactor any unrelated Phase B logic. The only production deltas are 4 string-literal canonicalisations + 2 explanatory comment lines documenting the r4 watcher LOW #2 rationale. Net `+6 -4` claim in run_phase_b.py is accurate within rounding. |
| File-touch discipline | PASS | `git status --short` shows the expected set: stage1/run_phase_b.py, stage1/tests/test_phase_b_patcher.py modified for r4 work; stage1/intervention/patcher.py + stage1/utils/wording.py modified at their r1/r2 deltas (unchanged in r4 — `git diff --stat` shows the same line counts cited in the handoff); notes/handoffs/writer_phase_b_revision_r4.md is the new handoff. No unrelated stage1/ files dirty. |
| Ready for external Codex re-audit | YES_WITH_LOW_FOLLOWUP | r4 directly addresses watcher r3 caveats[0] (test theater) and caveats[2] (reason-string canonicalisation). Caveat[1] (approach b vs c) is unchanged from r3, where it was judged as a one-time accept. Codex can be told the two carried-forward LOWs (W4-L1 brittleness of `inspect.getsource`-based source guard; W4-L2 bare `except Exception` in the counterfactual harness) are acknowledged and non-blocking. |

## Specific judgment on +680-line test diff

**Accept with LOW comments.**

The +680-line test diff IS large for a single-LOW fix, but the size is concentrated in three module-level helper functions (`_build_gate_ordering_harness`, `_build_buggy_gate_ordering_harness`, `_assert_run_phase_b_source_orders_precheck_before_emit`) and the two strengthened test bodies that consume them. Each layer addresses a structurally different failure mode the simpler 30-50 line alternative could not catch:

- A pure integration test with mocked anchors would exercise the control flow but would NOT catch a future writer who keeps the helper symbol and the call but accidentally puts `_emit_conditional_artifacts(...)` BEFORE the precheck (because the integration test still observes "skipped emit, no computed emit" if the failure path raises). The source-guard catches this.
- A pure source-guard catches structural regressions but does not exercise that the recorder/wiring actually reflects production behaviour. The mini-harness with REAL helpers catches this.
- Both source-guard and mini-harness could in principle be written such that they pass on a buggy code path nobody noticed. The counterfactual buggy harness proves discriminating power directly.

Given the Codex BLOCK history of this exact gate-ordering bug, the layered defence is proportionate. The two LOW findings (source-guard brittleness, bare except in counterfactual) are filed as W4-L1 and W4-L2 above and are not blockers.

## If simplification is required

N/A — simplification is not required. Optional future polish: convert `_assert_run_phase_b_source_orders_precheck_before_emit` to AST-based call-site detection if bytecode-only deployment becomes a concern.

## Recommendation

Proceed to external Codex re-audit.

Send the same review-target list as r3 plus the r4 handoff and the strengthened tests:
- notes/specs/phase_b_revision.md
- notes/handoffs/writer_phase_b_revision.md (r1)
- notes/handoffs/writer_phase_b_revision_r2.md
- notes/handoffs/writer_phase_b_revision_r3.md
- notes/handoffs/writer_phase_b_revision_r4.md
- notes/handoffs/watcher_phase_b_revision_r2.json
- notes/handoffs/watcher_phase_b_revision_r3.json
- notes/handoffs/watcher_phase_b_revision_r4.md (this artifact)
- stage1/run_phase_b.py
- stage1/utils/anchor_gate.py
- stage1/tests/test_phase_b_patcher.py

Areas to focus Codex on:
- Confirm acceptance of the underscore-canonical reason prefix end-to-end (no remaining space-form occurrence outside test negative assertions).
- Confirm the three-layer test defence is proportionate to the LOW concern, OR request simplification to a single integration test if Codex prefers a smaller surface.
- Confirm the carried-forward NIT (verbose r3 comment block at L1615-1639) is acceptable.
- Confirm approach (b) over (c) remains the right one-time judgment call (already accepted on r3; unchanged in r4).

## Confidence

High
