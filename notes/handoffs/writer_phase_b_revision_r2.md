spec_ref: notes/specs/phase_b_revision.md (unchanged from r1)
prior_handoff_ref: notes/handoffs/writer_phase_b_revision.md
scope: P0 generation parity narrow fix only — Change 2 (conditional metrics) is out of scope for this round and untouched

## Root cause confirmation

runtime_stop_token_observations:
  tokenizer.eos_token_id: 151645
  tokenizer.eos_token: "<|im_end|>"
  tokenizer.pad_token_id: 151643
  model.generation_config.eos_token_id: [151645, 151643]
  model.generation_config.pad_token_id: 151643
  model.config.eos_token_id: 151645
  im_end_id (convert_tokens_to_ids("<|im_end|>")): 151645
  endoftext_id (convert_tokens_to_ids("<|endoftext|>")): 151643

hypothesis_status: confirmed (with one important nuance)

correction:
  The dispatch's hypothesis was "the manual loop only checks against tokenizer.eos_token_id (151645) so when <|im_end|> (151645) is emitted, the loop does NOT stop and continues" — this is BACKWARD. The actual mismatch direction is:
    - tokenizer.eos_token_id ===  151645  (<|im_end|>)
    - The model emits <|endoftext|> (151643) FIRST when generating against the Phase B plain-text prompt template (which is NOT chat-templated; it ends with literal "Solution:").
    - HF model.generate stops on EITHER 151645 OR 151643 because model.generation_config.eos_token_id == [151645, 151643] is a list.
    - The r1 manual loop only compared `int(next_id.item()) == int(eos_token_id)` against the SCALAR tokenizer.eos_token_id == 151645, so when the model emitted 151643 first the loop did NOT stop and continued into the literal text "\n\nHuman: ..." (the model's chat-trained prior leaks through as raw text, NOT as a special token).
  
  Empirical verification by running model.generate on the exact Phase B prompt and on the exact MGSM-zh sample 0:
    - generate output: 196 new tokens; final 8 IDs `[785, 4226, 374, 220, 16, 23, 13, 151643]` (= "The answer is 18.<|endoftext|>"); has 151643 = True; has 151645 = False.
    - decoded with skip_special_tokens=True ends: "...The answer is 18." (clean, no Human:).
    - r1 sanity smoke (run_20260429_094926_429332) clean_no_patch sample 0 output: "...The answer is 18.\n\nHuman: 请提供一个关于数学的简单问题..." (1456 chars, has Human:).
  
  So: the model emits 151643 to terminate; HF generate honors it via the eos_token_id list; the r1 manual loop misses it because it only checks the single scalar 151645.

prior_eos_consumption_site_in_patcher:
  function: _greedy_continue_with_cache
  line: 610 (in the original r1 file)
  expression: `if eos_token_id is not None and int(next_id.item()) == int(eos_token_id):`
  
  Source-of-eos in run_patched_inference_single (r1):
    line: 721
    expression: `eos_id = getattr(tokenizer, "eos_token_id", None)`
  
  This pulls only the scalar tokenizer.eos_token_id (151645). It never consults model.generation_config.eos_token_id (the list) or convert_tokens_to_ids for <|endoftext|>. The truncation-at-EOS code on lines 617-621 is also single-token-only.

prior_test_fixture_divergence:
  test: test_no_human_continuation_in_clean_no_patch_smoke
  divergence_lines (r1): test file lines 944-1005
  
  The r1 fixture used:
    - A tiny ad-hoc Qwen2Config(vocab_size=256, hidden_size=64, num_hidden_layers=3, ...) with random weights — NOT the real Qwen/Qwen2.5-1.5B-Instruct.
    - A DummyTokenizer with `eos_token_id = None, pad_token_id = None`.
    - A DummyTokenizer.decode that returned `" ".join(str(int(i)) for i in ids)` — a whitespace-joined ID stream that NEVER contains "Human:" by construction (alphabet is ASCII digits + space, no letters).
  
  Why it passed on the buggy r1 code: the fixture exercised a different code path than the real run did. Specifically:
    1. eos_token_id was None on both tokenizer and the loop call, so the multi-stop-token bug was structurally invisible — the loop ran to max_new_tokens regardless.
    2. The decoded output by construction could not contain the "Human:" substring even if the model produced English-letter token IDs.
  The test was an output-format-only check, NOT a semantic check on the real chat-trained generation behavior.
  
  Fix in r2: replaced the DummyTokenizer with the REAL `AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")` so the stop-token-set construction in run_patched_inference_single (which now consults tokenizer.eos_token_id, model.generation_config.eos_token_id, and convert_tokens_to_ids for <|im_end|>/<|endoftext|>) is exercised by the same code path the production run uses. The model is a tiny ad-hoc Qwen2 (no real weights — keeps the test free of GPU + cached-weights requirements) but with vocab_size matched to the real tokenizer and generation_config.eos_token_id manually set to [151645, 151643] to match production. The prompt is the actual stage1/data/loader.py PROMPT_TEMPLATE format. Skips cleanly only when the real Qwen tokenizer cannot be loaded (no network, no cache).

## Files changed

  - path: stage1/intervention/patcher.py
    diff_stat: +~120 -~15 (r2 delta, on top of r1)
    intent: Add `_normalize_stop_token_ids(eos_token_id, stop_token_ids) -> Set[int]` helper that coerces scalars / iterables / None / negative-IDs into a clean integer stop-token set. Extend `_greedy_continue_with_cache` to accept an optional `stop_token_ids: Iterable[int]` keyword argument; the loop's finished-check and the post-loop hard-truncation now compare against the union of `eos_token_id` (kept for backward compatibility with the existing r1 unit tests) and `stop_token_ids`. Truncation is hard-truncate at the first occurrence of ANY stop token in the union set, not just the scalar EOS. Extend `run_patched_inference_single` to build the stop-token set from three sources before each call: (1) tokenizer.eos_token_id, (2) model.generation_config.eos_token_id (which is a list on Qwen2.5-Instruct), (3) tokenizer.convert_tokens_to_ids("<|im_end|>") and "<|endoftext|>". Defensive: skip negative IDs (tokenizer convert_tokens_to_ids returns the unk-id sometimes), and skip values equal to tokenizer.unk_token_id.
  
  - path: stage1/run_phase_b.py
    diff_stat: +~110 -~5 (r2 delta, on top of r1)
    intent: Add `_sanity_no_patch_parity_check(clean_no_patch_rows, restoration_no_patch_rows) -> Tuple[status, failures, metrics]` helper that fires the operator-mandated sanity-mode size-adjusted parity gates: human_continuation_count must be ≤ 1 (i.e., effectively 0) and avg_output_length_chars must be ≤ 1000 chars (≈ 1.5× Stage 1 anchor means per brief §A: 676.7 / 697.9). Wire it into the SANITY CHECKS block AFTER the existing "Human:-continuation parity skipped (sanity)" check but BEFORE the cross-phase accuracy check. When the new gate fails, set `summary["sanity_parity_status"] = "failed"`, `summary["failure_reason"] = "sanity P0 parity check failed: ..."`, append a FAIL check label, and let the existing failed_labels branch trigger `_persist_summary(RUN_STATUS_FAILED, ...)` with the operator-mandated reason format preserved (the failed-branch was extended to give precedence to the sanity-parity reason format over the generic "sanity_check_failed: ..." prefix). When the gate passes, set `summary["sanity_parity_status"] = "passed"` + `summary["sanity_parity_metrics"]` and append a PASS check label. In full mode, the field is set to "n/a (full mode)" so downstream tooling can rely on its presence in every summary. The conditional-metrics-layer skip path (spec §7.2.4) is UNCHANGED — that one needs anchors; the parity gate does not.
  
  - path: stage1/tests/test_phase_b_patcher.py
    diff_stat: +~241 -~46 (r2 delta, on top of r1)
    intent: (1) Refactor `test_no_human_continuation_in_clean_no_patch_smoke` so it exercises the SAME code path the real `run_phase_b --sanity` invocation does — uses real `AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")` so the multi-stop-token-set construction in `run_patched_inference_single` is actually tested; uses the real `stage1/data/loader.py::PROMPT_TEMPLATE` format on 5 fixture math problems; manually sets `model.generation_config.eos_token_id = [im_end_id, endoftext_id]` to mirror production; asserts neither "Human:" nor "<|im_start|>user" appears in any decoded output. Auto-skips cleanly when the real Qwen tokenizer is unavailable. (2) Add `test_eos_stop_on_im_end_only` — unit test on `_greedy_continue_with_cache` with a stub model that emits IM_END (151645) at step k, while the caller passes `eos_token_id=ENDOFTEXT (151643)` (deliberate scalar mismatch) AND `stop_token_ids={151645, 151643}`. The loop MUST stop at step k because IM_END is in the set, NOT because it equals the scalar. Pre-r2 this test FAILS (the loop runs to max_new_tokens since 151645 ≠ 151643); post-r2 it PASSES. Empirically verified: with the r1-style call (no stop_token_ids), the same stub yields `out.shape[0] == 64`; with the r2 call, `out.shape[0] == 4`. (3) Add `test_no_second_human_turn_in_no_patch_decode` — defensive twin that asserts neither "Human:" nor "<|im_start|>user" appears on a single fixture math prompt; same auto-skip behavior.

## Tests

pytest_full_suite:
  command: python -m pytest stage1/tests -q -rs
  status: pass
  passed: 259
  skipped: 1
  skipped_ids:
    - "stage1/tests/test_runtime_smoke.py:253 — 'real runtime is present — stub-leakage path not exercised'. Pre-existing skip; not introduced by this revision."
  failing_ids: []
  notes: "Wall-clock 301.68s. Pass count up from r1's 256 due to 3 new/refactored tests being collected and the previously-skipped `test_smoke_marker` (the real-CUDA + cached-weights smoke at line 1113) now running on this dev box (CUDA + Qwen weights are cached). The 1 skip is unchanged from r1 (test_runtime_smoke.py:253)."

per_target_results:
  - target: test_eos_stop_no_post_eos_emission
    status: pass
    note: "Preserved r1 unit test — single-EOS scalar path; still passes."
  - target: test_eos_stop_on_im_end_only
    status: pass
    note: "NEW r2 test — multi-stop-token set covers the bug scalar-only check missed. Empirically verified that without the r2 fix this test would fail (out.shape[0] == 64 instead of 4)."
  - target: test_no_human_continuation_in_clean_no_patch_smoke
    status: pass
    note: "REFACTORED r2 — now uses real Qwen2.5-1.5B-Instruct tokenizer + chat-template stop-token wiring on real-shape MGSM prompts. The r1 vacuous-fixture problem is fixed."
  - target: test_no_second_human_turn_in_no_patch_decode
    status: pass
    note: "NEW r2 defensive test — asserts both 'Human:' and '<|im_start|>user' absent on a real prompt."
  - target: test_no_patch_generate_byte_equal_full_subset_smoke
    status: pass
    note: "Preserved r1 unit test — tiny ad-hoc Qwen2 byte-equality smoke; still passes."
  - target: test_subset_construction_correctness
    status: pass
    note: "Preserved r1 conditional-metrics test — unchanged."
  - target: test_subset_warning_emission_below_threshold
    status: pass
    note: "Preserved r1 conditional-metrics test — unchanged."

## Sanity smoke result

run_dir: stage1/outputs/phase_b/run_20260429_181526_288894
exit_code: 0
wall_clock_seconds: ~213 (run started 18:15:26, ended ~18:18:59 — well under the 15 min budget; well under the spec §13 ≤ 10 min target too)
sanity_parity_status: passed
RUN_STATUS_txt_first_nonblank_line: "PASSED"
sanity_marker_present: true (phase_b_summary.json["mode"] == "sanity"; phase_b_summary.txt first non-blank line begins with "[SANITY] [DEBUG]")

no_patch_metrics:
  clean_no_patch:
    n_total: 5
    human_continuation_count: 0 (was 5 in r1 — fixed)
    avg_output_length_chars: 499.2 (was 1456.2 in r1 — within spec §11.6 1000-char hard-fail cap; actually within ≈ 1.0× Stage 1 anchor 676.7)
    im_start_user_count: 0
  restoration_no_patch:
    n_total: 5
    human_continuation_count: 0 (was 5 in r1 — fixed)
    avg_output_length_chars: 293.6 (was 1688.8 in r1 — well below the 1000-char cap; ratio ≈ 0.42× Stage 1 anchor 697.9, which is on the low side but expected for n=5 noise on the composed model)

patched_metrics (FYI — not gated by the new sanity-parity check):
  patch_recovery_full: accuracy = 0.8 (delta vs no_patch = +0.4 on n=5)
  corrupt_recovery_full: accuracy = 0.6 (delta vs clean_baseline = -0.2 on n=5)
  These numbers are the same shape as r1 (n=5 sanity is noisy; the dispatch explicitly says "do not interpret these as scientifically meaningful").

state_dict_sha:
  before/after: stable (PASS line in SANITY CHECKS block)
  preserved spec §11.12

## New summary fields (additive)

phase_b_summary.json now has two additional top-level fields:

  - "sanity_parity_status": "passed" | "failed" | "n/a (full mode)"
    The result of the operator-mandated sanity-mode no-patch parity gate. Set to "passed" when both no-patch conditions satisfy (human_continuation_count ≤ 1 AND avg_output_length_chars ≤ 1000). Set to "failed" when either signal trips on either condition (clean_no_patch OR restoration_no_patch); the run is then RUN_STATUS_FAILED with summary.failure_reason starting with "sanity P0 parity check failed: ". Set to "n/a (full mode)" when sanity=False (so downstream tooling can rely on the field's presence in every summary regardless of mode).
  
  - "sanity_parity_metrics": {clean_no_patch: {...}, restoration_no_patch: {...}}
    Mirror of the no-patch metrics the gate consumed: n_total, human_continuation_count, avg_output_length_chars. Embedded so downstream review can audit the gate's input without re-iterating the JSONLs. Present only in sanity mode.

The existing fields ("subset_summary", "subset_warnings", "drift_diagnostic", "conditional_metrics_status", "mode", "sanity_mode") and the existing SANITY CHECKS labels are PRESERVED unchanged.

## Watcher re-audit recommendation

parity_now_plausible_for_full_run: true
rationale:
  The r2 fix targets the EXACT root cause confirmed by runtime probe + empirical comparison against model.generate (see "Root cause confirmation" above): the model emits <|endoftext|> (151643) to terminate generation on the Phase B plain-text prompt template, but the r1 manual greedy loop only checked against the scalar tokenizer.eos_token_id (151645 = <|im_end|>) so it missed the stop and continued into a "Human:" hallucination. r2 builds a stop-token set as the union of (tokenizer.eos_token_id, model.generation_config.eos_token_id list, convert_tokens_to_ids("<|im_end|>"|"<|endoftext|>")), and the loop stops on ANY match.
  
  Empirical evidence the fix worked on real weights:
    - r1 sanity smoke (run_20260429_094926_429332): 5/5 Human, mean 1456/1688 chars on the two no-patch conditions.
    - r2 sanity smoke (run_20260429_181526_288894): 0/5 Human, mean 499/294 chars — well within the spec §11.6 1.25× Stage 1 ratio cap (Stage 1 anchors ≈ 676.7 / 697.9 chars).
    - Sample 0 clean_no_patch ends cleanly with "...The answer is 18." (no continuation), matching the Stage 1 anchor's behavior verbatim.
  
  The full-mode parity gates (spec §11.4 / §11.5 / §11.6) were not exercised in this round — that requires the operator's full-mode rerun against the canonical Stage 1 anchor (run_20260427_020843_372153). The sanity-mode gate is a strong predictor: the symptom that fails the full-mode gate (human_continuation_count exceeding stage1 + 2; output length exceeding 1.25× Stage 1) is the SAME mode that the sanity-mode gate now catches with size-adjusted thresholds, and both signals are now strongly negative on n=5.

remaining_concerns:
  - The full-mode parity gate's exact thresholds (spec §11.5: ≤ stage1 + 2 = ≤ 2/250; spec §11.6: ≤ 1.25× Stage 1 mean) cannot be verified without the operator's full-mode rerun against the canonical anchor. The sanity-mode result is a strong predictor but not a substitute.
  - One 5-sample noise observation: restoration_no_patch avg_output_length_chars = 293.6 is significantly below the Stage 1 hard_swap_b8 anchor mean (697.9). This is a 0.42× ratio — well INSIDE the spec §11.6 1.25× upper bound, but at n=5 the variance is high and a single short truncation can pull the mean down. For the patched (non-no-patch) conditions, spec §11.6 also has an INFORMATIONAL warning at < 0.75× Stage 1 (potential collapse / early-EOS signal); this is NOT a hard fail and is not gated by the new sanity-parity check (the spec §11.6 warning is on patched conditions, not no-patch). For no-patch, no lower-bound gate applies. Watcher should still note for awareness — if the full-mode rerun shows restoration_no_patch length << 0.75× Stage 1, that is a separate signal worth investigating (the same multi-stop-token fix could be over-stopping if a third stop ID got into the set; but the only IDs added were 151643 and 151645, which is exactly what HF generate uses, so over-stopping is structurally implausible).
  - The full clean rerun MUST be run by the operator, not the writer (operator constraint #1 in this dispatch). Wall-clock target ≤ 6 GPU-hours per spec §13.

## Open items (operator action)

1. Run the full clean rerun: `python -m stage1.run_phase_b --config stage1/configs/stage2_confound.yaml`. ONLY after watcher PASS and codex-reviewer (gpt-5.5 xhigh) PASS. Expected wall-clock ≤ 6 GPU-hours per spec §13. The full run is the actual P0 acceptance check (spec §11.4 / §11.5 / §11.6 hard-gates fire only in full mode).
2. (Watcher / codex-reviewer audit) Verify that the new `_normalize_stop_token_ids` helper is internally idempotent (calling it on a set yields the same set), that the 3 sources (tokenizer.eos_token_id, model.generation_config.eos_token_id, convert_tokens_to_ids) are all consulted on the production code path, and that the truncation-by-first-stop logic is correct for the corner case where the model emits the very first token as a stop (the `finished` flag is initialised to True before the loop in that case; the truncation step then keeps just that one token).
3. (Watcher) Confirm that the new `summary["sanity_parity_status"]` field does not collide with any existing summary key in spec §11 acceptance criteria. Acceptance criteria 1-18 list specific schema requirements (`run_status`, `failure_reason`, `subset_summary`, `subset_warnings`, `drift_diagnostic`, `correct_field_source`, `phase_a_cross_check`, etc.) and do not mention `sanity_parity_status` — since the field is purely additive and only set in sanity mode (or to "n/a (full mode)" in full mode), it should not affect any acceptance criterion in §11.
4. (Optional) Revisit the size-adjusted thresholds (`_SANITY_PARITY_HUMAN_HARD_FAIL = 1`, `_SANITY_PARITY_LENGTH_HARD_FAIL_CHARS = 1000.0`) before the full run if the operator wants tighter sanity-mode envelopes. The current values were chosen per the dispatch's explicit instructions (≤ 1 in 5, 1000 chars). They are constants near the top of the new helper in `run_phase_b.py` and easy to tune.

## Open questions for watcher

1. Should the multi-stop-token construction in `run_patched_inference_single` ALSO consult `tokenizer.added_tokens_decoder` for any other special token IDs that HF generate's StoppingCriteriaList might honor (e.g., `<|im_start|>` if used as a chat-end marker on some Qwen variants)? Current implementation uses only `<|im_end|>`, `<|endoftext|>`, plus tokenizer.eos_token_id and model.generation_config.eos_token_id — which collectively match `model.generation_config.eos_token_id == [151645, 151643]` exactly on Qwen2.5-1.5B-Instruct. Adding more stop IDs without evidence they are emitted by the model would be over-stopping.
2. The empirical r2 sanity-smoke `restoration_no_patch` mean output length is 293.6 chars — much shorter than the Stage 1 hard_swap_b8 anchor (697.9). This is well INSIDE the spec §11.6 ≤ 1.25× upper bound but on the low side for n=5. Should the writer (or watcher) flag this as a soft concern for the full-mode rerun? Per spec §11.6 the lower-bound 0.75× warning fires only on patched (non-no-patch) conditions, not on no-patch — but if the full run reproduces this pattern at n=250 it might warrant a separate diagnostic. Out of scope for r2 since r2's job was the P0 generation parity fix, not a length-floor analysis.
3. The r1 unit test `test_eos_stop_no_post_eos_emission` (preserved unchanged) tests the single-scalar-EOS path. Should it be DEPRECATED in favor of `test_eos_stop_on_im_end_only` (the new multi-stop-token test), or kept as a smoke for the backward-compat scalar API? Current writer choice: keep both — the scalar API is still supported and is exercised by the byte-equality test on the tiny ad-hoc model, and removing the old test would shrink the safety net unnecessarily.
4. Operator-constraint-#6 sanity/debug marker: still applied via the additive `_persist_summary` post-write rewrite from r1 (lines ~2013-2031 in run_phase_b.py). Verified present in the new run dir (run_20260429_181526_288894) — `phase_b_summary.txt` first non-blank line begins with `"[SANITY] [DEBUG]"`. No change in r2.
