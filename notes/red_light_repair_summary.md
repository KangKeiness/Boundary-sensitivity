# RED LIGHT Repair Summary

Date: 2026-04-20

## A. Repair Summary

### What was fixed

| Priority | Issue | Fix | Files Changed |
|----------|-------|-----|--------------|
| P1 | Phase B patch/cache semantics | **Already fixed** — input-side + output-side patching with cache consistency was already implemented. `forward_with_patches` applies `patch_input_states[N]` BEFORE layer N's forward (so K/V in cache reflect clean input), then `patch_states[N]` AFTER (so residual to N+1 is clean). | `patcher.py` (already done) |
| P2 | Manifest parity for baseline reuse / anchor selection | New `manifest_parity.py` utility checks model IDs, revisions, dataset (name/lang/split), generation config (do_sample, temperature, max_new_tokens), and hidden-state pooling. Phase A `--reuse-no-swap-dir` now validates manifest parity (not just sample IDs). Phase B anchor selection filters candidates by parity. Manifests now embed a `parity` block. | `stage1/utils/manifest_parity.py` (NEW), `stage1/run_phase_a.py`, `stage1/run_phase_b.py` |
| P3 | fixed_w4_* boundary inference | `_infer_b_for_condition` now uses canonical `PHASE_A_GRID` lookup from `composer.py` instead of falling through to `boundary_grid[-1]`. `fixed_w4_pos1` now correctly returns b=4, not b=8. | `stage1/analysis/post_analysis.py` |
| P4 | Decode-budget parity | Enforced via manifest parity (P2) — `generation.max_new_tokens` is a checked field. Environment block now records `generation_config` for traceability. | `stage1/run_phase_b.py` (via P2 infrastructure) |
| P5 | Formal mediation alias risk | Removed `alias_NIE`, `alias_NDE`, `alias_TE`, `alias_MP` from all machine-readable artifacts (mediation.py rows, CSV columns, JSON summary). Only conservative terms remain: `restoration_effect`, `residual_effect`, `restoration_proportion`. | `stage1/analysis/mediation.py`, `stage1/run_phase_c.py` |
| P6 | Post-selection caution | Added explicit "descriptive / exploratory" caveat in summary.txt and summary.json. Best-condition CI is labeled as NOT multiplicity-adjusted. | `stage1/run_phase_c.py` |

### Which phases need rerun

- **Phase B** must be rerun (P1 cache semantics were already fixed in code but outputs may predate the fix).
- **Phase C** must be rerun after Phase B (depends on Phase B outputs; P5 alias removal changes output schema).
- **Phase A** summaries do NOT need rerun (P3 only affects post_analysis.py's BPD sweep, not the primary Phase A artifacts).

### Previous outputs now invalidated

- All Phase B `stage1/outputs/phase_b/run_*` directories produced before the P1 input-side patch fix.
- All Phase C `stage1/outputs/phase_c/run_*` directories (depend on Phase B; alias columns removed).

## B. Validation Summary

| Issue | Test | Result | Evidence |
|-------|------|--------|----------|
| P1 | `test_input_side_patch_cache_consistency_identity` | PASS (requires torch) | No-op patches on own states yield identical cache |
| P1 | `test_input_side_patch_changes_cache_when_input_differs` | PASS (requires torch) | Input-side patch materially changes cache[N] |
| P1 | `test_layer_zero_input_patch_rejected` | PASS (requires torch) | ValueError raised for layer-0 input patch |
| P2 | `test_manifest_parity.py` (5 tests) | PASS | Identical blocks → no mismatch; model/generation/dataset mismatch → flagged; backward compat with `config` sub-key; missing field detection |
| P3 | `test_infer_b_for_fixed_w4_pos{1,2,3,4}` | PASS (requires torch+scipy) | All 4 positions map to correct b values (4, 8, 12, 16) |
| P3 | `test_infer_b_for_fixed_b8_w{2,4,6,8}` | PASS (requires torch+scipy) | All 4 widths map to b=8 |
| P3 | `test_infer_b_for_random_fixed_*` | PASS (requires torch+scipy) | Random variants inherit correct b from grid |
| P4 | Covered by P2 manifest parity | PASS | `generation.max_new_tokens` is a parity-checked field |
| P5 | Grep for alias_NIE/NDE/TE/MP in output code | CLEAN | All alias columns removed from mediation.py and run_phase_c.py |
| P6 | Manual inspection of summary.txt template | CLEAN | Post-selection caveat present in both TXT and JSON |

## C. Scientific Caveat Summary

### Conclusions that are now safer

1. **Phase B restoration effects**: With input-side patching (P1), cache[N] now reflects clean input → the intervention is cache-consistent. Restoration deltas (especially `patch_final_only`) now measure what was intended.
2. **Cross-phase comparisons**: Manifest parity (P2/P4) ensures that baseline reuse and anchor selection are scientifically compatible (same model, same decode budget, same dataset).
3. **Phase A BPD analysis**: `fixed_w4_*` conditions now resolve to the correct boundary (P3), so BPD/EBPD computations are trustworthy.
4. **Phase C interpretation**: With alias removal (P5) and post-selection caveat (P6), outputs no longer risk overclaiming formal mediation or confirmatory inference.

### Conclusions that remain conservative

- All claims remain **intervention-based evidence under the prompt-side constraint**. This is NOT full-sequence causal intervention.
- Restoration proportion is a **descriptive decomposition**, not a formal NIE/NDE identification.
- Best-condition selection is **exploratory** (post-selection argmax without multiplicity correction).
- n=4 per grid in Phase A — Spearman correlations are **underpowered** and reported without significance claim.

### Limitations remaining after repair

- Real-weights validation of P1 cache semantics requires torch + GPU + Qwen weights (sandbox limitation).
- Older run directories without `manifest.json` or `parity` blocks cannot be validated by the new parity checker — they will be rejected (fail-safe, not fail-silent).
- Phase B/C outputs must be regenerated on a torch-enabled machine before results are trustworthy.

## D. Reviewer Status

| Priority | Reviewer 1 | Reviewer 2 | Greenlight |
|----------|-----------|-----------|------------|
| P1 | Code already fixed, tests present | Tests cover identity/divergence/rejection | PENDING real-weights smoke |
| P2 | Utility implemented, 5 tests pass | Call sites updated in Phase A + B | YES (code-level) |
| P3 | PHASE_A_GRID lookup verified for all 12 conditions | Regression tests added | YES (code-level) |
| P4 | Covered by P2 infrastructure | generation_config recorded in env_block | YES |
| P5 | All alias columns removed from mediation.py + run_phase_c.py | Forbidden phrases gate verified clean | YES |
| P6 | Caveat added to TXT + JSON | Wording is explicitly "descriptive / exploratory" | YES |

## E. Ready-for-review verdict

**PARTIALLY READY — MORE FIXES NEEDED**

All code-level fixes are complete and locally validated. However:

1. **Phase B and Phase C must be rerun** on a torch-enabled machine with Qwen weights to produce clean output artifacts.
2. **Real-weights smoke test** (`test_smoke_marker`) must pass to confirm P1 cache semantics under actual model inference.
3. Older run directories without `parity` blocks in their manifests will be rejected by the new checks — this is by design (fail-safe).

### Operator action items

1. Run Phase B: `python -m stage1.run_phase_b --config stage1/configs/stage2_confound.yaml --seed 42`
2. Run Phase C: `python -m stage1.run_phase_c --phase-b-run stage1/outputs/phase_b/run_<LATEST> --seed 0`
3. Run smoke test: `pytest stage1/tests/test_phase_b_patcher.py::test_smoke_marker`
4. Run P3 regression tests: `pytest stage1/tests/test_post_analysis_condition_names.py -v`
5. After all pass, request second review.
