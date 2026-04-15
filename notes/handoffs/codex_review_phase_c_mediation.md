# Codex Normal Review — Phase C Mediation

**Fallback disclosure:** `/codex:review` (marketplace plugin) is unavailable
in this environment. Per the agent definition's fallback convention, this
artifact is a self-review produced by the `codex-reviewer` wrapper acting as
Codex stand-in. All findings below are strictly read-only observations
against the spec, writer handoff, and watcher JSON listed in STAGE inputs.
No source files were edited. Triage labels follow the agent-definition
schema (`agree-block | agree-nit | disagree-with-reason | already-known`).

Pass: 1 / 2 (normal review)
Slug: `phase_c_mediation`
Watcher precondition: **PASS** (Round 2, verdict=`PASS`, 0 must-fix,
2 can-ship-with-followup items) — confirmed.

---

## Verbatim review (self-review stand-in)

### Spec conformance — matched

1. **Paired bootstrap correctness** (spec §6, §7): `stage1/analysis/mediation.py::_paired_bootstrap`
   uses `np.random.default_rng(seed)` with `seed=0` default, draws
   `(n_resamples, n)` integer index matrix once, and applies the same index
   row to every array (paired-across-arrays by construction at lines 205-
   210). `n_resamples=1000` and `ci=0.95` are default; percentile
   computation at lines 220-222 uses `np.quantile` on float64 with
   `alpha=(1-ci)/2`. This matches spec §7 "2.5th / 97.5th percentile of
   resampled values."
2. **`sample_id` pairing is intersection, not positional** (spec §6, §7, H1):
   `align_by_sample_id` at lines 122-161 uses `set.intersection(*id_sets)`
   then `sorted(common)` — sorted intersection exactly. Positional
   fallback is absent. Drop logging uses the substring `"dropped"` at line
   150, satisfying the caplog assertion in test §10.3.
3. **Alphabetical tie-break on claim-eligible set** (spec §7, §11.8):
   `compute_decomposition_table` at lines 495-508 builds
   `eligible = [(name, effect), ...]` filtered by
   `CLAIM_ELIGIBLE_CONDITIONS` and sorts by `(-effect, name)` so ties
   collapse to ASCII-alphabetical. `CLAIM_ELIGIBLE_CONDITIONS` at lines
   29-34 is exactly the spec §7 four-tuple (`patch_boundary_local`,
   `patch_recovery_early`, `patch_recovery_full`, `patch_final_only`);
   `patch_all_downstream` is correctly excluded from claim-eligibility.
4. **Mandated caveat byte-exact** (spec scope-note, §11.3, §11.4):
   `MANDATED_CAVEAT` at `stage1/run_phase_c.py:50-53` is a two-line
   concatenation that renders exactly:
   `"Mediation analysis here decomposes accuracy deltas under prompt-side "`
   `"restoration intervention only. It is not a formal NIE/NDE decomposition."`
   — byte-equal to the spec literal.
5. **TXT header byte-equal** (spec §11.10): `PHASE_C_TXT_HEADER` at
   `stage1/run_phase_c.py:56-59` uses `\u2014` (U+2014 em dash), matching
   watcher's byte-level verification. Round-2 fix is present.
6. **FORBIDDEN_PHRASES gate** (spec §8): `FORBIDDEN_PHRASES_PHASE_B` at
   `stage1/utils/wording.py:30-40` matches spec §8 exactly. `FORBIDDEN_
   PHRASES_PHASE_C` at :43-59 omits the two literal tokens `"nie/nde"` and
   `"nde/nie"` with a documented rationale (caveat self-collision under
   case-insensitive rule). Backward-compat alias at :63 preserves
   `FORBIDDEN_PHRASES == FORBIDDEN_PHRASES_PHASE_B`, so
   `test_forbidden_phrases_gate` (Phase B) still passes bytewise (writer
   reports 3 passed).
7. **`check_artifacts_for_forbidden` phrases= kwarg** (spec §8): signature
   at `wording.py:66-70` is additive with default `None`; default branch
   selects `FORBIDDEN_PHRASES` (line 87-89), keeping Phase B call sites
   unchanged.
8. **Post_analysis backward compatibility** (spec §8, H3): new
   `CONDITION_NAME_PREFIXES` and `_enumerate_conditions` added with
   explicit legacy-subset-first iteration (writer handoff §files_changed).
   Pinned byte-identical-ordering test in
   `test_post_analysis_condition_names.py` covers H3. However the test
   file is torch-gated via `pytest.importorskip("torch")` at module top
   and was skipped in writer's sandbox — see finding #1 below.
9. **Restoration_proportion null branches** (spec §7, §11.9): both
   branches present at `mediation.py:316-326` (`denominator_below_epsilon`)
   and :345-356 (`unstable_denominator`); `epsilon_denom=EPSILON_DENOM=0.005`
   matches spec.
10. **Upstream provenance**: `phase_b_summary_sha256` recorded via
    hashlib (imported at `run_phase_c.py:23`) and `environment` block copy
    is part of the summary write path (writer handoff + watcher
    `matched` item).
11. **Determinism contract**: `n_resamples=1000`, `seed=0`,
    `np.random.default_rng` construction is single-site and deterministic
    — bytewise-equal CSV assertion in end-to-end test is well-founded.

### Spec conformance — extras (acceptable)

1. `--config` flag accepted-and-logged for cross-phase CLI parity
   (writer-agent hard rule). Writer disclosed; watcher accepted. Non-
   harmful — does not route into analysis.
2. `acc_cross_check_tolerance = 2e-3` in `environment` block (round-2 nit
   #3 fix) correctly documents the deviation from spec §10's 1e-6 tolerance
   (Phase B rounds to 4 decimals in `phase_b_summary.json`).

### Findings

#### Finding 1 (med) — torch-gated post_analysis tests unexercised in writer sandbox
- **File:** `stage1/tests/test_post_analysis_condition_names.py` (module top)
- **Claim:** `pytest.importorskip("torch")` at module level skips all 9
  backward-compat tests when torch is not installed. Writer's sandbox
  has no torch, so H3 byte-identical-ordering assertion did not execute
  in Round-1 or Round-2 test runs. The 1-skipped count in writer's
  tests_run output is this entire file.
- **Evidence:** Writer handoff "details: 13 passed, 1 skipped"; writer's
  deferred_nit #2 explicitly acknowledges and defers to Colab/GPU box.
  Watcher Round-2 re-raised the same finding at severity `med`.
- **Impact:** H3 (backward-compat of `compute_bpd_sweep`) is unverified
  on the writer's workstation. If `post_analysis.py` accidentally imports
  torch at module scope and the test file transitively inherits that,
  the skip is mechanically honest but hypothesis-silent.
- **Suggested fix:** Manager/CI should run this suite in a torch-enabled
  environment before merging Phase C. Non-blocking for writer handoff;
  **blocking for the CI acceptance smoke test** described in spec §10
  ("pytest -q ... must pass").

#### Finding 2 (nit) — Writer handoff counts "five" patch fixtures but directory has four
- **File:** `notes/handoffs/writer_phase_c_mediation.md:70-72` ("five
  `results_restoration_patch_*.jsonl` files") vs fixture directory listing.
- **Claim:** `stage1/tests/fixtures/phase_b_run_fixture/` contains only
  four `results_restoration_patch_*.jsonl` files (boundary_local,
  final_only, recovery_early, recovery_full) — not five. The missing one
  would be `patch_all_downstream`. This is a handoff-documentation nit,
  not a correctness issue: `patch_all_downstream` is excluded from
  `CLAIM_ELIGIBLE_CONDITIONS`, and spec §11.2 allows the CSV row count
  to be "5 or 6 full run, 2 sanity" — four present + one no_patch audit
  row = five rows, which is ≤ spec-compliant upper bound. The E2E test
  still exercises all four claim-eligible branches.
- **Suggested fix:** Either (a) update writer handoff line 72 from
  "five" → "four", or (b) add `results_restoration_patch_all_downstream.
  jsonl` to the fixture for completeness. Neither is blocking.

#### Finding 3 (nit) — Double summary.json write (deferred by writer, confirmed low-risk)
- **File:** `stage1/run_phase_c.py` (summary write path then gate rewrite).
- **Claim:** Summary is written, then gate-scanned, then rewritten with
  `forbidden_phrases_gate` field populated. Defensive but non-atomic
  (a crash between writes could leave a pre-gate summary on disk).
- **Evidence:** Watcher Round-2 finding #2 (same); writer's deferred_
  nit #5 rationale ("gate already reads summary.json as part of its
  contract").
- **Suggested fix:** Accept as-is (watcher-aligned). A future refactor
  could buffer the gate scan against an in-memory dict, then single-
  write; not worth the risk here.

#### Finding 4 (nit) — 2e-3 cross-check tolerance vs spec §10's 1e-6
- **File:** `stage1/run_phase_c.py::_cross_check_accuracies` call site.
- **Claim:** Writer documented deviation + recorded
  `acc_cross_check_tolerance: 2e-3` into `environment` block (Round-2
  nit #3 fix). Root cause is Phase B's 4-decimal rounding in
  `phase_b_summary.json`; tightening requires a Phase B spec errata.
- **Suggested fix:** Accept for Phase C; track Phase B spec errata
  (watcher's `can_ship_with_followup` #2 already captures this).

### Triage table

| codex_finding_id | claude_label       | reasoning |
|------------------|--------------------|-----------|
| 1                | agree-nit          | Matches watcher `med` finding; not a writer-actionable defect. Blocker for the CI smoke test only. Manager routes to acceptance-test runner (torch env), not back to writer. |
| 2                | agree-nit          | Pure handoff-text accounting error; fixture count is spec-compliant (4 claim-eligible + no_patch audit = 5 CSV rows, within §11.2 bound). Writer correction optional. |
| 3                | already-known      | Writer deferred (deferred_nit #5); watcher accepted (Round-2 finding #2). No re-litigation. |
| 4                | already-known      | Writer disclosed (deviations_from_spec #3); watcher accepted with `acc_cross_check_tolerance` persisted. Phase B spec errata, not a Phase C blocker. |

### Verdict — Pass 1

**PASS (with 1 med follow-up for CI-environment test exercise).**

No `agree-block` findings. The diff is spec-conformant on every
quantitative acceptance criterion verifiable from the static artifacts
(paired bootstrap semantics, sample_id pairing, 1000 resamples at seed=0,
95% CI, mandated caveat byte-exact, FORBIDDEN_PHRASES gate extension,
post_analysis backward-compat hook). The single `med` finding is an
environment-coverage gap that the watcher already surfaced and the
writer already deferred; it will be resolved by running the existing
post_analysis suite in a torch-enabled environment.

---

## Artifact paths

- This artifact:
  `C:\Users\system1\Boundary-sensitivity\notes\handoffs\codex_review_phase_c_mediation.md`
- Adversarial (Pass 2):
  `C:\Users\system1\Boundary-sensitivity\notes\handoffs\codex_adversarial_phase_c_mediation.md`
