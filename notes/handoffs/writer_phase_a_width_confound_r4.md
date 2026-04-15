spec_ref: notes/specs/phase_a_width_confound.md

files_changed:
  - path: stage1/run_phase_a.py
    diff_stat: +65 -65
    description: >
      Added module-level helper `_check_grid_intersection(rows, run_dir)` at line 201
      (between spearman helper and Main runner section). Replaced the 69-line inline
      block in run_phase_a() body (previously lines 702-770) with a 2-line delegation:
      call to `_check_grid_intersection(rows, run_dir)` plus the existing checks.append
      logic for the sanity-check ledger.
  - path: stage1/tests/test_grid_intersection_check.py
    diff_stat: +55 -75
    description: >
      Deleted inline shadow copy `_run_grid_intersection_check` (was lines 25-85).
      Added `from stage1.run_phase_a import _check_grid_intersection` import.
      Replaced all 4 test call sites with `_check_grid_intersection(rows, run_dir)`.
      Updated test fixtures to build proper `rows` lists (with "condition" key) via
      new `_make_rows` helper. Replaced false comment at lines 9-10 with accurate text.
      Replaced `test_scenario_b_both_fld_metrics_in_warning` with scenario (d):
      `test_scenario_d_only_one_condition_silent_skip` to satisfy the spec's required
      four scenarios (a)-(d). Total test count unchanged: 36.

tests_run:
  - command: /c/Users/system1/anaconda3/envs/datascience/python.exe -m pytest stage1/tests/ -x -q
    status: pass
    failing_ids: []
    output: "36 passed in 2.52s"

deviations_from_spec:
  - field: test_scenario_b_both_fld_metrics_in_warning
    reason: >
      The spec lists 4 required scenarios (a)-(d) where (d) is "only one intersection
      condition present → silent skip". The existing r3 test file had 4 tests but the
      4th covered a sub-scenario of (b) (both fld metrics triggering), not scenario (d).
      That test was replaced with `test_scenario_d_only_one_condition_silent_skip` to
      satisfy the spec. Test count remains 36 (unchanged from r3). The both-fld-metrics
      sub-case is already implicitly covered by scenario (b) assertion structure.

dependencies_added: []

reproducibility_audit:
  seed_wired: true
    note: >
      _check_grid_intersection is a pure deterministic check helper with no randomness.
      The run_phase_a() entrypoint seed wiring is unchanged from r3.
  config_logged: true
  run_name_required: true
  determinism_flags: true

fix_checklist:
  AV5-ADV-BLOCK: CLOSED
    - [x] Inline shadow copy deleted from test file
    - [x] Production helper `_check_grid_intersection` extracted to module level in run_phase_a.py
    - [x] Production helper signature: `(rows: List[Dict], run_dir: str) -> None`
    - [x] All 4 test call sites import and invoke production function
    - [x] False comment at lines 9-10 corrected
    - [x] Scenarios (a) no raise/no warning: PASS
    - [x] Scenarios (b) no raise/writes warning: PASS
    - [x] Scenarios (c) accuracy mismatch raises RuntimeError: PASS
    - [x] Scenarios (d) only one condition present, silent skip: PASS
    - [x] Tolerance 1e-3 unchanged
    - [x] Tier 1/Tier 2 logic behaviorally identical to r3
    - [x] 36/36 tests pass

open_questions_for_watcher:
  - The 4th test was swapped from `test_scenario_b_both_fld_metrics_in_warning` to
    `test_scenario_d_only_one_condition_silent_skip`. Watcher should confirm this
    replacement satisfies the AV5-ADV-BLOCK requirement and that the both-fld-metrics
    scenario does not need its own dedicated test given it is implicitly covered by (b).
  - No other open questions. AV1/AV2/AV3/AV7 nits remain deferred per r4 scope constraint.
