spec_ref: notes/specs/phase_a_width_confound.md
prior_handoff_ref: notes/handoffs/writer_phase_a_width_confound_r2.md
codex_block_ref: notes/handoffs/codex_phase_a_width_confound_adversarial.json
scope: AV-4 only

files_changed:
  - path: stage1/run_phase_a.py
    diff_stat: +51 -14
    change: |
      Replaced monolithic 1e-6 grid-intersection check (lines 700-723) with a
      two-tier check.
      Tier 1 (strict): accuracy and degradation compared with abs() > 0 guard —
        fully deterministic under greedy decoding (temperature=0.0, do_sample=False);
        raises RuntimeError with "logic bug" message on mismatch.
      Tier 2 (relaxed): fld_cos and fld_l2 compared at tolerance 1e-3; on mismatch
        prints a SANITY WARNING to stdout, writes grid_intersection_fld_warning key
        into phase_a_summary.json (re-opening the file, merging the key, re-saving),
        and does NOT raise — run continues.
      Also updated the phase_a_summary.txt grid-intersection note (one string literal
      near line 623) to mention the relaxed fld tolerance and the strict
      accuracy/degradation expectation.

  - path: stage1/tests/test_grid_intersection_check.py
    diff_stat: +175 -0 (new file)
    change: |
      New test file with 4 tests covering all three required scenarios plus one
      extra edge case:
        test_scenario_a_no_raise_no_warning — fld delta < 1e-3, no raise, no key
        test_scenario_b_no_raise_but_writes_warning — fld_cos delta = 1e-2, no raise,
          warning key written with correct structure
        test_scenario_c_mismatched_accuracy_raises — accuracy differs, RuntimeError raised
        test_scenario_b_both_fld_metrics_in_warning — both fld_cos and fld_l2 > 1e-3,
          both appear in warning details

tests_run:
  - command: /c/Users/system1/anaconda3/envs/datascience/python.exe -m pytest stage1/tests/ -x -q
    status: pass
    passed: 36
    failed: 0
    note: 32 prior passing + 4 new tests in test_grid_intersection_check.py

deviations_from_spec:
  - field: test count
    reason: |
      Prompt required 3 parametrized or separate tests (scenarios a, b, c).
      Implementation provides 4 tests (scenarios a, b, c + extra edge case covering
      both fld metrics firing simultaneously). The extra test does not expand scope;
      it verifies a natural code path. No spec deviation.

  - field: grid_intersection_fld_warning persistence mechanism
    reason: |
      phase_a_summary.json is written before the sanity-check block runs. The fix
      re-opens and updates the file if the fld warning fires. This is the minimal
      surgical approach; it avoids restructuring the run_phase_a function order.

reproducibility_audit:
  seed_wired: true (unchanged from r2)
  config_logged: true (unchanged from r2)
  run_name_required: false (unchanged from r2; optional CLI arg per spec)
  determinism_flags: true (temperature=0.0, do_sample=False, float32 cast; no
    torch.use_deterministic_algorithms added — codex Option B explicitly rejected)

fix_checklist:
  AV-4: CLOSED
    Mechanism: Two-tier check.
    - accuracy/degradation: strict equality (counts-based, greedy decoding);
      RuntimeError("logic bug") on any mismatch.
    - fld_cos/fld_l2: tolerance 1e-3 (flash-attention2 non-determinism);
      SANITY WARNING printed to stdout + grid_intersection_fld_warning written
      into phase_a_summary.json; run continues normally.
    - "skipped when not both present" branch: unchanged.
    - torch.use_deterministic_algorithms NOT set (Option B rejected per prompt).

open_questions_for_watcher:
  - All open questions from r2 remain (HF revision pins, mgsm_zh sha256 pending,
    bootstrap seed policy, scipy import). No new open questions introduced.
  - Watcher should confirm: is the re-open-and-update approach for
    grid_intersection_fld_warning acceptable, or should it be restructured to
    populate the key before the initial JSON write?
