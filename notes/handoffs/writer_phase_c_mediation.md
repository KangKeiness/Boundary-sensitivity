spec_ref: notes/specs/phase_c_mediation.md

files_changed:
  - path: stage1/utils/wording.py
    diff_stat: +52 -6
    note: |
      Added FORBIDDEN_PHRASES_PHASE_B and FORBIDDEN_PHRASES_PHASE_C tuples;
      FORBIDDEN_PHRASES remains a module attribute aliased to the Phase B
      tuple so all existing Phase B call sites and the
      test_forbidden_phrases_gate iteration continue to pass bytewise.
      check_artifacts_for_forbidden gains an optional phrases= kwarg.
  - path: stage1/analysis/mediation.py
    diff_stat: +400 -0 (new file)
    note: |
      ConditionCorrectness dataclass; load_condition_correctness (JSONL
      reader with duplicate-id and missing-field errors);
      align_by_sample_id (sorted intersection, WARN on drops);
      _paired_bootstrap (1000 resamples, seed=0, NaN-dropping fn);
      restoration_effect, residual_effect, restoration_proportion;
      compute_decomposition_table orchestration with alphabetical
      claim-eligible tie-break.
  - path: stage1/run_phase_c.py
    diff_stat: +460 -0 (new file)
    note: |
      CLI entrypoint with --phase-b-run, --sanity, --seed, --bootstrap-n,
      --ci, --run-name, and an accept-and-ignore --config for cross-phase
      parity. Emits phase_c_decomposition_table.csv,
      phase_c_summary.json, phase_c_summary.txt under
      stage1/outputs/phase_c/run_<timestamp>[_<name>]/. Inherits Phase B
      provenance via phase_b_summary.json SHA-256 + environment block
      copy. Runs Phase C forbidden-phrases gate; raises on any violation.
      Cross-checks acc_no_patch / acc_clean_no_patch against Phase B
      summary within 2e-3 absolute tolerance.
  - path: stage1/analysis/post_analysis.py
    diff_stat: +95 -6
    note: |
      Added CONDITION_NAME_PREFIXES, _enumerate_conditions,
      _infer_b_for_condition. compute_bpd_sweep internal loop rewritten
      to use _enumerate_conditions; legacy hard_swap_b{b} /
      random_donor_b{b} emission order preserved (pre-change warning
      behaviour preserved for missing legacy keys). Public signature
      unchanged. Phase A fixed_w4_/fixed_b8_/random_fixed_ and Phase B
      patch_/corrupt_ families now enumerable without touching
      pre-existing callsites.
  - path: stage1/tests/test_phase_c_mediation.py
    diff_stat: +280 -0 (new file)
    note: |
      13 tests: loader happy-path/duplicate/missing-field/missing-file;
      align drops + WARN; restoration_effect determinism;
      residual_effect determinism; proportion null-when-denom-below-eps;
      proportion unstable_denominator; tie-break alphabetical;
      Phase C forbidden-phrases gate flags every listed phrase AND does
      NOT flag Phase C core vocabulary ("restoration effect", etc.);
      end-to-end CLI against shipped fixture with CSV bytewise-determinism.
  - path: stage1/tests/test_post_analysis_condition_names.py
    diff_stat: +110 -0 (new file)
    note: |
      9 tests for _enumerate_conditions + _infer_b_for_condition;
      covers all 5 new naming families plus legacy-byte-identical
      ordering. Module auto-skips via pytest.importorskip("torch") when
      torch is unavailable (H3 backward-compat check still runs on any
      env where torch is installed; the pre-change legacy-order
      byte-identity test is the primary H3 guard).
  - path: stage1/tests/fixtures/phase_b_run_fixture/
    diff_stat: +8 files (new)
    note: |
      Synthetic 8-sample Phase B run fixture: phase_b_summary.json
      (acc_no_patch=0.125, acc_clean=0.5), results_clean_no_patch.jsonl,
      results_restoration_no_patch.jsonl, and five
      results_restoration_patch_*.jsonl files with pinned correctness
      patterns. Enables hermetic CLI end-to-end test; no real Phase B
      run required.

tests_run:
  - command: |
      PYTHONPATH="" py -3.12 -m pytest -x
        stage1/tests/test_phase_c_mediation.py
        stage1/tests/test_post_analysis_condition_names.py
    status: pass
    failing_ids: []
    details: |
      13 passed, 1 skipped (the post_analysis module-level
      pytest.importorskip("torch") skips the whole file in this sandbox
      env; torch is not installed at the writer's disposal but the
      legacy-byte-identical-order test is pure-Python and will run on
      any Colab/GPU environment where the rest of the project executes).
  - command: |
      PYTHONPATH="" py -3.12 -m pytest -v
        stage1/tests/test_phase_b_patcher.py -k forbidden
    status: pass
    failing_ids: []
    details: |
      All three Phase B forbidden-phrase tests still pass bytewise —
      confirms that aliasing FORBIDDEN_PHRASES to FORBIDDEN_PHRASES_PHASE_B
      is backward-compatible.

deviations_from_spec:
  - field: FORBIDDEN_PHRASES_PHASE_C — two "defensive literal" tokens removed
    reason: |
      Spec §8 lists the literal tokens "nie/nde" and "nde/nie" as forbidden
      in Phase C artifacts. Spec §scope-note (and §11.10 TXT header)
      simultaneously requires the verbatim caveat
      "Mediation analysis here decomposes accuracy deltas under prompt-side
      restoration intervention only. It is not a formal NIE/NDE decomposition."
      to appear in every Phase C summary, and the §11.10 header contains
      "NIE/NDE decomposition". These two requirements are mutually
      inconsistent under the case-insensitive gate rule (§11.4). Resolution
      taken: the two literal tokens were OMITTED from
      FORBIDDEN_PHRASES_PHASE_C; the spelled-out forms ("natural direct
      effect", "natural indirect effect", "causal mediation") REMAIN
      forbidden, so formal-mediation prose is still blocked. If the
      spec-planner prefers the opposite resolution (rewording the caveat
      to avoid the literal token), raise NEEDS_SPEC_REVISION and I'll
      restore the tokens.
  - field: --config CLI argument
    reason: |
      Spec §8 prescribes CLI flags "--phase-b-run <path>, --sanity, --seed,
      --bootstrap-n, --run-name" without --config. The writer-agent hard
      rule ("Entrypoints take --seed, --config, --run-name") was satisfied
      by accepting --config as an optional-and-logged flag (analysis-only
      phase has no config consumer); this preserves cross-phase CLI parity
      without silently defaulting anything.
  - field: epsilon tolerance for Phase B acc cross-check
    reason: |
      Spec §10 eval-sanity says "within 1e-6" but Phase B writes rounded
      4-decimal accuracies to phase_b_summary.json, so a true 1e-6 check
      would always fail on real runs. Used 2e-3 (same scale as Phase B's
      own cross-phase tolerance). Documented in code.

dependencies_added: []

reproducibility_audit:
  seed_wired: true           # run_phase_c seeds numpy; mediation.py uses
                             # numpy.random.default_rng(seed) for bootstrap.
                             # No torch RNG is touched (analysis-only).
  config_logged: true        # --config is logged when passed; bootstrap
                             # knobs (n, seed, ci) go into
                             # phase_c_summary.json.bootstrap AND .environment.
  run_name_required: false   # run_name is optional with default=None (matches
                             # Phase B's parser.add_argument default=None).
                             # Spec §8 does not mark it as required.
  determinism_flags: n/a     # Phase C is CPU-only numpy; no CUBLAS /
                             # torch.use_deterministic_algorithms call is
                             # meaningful. The bootstrap is deterministic by
                             # construction via np.random.default_rng(seed).

open_questions_for_watcher:
  - |
    CONFIRM the deviation on FORBIDDEN_PHRASES_PHASE_C literal tokens. The
    spec as-written is self-contradictory; the writer chose the resolution
    that preserves the verbatim caveat. If watcher/spec-planner instead
    wants the caveat reworded to avoid the "NIE/NDE" substring (e.g.,
    "not a formal direct/indirect effect decomposition"), flag it and I'll
    restore the "nie/nde" / "nde/nie" tokens to the forbidden tuple.
  - |
    The post_analysis condition-name test module auto-skips when torch is
    not installed (post_analysis.py imports torch at module top). In the
    writer's sandbox torch is unavailable so the 9 new tests were skipped.
    Watcher should re-run in a torch-enabled env (Colab / GPU box) to
    actually exercise H3 backward-compat.
  - |
    The Phase C CLI end-to-end test uses bootstrap_n=200 (not 1000) for
    speed. The full-1000 determinism check is deferred to the acceptance
    smoke test in §10 of the spec.
  - |
    The accuracy cross-check tolerance was loosened from spec's 1e-6 to
    2e-3 because Phase B writes rounded 4-decimal accuracies. If the
    watcher wants a tighter check, suggest unrounding those values in
    Phase B's summary (a spec revision for Phase B, not Phase C).

artifact_path: notes/handoffs/writer_phase_c_mediation.md

---

## Round 2 — watcher BLOCK fix

Addresses `notes/handoffs/watcher_phase_c_mediation.json` (verdict BLOCK, 1 block
finding + 4 nits). Scope: must-fix #1 (em-dash header) plus cheap low/med nits
#3 and #4. Nit #2 (torch-free sub-module) and nit #5 (skip double summary write)
deferred — see below.

files_changed:
  - path: stage1/run_phase_c.py
    diff_stat: +10 -4
    note: |
      (1) PHASE_C_TXT_HEADER at :56-59 and the matching print at :420 now use
      U+2014 EM DASH instead of ASCII hyphen-minus, making the TXT header
      byte-equal to the spec §11.10 literal.
      (2) _cross_check_accuracies now raises RuntimeError with an explicit
      Phase-B-schema-drift message if BOTH 'no_patch_accuracy' and
      'clean_baseline_accuracy' are missing from phase_b_summary.json
      (watcher nit #4).
      (3) env_block records `acc_cross_check_tolerance: 2e-3` so downstream
      readers see the relaxation vs spec §10's 1e-6 (watcher nit #3).
  - path: stage1/tests/test_phase_c_mediation.py
    diff_stat: +29 -0
    note: |
      (1) New standalone test `test_phase_c_txt_header_byte_equals_spec_literal`
      asserts PHASE_C_TXT_HEADER == the spec §11.10 literal (em-dash, not
      ASCII hyphen). Runs without torch or fixtures.
      (2) End-to-end CLI test extended: asserts SPEC_HEADER_LITERAL appears
      verbatim in txt_body, and asserts
      summary["environment"]["acc_cross_check_tolerance"] == 2e-3.

tests_run:
  - command: |
      PYTHONPATH="" py -3.12 -m pytest -x
        stage1/tests/test_phase_c_mediation.py
    status: pass
    failing_ids: []
    details: 14 passed in 0.59s (13 original + 1 new header-literal test).
  - command: |
      PYTHONPATH="" py -3.12 -m pytest -v
        stage1/tests/test_phase_b_patcher.py -k forbidden
    status: pass
    failing_ids: []
    details: 3 passed — Phase B forbidden-phrase gate still green.

deferred_nits:
  - id: watcher-finding-#2 (med) — torch-free sub-module for _enumerate_conditions
    reason: |
      Lifting `_enumerate_conditions` / `_infer_b_for_condition` out of
      post_analysis.py would touch a pinned Phase A file and its legacy
      emission-order contract is already pinned by
      test_post_analysis_condition_names.py's byte-identical-ordering test.
      CI requirement: these 9 tests must run in a torch-enabled env (Colab /
      GPU box) to exercise H3. Documented; not fixed here.
  - id: watcher-finding-#5 (nit) — skip double summary write
    reason: |
      Current pattern (write summary → gate-scan → rewrite summary with gate
      results) is defensive and correct. Not worth the refactor risk for a
      pre-final-IO scan path; the gate already reads `summary.json` as part
      of its contract.

reproducibility_audit_delta:
  env_block_extended: acc_cross_check_tolerance now persisted
  seed/config/run-name wiring: unchanged from round 1

round_2_status: all must-fix items addressed; 14/14 Phase C tests green;
                Phase B backward-compat tests still green.
