spec_ref: notes/specs/runtime_repro_v5.md
files_changed:
  - path: stage1/run.py
    diff_stat: +8 -8
    summary: |
      Commit A — rewrote the 8 bare sibling-package imports (L11–L25 per spec,
      actual L11–L18 after file shift) to the `stage1.`-qualified form. Matches
      the style already used in `stage1/run_phase_a.py`. Standard-library and
      third-party imports untouched. No other code in the file modified by me.
      (Note: the full `git diff` for this file also shows pre-existing revision
      arg additions in `main()` that were uncommitted on the working tree
      before this session — not authored here.)
  - path: stage1/utils/config.py
    diff_stat: +2 -1
    summary: |
      Commit B — added `encoding="utf-8"` to the `open(path)` call inside
      `load_config` (at L101 per spec, actual L101-L102 after the added
      comment). Added the required one-line comment immediately above:
      `# UTF-8 enforced: config loading must not depend on OS locale (Windows cp949).`
      No other `open` calls in `load_config`'s call chain; no widening.
  - path: stage1/run_phase_a.py
    diff_stat: +1 -1
    summary: |
      Commit C — inside `_load_reused_no_swap`, changed the single parity
      extraction call from `extract_parity_block(config)` to
      `extract_parity_block(config, sample_ids=[s["sample_id"] for s in samples])`.
      Verified `samples` is already in scope (function parameter), each element
      is a dict with `sample_id` (consistent with the fresh-write site at
      L781–L783 which uses the same expression). Kept as a single line so the
      spec §11.9 literal grep passes.
  - path: stage1/tests/test_runtime_smoke.py
    diff_stat: +287 -0
    summary: |
      Commit D — new file. Five tests (four per spec §10.1–10.4 plus a
      torch-free parity-check variant so the §10.4 regression signal still
      fires on machines without torch):
        - test_stage1_run_importable — `importlib.util.find_spec("stage1.run")`
          plus a subprocess `python -c "import stage1.run"`; keeps torch out
          of the pytest process.
        - test_stage1_run_m_help_exit_zero — `python -m stage1.run --help` from
          `Path(__file__).resolve().parents[2]` (repo root); asserts rc==0 and
          `--config` appears in stdout.
        - test_load_config_utf8 — calls `load_config` on stage2_confound.yaml,
          asserts `dataset.lang == "zh"`, and `inspect.getsource(load_config)`
          contains literal `encoding="utf-8"`.
        - test_phase_a_reuse_parity_has_sample_regime_pure — torch-free variant.
          Builds source manifest via `extract_parity_block(cfg, sample_ids=[...])`,
          compares against both the post-fix (with sample_ids) and pre-fix
          (no sample_ids) current blocks; asserts `ManifestParityError` with
          match="sample_regime" on the pre-fix branch. Fails on pre-fix code.
        - test_phase_a_reuse_parity_end_to_end — gated via
          `pytest.importorskip("torch")`. Calls `_load_reused_no_swap` on a
          real tmp reuse dir with stub `.pt`, `.jsonl`, and manifest; accesses
          the private function via `importlib.import_module` + `getattr`.
tests_run:
  - command: python -c "import stage1.run"
    status: pass
  - command: python -m stage1.run --help
    status: pass
  - command: python -c "from stage1.utils.config import load_config; load_config('stage1/configs/stage2_confound.yaml')"
    status: pass
  - command: pytest -x stage1/tests/test_runtime_smoke.py -v
    status: pass
    failing_ids: []
  - command: pytest -x stage1/tests/ -k "parity or runtime" -v
    status: pass
    failing_ids: []
  - command: pytest stage1/tests/ (full suite)
    status: fail
    failing_ids:
      - stage1/tests/test_phase_b_patcher.py::test_input_side_patch_cache_consistency_identity
      - stage1/tests/test_phase_b_patcher.py::test_input_side_patch_changes_cache_when_input_differs
      - stage1/tests/test_phase_b_patcher.py::test_run_phase_b_module_imports
      - stage1/tests/test_phase_b_patcher.py::test_phase_a_loader_no_match_returns_none
      - stage1/tests/test_phase_b_patcher.py::test_phase_a_loader_resolves_repo_relative
    note: |
      These 5 failures are pre-existing at baseline — confirmed by stashing
      my three source edits and re-running the test file: all 5 still fail
      with the same `AttributeError: module 'stage1.run_phase_b' has no
      attribute '_load_latest_phase_a_summary'`. They are NOT caused by this
      runtime-repro sprint and are outside the §9 files-to-touch scope.
deviations_from_spec:
  - spec said the bare imports span L11–L25 of stage1/run.py; the actual bare
    import block was L11–L25 inclusive (8 distinct statements ending with a
    multi-line paren block). Rewritten in place without reflow. No semantic
    drift; intent matched exactly.
  - spec said `open(path)` is at L101 of stage1/utils/config.py; actual line
    was also L101. After adding the required preceding comment the call moved
    to L102. No semantic drift.
  - spec said `extract_parity_block(config)` is at L344 of stage1/run_phase_a.py;
    actual line was 344. After edit it remains L344 as a single line.
  - spec §10.4 allowed splitting test 4 into a pure-parity variant (always
    runs) plus a torch-gated end-to-end variant when torch is not guaranteed.
    Implemented both: the pure variant (`test_phase_a_reuse_parity_has_sample_regime_pure`)
    runs unconditionally and still fails on pre-fix code; the end-to-end
    variant (`test_phase_a_reuse_parity_end_to_end`) is gated via
    `pytest.importorskip("torch")`. Total smoke-test count is therefore 5,
    not 4. This is explicitly sanctioned by the spec's "(a) + (b)" fallback.
  - Verification used `C:\Users\system1\AppData\Local\Programs\Python\Python312\python.exe`
    with `PYTHONPATH` unset. The sandbox's default Python 3.14 has a
    `PYTHONPATH=C:\Users\system1\anaconda3\Lib` pollution causing an
    `SRE module mismatch` on `import re`, and the Anaconda Python 3.11
    lacks `transformers`. The Py3.12 install has both torch (2.11.0+cpu)
    and transformers (5.5.4) and was the only viable interpreter. No spec
    impact — this is purely a local sandbox quirk.
acceptance_criteria_status:
  - §11.1 import stage1.run exits 0: pass
  - §11.2 python -m stage1.run --help exits 0 (and prints --config): pass
  - §11.3 load_config UTF-8 (exits 0 under any active code page since
    encoding="utf-8" is explicit): pass
  - §11.4 four smoke tests (five after the §10.4 split): pass
  - §11.5 full test suite not regressed (5 pre-existing test_phase_b_patcher
    failures confirmed present at baseline via stash-and-retest; no new
    failures introduced): pass
  - §11.6 no diff in do-not-modify list caused by this session (the
    pre-existing uncommitted diff in stage1/configs/stage2_confound.yaml was
    authored before this sprint and not touched here): pass
  - §11.7 parity fields not weakened (_PARITY_FIELDS and _NULL_EQUIV_PATHS
    in manifest_parity.py untouched): pass
  - §11.8 grep encoding="utf-8" in load_config: pass
  - §11.9 grep `extract_parity_block(config, sample_ids=` in
    _load_reused_no_swap: pass
open_questions_for_watcher:
  - The 5 pre-existing test_phase_b_patcher.py failures reference an
    `_load_latest_phase_a_summary` symbol that is missing from the current
    `stage1/run_phase_b.py` working copy. This is an OPEN BUG unrelated to
    runtime_repro_v5 and should be tracked separately. Recommend the
    watcher/reviewer confirm it was not a regression introduced by an
    earlier sprint that never landed.
  - The working tree has several unrelated uncommitted edits (run_phase_b.py,
    run_phase_c.py, analysis/mediation.py, analysis/post_analysis.py, etc.)
    that were already in place when this session started. These are outside
    the spec's §9 scope; do not flag as writer-introduced.
  - Verification Python was Py3.12 because Py3.14 (default) had a sandbox
    PYTHONPATH pollution and Py3.11 (anaconda) lacked transformers.
    All tests pass on Py3.12; repeat on the canonical interpreter if the
    reviewer prefers.
