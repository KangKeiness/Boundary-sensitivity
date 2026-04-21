scope: portability-only hardening per user instruction (option b)
files_changed:
  - stage1/run.py (+12 docstring lines — module contract banner replacing the single-line docstring)
  - stage1/tests/test_runtime_smoke.py (+6 lines — two `pytest.importorskip` gates for transformers + torch at the top of `test_stage1_run_importable` and `test_stage1_run_m_help_exit_zero`)
tests_run:
  - command: PYTHONPATH="" C:/Users/system1/AppData/Local/Programs/Python/Python312/python.exe -m pytest -x stage1/tests/test_runtime_smoke.py -v
    result: 5 passed, 0 skipped, 0 failed (17.04s → 15.99s across two runs)
  - command: PYTHONPATH="" C:/Users/system1/AppData/Local/Programs/Python/Python312/python.exe -c "<install sys.meta_path blocker that raises ImportError for 'transformers*'; pop any cached transformers entries from sys.modules>; import pytest; raise SystemExit(pytest.main(['-x','--noconftest','stage1/tests/test_runtime_smoke.py::test_stage1_run_importable','-v']))"
    result: 1 skipped, 0 failed, 1 warning (PytestDeprecationWarning about default ImportError acceptance in 9.1 — benign)
  - command: same, with target test_stage1_run_m_help_exit_zero
    result: 1 skipped, 0 failed, 1 warning (same deprecation notice)
verification:
  - docstring present: yes ("Runtime contract:" on line 3 of stage1/run.py)
  - full-env all-pass: yes (5/5, all target tests PASSED; no skips)
  - lightweight-env clean-skip (no FAIL): yes (both gated tests SKIPPED cleanly, the 3 other tests were not invoked in the simulation — they remain unmodified)
  - no other files touched: my session edits are limited to stage1/run.py (docstring-only) and stage1/tests/test_runtime_smoke.py (importorskip gates only). `git diff --stat` includes many other files that were already uncommitted from prior phases (v5 runtime_repro, phase B/C, etc.) and are out-of-scope for this pass.
deviations:
  - simulation technique: the spec's `sys.modules['transformers']=None` trick does NOT work in this repo because `stage1/tests/conftest.py` installs a deterministic transformers STUB in sys.modules at collection time (see conftest lines 79–80). I therefore used `pytest.main(['--noconftest', ...])` combined with a sys.meta_path `_Blocker` that raises ImportError on any `transformers*` find_spec. This reliably demonstrates clean SKIP and is portable (no file renames, no site-packages surgery). Flagging this because it means the new `importorskip` gates do NOT protect against a scenario where the conftest stub is present but the REAL transformers is missing — in that pathological state the gate would succeed (stub exists) but the subprocess `import stage1.run` would still fail. For the user's stated goal (gating smoke tests on dependency availability for a lightweight CI lane), the correct lightweight-CI invocation must either use `--noconftest` OR have the conftest itself skip stub installation when real transformers is absent. That is a conftest-level concern outside this option-b scope.
