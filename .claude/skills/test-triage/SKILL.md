---
name: test-triage
description: Deterministic triage of pytest collection and failures. Distinguishes flakes, fixture errors, real regressions, and environment mismatches. Enforces coverage rules for new files.
allowed-tools: Read, Grep, Glob, Bash
disable-model-invocation: false
---

# test-triage

## Procedure

1. Run `pytest --collect-only -q` on the touched test modules. Any collection
   error → root-cause it before running tests.
2. Run `pytest -x --lf` on the touched modules.
3. Categorize each failure:
   - **regression** — new code path broken
   - **flake** — passes on rerun, no code change; requires an issue link
   - **fixture** — shared fixture broken; block all dependent tests
   - **env** — missing optional dep or CUDA; N/A if not in CI target
4. Check coverage: every new file in the writer diff must have at least one
   test under `tests/` exercising it (even a smoke test).
5. Verify no test imports from `runs/` or `outputs/`.
6. Verify no new `@pytest.mark.skip` or `@pytest.mark.xfail` without a
   linked issue id in the reason string.

## Bash allowlist

`pytest --collect-only`, `pytest -x --lf`, `git log`, `git diff`. No
mutation.

## Output

```json
{
  "collection_ok": true,
  "failures": [
    {"test_id": "tests/...", "category": "regression|flake|fixture|env",
     "root_cause": "...", "evidence": "..."}
  ],
  "coverage_gaps": [{"new_file": "src/...", "reason": "no corresponding test"}],
  "new_skips_or_xfails": [{"test_id": "...", "issue_link": "..."}],
  "verdict": "CLEAN|DIRTY"
}
```

## Forced checks

- No test may be marked skip/xfail without an issue link.
- New files without tests → `DIRTY`.
- Flakes without issue links → `DIRTY`.

## Example invocation

```
Run test-triage on tests/xling/ after writer_xnli_contrastive.md.
```
