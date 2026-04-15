# Codex Adversarial Review — Phase B Rewrite (Round 2, Pass 2)

**Header — fallback disclosure:** The `/codex:adversarial-review`
slash-command is NOT provisioned in this Claude Code sandbox. This artifact
is a **codex-equivalent adversarial review** performed by the reviewer
agent under the discipline "assume the R2 fix re-introduced something
until proven otherwise; bias toward false-positives." It is explicitly NOT
a run of the real Codex xhigh adversarial reviewer.

**Adversarial framing:**
- Assume the R2 patch silently broke a round-1 matched item.
- Assume A1/A2/A3 fixes have edge cases not covered by the new tests.
- Assume the pytest pass of the R2 sandbox (7 pass, 4 skip) masks a real
  defect — 4 of the 12 tests did not execute.
- Scrutinize path resolution on Windows (separators, drive letters,
  symlinks, UNC network paths), import-order side effects, and CI-vs-local
  CWD divergence for the smoke test.

**Inputs consulted:** same as Pass 1 + round-1 adversarial artifact
(`notes/handoffs/codex_adversarial_phase_b_rewrite.md`).

---

## Adversarial findings (R2 patch scrutiny)

### X1 (NIT, correctness) — `_phase_a_outputs_dir` double-nests `"stage1"`

`stage1/run_phase_b.py:125-136`:

```python
return str(
    pathlib.Path(__file__).resolve().parents[1]
    / "stage1" / "outputs" / "phase_a"
)
```

`__file__` is `<repo>/stage1/run_phase_b.py`. `.parents[0]` is
`<repo>/stage1`; `.parents[1]` is `<repo>`. Composed with
`"stage1"/"outputs"/"phase_a"`, the final path is
`<repo>/stage1/outputs/phase_a` — correct.

Adversarial concern: if the repo is ever vendored into a larger project
as a subdirectory (e.g., `<super>/external/Boundary-sensitivity/
stage1/run_phase_b.py`), `parents[1]` becomes
`<super>/external/Boundary-sensitivity`, and
`<super>/external/Boundary-sensitivity/stage1/outputs/phase_a` is still
the right path — so the fix is actually vendoring-resilient.

Equal-adversarial: if anyone ever moves `run_phase_b.py` into a nested
subdirectory (e.g., `stage1/cli/run_phase_b.py`), `parents[1]` becomes
`<repo>/stage1` and the composed path becomes
`<repo>/stage1/stage1/outputs/phase_a` — wrong. `test_phase_a_loader_
resolves_repo_relative` would NOT catch this because it monkeypatches
`_phase_a_outputs_dir` directly, bypassing the `__file__`-based
resolution.

Not a blocker; spec §9 files-to-touch keeps `run_phase_b.py` at
`stage1/run_phase_b.py`. Flagging for future refactor hygiene.

### X2 (LOW, correctness) — Windows `pathlib.Path(__file__).resolve()` on symlinks

`Path.resolve()` on Windows follows symlinks and returns a canonical
path. If the dev box has a symlinked `stage1` or `run_phase_b.py` (not
unusual in a monorepo with shared modules), `.resolve()` canonicalizes
to the symlink target, which may not be under
`<repo>/stage1/outputs/phase_a` at all.

This is a hypothetical; production deployment uses a git checkout
without symlinks. But `pathlib.Path(__file__).resolve()` is more
aggressive than `os.path.abspath(__file__)` (which does not follow
symlinks). If adversarial framing is applied strictly, using
`Path(__file__).absolute()` or `Path(__file__).resolve(strict=False)`
(current behavior, no-op) would be safer for symlink-heavy setups.

Non-blocking adversarial nit.

### X3 (LOW, correctness) — UNC / network-drive glob on Windows

If `<repo>` is on a Windows UNC path like `\\fileserver\home\user\
Boundary-sensitivity\stage1`, `pathlib.Path(__file__).resolve()` returns
the UNC path. `glob.glob` on a UNC path with `run_*` wildcard works on
modern Windows (PEP 519 / pathlib), but bare `os.path.join` on UNC
paths has historically (pre-3.8) had edge cases. Python 3.12 is fine;
just noting.

The writer handoff documents the sandbox runs on Python 3.12 (`py -3.12
-m pytest ...`) — no UNC path concern in production. Non-blocking.

### X4 (NIT, test-coverage) — `test_phase_a_loader_resolves_repo_relative` monkeypatches the helper

`stage1/tests/test_phase_b_patcher.py:475-504` does
`monkeypatch.setattr(mod, "_phase_a_outputs_dir", lambda: str(tmp_path))`.
That means the test validates the **caller-of-helper** (the loader) is
CWD-invariant GIVEN a correct outputs-dir, but it does NOT validate
`_phase_a_outputs_dir` itself. Adversarially: if a future refactor
mis-writes `_phase_a_outputs_dir` to use `Path.cwd()` instead of
`Path(__file__)`, these tests still pass (the monkeypatch overrides the
broken helper). The regression would only be caught by the R2 smoke
test which auto-skips in the sandbox.

Fix (non-blocking): add a direct assertion, e.g.:

```python
def test_phase_a_outputs_dir_is_cwd_invariant(monkeypatch, tmp_path):
    from stage1.run_phase_b import _phase_a_outputs_dir
    monkeypatch.chdir(tmp_path)
    resolved = _phase_a_outputs_dir()
    assert resolved.endswith(os.path.join("stage1", "outputs", "phase_a"))
    # Ensure CWD change did not affect the resolution:
    monkeypatch.chdir(tmp_path.parent)
    assert _phase_a_outputs_dir() == resolved
```

Non-blocking. Flag for possible R3 if Phase C touches this surface.

### X5 (LOW, test-coverage) — `test_smoke_marker` asserts exit 0 but not the wording-gate output

When the smoke test auto-runs, it asserts `proc.returncode == 0` and the
four canonical artifacts exist. It does NOT grep the TXT/JSON for the
FORBIDDEN_PHRASES (the wording gate is enforced by `run_phase_b` itself
via `check_artifacts_for_forbidden` + RuntimeError, so a forbidden phrase
WOULD cause non-zero exit — indirectly covered). It also does NOT
verify the spec §11.10 comparative-sentence literal or the §11.11 TXT
header literal.

Adversarially: if a subtle bug emits a forbidden phrase but the gate
also has a subtle bug that fails to detect it, the smoke test gives a
false-green. The likelihood is low (two independent bugs), but a defense-
in-depth grep in the test (e.g.,
`assert "of performance (rough estimate)" not in txt_contents`) would
harden.

Non-blocking. The existing gate + hard-FAIL path is adequate for R2.

### X6 (NIT, correctness) — Windows `subprocess.run([sys.executable, "-m", ...])` PATH pollution

`test_smoke_marker` uses `sys.executable` — correct, no PATH
dependency. The subprocess sees parent env (including any
`CUBLAS_WORKSPACE_CONFIG` the parent exported). Fine.

Adversarial concern: on a CI runner, the parent pytest may be launched
by a test harness that sets `PYTHONPATH` to include test utilities that
shadow `stage1.*`. If that happens, the subprocess's `-m
stage1.run_phase_b` may import a shadow copy. Unlikely but possible.
Mitigation would be `env={"PYTHONPATH": repo_root, ...}` but that
conflicts with inheriting CUDA-related env. Not actionable without
knowing the CI spec. Flagging for completeness.

### X7 (NIT, correctness) — import-order side effect of `_sys.modules.pop`

`test_run_phase_b_module_imports` at line 417 does
`_sys.modules.pop("stage1.run_phase_b", None)` before `importlib.
import_module`. This correctly forces a fresh import, but it does NOT
pop transitive submodules (`stage1.utils.config`, etc.). If a later
test in the same session monkeypatches one of those submodules and the
module-level import of `stage1.run_phase_b` already cached the
name-binding, the monkeypatch is invisible to `run_phase_b`'s internal
references.

The two other new tests (`test_phase_a_loader_*`) use
`_import_run_phase_b_or_skip()` which also pops only
`stage1.run_phase_b`; they then `monkeypatch.setattr(mod,
"_phase_a_outputs_dir", ...)` — this is direct attribute monkeypatching
on the reloaded module, so it IS visible. Fine for the current three
tests; noting for future tests that might want to monkeypatch
transitives.

Non-blocking.

### X8 (NIT, repro) — `test_smoke_marker` timeout is 1800s; spec §10 says "under 10 min"

Spec §10 smoke-test deadline is 10 min on dev GPU. The R2 test uses
`timeout=1800` (30 min). Adversarially: a spec §10 compliance drift
(e.g., a regression that makes sanity mode take 15 min) would pass the
test but violate the spec. Non-blocking — 30 min is a reasonable upper
bound for slower dev boxes, and a 15-min sanity run is still a practical
red flag during manual operation.

### X9 (LOW, correctness) — `_apply_determinism` is called AFTER `setup_logging`

Verified at `stage1/run_phase_b.py` around lines 242-245 (per watcher
line 103 and round-1 A11). Determinism warnings are appended to a list
and later written into the summary. Not a regression; round-1 A11 was
correctly marked disagree-with-reason.

Adversarial recheck: `setup_logging` itself should not consume any RNG.
If a future refactor adds logging-side randomness (e.g., a random
logger name, which nobody does), the order would matter. No action.

### X10 (LOW, correctness, NEW) — `run_20991231_000000` fake timestamp is the newest forever

`test_phase_a_loader_resolves_repo_relative` plants a
`run_20991231_000000` fake (year 9999). This is sorted-descending
newest. If a future test in the same session leaves a real
`run_20260401_...` fixture under the same tmp_path, the fake still
wins. The test's tmp_path is ephemeral so this doesn't bite, but the
pattern of using 9999-year timestamps for "newest wins" fixtures is
load-bearing on the sort being lexicographic-descending — which it is
(`sorted(..., reverse=True)`). No action; noting for literacy.

### X11 (LOW, repro, NEW) — `_state_dict_sha256` uses `numpy().tobytes()` — fails on bf16

`stage1/run_phase_b.py:119`:
`h.update(t.detach().to("cpu").contiguous().numpy().tobytes())`. Spec
§5 pins fp16; numpy supports fp16. If a future config drift to bf16
happens (Phase C hardware change), `.numpy()` raises TypeError.
Adversarial: round-1 A5 raised this; writer noted "flagged for
spec-planner". No R2 change. No action; carry forward.

### X12 (LOW, correctness, NEW) — `test_run_phase_b_module_imports` skip-vs-fail ambiguity

The new guard distinguishes bare-name ModuleNotFoundError (fail) from
unrelated (`pytest.skip`). Adversarial: a regression that imports
`from stage1.utils import config` (dotted but wrong — should be `from
stage1.utils.config import ...`) raises
`ImportError: cannot import name 'config' from 'stage1.utils'` — which
is an `ImportError`, not a `ModuleNotFoundError`. The `except
ModuleNotFoundError` clause at line 421 does NOT catch `ImportError`
(they're siblings under `ImportError`, but `ModuleNotFoundError` is a
subclass of `ImportError`, not the other way around). So an
`ImportError` from a wrong-name import would propagate up and the test
would fail — which is correct behavior (test fails loudly on unexpected
import error). No issue; explicitly verifying.

### X13 (NIT, test-hygiene, NEW) — `test_smoke_marker` uses `glob` for "latest run_dir"

Line 545-549:
```python
run_dirs = sorted(_glob.glob(
    _os.path.join(repo_root, "stage1", "outputs", "phase_b", "run_*")
))
latest = run_dirs[-1]
```

Adversarial: if the CI runner has a pre-existing `run_*` directory from
a prior test run, `run_dirs[-1]` may be a stale directory (if
timestamp-sorted ascending) — but the subprocess just ran, so the newly
created dir should be the last. The sort is lexicographic; timestamped
run_YYYYMMDD_HHMMSS sorts correctly newest-last. OK.

But if the CI wall-clock is skewed (e.g., two runs in the same second,
or a stale run from a future date), `latest` may be wrong. Mitigation:
record the run_dir returned from the subprocess (via stdout parsing or
a dedicated marker file). Non-blocking — no evidence of clock skew in
writer handoff.

---

## Triage

| codex_finding_id | claude_label          | reasoning |
|------------------|-----------------------|-----------|
| X1               | agree-nit             | `parents[1]` is robust for current file layout. Future refactor to `stage1/cli/run_phase_b.py` would break it; no current risk. |
| X2               | agree-nit             | `Path.resolve()` follows symlinks; production uses git checkout without symlinks. Hypothetical. |
| X3               | disagree-with-reason  | Python 3.12 handles UNC paths correctly. Not a real concern for the pinned dev env. |
| X4               | agree-nit             | Monkeypatching the helper bypasses the helper's own correctness check; a direct `_phase_a_outputs_dir` test would belt-and-suspenders. Deferral is justified by R2 scope. |
| X5               | agree-nit             | Indirect coverage via `check_artifacts_for_forbidden` + hard-FAIL in the driver. Defense-in-depth grep in the smoke test is optional. |
| X6               | disagree-with-reason  | `sys.executable` is correct; PYTHONPATH pollution is a CI-spec concern, not a patcher-or-driver defect. |
| X7               | agree-nit             | Only `stage1.run_phase_b` is popped; transitive submodule monkeypatching is not currently required by any test. Noted for future. |
| X8               | agree-nit             | 1800s vs 600s — acceptable upper bound; spec §10 "under 10 min" is a performance target, not a hard assertion. |
| X9               | disagree-with-reason  | Verified ordering correct (`setup_logging` at 242, `_apply_determinism` at 245); not a regression. |
| X10              | disagree-with-reason  | 9999-year fixture is a common test idiom for "newest wins"; not an issue. |
| X11              | already-known         | Round-1 A5; writer carried forward. Not R2's concern. |
| X12              | disagree-with-reason  | `ImportError` from a wrong-name import propagates (not caught by `except ModuleNotFoundError`); test fails loudly — correct behavior. |
| X13              | agree-nit             | CI clock-skew is a stretch; `sorted(...)[-1]` is fine for timestamped dirs in practice. |

---

## Adversarial verdict

**PASS (Pass 2, Adversarial).**

- All three round-1 BLOCKs (A1, A2, A3) survive adversarial re-examination;
  the R2 fixes do not re-introduce the failure modes they claim to fix.
- No new agree-block findings. The strongest new findings (X1, X4) are
  future-refactor hygiene, not current defects.
- Round-1 still-open non-blockers (A5 state_dict hash fp16 assumption,
  A7 bytewise-test decode round-trip, A10 batch=1 hardcode, A12 token-count
  shim) are unchanged from round 1, still non-blocking, and correctly
  documented by the writer in `cheap_codex_nits_not_addressed`.
- No regressions introduced by the R2 diff (+176/-14 across 2 files).

Must-fix-before-merge: none.

Can-ship-with-followup recommendations (additive to watcher's list):
1. Consider a direct `_phase_a_outputs_dir` CWD-invariance test
   (X4) if Phase C touches path resolution.
2. Consider defense-in-depth forbidden-phrase grep in `test_smoke_marker`
   (X5) if a Phase C extension to wording is anticipated.
3. Record the effective `CUBLAS_WORKSPACE_CONFIG` (not the hardcoded
   literal) into `environment` for logged-vs-effective parity (see Pass 1
   N5).

---

**Cross-reference to round-1 adversarial verdict:** round-1 flagged
A1/A2/A3 as agree-block; all three are now resolved. Adversarial
findings A4 (loader integrity), A5 (bf16 hazard), A6 (bootstrap
sample-id pairing), A7 (bytewise decode round-trip), A8 (fp32 tolerance
— RESOLVED in R2), A10 (batch=1 assertion), A11 (logging order —
already not a finding), A12 (token-count shim), A13/A14 (non-findings),
A15 (module-scope assert — RESOLVED in R2) are either resolved or
remain non-blocking as before. No adversarial upgrades from nit-to-block
in R2.
