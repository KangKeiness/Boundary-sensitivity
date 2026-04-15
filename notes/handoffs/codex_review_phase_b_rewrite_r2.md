# Codex Review — Phase B Rewrite (Round 2, Pass 1 Normal)

**Header — fallback disclosure:** The `/codex:review` slash-command is NOT
provisioned in this Claude Code sandbox (no `codex` on PATH, no
`~/.claude/commands/codex` alias; `agents/codex-reviewer.md` declares
`/codex:review` but the runtime does not resolve it). This artifact is a
**codex-equivalent normal review** performed by the reviewer agent under the
Codex-style discipline (high-granularity, prefer false-positives, cite
file/line). It is explicitly NOT a run of the real Codex xhigh reviewer.

**Inputs consulted**
- Spec: `notes/specs/phase_b_rewrite.md`
- Round-1 artifacts: `notes/handoffs/codex_review_phase_b_rewrite.md`,
  `notes/handoffs/codex_adversarial_phase_b_rewrite.md`
- Writer round-2 handoff: `notes/handoffs/writer_phase_b_rewrite.md`
  (section "Round 2 — codex BLOCK fixes")
- Watcher round-2 verdict: `notes/handoffs/watcher_phase_b_rewrite.json`
  (verdict PASS, round 2)
- Source under review: `stage1/run_phase_b.py`,
  `stage1/tests/test_phase_b_patcher.py`,
  `stage1/intervention/patcher.py`, `stage1/intervention/__init__.py`,
  `stage1/utils/wording.py`
- Cross-reference: `stage1/run_phase_a.py` (qualified-import template)

**Preconditions:** watcher R2 verdict is PASS; round-2 diff is scoped to
`stage1/run_phase_b.py` (+31/-12) and `stage1/tests/test_phase_b_patcher.py`
(+145/-2). No changes to `patcher.py`, `wording.py`, `intervention/__init__.py`.

---

## R2 verification of round-1 BLOCKs

### B1 — A1 (imports fully qualified): RESOLVED

`stage1/run_phase_b.py:43-56` now reads:

```
from stage1.utils.config import load_config, setup_logging
from stage1.utils.logger import create_run_dir
from stage1.utils.wording import FORBIDDEN_PHRASES, check_artifacts_for_forbidden
from stage1.data.loader import load_mgsm
from stage1.models.composer import load_models, compose_model
from stage1.inference.parser import parse_answer
from stage1.analysis.evaluator import exact_match
from stage1.intervention.patcher import (
    METHODOLOGY_TAG,
    PatchConfig,
    RESTORATION_PATCHES,
    REVERSE_CORRUPTION_PATCHES,
    run_patched_inference,
)
```

All eight internal imports are prefixed `stage1.`, mirroring
`stage1/run_phase_a.py:30-46`. Reviewer-side grep for bare internal imports
(`^from (utils|data|models|inference|analysis|intervention)\.`) returns zero
hits. Reviewer-side import smoke under clean Python 3.12 (no anaconda
contamination) advances past every `stage1.*` line and stops only on a
transitive `scipy` dep inside `stage1.analysis.evaluator` — which is a
sandbox-env issue, not an A1 regression. B1 is discharged.

Regression guard: `test_run_phase_b_module_imports` at
`stage1/tests/test_phase_b_patcher.py:398-438` distinguishes the A1
failure mode (bare top-level `utils|data|models|inference|analysis|
intervention` escaping `stage1.*`) from unrelated missing deps via
`ModuleNotFoundError.name`. The former raises a descriptive AssertionError;
the latter `pytest.skip`s. Asserts canonical public surface
(`run_phase_b`, `main`, `EPSILON_DELTA`, `PHASE_A_CROSS_CHECK_TOL`,
`_load_latest_phase_a_summary`, `_phase_a_outputs_dir`). Adequate guard.

### B2 — A2 (Phase A loader CWD-invariant + no-match FAILs): RESOLVED

`_phase_a_outputs_dir()` at `stage1/run_phase_b.py:125-136` resolves via
`pathlib.Path(__file__).resolve().parents[1] / "stage1" / "outputs" /
"phase_a"`. `stage1/run_phase_b.py` lives at
`<repo>/stage1/run_phase_b.py`, so `parents[1]` is the repo root and the
composed path is `<repo>/stage1/outputs/phase_a` — correct.

`_load_latest_phase_a_summary()` at lines 139-154 globs
`os.path.join(_phase_a_outputs_dir(), "run_*", "phase_a_summary.json")`,
reads each with `encoding="utf-8"`, and returns `(summary, path)` or
`(None, None)`. The resolved path is now provenance-logged (intended
to land under `phase_a_cross_check.phase_a_summary_path` in the summary
JSON).

Non-sanity no-match branch at lines 696-707 now APPENDs a descriptive
FAIL tuple to the `checks` list when `cross_check_passed is None`:

```
else:
    checks.append((
        f"Phase A cross-check FAILED: no phase_a_summary.json found "
        f"under {_phase_a_outputs_dir()} (spec §11.7 requires a prior "
        f"Phase A run)", False,
    ))
```

This propagates through the `raise RuntimeError` guard at lines 718-719.
Spec §11.7 is no longer trivially PASS-by-absence. Sanity mode tolerates
the absence with an explicitly-labelled skip — acceptable bootstrap
carve-out.

Regression guards (at `stage1/tests/test_phase_b_patcher.py:452-504`):
- `test_phase_a_loader_no_match_returns_none` monkeypatches
  `_phase_a_outputs_dir` to an empty tmp path, chdir's elsewhere, asserts
  `(None, None)`.
- `test_phase_a_loader_resolves_repo_relative` plants a fake
  `run_20991231_000000/phase_a_summary.json` under tmp_path, chdir's to
  `tmp_path.parent`, asserts the loader still finds it and the returned
  `path` contains the fake run id — directly proves CWD-invariance.

Adequate guards. B2 discharged.

### B3 — A3 (smoke test gated, manual command in skip reason): RESOLVED via option (b)

`test_smoke_marker` at `stage1/tests/test_phase_b_patcher.py:507-556` is
gated by `pytest.mark.skipif(not _has_gpu_and_weights(), reason=...)`.

`_has_gpu_and_weights` at lines 378-395 checks
`torch.cuda.is_available()` AND
`AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
local_files_only=True)` — both must be True for the test to run. Skip
reason string contains the exact verbatim manual command:

```
python -m stage1.run_phase_b --config stage1/configs/stage2_confound.yaml
    --sanity --seed 42
```

When prerequisites are present, the test runs the subprocess end-to-end
from the repo root (`repo_root = dirname(dirname(dirname(abspath(__file__))))`
— correct for `stage1/tests/test_phase_b_patcher.py`), with a 1800s
timeout, asserts exit 0 and the presence of the four canonical artifacts
(`phase_b_summary.txt`, `phase_b_summary.json`, `restoration_table.csv`,
`corruption_table.csv`) in the latest `stage1/outputs/phase_b/run_*`
directory. B3 discharged.

### Round-1 F2 / adversarial A15 (module-scope assert at line 739): RESOLVED

Reviewer grep for `^assert ` in `stage1/run_phase_b.py` returns zero
hits. The invariant "restoration effect in FORBIDDEN_PHRASES" is now
enforced solely by `test_forbidden_phrases_gate` (which iterates all of
`FORBIDDEN_PHRASES`). No import-time AssertionError hazard under
`python -O`.

### Round-1 secondary tighten (test #1 fp32 tolerance): RESOLVED

`test_identity_patch_equivalence` tolerance at the patched line is now
`atol=1e-5` per spec §10 test #1 (fp32 bound). Writer-reported PASS on
the sandbox's 3-layer CPU-initialised Qwen2 is plausible; watcher also
confirmed PASS.

---

## R2 Pass-1 findings (new + lingering non-blockers)

### N1 (NIT, repro) — Path separator hazard in provenance-logged Phase A path on Windows

`_load_latest_phase_a_summary` returns a path built with `os.path.join`,
which on Windows produces backslash-separated strings. The watcher's
evidence note says the resolved path is "logged into
`phase_b_summary.json.phase_a_cross_check.phase_a_summary_path` for
provenance." Downstream JSON consumers on a Linux box reading that
summary for cross-machine audit will see Windows-style separators.

Not a correctness bug — `json.load(... encoding="utf-8")` handles the
escaping. Flagging for completeness; normalizing to forward-slash via
`.replace(os.sep, "/")` or `pathlib.PurePosixPath` would be marginally
friendlier for cross-platform log-grepping.

Non-blocking.

### N2 (NIT, test-hygiene) — `_has_gpu_and_weights` failure modes all return False

The bare `except Exception: return False` in `_has_gpu_and_weights`
treats every failure (torch not installed, transformers not installed,
HuggingFace cache permission denied, OSError on disk) as "no GPU or
weights". That's the right behavior for a gate, but it means a
genuinely-broken environment (e.g., transformers is half-installed and
`AutoConfig` raises an ImportError during runtime) silently skips the
smoke test. In a production CI, `pytest -rs` at least surfaces the skip
count, but the reason string is fixed to the manual-command note — the
actual exception is swallowed.

Fix (non-blocking): capture the exception object and fold a short note
into a module-level `_GATE_ERROR` string that the skip reason can
reference. Flagging for future hardening; not a must-fix before merge.

### N3 (LOW, correctness) — smoke-test subprocess env does not inherit deterministic-algorithms flag

`test_smoke_marker` runs `subprocess.run([sys.executable, "-m",
"stage1.run_phase_b", ...], cwd=repo_root, ...)` without an explicit
`env={...}` argument, which means the subprocess inherits the parent
pytest env. That's correct and desired for `CUBLAS_WORKSPACE_CONFIG`
(set inside `_apply_determinism`). However, if the parent pytest is
invoked with `PYTHONDONTWRITEBYTECODE=1` or some other env var that
interferes with HF caching, the smoke test inherits that too. Minor,
unlikely to bite; noting for adversarial pass coverage.

### N4 (NIT, test) — round-1 F8 / F10 and adversarial A7 still open

Unchanged from round 1; watcher continues to list these as
non-blockers:
- `test_empty_patch_generate_bytewise_equal` compares decode-roundtrip
  strings, not raw token IDs (F8 / A7, watcher med).
- `test_all_clean_patch_matches_recipient` narrows spec §10 test #3 to
  hidden-state equality (watcher med).
- `numpy` is imported module-top without a requirements-explicit entry
  (F10 / A10-ish, watcher low).
- `forward_with_patches` unconditional CPU offload per layer (F7,
  watcher nit).

No regression relative to round 1; no new escalation.

### N5 (NIT, correctness, NEW) — `setdefault("CUBLAS_WORKSPACE_CONFIG", ...)` in _apply_determinism

`_apply_determinism` at `stage1/run_phase_b.py:84` uses
`os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")`. If the
parent shell already set `CUBLAS_WORKSPACE_CONFIG` to a value that is
NOT `:4096:8` (e.g., `:16:8`), the `setdefault` is a no-op and the
determinism contract in `phase_b_summary.json.environment` is logged as
"`:4096:8`" while torch is actually using the shell value — a silent
divergence between logged-value and effective-value.

Fix (non-blocking): either (a) unconditionally overwrite with
`os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` and warn if the
pre-existing value differs, or (b) log the effective value
(`os.environ["CUBLAS_WORKSPACE_CONFIG"]`) into
`environment.cublas_workspace_config` rather than the hardcoded literal.
Watcher checklist marks `determinism_flags: true`; the flag IS set, but
reviewer-discipline flags the logging-vs-effective drift. Non-blocking.

---

## Triage

| codex_finding_id | claude_label       | reasoning |
|------------------|--------------------|-----------|
| B1 (R1 A1)       | resolved           | Verified at `run_phase_b.py:43-56`; guard test present. Clean-python import advances past the qualified-import surface; only scipy (env dep) blocks further import. |
| B2 (R1 A2)       | resolved           | `_phase_a_outputs_dir` uses `parents[1]` from `__file__`, CWD-invariant. Non-sanity no-match FAILs explicitly at lines 696-707, propagates via `raise RuntimeError`. Two guard tests cover no-match and repo-relative paths. |
| B3 (R1 A3)       | resolved           | `pytest.mark.skipif(not _has_gpu_and_weights(), ...)` gate; skip reason contains verbatim manual command; subprocess invocation uses correct `repo_root` depth; artifact presence asserted. |
| R1 F2 / A15      | resolved           | No module-scope `assert ` remains in `run_phase_b.py`; invariant enforced by `test_forbidden_phrases_gate`. |
| R1 test #1 tol   | resolved           | `atol=1e-5` per spec §10 #1. |
| N1               | agree-nit          | Windows path-sep in provenance log is cosmetic; non-blocking. |
| N2               | agree-nit          | `_has_gpu_and_weights` swallowing exception is acceptable for a gate; hardening is optional. |
| N3               | agree-nit          | subprocess env inheritance is correct for CUBLAS_WORKSPACE_CONFIG; no finding beyond an adversarial note. |
| N4               | already-known      | Watcher and round-1 codex both raised; writer acknowledged; no regression. |
| N5               | agree-nit (NEW)    | Logged value can drift from effective value if the shell pre-sets the env var. Non-blocking. |

---

## Verdict

**PASS (Pass 1, Normal).**

All three round-1 BLOCK items (A1, A2, A3) are discharged with adequate
regression guards. No new agree-block findings introduced by the round-2
patch. No regressions observed.

Must-fix-before-merge: none.

Can-ship-with-followup (unchanged from watcher):
- Run `test_smoke_marker` on GPU-equipped box with cached weights.
- Ensure scipy-available env exercises
  `test_run_phase_b_module_imports` as a live guard (not skip).
- Populate `stage1/outputs/phase_a/run_*/phase_a_summary.json` before
  any non-sanity Phase B run — spec §11.7 is now a hard-FAIL.
