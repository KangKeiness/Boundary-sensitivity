# runtime_repro_v5

## 1. Goal
Make the Stage 1 / Phase A execution path reliably runnable and reproducible
across the two documented module invocation styles (`python -m stage1.run` and
`import stage1.run`), under non-UTF-8 OS locales (Windows cp949), and across
Phase A no-swap reuse against the current sample-regime parity contract. This
is a pure runtime-hardening sprint; no science, no metrics, no new behavior.

## 2. Hypothesis and falsification
Hypothesis: four narrow plumbing defects are the sole remaining runtime
blockers, namely (a) bare non-package-qualified imports in `stage1/run.py`,
(b) locale-dependent YAML loading in `stage1/utils/config.py`, (c) a
sample-regime-less parity block in the Phase A no-swap reuse path, and
(d) absence of smoke coverage for these three paths.

Falsification: after fix, any of the following still fails —
  - `python -c "import stage1.run"` exits non-zero
  - `python -m stage1.run --help` exits non-zero
  - loading `stage1/configs/stage2_confound.yaml` under `PYTHONUTF8=0`
    (cp949 ambient) raises `UnicodeDecodeError`
  - `run_phase_a` with `--reuse-no-swap-dir` pointing at a v4+ manifest
    raises `ManifestParityError` for `sample_regime.*` missing-in-current

## 3. Prior art and delta
N/A — pure runtime hardening, RED LIGHT repair follow-up. This spec is the
plumbing closure for the v4 PRIORITY 2 sample-regime parity addition (see
`notes/boundary_sensitivity_final_blocking_fix_prompt_v4.md`) and the earlier
RED LIGHT P2 manifest parity work, neither of which updated the Phase A
reuse code path to match.

## 4. Datasets
N/A — unchanged. MGSM-zh, 250 samples, same split and ordering. No data file
is read, hashed, or re-loaded as part of this fix.

## 5. Models and tokenizers
N/A — unchanged. Qwen2.5-1.5B-Instruct / Qwen2.5-1.5B at pinned revisions per
`stage1/configs/stage2_confound.yaml`.

## 6. Training config
N/A — unchanged. Decoding defaults (`do_sample=False`, `temperature=0.0`,
`max_new_tokens=512`), prompt template, and `Solution:` behavior MUST NOT
change.

## 7. Evaluation protocol
N/A — unchanged. No metric, no evaluator, no parser, no BDS path touched.

## 8. Interfaces to add or change

### 8.1 `stage1/run.py`
No signature changes. Only import statements at L11–L25 are rewritten from
bare form to `stage1.`-qualified form.

### 8.2 `stage1/utils/config.py`
`load_config(config_path: str) -> Stage1Config` — signature unchanged. Only
the `open(path)` call at L101 gains `encoding="utf-8"`.

### 8.3 `stage1/run_phase_a.py`
```
def _load_reused_no_swap(
    reuse_dir: str,
    samples: List[Dict],
    config=None,
) -> Tuple[Dict[str, torch.Tensor], List[Dict], List[float]]:
```
Signature unchanged. Internal body at L344 changes from
`extract_parity_block(config)` to
`extract_parity_block(config, sample_ids=[s["sample_id"] for s in samples])`.
No new parameter — `samples` is already in scope.

No other public API changes. Reviewer-check: `extract_parity_block` in
`stage1/utils/manifest_parity.py` already accepts `sample_ids=...`; no
helper addition is required there.

## 9. Files-to-touch (exhaustive)

| Path | Action | Rationale |
|------|--------|-----------|
| `stage1/run.py` | modify | Rewrite L11–L25 bare imports to `stage1.xxx` package-qualified form so `python -m stage1.run` and `import stage1.run` resolve. |
| `stage1/utils/config.py` | modify | Add `encoding="utf-8"` to `open(path)` on L101 of `load_config`. Short comment: UTF-8 enforced so config loading is locale-independent on Windows cp949. |
| `stage1/run_phase_a.py` | modify | In `_load_reused_no_swap` at L344, pass `sample_ids=[s["sample_id"] for s in samples]` to `extract_parity_block(config)` so the reuse parity block matches the v4 sample-regime contract and aligns with the fresh-manifest write at L781–L783. |
| `stage1/tests/test_runtime_smoke.py` | add | New smoke/regression module covering the four previously uncovered runtime paths (import, `-m --help`, UTF-8 config load, reuse parity regression). |

**Do-not-touch confirmation** (HARD list; read-only through this fix):
- `stage1/inference/runner.py`
- `stage1/inference/parser.py`
- `stage1/analysis/bds.py`
- `stage1/analysis/evaluator.py`
- `stage1/data/loader.py`
- `stage1/utils/manifest_parity.py` — schema already correct; do NOT weaken
  `_PARITY_FIELDS`, do NOT add `sample_regime.*` to `_NULL_EQUIV_PATHS`, do
  NOT add new helpers.
- `stage1/configs/stage2_confound.yaml` — verified ASCII, no edit needed.
- `stage1/__init__.py`, `stage1/utils/__init__.py` — empty package markers
  already present; do NOT add code.

## 10. Test plan

All tests live in `stage1/tests/test_runtime_smoke.py` (new file) and must
be torch-free / GPU-free to run on any developer box.

### 10.1 `test_stage1_run_importable`
`import importlib; importlib.import_module("stage1.run")` must not raise.

### 10.2 `test_stage1_run_m_help_exit_zero`
`subprocess.run([sys.executable, "-m", "stage1.run", "--help"], cwd=<repo_root>,
check=True)` must complete with returncode 0 within a short timeout.
`cwd` must be the repo root, NOT `stage1/`, so any regression to bare
imports is caught.

### 10.3 `test_load_config_utf8`
`stage1.utils.config.load_config("stage1/configs/stage2_confound.yaml")`
completes and returns a valid `Stage1Config` with `config.dataset.lang == "zh"`.
Asserts `encoding="utf-8"` is used by inspecting the source of `load_config`
via `inspect.getsource` — the string `encoding="utf-8"` must appear in the
function body. (Source-string belt-and-braces on top of the behavioral call.)

### 10.4 `test_phase_a_reuse_parity_has_sample_regime`
Builds a tmp reuse dir with a stub `hidden_states_no_swap.pt`
(torch-free fallback: write an empty `.pt` path behind a patched
`torch.load`), a stub `results_no_swap.jsonl` with 3 sample ids, and a
`manifest.json` whose `parity` block was produced by
`extract_parity_block(cfg, sample_ids=[...])` — i.e. includes the full
`sample_regime` sub-dict. Calls `_load_reused_no_swap(tmp_dir, samples, config=cfg)`
and asserts no exception. Then mutates the manifest to drop
`sample_regime` and asserts `ManifestParityError` is raised with a message
mentioning `sample_regime.mode`. This test must fail on pre-fix code
(the pre-fix path builds `extract_parity_block(config)` without
`sample_ids`, which produces a target block with NO `sample_regime`, which
causes the check against any v4-compliant source to raise
`"sample regime mode ...: missing in current config"`).

### 10.5 Test commands
```
pytest -x stage1/tests/test_runtime_smoke.py -v
pytest -x stage1/tests/test_sample_regime_parity.py -v   # regression against v4 P2
pytest -x stage1/tests/ -k "parity or runtime" -v        # combined
```

## 11. Acceptance criteria
1. `python -c "import stage1.run"` exits 0.
2. `python -m stage1.run --help` exits 0 and prints usage containing
   `--config`.
3. `python -c "from stage1.utils.config import load_config;
   load_config('stage1/configs/stage2_confound.yaml')"` exits 0 under a
   shell whose active code page is 949 (simulated via
   `PYTHONLEGACYWINDOWSFSENCODING=1` or explicit `encoding="cp949"` not
   interfering — the `open(...)` call has explicit `encoding="utf-8"`, so
   the ambient locale cannot leak in).
4. All four smoke tests in §10 pass.
5. `pytest -x stage1/tests/` completes with no new failures vs the current
   baseline.
6. No diff touches any path in the HARD do-not-modify list (§9).
7. No diff weakens `_PARITY_FIELDS` or `_NULL_EQUIV_PATHS` in
   `manifest_parity.py`.
8. Grep `encoding="utf-8"` appears in `load_config` body.
9. Grep `extract_parity_block(config, sample_ids=` appears inside
   `_load_reused_no_swap`.

## 12. Risks and ablations

### 12.1 Risks specific to the parity fix
- **Risk: false-positive parity failures on legacy reuse dirs.** Manifests
  written before the v4 P2 addition do not have a `sample_regime` block.
  A strict check will refuse to reuse them. This is the INTENDED behavior
  per the v4 P2 design — legacy manifests must be regenerated. Do NOT
  "soften" this by adding `sample_regime.*` to `_NULL_EQUIV_PATHS`. Mitigation:
  document in the commit message that pre-v4 anchor runs must be re-produced.
- **Risk: the fresh-manifest write path and the reuse-read path diverge
  again later.** Mitigation: the smoke test at §10.4 directly compares the
  shape of both blocks, and the test will fail if either side regresses.

### 12.2 General risks
- **Risk: an import-path fix could accidentally break the `if __name__ ==
  "__main__"` path.** Mitigation: the `-m --help` subprocess test covers it.
- **Risk: a UTF-8 change triggers `yaml.safe_load` reading bytes differently
  on files with BOM.** Mitigation: `stage2_confound.yaml` has no BOM (ASCII,
  verified); Python's `open(..., encoding="utf-8")` handles BOM-less UTF-8
  identically to the previous default on UTF-8 locales.

### 12.3 Absolute non-negotiables (verbatim from v5 prompt §6)
- Do not modify parser behavior.
- Do not modify prompt template or `Solution:` behavior.
- Do not change decoding defaults.
- Do not weaken parity checks.
- Do not add unrelated features.
- Do not proceed past a priority block until both internal reviewers return
  green lights.

## 13. Compute budget
Local only, under one minute wall clock. No GPU required for any test in
§10. No model loads, no dataset loads beyond YAML parse.

## 14. Rollback
Each priority is a self-contained commit:
  - commit A: `stage1/run.py` imports
  - commit B: `stage1/utils/config.py` UTF-8
  - commit C: `stage1/run_phase_a.py` reuse parity
  - commit D: `stage1/tests/test_runtime_smoke.py` tests

To roll back any one fix: `git revert <sha>`. The four commits are ordered
A -> B -> C -> D and are independent; reverting C will re-break only the
reuse path and leave the other three intact. If the tests in D catch a
regression after C lands, prefer a forward-fix commit over amending C
(per project git-safety protocol: create NEW commits, do not amend).
