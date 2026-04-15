# Codex Review — Phase B Rewrite (Pass 1, Normal)

**Header — fallback disclosure:** The `codex:review` slash-command is NOT
available in this Claude Code sandbox (no `codex` entry on PATH, no
`~/.claude/commands/codex` alias, `agents/codex-reviewer.md` defines only
`/codex:review` which is absent at runtime). This artifact is a
**codex-equivalent normal review** performed by the reviewer agent with the
Codex-style discipline: high-granularity, prefer false positives, cite file
paths and line numbers. It is explicitly NOT a run of the real Codex xhigh
reviewer.

**Inputs consulted**
- Spec: `notes/specs/phase_b_rewrite.md`
- Writer: `notes/handoffs/writer_phase_b_rewrite.md`
- Watcher: `notes/handoffs/watcher_phase_b_rewrite.json` (verdict PASS)
- Source: `stage1/intervention/patcher.py`, `stage1/intervention/__init__.py`,
  `stage1/run_phase_b.py`, `stage1/utils/wording.py`,
  `stage1/tests/test_phase_b_patcher.py`
- Cross-reference: `stage1/run_phase_a.py`, `stage1/models/composer.py`

---

## Verbatim review

### F1 (BLOCK, correctness) — Phase B import paths are unqualified; breaks `python -m stage1.run_phase_b`

`stage1/run_phase_b.py:43-50` uses unqualified imports:

```python
from utils.config import load_config, setup_logging
from utils.logger import create_run_dir
from utils.wording import FORBIDDEN_PHRASES, check_artifacts_for_forbidden
from data.loader import load_mgsm
from models.composer import load_models, compose_model
from inference.parser import parse_answer
from analysis.evaluator import exact_match
from intervention.patcher import (...)
```

Phase A by contrast uses `from stage1.utils.config import ...` etc.
(`stage1/run_phase_a.py:30-46`). The spec §10 smoke test requires
`python -m stage1.run_phase_b --config ...`. Under `-m stage1.run_phase_b`,
the package root is the repo root and top-level names are `stage1`, `eval`,
`notes`, etc. — there is no top-level `utils`, `data`, `models`,
`inference`, `analysis`, or `intervention` package. The module will raise
`ModuleNotFoundError: No module named 'utils'` at import time, before any
Phase B logic runs. This is a hard blocker for the spec §10 smoke test and
spec §11 acceptance items 2/3/5/9/10/11 (all of which require the run to
complete and produce artifacts).

It works for the unit-test module because `stage1/tests/test_phase_b_patcher.py`
imports only `stage1.utils.wording` and `stage1.intervention.patcher` (not
`run_phase_b`), so the bad import never executes.

Fix: change every Phase B unqualified import to `from stage1.xxx` to match
Phase A. Spec §9 allows modifying `stage1/run_phase_b.py`.

### F2 (BLOCK, correctness) — `assert` at module scope compiles away under `python -O`

`stage1/run_phase_b.py:739`:

```python
assert "restoration effect" in FORBIDDEN_PHRASES
```

Under `python -O` or `PYTHONOPTIMIZE=1`, this assertion is stripped. More
importantly, a failed assertion at *import time* produces an `AssertionError`
that aborts module load — any code path that imports `stage1.run_phase_b` for
testing or for IDE inspection under `-O` will behave inconsistently vs.
default mode. The watcher flagged this as "low", but for a codex-discipline
review this is a medium correctness issue: import-time assertions that
exercise an invariant of another module are the wrong mechanism. Either
(a) delete it (the invariant is already enforced by
`test_forbidden_phrases_gate` which iterates `FORBIDDEN_PHRASES`) or
(b) move it into the test module.

### F3 (BLOCK, repro) — `find_latest_phase_a_summary` glob is CWD-relative

`stage1/run_phase_b.py:125-141`:

```python
candidates = sorted(
    glob.glob("stage1/outputs/phase_a/run_*/phase_a_summary.json"),
    reverse=True,
)
```

`_git_sha` a few lines above computes `_repo_root` from `__file__` and uses
it as `cwd`; this function does not. If the user launches from any directory
other than the repo root (e.g. `cd stage1 && python -m run_phase_b`, or a
scheduled job with an absolute config path and unspecified CWD), the glob
returns `[]` silently and the cross-check (spec §7, §11.7 — which is an
ACCEPTANCE criterion) is downgraded to a skip — `checks.append(("Phase A
cross-check skipped (no prior summary)", True))`. This turns a numeric
acceptance check into a pass-by-absence. Spec §11.7 explicitly requires
`|no_patch_accuracy − phase_a.hard_swap_b8_accuracy| ≤ 0.008` to be verified,
not waived.

Fix: resolve the glob relative to `_repo_root`:

```python
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pattern = os.path.join(repo_root, "stage1", "outputs", "phase_a",
                       "run_*", "phase_a_summary.json")
candidates = sorted(glob.glob(pattern), reverse=True)
```

### F4 (NIT, correctness) — `_annotate_correct` uses `dict.update(parsed)`; silently overwrites

`stage1/run_phase_b.py:212-224`:

```python
parsed = parse_answer(r["output_text"])
r.update(parsed)
```

If `parse_answer` ever returns a dict that contains any of the existing keys
`output_text`, `patched_layers`, `patch_name`, `direction`, `methodology`,
`sample_id`, `patch_condition`, they will be silently overwritten. This is
not a current bug (the current `parser.parse_answer` keys are scoped), but
it's load-bearing on implementation detail of a file that is on the
DO-NOT-MODIFY list (`stage1/inference/parser.py`). A defensive guard is
cheap:

```python
for k, v in parsed.items():
    if k in r and k not in ("correct",):
        continue
    r[k] = v
```

Non-blocking; flag for future hardening.

### F5 (NIT, correctness) — `eos_token_id` early-stop uses `int(current.item())` against an integer

`stage1/intervention/patcher.py:319-321`:

```python
for step in range(remaining):
    if eos_token_id is not None and int(current.item()) == int(eos_token_id):
        break
```

`current` at step 0 is `first_token_id`, and `first_token_id` was produced
by `argmax` over the lm_head logits. If that very first token IS eos,
`_greedy_continue_with_cache` returns a 1-token sequence containing eos —
which is correct. Fine. But note that when `eos_token_id` is a **list**
(some tokenizer configs expose a list for eos to cover multiple stop ids),
`int(eos_token_id)` raises `TypeError`. Qwen2.5 tokenizer eos is a single
int so this doesn't bite today; flagging for the general case.

### F6 (NIT, correctness) — `use_cache=(cache is not None)` does not guarantee the layer writes to cache

`stage1/intervention/patcher.py:225-234`:

The manual forward passes `past_key_values=cache, use_cache=(cache is not None)`
into each Qwen2 decoder layer. The layer-level contract in transformers
4.45+/5.x is that with `use_cache=True` and a `DynamicCache` passed in, the
layer appends via `past_key_value.update(key_states, value_states, layer_idx,
cache_kwargs)`. This relies on the layer's own `layer_idx` being set at
construction time (`Qwen2DecoderLayer.__init__` takes `layer_idx`). That's
standard for Qwen2. However, the patcher never verifies this — if a future
transformers refactor changes the cache-update contract, the bytewise-equal
test (#2) will catch it, but nothing in production code fails fast with a
descriptive message. Defensive assertion after the loop would help, e.g.
`assert len(cache) == n_layers` if the DynamicCache exposes `__len__`.
Non-blocking; test #2 provides the safety net.

### F7 (NIT, perf/memory) — Unconditional per-layer CPU offload in `forward_with_patches`

`stage1/intervention/patcher.py:251`:

```python
all_outputs.append(hidden.detach().to("cpu"))
```

This is on the critical path for every patched forward; on GPU it's a D2H
copy of `[1, S, H]` × 28 layers per sample. For 2,750 sample-runs in the
full Phase B (spec §13) that is ~77,000 D2H syncs. Writer marked this as
non-critical in the watcher; the `all_outputs` return value is only used by
`test_all_clean_patch_matches_recipient`. The production path in
`run_patched_inference_single` discards it (`_` in
`final_hidden, _, cache = forward_with_patches(...)`). Making the offload
conditional (`return_outputs_device` kwarg, default-off) would save
measurable wall-clock at zero semantic cost. Non-blocking but worth filing.

### F8 (NIT, test) — `test_empty_patch_generate_bytewise_equal` round-trips through string decode

`stage1/tests/test_phase_b_patcher.py:221`:

```python
patched_new = [int(x) for x in result["output_text"].split()]
```

The test claims byte-equality on token IDs but actually parses IDs out of
the DummyTokenizer's space-joined decode. The DummyTokenizer's `decode` is
bijective-ish (every integer becomes its base-10 string, joined by spaces),
but the test contract would be stronger if `run_patched_inference_single`
exposed a `return_ids` hook (or if the test called
`_greedy_continue_with_cache` directly to retrieve token IDs). The watcher
flagged the same concern (med severity). I agree med, not block: the current
DummyTokenizer round-trip IS bijective, so the assertion holds for the
current code. If anyone later changes the decode join character or the
id-to-string mapping, the test goes silently green on non-equal IDs. Worth
hardening.

### F9 (NIT, other) — `Phase A summary loader` may accept stale runs silently

`stage1/run_phase_b.py:125-141`: `_load_latest_phase_a_summary` picks the
newest `phase_a_summary.json` by directory-name sort (run_<timestamp>).
If a broken/experimental Phase A run directory exists (e.g., a partial
rerun that wrote summary.json but recorded garbage accuracies), Phase B
will cross-check against garbage and either falsely PASS or falsely FAIL.
Spec §4 says Phase B should pin to the **same** Phase A manifest that Phase
A emitted; there is no integrity check that the selected run is the one
the user intended. A `--phase-a-run` CLI override (pointing at a specific
run_dir) would be strictly better. Non-blocking; reasonable default, but
surfacing the chosen Phase A path in `phase_b_summary.json.phase_a_cross_check`
would help provenance.

### F10 (NIT, repro) — `numpy` import at module top without a requirements declaration

`stage1/run_phase_b.py:39`: `import numpy as np`. numpy is a transitive dep
of torch, so the import works today. Spec §13 does not list numpy as a
new top-level dep; writer noted the same. Watcher flagged it as low. I
agree low, not block.

---

## Triage

| codex_finding_id | claude_label           | reasoning |
|------------------|------------------------|-----------|
| F1               | agree-block            | Confirmed against spec §10 smoke-test invocation `python -m stage1.run_phase_b ...`. `stage1/run_phase_b.py:43-50` uses unqualified `from utils.config import ...` while the module must be importable as `stage1.run_phase_b`. Phase A at `stage1/run_phase_a.py:30` uses `from stage1.utils.config import ...` — same repo layout, so this is a writer oversight. Hard blocker for spec §11.2/§11.3/§11.5. |
| F2               | agree-nit              | Confirmed at `stage1/run_phase_b.py:739`. Real but low-impact; codex-discipline upgrade to BLOCK is defensible but the test in `test_phase_b_patcher.py:63` already enforces the invariant, so functionally it is belt-and-suspenders. Labeling as nit (Codex agree-block rule does not apply — this is reviewer's own finding, not Codex's). |
| F3               | agree-block            | Confirmed at `stage1/run_phase_b.py:131-134`. Spec §11.7 is an ACCEPTANCE criterion with a numeric tolerance (0.008); silently skipping under a non-repo-root CWD converts the acceptance check into a pass-by-absence. Writer also flagged this in `open_questions_for_watcher`. Must-fix before the full run. |
| F4               | agree-nit              | Defensive hardening; non-blocking per writer's note on parser being on the DO-NOT-MODIFY list. |
| F5               | agree-nit              | Hypothetical (Qwen2.5 eos is scalar); flagging for completeness. |
| F6               | agree-nit              | Covered by test #2 empirically. |
| F7               | agree-nit              | Perf only; spec §13 budget not violated. |
| F8               | already-known          | Watcher raised the same concern (med, line 221); writer acknowledged. Non-blocking. |
| F9               | agree-nit              | Provenance improvement, not a correctness bug. |
| F10              | already-known          | Watcher and writer both flagged; agreed low. |

---

**Summary of must-fix-before-merge (from this pass):** F1, F3.
**F2 recommended-fix** (strict reading; not strict-blocker since the test
enforces the invariant).
