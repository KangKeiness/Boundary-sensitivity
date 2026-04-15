# Codex Adversarial Review — Phase B Rewrite (Pass 2)

**Header — fallback disclosure:** The `codex:adversarial-review` slash-command
is NOT available in this Claude Code sandbox (no `codex` on PATH; no
`~/.claude/commands/codex` provisioning). This artifact is a
**codex-equivalent adversarial review** performed by the reviewer agent
under the discipline "assume the diff is wrong until proven otherwise",
biased toward false positives over false negatives. It is explicitly NOT a
run of real Codex xhigh in adversarial mode.

**Inputs consulted**
- Pass 1 artifact: `notes/handoffs/codex_review_phase_b_rewrite.md`
- Watcher: `notes/handoffs/watcher_phase_b_rewrite.json` (verdict PASS)
- Writer: `notes/handoffs/writer_phase_b_rewrite.md`
- Spec: `notes/specs/phase_b_rewrite.md`
- Source: full surface listed in Pass 1.

Assumption framing: every positive claim in writer + watcher is provisional.
Every "passes" is read as "passes the tests that were written"; the tests
themselves are subject to adversarial scrutiny. The 7-passing pytest result
does NOT discharge the spec §10 smoke-test requirement, which was
explicitly deferred to watcher (writer `deviations_from_spec` §4).

---

## Verbatim adversarial review

### A1 (BLOCK, correctness) — Import-path regression breaks the spec's canonical invocation

Same root cause as Pass 1 F1 but reframed adversarially: the writer wrote
tests that exercise `stage1.intervention.patcher` and `stage1.utils.wording`
directly, and those tests pass. Neither test imports `stage1.run_phase_b`.
Therefore the 7/7 pytest result provides **zero evidence** that
`run_phase_b.py` even imports successfully. The diff introduces an
import-path regression relative to Phase A (`stage1/run_phase_b.py:43-50`
uses unqualified `from utils.config import ...`, while
`stage1/run_phase_a.py:30-46` uses `from stage1.utils.config`), and the
only code path that would have caught this is the spec §10 smoke test
(`python -m stage1.run_phase_b --config ...`), which writer explicitly did
not run. This is a latent ModuleNotFoundError in the merge.

Adversarial upgrade rationale: the watcher noted "7 passed per writer" in
the `matched` list (watcher line 28), treating the pytest pass as evidence
of acceptance. Under codex adversarial framing, green unit tests for
non-imported modules MUST NOT be counted as evidence that the importing
module works. BLOCK.

### A2 (BLOCK, acceptance) — Pass 1 F3 re-escalated: acceptance criterion silently waived

Repeating Pass 1 F3 at BLOCK: spec §11.7 (`|no_patch_acc −
phase_a.hard_swap_b8| ≤ 0.008`) is an acceptance criterion. The
implementation's CWD-relative glob at `stage1/run_phase_b.py:131-134` will
return `[]` from any non-repo-root CWD; the run then emits
`"Phase A cross-check skipped (no prior summary)"` (line 678) and the
sanity-check block marks the skip as PASS (line 678). This converts a
numeric acceptance check into "pass by absence." Adversarially: an attacker
(or a hurried operator) who launches the job from `stage1/` or from `/tmp/`
will get a green run that trivially satisfies §11.7 without ever comparing
to Phase A. BLOCK.

### A3 (BLOCK, test-coverage) — Test #3 narrowed scope not recorded in acceptance artifact

`stage1/tests/test_phase_b_patcher.py:232-301` implements
`test_all_clean_patch_matches_recipient` as a hidden-state-equality
assertion on a 3-layer random-weight Qwen2 pair. The spec §10 test #3
requires a `≥ 95% of 5 fixture prompts` token-sequence-match sub-claim
against pretrained Qwen2.5-1.5B. Writer's deviation #3 documents the
narrowing: "the 5-fixture-prompt clause requires a pretrained Qwen2.5-1.5B
download which is out of scope for a sandboxed unit test." This is a
legitimate sandbox limitation — but nothing in the CURRENT acceptance path
compensates for the missing sub-claim in a real-weights environment. Spec
§10 smoke test is the backstop, and that has not yet been run. Under
adversarial discipline, a narrowed test + a deferred smoke test = an
acceptance gap, not a satisfied criterion. BLOCK the "ready to merge"
status until the smoke test runs on real weights.

Watcher raised this as `severity: med` (watcher finding line 36-42).
Adversarial upgrade: BLOCK of merge (not BLOCK of writer rewrite).

### A4 (MED, correctness) — `_load_latest_phase_a_summary` accepts the freshest JSON without integrity check

`stage1/run_phase_b.py:125-141`: the loader picks the newest matching file
and fails soft on exceptions. Any partial/corrupted prior run that wrote a
valid-JSON summary with garbage numbers will silently become the reference.
Adversarially: if a developer ran a broken Phase A once with a stale
checkpoint, then ran Phase A correctly, and the broken run was later (the
timestamp happens to be newer for any reason — manual mv, clock skew on a
shared box, etc.), the Phase B cross-check uses the wrong reference. Spec
§4 implies the dataset manifest is pinned to Phase A's emitted manifest,
but nothing enforces "the Phase A run I care about."

Fix: accept a `--phase-a-run <path>` CLI override that skips the glob
entirely; log the resolved path to `phase_b_summary.json.phase_a_cross_check.source_path`.
Non-blocking but noted adversarially.

### A5 (MED, repro) — `hash` of state_dict via `numpy.tobytes` assumes fp16 numpy compat

`stage1/run_phase_b.py:111-120`: `t.detach().to("cpu").contiguous().numpy().tobytes()`.
Spec §5 pins weights in fp16. `numpy` supports float16 since 1.6, so this
works on every conceivably-installed numpy. However, future config drift
to bf16 (which numpy < 1.24 does NOT support; `.numpy()` will raise
`TypeError: Got unsupported ScalarType BFloat16`) will break the hash
silently for the two-phase compare. Not an immediate bug; record as
hazard.

Also: `.to("cpu")` on a large state_dict is a full D2H copy of the entire
1.5B-parameter model. This runs twice per Phase B full invocation (before
inference and after). On a shared GPU box this is a multi-GB memory spike
that may OOM if CPU RAM is tight. Spec §13 budgets "6 GB weights" CPU
RAM; this adds ~3 GB momentarily for the hash. Non-blocking.

### A6 (MED, correctness) — Bootstrap pairing assumes "sample order matches across conditions by construction"

`stage1/run_phase_b.py:521-525`:

```python
best_corr = [int(x.get("correct", False))
             for x in restoration_results[best["condition"]]]
bl_corr = [int(x.get("correct", False))
           for x in restoration_results["patch_boundary_local"]]
# Pair on sample ordering, which matches across conditions by construction.
point, ci_lo, ci_hi = _paired_bootstrap_diff_ci(best_corr, bl_corr, ...)
```

The claim "matches across conditions by construction" depends on:
1. `samples` being the same object across calls (yes — Phase B passes the
   same `samples` list to every `run_patched_inference`).
2. `run_patched_inference` iterating that list in order (yes — line 481:
   `for idx, sample in enumerate(samples)`).
3. No condition dropping samples (yes — no early-continue in the loop).

So the invariant holds today. Adversarially: there is no `sample_id`-based
pairing; if ANY of (1)/(2)/(3) change in a future refactor, the bootstrap
will silently pair mismatched samples and the CI becomes meaningless.

Fix: pair by `sample_id`:

```python
bl_by_id = {r["sample_id"]: int(r.get("correct", False))
            for r in restoration_results["patch_boundary_local"]}
best_by_id = {r["sample_id"]: int(r.get("correct", False))
              for r in restoration_results[best["condition"]]}
shared = sorted(set(bl_by_id) & set(best_by_id))
bl_corr = [bl_by_id[s] for s in shared]
best_corr = [best_by_id[s] for s in shared]
```

Non-blocking (invariant holds today); hardening for future-proofing.

### A7 (MED, test) — `test_empty_patch_generate_bytewise_equal` byte-equality is a decode round-trip, not a token-ID comparison

Pass 1 F8 / Watcher line 43 flagged this. Adversarially it is MED: spec
§10 test #2 literally says "byte-identical on token IDs." The test asserts
equality on ints parsed out of a dummy-decoded whitespace-joined string.
If `DummyTokenizer.decode` ever used a delimiter that collided with the
`str()` of an integer, equality would silently go green on non-equal IDs.
The test is **brittle** in a way the spec wording does not anticipate.

Fix: adjust `run_patched_inference_single` to optionally return raw IDs,
OR rewrite the test to call `_greedy_continue_with_cache` directly and
compare the returned LongTensor.

### A8 (MED, correctness) — `test_identity_patch_equivalence` tolerance is permissive

`stage1/tests/test_phase_b_patcher.py:146`: `assert max_abs_diff < 1e-4`
with a 3-layer fp32 CPU model. Spec §2 says
`max-abs-diff < 1e-4` in fp16, `< 1e-5` in fp32. Writer's deviation #2
explains the choice but the fp32 tolerance is literally **10×** looser than
the spec. On a 3-layer fp32 model the diff should be near machine epsilon;
tightening to `< 1e-5` would catch subtle bugs (e.g., a position_ids
off-by-one, a rotary-embedding dtype mismatch) that `1e-4` masks.

Non-blocking adversarially (the test passes with either tolerance), but
spec-compliance adversarial reviewer position: the explicit fp32 number
in the spec is `1e-5`, and the test should use it. Writer's deviation
justification ("stricter than fp16 bound and still passes") is wrong on
the strict vs. loose distinction — `1e-4` is LOOSER than `1e-5`, not
stricter.

### A9 (LOW, correctness) — `forward_with_patches` patches the residual stream AFTER the layer; spec says "patch at prompt tokens only" — verify residual-stream semantics

`stage1/intervention/patcher.py:242-249`: after `layer(hidden, ...)` returns,
`hidden = patch` replaces the entire [1, S, H] residual tensor. Spec §3
prior-art delta (b): "our 'donor' and 'target' are two variants of the
same model compiled by two-cut swap." The patch overwrites post-layer
residual stream, which is the correct ROME-style lever. Adversarially
confirmed: correct. No finding, just explicit accounting.

### A10 (LOW, correctness) — `attention_mask_2d = torch.ones(...)` hard-codes no padding

`stage1/intervention/patcher.py:153`: the manual forward assumes `batch=1`
with no padding (all tokens attended). Spec §7 says `tokenizer(...,
padding=False)` so this is consistent. Adversarially: any future batch>1
call path would produce wrong attention behavior. Add a `batch=1` assertion
for defense-in-depth:

```python
if input_ids.shape[0] != 1:
    raise NotImplementedError("forward_with_patches supports batch=1 only")
```

Non-blocking.

### A11 (LOW, other) — `setup_logging()` called after `_apply_determinism`; logger config misses determinism warnings

`stage1/run_phase_b.py:242-245`: `setup_logging()` is called BEFORE
`_apply_determinism` — check order. Looking again: line 242 is
`setup_logging()`, line 245 is `_apply_determinism(seed)`. Good. The
determinism warnings are captured and logged into `environment`. Fine.

### A12 (LOW, test) — `test_state_dict_hash_stable` uses max_new_tokens=4

`stage1/tests/test_phase_b_patcher.py:368`: The test runs 2 samples × 4
generated tokens. Spec §6 says `max_new_tokens=512`. If any in-place
weight mutation were triggered by something that only fires deep in the
generation loop (e.g., RoPE cache expansion past `max_position_embeddings`,
a scalar-param update in a conditional branch), 4 tokens wouldn't exercise
it. Adversarially: the test is a minimum-coverage shim. Non-blocking but
flagged.

### A13 (NIT, other) — `methodological_constraint` string in summary contains "restoration intervention"

`stage1/run_phase_b.py:67-72`: constraint = "This is prompt-side restoration
intervention, NOT full-sequence causal intervention." The phrase
"restoration intervention" is NOT in FORBIDDEN_PHRASES (which blocks
"restoration effect", "restoration proportion" as substrings). Correct;
no finding. Explicitly recorded because it is adversarially tempting to
confuse the two.

### A14 (NIT, correctness) — EOS-check on `int(current.item())` when `current` is a [1,1] tensor

`stage1/intervention/patcher.py:320`: `int(current.item())`. `current` was
assigned `first_token_id.view(1, 1)` or `next_id` which is also `[1,1]`.
`.item()` on a 1-element tensor returns a Python scalar. Fine.

### A15 (NIT, correctness) — Pass 1 F2 (module-scope assert at line 739) re-confirmed

No change; still a nit. Adversarially recommend deleting.

---

## Triage

| codex_finding_id | claude_label           | reasoning |
|------------------|------------------------|-----------|
| A1               | agree-block            | Confirmed via source inspection: `stage1/run_phase_b.py:43-50` unqualified imports; `stage1/run_phase_a.py:30-46` qualified. Spec §10 canonical invocation is `python -m stage1.run_phase_b ...` (spec §10 "Smoke test — CLI"). Unit tests do not exercise `run_phase_b` import path. BLOCK. |
| A2               | agree-block            | Spec §11.7 numeric acceptance criterion; CWD-relative glob silently skips; skip is marked PASS at `stage1/run_phase_b.py:678`. BLOCK. |
| A3               | agree-block            | Spec §10 explicitly lists the smoke test as a test that MUST pass; writer deferred to watcher; watcher marked PASS but flagged the deferral in `can_ship_with_followup` (watcher lines 125-129). Adversarial reading: "can ship with followup" is not the same as "passed spec §10". BLOCK merge until smoke test runs. |
| A4               | agree-nit              | Provenance/robustness; non-blocking. |
| A5               | agree-nit              | Future-proofing against bf16 drift + CPU RAM spike; non-blocking. |
| A6               | disagree-with-reason   | Invariant holds by construction per `stage1/run_phase_b.py:481` (single shared `samples` list, in-order iteration, no sample skipping). Spec §7 does not mandate sample_id-based pairing — it says "paired bootstrap over sample-level correct booleans." The current implementation satisfies the spec. Hardening is nice-to-have, not a finding. |
| A7               | already-known          | Watcher severity: med (watcher line 43-51); writer acknowledged. |
| A8               | agree-nit              | Spec §2 says `< 1e-5` (fp32); test uses `< 1e-4`. Writer's deviation #2 justifies it but is wrong on "stricter". Non-blocking because 1e-4 still catches gross bugs, but spec-literal compliance argues for 1e-5. |
| A9               | disagree-with-reason   | Not a finding — this is my own accounting of correctness. Spec §3 lines 29-31 explicitly endorse residual-stream patching. No action. |
| A10              | agree-nit              | Defense-in-depth. |
| A11              | disagree-with-reason   | Checked source: `setup_logging` at line 242 followed by `_apply_determinism` at line 245. Ordering is correct (setup logging before we might log warnings). Self-correcting; no action. |
| A12              | agree-nit              | Minimum-coverage shim; non-blocking. |
| A13              | disagree-with-reason   | Explicit non-finding. Verified against `stage1/utils/wording.py:18-28` — FORBIDDEN_PHRASES contains "restoration effect", "residual effect", "restoration proportion"; "restoration intervention" is not a substring of any of those and none of those are a substring of "restoration intervention". No violation. No action. |
| A14              | disagree-with-reason   | Source check at `stage1/intervention/patcher.py:316-320`: `current = first_token_id.view(1, 1)`; `next_id = torch.argmax(..., keepdim=True)` yields [1,1]; so `current.item()` is always 1-element safe. No action. |
| A15              | agree-nit              | Same as Pass 1 F2. |

---

## Adversarial verdict

**Blocks for merge (AGREE-BLOCK):** A1, A2, A3.

A1 and A2 are code fixes writer can do in under 15 lines total. A3 is an
environmental gate (run the spec §10 smoke test on a GPU box with real
Qwen2.5-1.5B weights + a Phase A summary present) that the watcher already
flagged in `can_ship_with_followup`. The writer's pytest suite is
necessary-but-not-sufficient evidence of Phase B correctness.

Codex-discipline summary: watcher PASS + 7/7 pytest is **not** sufficient
evidence of §10 smoke-test readiness given (a) the import-path regression,
(b) the CWD-relative glob silently waiving §11.7, and (c) the deferred
real-weights end-to-end run. These three together are a classic
"tests-that-were-written-all-pass" false-green.
