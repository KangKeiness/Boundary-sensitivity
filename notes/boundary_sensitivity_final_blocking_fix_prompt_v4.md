# Boundary Sensitivity Project — Validity Leak Fix Prompt (v4)
# Apply the remaining fail-closed scientific integrity fixes after latest RED LIGHT review

You are the implementation and research-validation agent team for the Boundary Sensitivity project.

The latest review returned an overall RED LIGHT again, but the meaning of this RED is now narrower:
- the core scientific design is much stronger than before
- major patching and provenance bugs were already improved
- the remaining blockers are specific validity leaks and reproducibility-hardening failures

Your task is NOT to expand the project.
Your task is to close the remaining scientific-integrity leaks, strengthen fail-closed behavior, and prepare the codebase for another final review.

This is a research-grade hardening pass.
Prioritize scientific validity and fail-closed behavior over convenience.

---

## 1. CURRENT REVIEW STATE

Latest review summary:
- Overall verdict: RED LIGHT
- Phase A: GREEN
- Phase B: YELLOW
- Phase C: RED

Interpretation:
- Phase A is now methodologically sound enough
- Phase B intervention mechanics are much improved
- Phase C still has a critical upstream-validity leak
- Remaining blockers are mostly about validity gating, parity completeness, and test isolation

The latest review identified the following key issues:

### HIGH severity
1. Phase C can silently consume a failed Phase B run when `--phase-b-run` is omitted
   - it selects latest `run_*`
   - but does not require `phase_b_summary.run_status == "passed"`
   - this can propagate invalid upstream evidence into plausible-looking decomposition outputs

2. Anchor parity logic does not include sample-regime parity
   - debug/sanity vs full, sample count, or sample-order regime may still differ
   - a run may look “compatible” while being methodologically non-comparable

### MEDIUM severity
3. `post_analysis.py` boundary inference for `fixed_w4_*` / `random_fixed_w4_*` depends on optional heavy imports and can become environment-sensitive
4. tests show order-dependent behavior due to stub / dependency-mocking interactions, so subset runs and full-suite runs can disagree

The reviewer's bottom line:
- Not safe to trust until these remaining validity leaks are fixed

---

## 2. CORE TASK

Close the remaining validity leaks and hardening gaps.

Do not add new features.
Do not change research framing.
Do not modify parser/prompt/decoding behavior.
Do not weaken existing fail-closed behavior.

Your goal is:
- Phase C must reject invalid upstream Phase B runs by default
- anchor parity must include sample-regime parity
- analysis behavior must not depend on optional heavy-import side effects
- tests must become more order-robust and trustworthy

---

## 3. REQUIRED FIXES IN PRIORITY ORDER

## PRIORITY 1 — Hard-gate Phase C on passed Phase B upstream
Problem:
Phase C currently can select a latest Phase B run without confirming that upstream `run_status == "passed"`.

Why this matters:
A failed Phase B run can still leave plausible-looking JSONLs or summaries.
If Phase C consumes them, decomposition outputs may look valid while being scientifically invalid.

Required action:
- inspect `run_phase_c.py`
- inspect how Phase B summaries are loaded and validated
- after loading Phase B summary, hard-fail unless:
  - `run_status == "passed"`
- default behavior must be strict
- if you want a debugging escape hatch, it must be explicit, e.g.:
  - `--allow-failed-upstream`
- but default full-mode behavior must reject failed upstream runs

Required deliverables:
- code fix in `run_phase_c.py`
- regression tests:
  - failed Phase B upstream → hard fail
  - passed Phase B upstream → allowed
  - optional explicit override (if implemented) behaves as documented
- clear summary/log wording explaining why the run was rejected

## PRIORITY 2 — Extend parity contract with sample-regime parity
Problem:
Current parity logic is stronger than before, but it still does not include sample-regime parity.

At minimum, parity should distinguish:
- debug/sanity vs full
- sample cardinality
- sample ordering regime

Why this matters:
Cross-phase anchor acceptance can still allow runs that are methodologically non-comparable.

Required action:
Update parity logic so full-mode anchor acceptance checks include sample-regime identifiers.

At minimum include:
- `dataset.debug_n` or equivalent
- sample count
- sample ordering hash or sample-cardinality/signature field
- if available, explicit sample-regime label

Required deliverables:
- updated manifest/parity checker
- full-mode rejection on sample-regime mismatch
- regression tests covering:
  - debug vs full mismatch
  - count mismatch
  - ordering/signature mismatch (if implemented)
- clear parity logs indicating which field caused rejection

Important:
This must apply to anchor compatibility where scientific comparability is claimed.

## PRIORITY 3 — Make post_analysis boundary inference independent of optional heavy imports
Problem:
`post_analysis.py` currently has environment-sensitive inference for:
- `fixed_w4_*`
- `random_fixed_w4_*`

If optional heavy imports are unavailable, inference may break or fallback incorrectly.

Why this matters:
Analysis results should not depend on whether a heavy import path happened to succeed.
That creates fragile or silent analysis behavior.

Required action:
- remove dependence on optional heavy imports for these condition-name families
- implement minimal static parsing / mapping logic directly
- ensure:
  - `fixed_w4_pos1`
  - `fixed_w4_pos2`
  - `fixed_w4_pos3`
  - `fixed_w4_pos4`
  - random variants
  are inferred correctly in lightweight analysis environments

Required deliverables:
- corrected inference logic
- regression tests for all fixed-width position conditions
- explicit failure only when inference is truly impossible

## PRIORITY 4 — Remove order-dependent test behavior
Problem:
Some tests behave differently depending on subset/full run order due to dependency stubbing or shared-state mutation.

Why this matters:
This undermines confidence in CI and may hide regressions.

Required action:
- inspect test fixtures, stubs, and dependency mocking
- remove global stub mutation / order-sensitive side effects
- isolate dependency mocking per test module or fixture
- ensure key suites behave consistently in:
  - subset runs
  - full-suite runs
  - different orderings where possible

Required deliverables:
- test hardening changes
- evidence that problematic subset/full combinations now behave consistently
- if feasible, add randomized-order CI note or at least a deterministic local reproducibility recipe

---

## 4. EXECUTION ORDER

Follow this order exactly:

### Step 1
Inspect:
- `run_phase_c.py`
- Phase B summary schema / loading path
- parity checker / manifest compatibility logic
- `post_analysis.py`
- relevant tests and fixtures

### Step 2
Fix PRIORITY 1
- run targeted tests
- reviewer loop until both greenlight

### Step 3
Fix PRIORITY 2
- run targeted tests
- reviewer loop until both greenlight

### Step 4
Fix PRIORITY 3
- run targeted tests
- reviewer loop until both greenlight

### Step 5
Fix PRIORITY 4
- run targeted tests
- verify subset/full consistency as best as possible
- reviewer loop until both greenlight

### Step 6
Run a final relevant test bundle and summarize outcomes

### Step 7
Prepare final repair summary for next review

---

## 5. REQUIRED OUTPUT FORMAT

At the end, provide:

### A. Fix summary
- what was changed
- which files were modified
- why each change matters scientifically

### B. Test summary
- which new tests were added
- which existing tests were hardened
- which suites were run
- pass/fail results
- what each test now protects against

### C. Validity-gating summary
Explain clearly:
- how Phase C now rejects invalid upstream Phase B runs
- how sample-regime parity is now enforced
- why anchor comparability is now stronger than before

### D. Analysis robustness summary
Explain:
- how post-analysis condition inference is now independent of optional heavy imports
- how test order sensitivity was reduced or removed

### E. Reviewer status
For each priority block:
- reviewer 1 verdict
- reviewer 2 verdict
- whether both returned green lights

### F. Final readiness verdict
Choose one:
- READY FOR FINAL REVIEW
- STILL NOT SAFE — MORE FIXES REQUIRED

---

## 6. ABSOLUTE NON-NEGOTIABLES

- Do not overclaim causality
- Do not modify parser, prompt template, `Solution:` behavior, or greedy decoding defaults
- Do not allow Phase C to consume failed Phase B runs by default
- Do not keep parity checks blind to sample regime
- Do not leave fixed-width condition inference dependent on optional heavy imports
- Do not leave order-sensitive test behavior unaddressed
- Do not proceed past a priority block until both internal reviewers return green lights

---

## 7. START NOW

Start by:
1. locating the exact Phase C upstream-validity leak
2. locating where parity currently omits sample-regime information
3. locating where `post_analysis.py` still depends on optional heavy imports for fixed-width condition inference
4. identifying the tests/fixtures causing order sensitivity
5. summarizing the repair plan before editing
6. fixing PRIORITY 1 first
