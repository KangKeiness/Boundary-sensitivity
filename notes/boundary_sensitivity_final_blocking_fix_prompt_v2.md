# Boundary Sensitivity Project — Final Hardening Prompt (v3)
# Apply final robustness / reproducibility fixes after YELLOW LIGHT review

You are the implementation and research-validation agent team for the Boundary Sensitivity project.

The latest review returned an overall YELLOW LIGHT.

Interpretation of that review:
- The major scientific blockers have mostly been fixed
- Phase A is GREEN
- Phase C is GREEN
- Phase B is no longer blocked by core scientific-invalidity issues, but still needs stronger guardrails
- Remaining issues are mainly about robustness, reproducibility, and fail-closed behavior
- This is now a final hardening pass, not a research redesign pass

Your task is NOT to add new features or expand the experiment.
Your task is to make the current implementation safer to trust operationally and scientifically.

The goal of this pass is:
- stronger integration-level protection for Phase B anchor gating
- explicit parity-compatible anchor-generation workflow
- clearer failure-state handling in saved artifacts
- readiness for final review / final run

---

## 1. CURRENT REVIEW STATE

Latest review summary:
- Overall verdict: YELLOW LIGHT
- Phase A: GREEN
- Phase B: YELLOW
- Phase C: GREEN

The review says the following are already improved:
- Stage1 now forwards model revisions correctly
- Phase B full runs require both anchors
- Phase C enforces strict sample-ID equality in non-sanity mode
- parser / prompt / Solution: suffix / greedy decoding defaults were not silently changed
- tests are currently clean at a high level

Remaining issues:

### MEDIUM
1. Phase B anchor-gate regression tests are still too shallow
   - current tests are mostly logic/source-structure tests
   - they are not yet true integration-level gate assertions

2. Full-mode anchor validation depends on parity-compatible upstream artifacts
   - this is fail-closed, which is good scientifically
   - but the workflow / config contract is not explicit enough
   - users may misinterpret this as random pipeline breakage or bypass checks manually

### LOW
3. Failed Phase B runs can still leave plausible-looking output summaries/artifacts
   - downstream users may get confused if they ignore exit status

Your job is to fix these remaining issues cleanly and conservatively.

---

## 2. CORE PRINCIPLES FOR THIS PASS

A. No new science scope
- no new phases
- no new experiments
- no new metrics
- no changes to research framing

B. Preserve non-negotiable invariants
Do NOT modify:
- parser behavior
- prompt template
- "Solution:" prefix
- greedy decoding defaults
- core evaluation semantics

C. Hardening only
This pass is about:
- integration reliability
- reproducibility contract clarity
- fail-closed behavior
- artifact trustworthiness

D. Reviewer-gated iteration remains active
For each hardening block:
1. inspect relevant files
2. summarize planned edits
3. implement
4. run tests
5. run internal reviewer loop
6. revise until both reviewers return green lights

---

## 3. REQUIRED FIXES

### PRIORITY 1 — Add true integration-level regression tests for Phase B anchor gate
Problem:
Current regression coverage for Phase B anchor acceptance is still too shallow.
Critical gate behavior should be tested at a more integration-like level.

Required action:
Add mocked or controlled integration-level tests that exercise Phase B full-run gate logic and verify:

- missing hard_swap_b8 anchor → hard fail
- missing no_swap anchor → hard fail
- both anchors present and parity-compatible → pass
- anchor exists but parity-incompatible → hard fail

Important:
These tests should verify actual gate behavior, not just helper predicates or source-shape assumptions.

Required deliverables:
- new integration-style regression tests
- explicit test names
- clear pass/fail assertions
- evidence in test output that gate logic is now protected end-to-end at the Phase B entry point

### PRIORITY 2 — Add explicit anchor-generation workflow / config contract
Problem:
Full-mode anchor validation is scientifically correct but operationally opaque unless users know exactly how parity-compatible anchors must be generated.

Required action:
Create an explicit workflow contract for anchor generation and reuse.

This should include:
- what config must match
- what manifest fields must match
- which run(s) are allowed to serve as anchors
- how to generate parity-compatible Stage1 / anchor runs
- how users should invoke Phase B safely
- what failure message means and how to resolve it

Implementation options:
- a short dedicated markdown doc
- CLI help text
- config comments
- run-time hints when anchor parity fails
- preferably some combination of the above

Required deliverables:
- a concrete documented anchor-generation recipe
- explicit parity-compatible config contract
- improved error/help message for anchor mismatch cases

### PRIORITY 3 — Mark failed Phase B runs explicitly in artifacts
Problem:
A failed Phase B run can leave plausible-looking artifacts or summaries, which may confuse downstream users.

Required action:
Make failure state explicit.

Choose one or both:
- write artifacts only after sanity checks / acceptance checks pass
- or write artifacts with explicit `run_status: failed` and failure reason

Requirements:
- downstream user must not be able to mistake a failed Phase B run for a valid completed run
- summary files must clearly indicate pass/fail status
- logs must include reason for failure

Required deliverables:
- explicit run-status handling
- example failure-path summary or test
- no ambiguous “looks valid” summary on failure paths

---

## 4. EXECUTION ORDER

Follow this order exactly:

### Step 1
Inspect:
- current Phase B gate logic
- current regression tests
- current artifact-writing behavior
- current docs / CLI help / config comments related to anchors

### Step 2
Implement PRIORITY 1
- add integration-level Phase B gate tests
- run tests
- reviewer loop until both greenlight

### Step 3
Implement PRIORITY 2
- add anchor-generation workflow / config contract
- improve mismatch help text if needed
- reviewer loop until both greenlight

### Step 4
Implement PRIORITY 3
- add explicit failed-run marking or delayed artifact writing
- test failure-path behavior
- reviewer loop until both greenlight

### Step 5
Run the relevant tests and summarize outcomes

### Step 6
Prepare final hardening summary for the next review

---

## 5. REQUIRED OUTPUT FORMAT

At the end, provide:

### A. Hardening summary
- what was changed
- which files were modified
- what robustness issue each change addresses

### B. Test summary
- which new tests were added
- which suites were run
- pass/fail results
- what each test now protects against

### C. Workflow summary
Explain clearly:
- how parity-compatible anchors should now be generated
- how users should run Phase B safely
- what happens if anchors are missing or incompatible

### D. Failure-state summary
Explain:
- how failed Phase B runs are now marked
- why failed artifacts can no longer be mistaken for successful outputs

### E. Reviewer status
For each priority block:
- reviewer 1 verdict
- reviewer 2 verdict
- whether both returned green lights

### F. Final readiness verdict
Choose one:
- READY FOR FINAL REVIEW
- STILL YELLOW — MORE HARDENING NEEDED

---

## 6. ABSOLUTE NON-NEGOTIABLES

- Do not overclaim causality
- Do not add unrelated features
- Do not weaken fail-closed behavior
- Do not leave Phase B gate coverage shallow
- Do not leave anchor workflow implicit
- Do not allow failed runs to look valid
- Do not proceed past a hardening block until both internal reviewers return green lights

---

## 7. START NOW

Start by:
1. identifying where current Phase B gate behavior is tested too shallowly
2. identifying where anchor workflow is currently underdocumented
3. identifying where failed-run artifacts can still look valid
4. summarizing the hardening plan before editing
5. implementing PRIORITY 1 first