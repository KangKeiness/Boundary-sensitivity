# Boundary Sensitivity Project — Runtime & Repro Fix Prompt (v5)
# Final stabilization pass for execution-path and parity-plumbing blockers

You are the implementation and research-validation agent team for the Boundary Sensitivity project.

At this stage, do NOT work on new experiments, new metrics, new analyses, or new paper framing.
This is purely a code-fix and pipeline-stabilization pass.

The current goal is simple:
make the project reliably executable and reproducible across the documented runtime paths.

This is a short, focused hardening sprint.
Prioritize runtime correctness, portability, and fail-closed behavior.

---

## 1. CURRENT STATE

Latest review conclusion:
- The remaining blockers are primarily runtime / reproducibility / plumbing issues
- The biggest remaining problems are:
  1. `stage1.run` cannot be reliably executed/imported through documented module paths
  2. config loading is not UTF-8 safe across locales
  3. Phase A no-swap reuse parity is inconsistent with the current sample-regime parity design
  4. tests do not yet sufficiently cover these runtime paths

You are NOT being asked to redesign the science.
You are being asked to make the codebase trustworthy to execute.

---

## 2. FIX SCOPE

Only fix the following classes of problems:

### A. Runtime entrypoint correctness
- `python -m stage1.run`
- `import stage1.run`

### B. Config portability
- UTF-8-safe loading of YAML/config files
- especially `stage2_confound.yaml`

### C. Phase A reuse parity plumbing
- current parity extraction must match the newer sample-regime parity contract
- no-swap reuse path must not internally contradict the manifest schema

### D. Runtime smoke-test coverage
- add tests for actual execution/import/config-load paths
- do not rely only on source inspection or stubs

Do NOT expand beyond this scope unless absolutely required to complete these fixes.

---

## 3. REQUIRED FIXES IN PRIORITY ORDER

## PRIORITY 1 — Fix `stage1.run` module execution/import path
Problem:
`stage1.run` currently fails under documented module execution/import paths because of broken import style.

Required action:
- inspect `stage1/run.py`
- replace fragile bare imports with robust package-qualified imports where needed
- ensure both of the following work:
  - `python -m stage1.run --help`
  - `import stage1.run`

Required deliverables:
- corrected imports in `stage1/run.py`
- any minimal supporting import-path fixes if necessary
- regression tests or smoke tests for:
  - module import
  - module-based CLI invocation

Important:
Do not rewrite unrelated logic.
Do the minimal clean fix.

## PRIORITY 2 — Force UTF-8 config loading
Problem:
Config loading currently depends on locale defaults, which can fail on non-UTF-8 environments (for example Windows cp949).

Required action:
- inspect the config loading utility (e.g. `stage1/utils/config.py`)
- ensure config files are opened with explicit `encoding="utf-8"`
- verify that `stage2_confound.yaml` loads correctly under the intended path

Required deliverables:
- explicit UTF-8-safe config loading
- regression/smoke test that loads `stage2_confound.yaml`
- if helpful, add a short code comment explaining why UTF-8 is enforced

Important:
Do not silently depend on OS locale defaults.

## PRIORITY 3 — Fix Phase A no-swap reuse parity plumbing
Problem:
The no-swap reuse path still builds the current parity block without `sample_ids`, while newer manifests include sample-regime parity information.

Required action:
- inspect the Phase A no-swap reuse path
- inspect how parity blocks are constructed
- ensure current parity extraction for reuse includes the correct sample-regime information
- align the reuse path with the current parity contract already used elsewhere

Required deliverables:
- corrected reuse parity logic
- regression test for no-swap reuse parity
- explicit confirmation that the current parity block and source manifest are now constructed on compatible terms

Important:
This is a plumbing fix.
Do not weaken parity checks to make runs pass more easily.

## PRIORITY 4 — Add runtime smoke/integration coverage
Problem:
The test suite currently passes at a high level, but critical runtime paths were not covered well enough.

Required action:
Add lightweight but real runtime coverage for:
- `import stage1.run`
- `python -m stage1.run --help`
- loading `stage2_confound.yaml`
- Phase A no-swap reuse parity regression

If possible, prefer true smoke/integration-style tests over source-string assertions.

Required deliverables:
- new tests
- clear test names
- final test summary showing these paths are now covered

---

## 4. EXECUTION ORDER

Follow this order exactly:

### Step 1
Inspect:
- `stage1/run.py`
- config loading utility
- Phase A reuse parity path
- current tests

### Step 2
Fix PRIORITY 1
- run targeted tests
- internal reviewer loop until both reviewers greenlight

### Step 3
Fix PRIORITY 2
- run targeted tests
- internal reviewer loop until both reviewers greenlight

### Step 4
Fix PRIORITY 3
- run targeted tests
- internal reviewer loop until both reviewers greenlight

### Step 5
Implement PRIORITY 4 smoke/integration coverage
- run tests
- internal reviewer loop until both reviewers greenlight

### Step 6
Run a final targeted test bundle and summarize outcomes

---

## 5. REQUIRED OUTPUT FORMAT

At the end, provide:

### A. Fix summary
- what was changed
- which files were modified
- why each fix matters for runtime trust or reproducibility

### B. Test summary
- which new tests were added
- which test commands were run
- pass/fail results
- what runtime path each test now protects

### C. Reproducibility summary
Explain clearly:
- how module execution/import is now supported
- how UTF-8 config loading is now enforced
- how Phase A reuse parity is now aligned with the current contract

### D. Reviewer status
For each priority block:
- reviewer 1 verdict
- reviewer 2 verdict
- whether both returned green lights

### E. Final readiness verdict
Choose one:
- READY FOR REVIEW
- STILL NOT STABLE — MORE FIXES NEEDED

---

## 6. ABSOLUTE NON-NEGOTIABLES

- Do not modify parser behavior
- Do not modify prompt template or `Solution:` behavior
- Do not change decoding defaults
- Do not weaken parity checks
- Do not add unrelated features
- Do not proceed past a priority block until both internal reviewers return green lights

---

## 7. START NOW

Start by:
1. identifying the exact import-path failure in `stage1.run`
2. identifying the exact config-loading function that lacks UTF-8 enforcement
3. identifying where Phase A reuse parity omits `sample_ids`
4. summarizing the repair plan before editing
5. fixing PRIORITY 1 first
