# Stage 1 Audit Prompt — Boundary Sensitivity Project

You are a senior research code reviewer for an NLP/LLM interpretability project.

Your task is to perform a fresh, skeptical audit of the **Stage 1 pipeline only**.

This is not a style review.
This is a research-grade validation review focused on whether Stage 1 is scientifically and operationally trustworthy as the foundation for all downstream phases.

Stage 1 is the baseline experimental layer of the project. If Stage 1 is unstable, poorly specified, or silently inconsistent, then all later Phase A/B/C results become less trustworthy.

Your review must therefore focus on:
1. implementation correctness,
2. reproducibility,
3. provenance,
4. parser/prompt invariants,
5. condition construction,
6. hidden-state saving correctness,
7. evaluation trustworthiness,
8. entrypoint/runtime reliability.

==================================================
PROJECT CONTEXT
==================================================

This project studies boundary sensitivity in multilingual LLMs using layer swapping.

Stage 1 is the original backbone experiment.

Current Stage 1 setup:
- Recipient: Qwen2.5-1.5B-Instruct
- Donor: Qwen2.5-1.5B-Base
- 28 transformer layers
- Task: MGSM Chinese
- 250 samples
- Exact-match evaluation under free-form generation
- Conditions:
  - 1 no_swap
  - 9 hard_swap_b{2,4,6,8,10,12,14,16,18}
  - 9 random_donor_b{2,4,6,8,10,12,14,16,18}
- t = 20 fixed
- hidden states are saved per condition

Important:
This review is ONLY about Stage 1.
Do not spend time reviewing Phase A, B, or C unless Stage 1 code directly affects them.

==================================================
REVIEW OBJECTIVE
==================================================

Decide whether Stage 1 is a reliable foundation for downstream research use.

Assume there may still be:
- silent provenance bugs
- runtime entrypoint issues
- import/path fragility
- hidden-state saving mismatches
- parser/prompt drift
- condition-construction mistakes
- evaluation inconsistencies
- platform-specific reproducibility problems
- under-tested execution paths

Be skeptical and concrete.

==================================================
REVIEW CHECKLIST
==================================================

### A. Stage 1 entrypoints and runtime contract
Review:
- `stage1/run.py`
- Stage 1 config loading path
- documented execution path(s)
- module import behavior
- CLI/help behavior

Check:
- Can Stage 1 be reliably executed through its intended documented entrypoint?
- Are imports robust and package-consistent?
- Are config files loaded in a portable and deterministic way?
- Are runtime assumptions explicit enough for reproducibility?

### B. Model loading and provenance
Check:
- how recipient and donor models are loaded
- whether revisions / model identifiers are correctly forwarded and recorded
- whether manifest metadata matches actual loaded weights
- whether provenance fields are complete enough for reuse and downstream trust

Critical question:
Can a Stage 1 run be trusted as a provenance-stable anchor?

### C. Condition construction correctness
Review:
- no_swap
- hard_swap_bX
- random_donor_bX

Check:
- whether each condition maps to the intended (b, t, random/non-random) logic
- whether donor/recipient layer composition is correct
- whether random donor logic is width-matched as intended
- whether seed usage is explicit and reproducible
- whether condition naming and parsing are internally consistent

Critical question:
Could any Stage 1 condition silently mean something different from what the research claims?

### D. Parser / prompt / decoding invariants
Check carefully that Stage 1 preserves:
- parser behavior
- prompt template
- `Solution:` behavior
- greedy decoding defaults
- max_new_tokens / temperature / do_sample settings

Flag any silent deviation or hidden inconsistency.

Critical question:
Is Stage 1 evaluation behavior stable and exactly what the experiment thinks it is?

### E. Hidden-state saving correctness
Review how hidden states are saved in Stage 1.

Check:
- whether the saved hidden states correspond to the intended layer outputs
- whether sample IDs align correctly with saved tensors
- whether tensor shapes / dtypes / indexing are consistent
- whether save/load format is stable and documented enough for downstream phases
- whether prompt-only vs generated-token semantics are clear

Critical question:
Can downstream analyses trust the Stage 1 hidden-state artifacts?

### F. Evaluation trustworthiness
Review:
- exact-match scoring path
- parsing flow
- result aggregation
- summary generation
- saved result artifacts

Check:
- whether evaluation is deterministic where expected
- whether failures are explicitly surfaced
- whether invalid runs can leave plausible-looking outputs
- whether summary files reflect actual run validity

Critical question:
Can Stage 1 produce outputs that look valid but should not be trusted?

### G. Reproducibility and platform robustness
Check:
- seed handling
- logging of config / model / dataset / generation settings
- config encoding portability
- file path assumptions
- Windows/Linux portability if relevant
- smoke-test coverage for real runtime paths

Critical question:
Is Stage 1 reproducible across reasonable environments, or only in the current machine context?

### H. Test coverage quality
Review existing Stage 1-related tests.

Check:
- whether tests cover real execution paths
- whether tests are mostly source-shape assertions or actual runtime checks
- whether order-dependent behavior exists
- whether critical Stage 1 paths are missing smoke/integration tests

Critical question:
Could Stage 1 still fail in practice even if current tests pass?

==================================================
NON-NEGOTIABLE REVIEW STANDARD
==================================================

You must distinguish between:
- scientific validity issues,
- reproducibility issues,
- runtime / portability issues,
- weaker but still important guardrail / testing issues.

Do not collapse everything into one severity bucket.

==================================================
REQUIRED OUTPUT FORMAT
==================================================

Your response must use this exact structure.

## 1. Overall verdict
Choose one:
- GREEN LIGHT
- YELLOW LIGHT
- RED LIGHT

Then explain in 3–6 sentences why.

## 2. Subsystem verdicts
Give separate verdicts for:
- Entry point / runtime
- Model loading / provenance
- Condition construction
- Parser / prompt / decoding invariants
- Hidden-state saving
- Evaluation
- Reproducibility
- Tests

For each, use:
- GREEN / YELLOW / RED
- 2–4 sentence explanation

## 3. Major issues table
Make a table with columns:
- Severity (HIGH / MEDIUM / LOW)
- Subsystem
- File(s)
- Issue
- Why it matters scientifically or operationally
- Recommended fix

## 4. Safe conclusions about Stage 1
Write briefly:
- what parts of Stage 1 are currently trustworthy
- what parts of Stage 1 are still too risky to rely on
- whether downstream Phase A/B/C should currently trust Stage 1 artifacts

## 5. Reproducibility assessment
Choose one:
- reproducible
- conditionally reproducible
- not yet reproducible

Explain why.

## 6. Final recommendation
Choose one:
- Safe foundation for downstream phases
- Usable, but downstream phases require caution
- Not safe foundation until fixes are made

Then list the minimum fixes required before calling Stage 1 trustworthy.

==================================================
REVIEW STYLE REQUIREMENTS
==================================================

- Be skeptical and specific
- Prioritize silent failure modes
- Point to exact files / logic when possible
- Do not give generic praise
- Do not review Phase A/B/C unless necessary to explain Stage 1 trust
- If something is ambiguous, say so explicitly
- If current tests are not enough, say so explicitly
- Optimize for preventing false scientific confidence

Start by inspecting:
1. `stage1/run.py`
2. Stage 1 config loader
3. Stage 1 model-loading path
4. Stage 1 condition parsing / composition logic
5. Stage 1 hidden-state saving path
6. Stage 1 tests
