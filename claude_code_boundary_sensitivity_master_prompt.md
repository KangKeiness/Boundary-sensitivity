# Boundary Sensitivity Project — Master Execution Prompt
# Sequential Phase A → B → C Expansion (Claude Code Agent Team Version)

You are an implementation and research-validation agent team working on an NLP/LLM interpretability project.

Your job is to extend an existing codebase for a multilingual layer-swapping study, while preserving methodological rigor and avoiding overclaiming. This is not just an engineering task. You must implement the experiments in a way that supports careful causal interpretation and clear research reporting.

The work should be executed **in one overall session**, but **strictly sequentially**:
- Phase A first
- then Phase B
- then Phase C

Do **not** execute all phases blindly in parallel.
Progression is **phase-gated**.

For each phase:
1. Inspect the current relevant code and files
2. Summarize the planned edits before coding
3. Implement the phase
4. Run sanity checks and validation
5. Submit the phase to the internal reviewer loop
6. If either internal code reviewer does **not** give a green light, revise and repeat
7. Only proceed to the next phase when **both** internal code reviewers give a green light

Do **not** stop permanently at the first issue.
Instead, iterate until the current phase is methodologically and technically sound enough to receive **two green lights** from the internal reviewer pair.

---

## 1. PROJECT GOAL

We are studying boundary sensitivity in multilingual LLMs using layer swapping.

Current study framing:
- Not a transfer-optimization project
- Not a “find the best swap recipe” project
- This is an intervention-based diagnostic study
- Main goal: move from performance observation toward stronger explanation of why degradation happens

Important current findings from Stage 1:
- Hard swap causes mild but consistent degradation relative to the no-swap baseline
- Random donor causes near-collapse, suggesting position-sensitive structural compatibility
- Existing BDS does not explain degradation well
- Recovery-side and geometry-level metrics correlate more strongly with degradation than local boundary metrics
- Current design has a critical limitation: boundary position and swap width are confounded

Your mission:
Implement a stronger experimental program that moves from correlation toward more intervention-based evidence, while keeping all claims conservative and methodologically explicit.

---

## 2. EXISTING PROJECT CONTEXT

Repository:
https://github.com/KangKeiness/Boundary-sensitivity

Backbone / models:
- Recipient: Qwen2.5-1.5B-Instruct
- Donor: Qwen2.5-1.5B-Base
- 28 transformer layers
- hidden_dim = 1536

Current infrastructure:
- `stage1/run.py` orchestrates inference
- `stage1/models/composer.py` handles model composition
- `stage1/inference/runner.py` runs generation
- `stage1/inference/parser.py` parses answers
- `stage1/analysis/` contains evaluator, BDS, post_analysis
- Hidden states are already saved per condition as `.pt` files

Current Stage 1 setup:
- 19 conditions total:
  - 1 `no_swap`
  - 9 `hard_swap_b{2,4,6,8,10,12,14,16,18}`
  - 9 `random_donor_b{2,4,6,8,10,12,14,16,18}`
- `t = 20` fixed
- Task = MGSM Chinese
- 250 samples
- Exact-match evaluation under free-form generation
- Current output directory root = `/workspace/outputs`

Key limitation to address:
- In current design, lower boundary `b` and swap width `(t - b)` co-vary

---

## 3. GLOBAL IMPLEMENTATION PRINCIPLES

You must obey the following principles throughout implementation.

### A. Do not overclaim causality
- Never write code comments or output summaries saying “proved causal mechanism”
- Use wording such as:
  - causally relevant
  - stronger causal evidence
  - restoration-based evidence
  - mediation-style decomposition
- Do **not** use strong formal causal language unless explicitly justified

### B. Preserve parser and generation behavior
Do **not** modify:
- `stage1/inference/parser.py`
- prompt template
- `"Solution:"` prefix
- greedy decoding defaults

Keep:
- `temperature = 0.0`
- `do_sample = False`
- `max_new_tokens = 512`

### C. Preserve distance consistency
- Use the same `cosine_distance` function already used in `stage1/analysis/bds.py` wherever cosine-based distance is needed
- Cast all tensors to `float32` before distance computations

### D. New code must be:
- runnable as CLI
- importable from Jupyter
- clearly commented
- equipped with a sanity-check mode

### E. Save intermediate outputs and summaries
Every phase should produce:
- machine-readable CSV / JSON summaries
- human-readable text summary
- sanity-check logs
- saved per-condition outputs

### F. Sequential but iterative execution
Execution order is strict:
- Phase A first
- Phase B second
- Phase C third

But progression is not “single pass only.”
For each phase, keep revising until:
- sanity checks pass, and
- both internal code reviewers give a green light

### G. Reviewer-gated progression rule
For each phase:
- reviewer 1 evaluates code and method
- reviewer 2 evaluates code and method
- if either reviewer raises issues, revise and rerun review
- proceed only when **both** reviewers return green lights

This reviewer loop is mandatory.

---

## 4. PHASE A — WIDTH CONFOUND SEPARATION GRID

### Purpose
Separate boundary-position effects from swap-width effects more cleanly than the current Stage 1 sweep.

### 4.1 Experimental Design

#### Grid 1: fixed width, varying position
Width = 4

Conditions:
- `fixed_w4_pos1` : `(b=4,  t=8)`
- `fixed_w4_pos2` : `(b=8,  t=12)`
- `fixed_w4_pos3` : `(b=12, t=16)`
- `fixed_w4_pos4` : `(b=16, t=20)`

#### Grid 2: fixed boundary, varying width
Boundary fixed at `b=8`

Conditions:
- `fixed_b8_w2` : `(b=8, t=10)`
- `fixed_b8_w4` : `(b=8, t=12)`
- `fixed_b8_w6` : `(b=8, t=14)`
- `fixed_b8_w8` : `(b=8, t=16)`

For each hard-swap condition, create a width-matched random donor control:
- `random_fixed_w4_pos1`, ..., `random_fixed_w4_pos4`
- `random_fixed_b8_w2`, ..., `random_fixed_b8_w8`

Total new conditions:
- 8 hard swap
- 8 random donor
- reuse existing `no_swap` baseline if compatible

Random donor seed rule:
```python
seed = seed_base * 1000 + b * 100 + t
```
Default `seed_base = 42`

### 4.2 Mandatory methodological rule

Because `t` varies in Phase A, recovery-zone length may differ across conditions.

Therefore:
- Primary comparable metrics for Phase A must be metrics that remain directly comparable across conditions
- Use the following as **PRIMARY** Phase A metrics:
  1. Accuracy / degradation
  2. Final-layer divergence (FLD)
  3. Final-layer geometry metrics, if implemented consistently
- Treat recovery-zone mean metrics as **SECONDARY / EXPLORATORY** when recovery-zone length differs

This rule is mandatory.

Explicitly document in outputs:
- which metrics are primary
- which metrics are secondary / exploratory
- why

### 4.3 Required implementation changes

Create:
- `stage1/configs/stage2_confound.yaml`
- `stage1/run_phase_a.py`

Modify:
- `stage1/models/composer.py`
- `stage1/analysis/post_analysis.py`

Add helper:
- `parse_condition_bt(condition_name, config)`

This helper must support:
- `no_swap`
- `hard_swap_b{X}`
- `random_donor_b{X}`
- `fixed_w4_pos{N}`
- `fixed_b8_w{W}`
- `random_fixed_w4_pos{N}`
- `random_fixed_b8_w{W}`

Composition logic itself does not change:
- bottom = `recipient[0:b]`
- middle = `donor[b:t]`
- top = `recipient[t:L]`

### 4.4 Required Phase A outputs

Produce at minimum:

#### A. Position-effect table (fixed width)
Columns:
- `condition`
- `b`
- `t`
- `width`
- `accuracy`
- `degradation`
- `fld_cos`
- `fld_l2`
- `optional_final_geom`

#### B. Width-effect table (fixed boundary)
Columns:
- `condition`
- `b`
- `t`
- `width`
- `accuracy`
- `degradation`
- `fld_cos`
- `fld_l2`
- `optional_final_geom`

#### C. Summary note
Must state:
- whether position variation matters under fixed width
- whether width variation matters under fixed boundary
- whether recovery-side metrics are comparable or exploratory only

#### D. Sanity mode
- run `2 conditions × 5 samples`

---

## 5. PHASE B — RESTORATION INTERVENTION

### Purpose
Test whether recovery-side disruption is not only correlated with degradation, but causally relevant.

This phase must be implemented conservatively:
- We are testing causal relevance via restoration intervention
- We are **not** claiming a complete causal mechanism proof

### 5.1 Target condition

Primary treatment condition:
- `hard_swap_b8`

Primary reference condition:
- `no_swap`

### 5.2 Intervention conditions

Implement the following patch conditions:
- `no_patch`
- `patch_boundary_local`      layers `[7, 8, 9]`
- `patch_recovery_early`      layers `[20, 21, 22]`
- `patch_recovery_full`       layers `[20..27]`
- `patch_final_only`          layer `[27]`
- `patch_all_downstream`      layers `[8..27]`

Also implement reverse interventions where feasible:
- `corrupt_recovery_full` into clean recipient
- `corrupt_boundary_local` into clean recipient

### 5.3 Critical methodological constraints

Clean hidden states are available for prompt tokens only.

Therefore:
- patching only applies to prompt-side hidden-state processing
- do **not** claim this is full-sequence causal intervention
- all output summaries must explicitly say this is **prompt-side restoration intervention**

Primary implementation approach:
- Use a manual layer-by-layer prompt forward pass
- Patch hidden states after the output of specified layers
- Then continue normal greedy generation from the patched prompt state

Do **not** implement multiple competing patching strategies.

Use one clear strategy only:
- **manual layer-by-layer prompt forward + patch + greedy generation**

### 5.4 Required implementation changes

Create:
- `stage1/intervention/__init__.py`
- `stage1/intervention/patcher.py`
- `stage1/run_phase_b.py`

The patcher must:
- load clean hidden states from `no_swap`
- load corrupt hidden states from `hard_swap_b8` if reverse intervention is used
- support per-sample patching
- save detailed outputs:
  - generated text
  - parsed answer
  - correctness
  - patch condition

### 5.5 Required Phase B outputs

Produce:

#### A. Restoration table
Columns:
- `condition`
- `accuracy`
- `delta_from_no_patch`

#### B. Reverse-corruption table
Columns:
- `condition`
- `accuracy`
- `delta_from_clean_baseline`

#### C. Interpretation summary using conservative language
Examples:
- recovery-side intervention yields stronger restoration than boundary-local intervention
- recovery-zone states appear more causally relevant than local boundary states
- final-layer-only patch is insufficient / partially sufficient
- prompt-side restoration recovers X% of lost performance

Do **not** use phrases like:
- proves the mechanism
- fully explains the failure
- identifies the true cause

#### D. Sanity mode
- `2 patch conditions × 5 samples`

---

## 6. PHASE C — MEDIATION-STYLE DECOMPOSITION

### Purpose
Quantify how much of the treatment effect is restored when specific internal states are repaired.

Important:
- This is **mediation-style** analysis
- Do **not** use strong formal causal-mediation terminology unless rigorously justified
- Avoid labels like:
  - Natural Indirect Effect
  - Natural Direct Effect
- Instead use:
  - restoration effect
  - residual effect
  - restoration proportion

### 6.1 Definitions

For a given treatment condition:
- baseline accuracy = `no_swap` accuracy
- treatment accuracy = hard swap accuracy
- patched accuracy = intervention condition accuracy

Compute:
- **total effect** = `baseline_accuracy - treatment_accuracy`
- **restoration effect** = `patched_accuracy - treatment_accuracy`
- **residual effect** = `total_effect - restoration_effect`
- **restoration proportion** = `restoration_effect / total_effect`

### 6.2 Bootstrap requirement

Implement bootstrap confidence intervals:
- resample 250 samples with replacement
- recompute accuracies
- recompute restoration quantities
- 1000 bootstrap samples
- 95% CI

### 6.3 Continuous exploratory analysis

You may also implement exploratory continuous analyses, such as:
- partial correlation
- regression-based association checks

But:
- these must be clearly labeled **exploratory**
- they must **not** be the central result of Phase C

Primary Phase C result must be restoration-based decomposition using patch interventions.

### 6.4 Required implementation changes

Create:
- `stage1/analysis/mediation.py`
- `stage1/run_phase_c.py`

### 6.5 Required Phase C outputs

Produce:

#### A. Restoration decomposition table
Columns:
- `patch_condition`
- `total_effect`
- `restoration_effect`
- `residual_effect`
- `restoration_proportion`
- `CI_low`
- `CI_high`

#### B. Summary paragraph
State:
- which intervention restores the largest share of lost performance
- whether recovery-side restoration dominates local restoration
- whether final-layer-only intervention is sufficient or insufficient

#### C. Explicit caveat section
State clearly:
- restoration-based decomposition is not identical to formal identified mediation
- current claims remain intervention-based but conservative

#### D. Sanity mode
- verify math on mock data before real run

---

## 7. FILE STRUCTURE TARGET

### New files
- `stage1/configs/stage2_confound.yaml`
- `stage1/intervention/__init__.py`
- `stage1/intervention/patcher.py`
- `stage1/analysis/mediation.py`
- `stage1/run_phase_a.py`
- `stage1/run_phase_b.py`
- `stage1/run_phase_c.py`

### Files to modify
- `stage1/models/composer.py`
- `stage1/analysis/post_analysis.py`

### Files NOT to modify
- `stage1/inference/runner.py`
- `stage1/inference/parser.py`
- `stage1/analysis/bds.py`
- `stage1/analysis/evaluator.py`
- `stage1/data/loader.py`

---

## 8. TESTING AND VALIDATION

Before any full run:
- run Phase A sanity mode
- run Phase B sanity mode
- run Phase C mock-data sanity mode

After each phase:
- print a concise pass/fail checklist
- confirm outputs saved successfully
- confirm metrics have no NaN issues
- confirm parser behavior unchanged
- confirm seeds logged
- confirm tensor dtype = float32 in analysis paths

For each phase, reviewer loop must verify:
- implementation correctness
- methodological consistency
- comparability caveats handled explicitly
- output summaries use conservative language

If either reviewer finds issues:
- revise
- rerun checks
- resubmit to both reviewers
- continue until both greenlight

---

## 9. DELIVERABLES

At the end, provide:

1. A short implementation summary
2. A list of files created / modified
3. A list of commands to run each phase
4. A short methodological caveat summary
5. A unified result summary combining:
   - existing Stage 1
   - Phase A
   - Phase B
   - Phase C

The unified result summary must explicitly distinguish:
- correlational evidence
- restoration / intervention evidence
- remaining limitations

---

## 10. ABSOLUTE NON-NEGOTIABLES

- Do not overclaim causality
- Do not silently change parser or prompting
- Do not treat exploratory metrics as primary when comparability is compromised
- Do not skip sanity checks
- Do not ignore recovery-zone comparability issues when `t` varies
- Do not replace conservative wording with stronger causal terminology
- Do not proceed to the next phase until the current phase receives **two internal green lights**

---

## 11. STARTING PROCEDURE

Start by:
1. Inspecting the current repository structure
2. Verifying existing hidden-state file format
3. Confirming how current condition parsing works
4. Summarizing planned edits for Phase A before coding
5. Implementing Phase A first
6. Running sanity checks
7. Entering the internal reviewer loop
8. Proceeding to Phase B only after both reviewers greenlight Phase A
9. Repeating the same process for Phase B and then Phase C
