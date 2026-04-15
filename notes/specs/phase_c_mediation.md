# phase_c_mediation

Scope note: Phase C is an **analysis-only** phase. It consumes the per-condition sample-level JSONLs produced by the Phase B rewrite (`notes/specs/phase_b_rewrite.md`, handoff `notes/handoffs/writer_phase_b_rewrite.md`) and produces a decomposition of accuracy deltas under prompt-side restoration intervention. No new inference, no new models, no training. Phase B (frozen per strict A → B → C order) is the only upstream dependency.

Conservative-wording scope: the Phase-C-reserved terms `"restoration effect"`, `"residual effect"`, `"restoration proportion"` are **allowed** in Phase C artifacts (they are Phase C's core vocabulary). The Phase B FORBIDDEN_PHRASES gate forbids them in Phase B output; Phase C extends the shared `stage1/utils/wording.py` list with new, Phase-C-specific forbiddens (`"natural direct effect"`, `"natural indirect effect"`, `"NIE"`, `"NDE"`, `"causal mediation"`) so that even Phase C text cannot slip into formal causal-mediation language the intervention does not support.

Mandatory caveat echoed in every Phase C summary artifact (verbatim): `"Mediation analysis here decomposes accuracy deltas under prompt-side restoration intervention only. It is not a formal NIE/NDE decomposition."`

## 1. Goal

Implement an analysis-only Phase C that loads a completed Phase B run directory, computes `restoration_effect`, `residual_effect`, and `restoration_proportion` with paired-bootstrap 95% CIs (1000 resamples) keyed by `sample_id`, and emits a decomposition table + JSON + TXT summary under `stage1/outputs/phase_c/run_<timestamp>/` that passes the extended `check_artifacts_for_forbidden` gate.

Testable: a single `pytest -q stage1/tests/test_phase_c_mediation.py stage1/tests/test_post_analysis_condition_names.py` run plus one `python -m stage1.run_phase_c --phase-b-run <latest> --sanity` CLI invocation must both pass the acceptance criteria in §11.

## 2. Hypothesis and falsification

Hypothesis H1 (engineering): given a Phase B run directory with `clean_no_patch`, `no_patch`, and at least one `patch_*` restoration condition's JSONLs populated and sample_id-aligned, the Phase C entrypoint produces a decomposition table whose point estimates and 95% CIs are deterministic under fixed seed=0 across re-runs (bytewise-equal `phase_c_decomposition_table.csv` across two invocations on the same inputs).

Falsification H1: any non-zero diff in `phase_c_decomposition_table.csv` between two seed=0 invocations on the same Phase B inputs; OR bootstrap CI widths change across runs; OR sample pairing falls back to positional pairing when `sample_id` fields are present on both sides.

Hypothesis H2 (reporting discipline): the extended `check_artifacts_for_forbidden` gate returns `[]` over every Phase C artifact, AND the mandated caveat string from the scope note above appears verbatim in both `phase_c_summary.json` and `phase_c_summary.txt`.

Falsification H2: the gate returns a non-empty list on any Phase C artifact; OR the caveat substring is missing or altered (case-sensitive for the JSON field, case-insensitive for the TXT body).

Hypothesis H3 (backward compat of `compute_bpd_sweep`): after extending the condition-name recognition set to include `fixed_w4_*`, `fixed_b8_*`, `random_fixed_*`, `patch_*`, `corrupt_*`, all existing Phase A and Phase B callers that pass `hard_swap_b{b}` / `random_donor_b{b}` condition names continue to receive byte-identical `bpd_sweep` results (verified by a pinned-input unit test).

Falsification H3: any divergence in the new unit test's pinned reference values vs. the pre-change output on the same Phase A fixture.

No scientific claim about the recipient model or about causation is asserted; all three hypotheses are engineering/reporting/backward-compat properties.

## 3. Prior art and delta

Two primary citations (both verified resolvable on arxiv.org; no invented IDs):

- **Vig, Gehrmann, Belinkov, Qian, Nevo, Sakenis, Huang, Singer, Shieber 2020, "Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias," arXiv:2004.12265.** Canonical application of Pearl's mediation framework to neural LMs. Defines natural direct effect (NDE), natural indirect effect (NIE), and the indirect/total proportion via component-level activation substitution. Delta vs. our Phase C: (a) we do **not** estimate NIE/NDE — Phase B patches prompt-side hidden states only, not generation-side, so the mediator-on-path assumption required by the formal decomposition is violated; (b) we therefore report `restoration_effect = acc_patched - acc_no_patch` and `residual_effect = acc_clean_baseline - acc_patched(best)` as **marginal accuracy deltas under the restoration intervention**, not NIE/NDE; (c) `restoration_proportion = restoration_effect / (acc_clean_baseline - acc_no_patch)` is an accuracy-recovery ratio, not a causal proportion-mediated. The FORBIDDEN_PHRASES gate extension (§8) forbids the formal terms so Phase C text cannot drift into claims the intervention does not license.
- **Meng, Bau, Andonian, Belinkov 2022, "Locating and Editing Factual Associations in GPT" (ROME / causal tracing), arXiv:2202.05262.** Introduces causal tracing via activation patching between clean and noised runs at specific (token, layer) coordinates and reports the indirect effect as the accuracy / probability shift induced by patching. Delta vs. our Phase C: (a) our donor vs. target pairing is `recipient` (clean baseline) vs. `hard_swap_b8` composed model (treatment), not clean-vs.-noised-embedding; (b) Phase B already performed the patching — Phase C only decomposes the resulting accuracy deltas and their bootstrap CIs; (c) we patch the full post-block residual stream across a set of layers (`[20..27]` at fixed `t=20`), not MLP-only at a single subject-token coordinate.

Delta of THIS Phase C spec vs. the existing codebase:
1. New analysis module `stage1/analysis/mediation.py` with `restoration_effect`, `residual_effect`, `restoration_proportion` and a top-level `compute_decomposition_table(phase_b_run_dir, *, bootstrap_n=1000, seed=0)` function.
2. New CLI entrypoint `stage1/run_phase_c.py` that locates a Phase B run directory, loads its JSONLs, delegates to the mediation module, and emits three artifacts (CSV + JSON + TXT) under `stage1/outputs/phase_c/run_<timestamp>/`.
3. Extended `FORBIDDEN_PHRASES` in `stage1/utils/wording.py` adding the five Phase-C-specific terms listed in §8 AND removing the three previously Phase-C-reserved terms (`"restoration effect"`, `"residual effect"`, `"restoration proportion"`) from the forbidden list since they are now Phase C's core vocabulary. Phase B gate semantics are preserved by routing Phase B through a dedicated `FORBIDDEN_PHRASES_PHASE_B` tuple (see §8) — backward-compatible, no Phase B artifact revalidation required.
4. Backward-compatible extension of `stage1/analysis/post_analysis.py::compute_bpd_sweep` to recognize Phase A (`fixed_w4_*`, `fixed_b8_*`, `random_fixed_*`) and Phase B (`patch_*`, `corrupt_*`) condition names, with unit tests covering all five naming families plus the existing `hard_swap_b{b}` / `random_donor_b{b}`.

## 4. Datasets

n/a — analysis-only on Phase B outputs.

Input pointer (schema, not a dataset):
- Source: per-condition JSONL files under `stage1/outputs/phase_b/run_<TS>/results_<name>.jsonl` produced by `stage1/run_phase_b.py::_save_condition_results`. Each row contains at least `sample_id` (str), `correct` (bool), `parsed_answer` (str), `gold_answer` (str), `prompt` (str) — `sample_id` and `correct` are the only fields Phase C reads.
- Condition coverage consumed by Phase C (full run): `results_clean_no_patch.jsonl`, `results_restoration_no_patch.jsonl`, `results_restoration_patch_boundary_local.jsonl`, `results_restoration_patch_recovery_early.jsonl`, `results_restoration_patch_recovery_full.jsonl`, `results_restoration_patch_final_only.jsonl`, `results_restoration_patch_all_downstream.jsonl`.
- Corruption JSONLs (`results_corruption_*.jsonl`) are NOT consumed by Phase C — they are kept for Phase B reporting only.
- Dataset provenance is inherited: Phase C copies `dataset` block from `phase_b_summary.json` verbatim into `phase_c_summary.json.dataset`.

Hash/version: no new hash; Phase C logs `phase_b_run_path` (absolute), the SHA-256 of `phase_b_summary.json`, and the Phase B `git_sha` under `phase_c_summary.json.upstream_provenance`.

License: inherited from Phase B (MGSM → GSM8K MIT + Google translations Apache-2.0). No new obligations.

## 5. Models and tokenizers

n/a — no model load in Phase C. The Phase B `environment` block (model revisions, tokenizer revision, composed `state_dict` sha256) is copied verbatim into `phase_c_summary.json.upstream_provenance.environment` for traceability.

## 6. Training config

n/a — analysis-only.

Analysis-side determinism and seed policy:
- `bootstrap_seed = 0` (fixed; matches Phase B's comparative-sentence bootstrap seed and ensures bytewise-deterministic CI reproduction).
- `bootstrap_n = 1000` (from `stage1/configs/stage2_confound.yaml::evaluation.bootstrap_n`).
- `bootstrap_ci = 0.95` (from YAML).
- Random state: `numpy.random.default_rng(bootstrap_seed)` used for resampling; no `torch` RNG touched by Phase C.
- Tensor dtype in analysis: all intermediate correctness arrays are `np.float32`; reported estimates and CI bounds are python `float` (for JSON stability); CSV cells are 6-decimal formatted strings.
- Pair sampling: paired bootstrap over the **intersection of sample_ids** across every condition consumed in a given comparison; mismatched or missing ids are dropped with a WARNING logged and the dropped count recorded under `phase_c_summary.json.sample_pairing`.

Logged into `phase_c_summary.json.environment`:
- `python_version`, `numpy_version`, `pandas_version` (CSV writer), `git_sha` (or `"unknown"` fail-soft), `bootstrap_seed`, `bootstrap_n`, `bootstrap_ci`, `phase_b_run_path`, `phase_b_summary_sha256`.

## 7. Evaluation protocol

Metric definitions (all on per-condition sample-level correctness boolean vectors, paired by `sample_id`):

- `acc(C) := mean(correct_i for i in aligned_sample_ids[C])`
- `restoration_effect(C) := acc(C) - acc(no_patch)` for each restoration condition `C ∈ {patch_boundary_local, patch_recovery_early, patch_recovery_full, patch_final_only, patch_all_downstream}` (plus `no_patch` itself, which trivially has `restoration_effect = 0` and is included in the table for audit).
- Best-restoration selection: `C_best := argmax_{C ∈ claim_eligible_set} restoration_effect(C)` where `claim_eligible_set` is the four corruption-mirrored conditions (`patch_boundary_local`, `patch_recovery_early`, `patch_recovery_full`, `patch_final_only`) — matches Phase B spec §11.10's claim-eligible restriction. Ties broken by alphabetical condition name (deterministic).
- `residual_effect := acc(clean_no_patch) - acc(C_best)`
- `total_gap := acc(clean_no_patch) - acc(no_patch)` (denominator)
- `restoration_proportion := restoration_effect(C_best) / total_gap` when `|total_gap| >= epsilon_denom` (default `epsilon_denom = 0.005`, i.e. half of Phase A's 0.008 cross-phase tolerance); otherwise emit `null` with reason `"denominator_below_epsilon"`.

Paired bootstrap (1000 resamples, seed=0, 95% percentile CI):
- For each metric `M` above, resample the **aligned index set** (not the condition pair) 1000 times with replacement; recompute `M` on each resample; report point estimate = metric on full aligned set, CI = 2.5th / 97.5th percentile of resampled values.
- `restoration_effect` CI: paired bootstrap on `(correct_C, correct_no_patch)` pairs.
- `residual_effect` CI: paired bootstrap on `(correct_clean_no_patch, correct_C_best)` pairs.
- `restoration_proportion` CI: paired bootstrap on the joint `(correct_clean_no_patch, correct_no_patch, correct_C_best)` triples; compute `numerator / denominator` per resample, drop resamples with `|denominator| < epsilon_denom` (log the drop fraction; if >5% of resamples dropped, emit `null` CI with reason `"unstable_denominator"`).

Reduction axes: per restoration condition (one row); per-language breakdown n/a (MGSM-zh only, inherited from Phase B).

Baselines: `no_patch` (composed, no intervention) and `clean_no_patch` (recipient, no intervention) — both loaded directly from Phase B JSONLs. No re-inference.

Statistical test & alpha: 95% percentile CI only; no null-hypothesis test is performed in Phase C (single-seed, single-dataset, matches project memory rule against premature multi-seed variance claims).

## 8. Interfaces to add/change

All paths absolute under `C:\Users\system1\Boundary-sensitivity\`. Python type-hinted.

### New module: `stage1/analysis/mediation.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import json, os
import numpy as np

CLAIM_ELIGIBLE_CONDITIONS: Tuple[str, ...] = (
    "patch_boundary_local",
    "patch_recovery_early",
    "patch_recovery_full",
    "patch_final_only",
)
EPSILON_DENOM: float = 0.005    # below this |acc_clean - acc_no_patch|, proportion is null

@dataclass(frozen=True)
class ConditionCorrectness:
    condition: str
    sample_ids: Tuple[str, ...]        # ordered; parallel to correct
    correct: Tuple[bool, ...]

def load_condition_correctness(jsonl_path: str) -> ConditionCorrectness: ...
    """Reads a Phase B results_*.jsonl with encoding='utf-8'. Requires fields
    'sample_id' (str) and 'correct' (bool). Raises FileNotFoundError if missing.
    Duplicate sample_ids raise ValueError."""

def align_by_sample_id(
    *conditions: ConditionCorrectness,
) -> Tuple[List[str], List[np.ndarray]]:
    """Intersect sample_ids across all inputs; return (aligned_ids, list of
    np.ndarray[int8] in the same order as inputs). Ids present in one condition
    but missing from another are dropped; dropped count is returned via logger
    WARNING. Order of aligned_ids is deterministic (sorted)."""

def _paired_bootstrap(
    fn,                               # callable(list of np.ndarray) -> float
    arrays: Sequence[np.ndarray],
    *, n_resamples: int = 1000, seed: int = 0, ci: float = 0.95,
) -> Tuple[float, float, float, int]:
    """Returns (point, ci_lo, ci_hi, n_dropped). n_dropped counts resamples
    where fn returned NaN (e.g., unstable denominator); excluded from
    percentile computation."""

def restoration_effect(
    patched: ConditionCorrectness,
    no_patch: ConditionCorrectness,
    *, bootstrap_n: int = 1000, seed: int = 0, ci: float = 0.95,
) -> Dict:
    """Point estimate = acc(patched) - acc(no_patch) on aligned ids.
    Returns {condition, point, ci_lo, ci_hi, n_aligned, n_dropped_ids}."""

def residual_effect(
    clean: ConditionCorrectness,
    best_patched: ConditionCorrectness,
    *, bootstrap_n: int = 1000, seed: int = 0, ci: float = 0.95,
) -> Dict: ...
    """Point estimate = acc(clean) - acc(best_patched). Same return shape as
    restoration_effect."""

def restoration_proportion(
    clean: ConditionCorrectness,
    no_patch: ConditionCorrectness,
    best_patched: ConditionCorrectness,
    *, bootstrap_n: int = 1000, seed: int = 0, ci: float = 0.95,
    epsilon_denom: float = EPSILON_DENOM,
) -> Dict: ...
    """Point estimate = (acc(best_patched) - acc(no_patch)) /
    (acc(clean) - acc(no_patch)) when |denom| >= epsilon_denom else None.
    Returns {point, ci_lo, ci_hi, n_aligned, denom_point,
    n_resamples_dropped_denominator, ci_reason} where ci_reason ∈
    {None, 'denominator_below_epsilon', 'unstable_denominator'}."""

def compute_decomposition_table(
    phase_b_run_dir: str,
    *, bootstrap_n: int = 1000, seed: int = 0, ci: float = 0.95,
    epsilon_denom: float = EPSILON_DENOM,
) -> Dict:
    """Orchestrates: loads clean_no_patch + restoration_no_patch + all
    restoration_patch_* JSONLs; computes restoration_effect per restoration
    condition; picks C_best from CLAIM_ELIGIBLE_CONDITIONS; computes
    residual_effect and restoration_proportion against C_best. Returns a dict
    with keys: rows (list of per-condition dicts), best_condition (str),
    residual, proportion, sample_pairing (dict with aligned_n,
    dropped_ids_per_condition), acc_no_patch, acc_clean_no_patch."""
```

### New entrypoint: `stage1/run_phase_c.py`

```python
def run_phase_c(
    phase_b_run: Optional[str] = None,
    *, sanity: bool = False,
    bootstrap_n: int = 1000, seed: int = 0, ci: float = 0.95,
    run_name: Optional[str] = None,
) -> str:
    """(1) resolve phase_b_run (explicit arg wins; else auto-pick latest
    stage1/outputs/phase_b/run_* matching the same CWD-invariant rule as
    Phase B's _phase_a_outputs_dir helper; fail with RuntimeError if none);
    (2) validate required JSONLs exist (sanity mode: {clean_no_patch,
    restoration_no_patch, restoration_patch_recovery_full}); (3) delegate to
    compute_decomposition_table; (4) write three artifacts with
    encoding='utf-8': phase_c_decomposition_table.csv,
    phase_c_summary.json, phase_c_summary.txt; (5) run
    check_artifacts_for_forbidden — raise RuntimeError on non-empty;
    (6) return run_dir (absolute).

    CLI: --phase-b-run <path>, --sanity, --seed, --bootstrap-n, --run-name."""
```

### Modified: `stage1/utils/wording.py`

```python
# New public names alongside existing FORBIDDEN_PHRASES / check_artifacts_for_forbidden.

FORBIDDEN_PHRASES_PHASE_B: Tuple[str, ...] = (
    "proves the mechanism",
    "proves mechanism",
    "causal proof",
    "identifies the true cause",
    "fully explains",
    "demonstrates causation",
    "restoration effect",        # Phase C reserved — forbidden in Phase B
    "residual effect",
    "restoration proportion",
)

FORBIDDEN_PHRASES_PHASE_C: Tuple[str, ...] = (
    "proves the mechanism",
    "proves mechanism",
    "causal proof",
    "identifies the true cause",
    "fully explains",
    "demonstrates causation",
    "natural direct effect",     # formal mediation terms Phase C intervention does not support
    "natural indirect effect",
    "causal mediation",
    "nie/nde",                   # defensive: literal token that sometimes slips into prose
    "nde/nie",
)

# Backward-compat: existing FORBIDDEN_PHRASES remains a module attribute for any
# import sites already bound to it. Its value is set to FORBIDDEN_PHRASES_PHASE_B
# so that all pre-existing Phase B call sites (and the Phase B test
# test_forbidden_phrases_gate which iterates this exact tuple) continue to pass
# bytewise.
FORBIDDEN_PHRASES: Tuple[str, ...] = FORBIDDEN_PHRASES_PHASE_B

def check_artifacts_for_forbidden(
    paths: Sequence[str],
    *, phrases: Optional[Sequence[str]] = None,
) -> List[str]:
    """Existing signature extended with optional `phrases` keyword. When
    `phrases is None`, uses FORBIDDEN_PHRASES (== FORBIDDEN_PHRASES_PHASE_B).
    Phase C call sites pass phrases=FORBIDDEN_PHRASES_PHASE_C explicitly."""
```

Existing behaviour (match semantics, UTF-8 read, missing-path skip) is preserved; only the keyword argument is additive. Phase B code (`stage1/run_phase_b.py` calls `check_artifacts_for_forbidden(paths)` with no `phrases=`) is unchanged.

### Modified: `stage1/analysis/post_analysis.py`

`compute_bpd_sweep` — internal change only; public signature unchanged.

```python
# New module-level constant
CONDITION_NAME_PREFIXES: Tuple[str, ...] = (
    "hard_swap_b",       # Phase A / Stage 2 (pre-existing)
    "random_donor_b",    # Phase A / Stage 2 (pre-existing)
    "fixed_w4_",         # Phase A width-confound grid
    "fixed_b8_",         # Phase A width-confound grid
    "random_fixed_",     # Phase A random-donor variant of the above
    "patch_",            # Phase B restoration conditions
    "corrupt_",          # Phase B reverse-corruption conditions
)

def _enumerate_conditions(hs: Dict, boundary_grid: List[int]) -> List[str]:
    """Returns the list of condition-name keys in `hs` that match any prefix
    in CONDITION_NAME_PREFIXES. For the two 'b'-suffixed prefixes
    (hard_swap_b, random_donor_b), enumeration still goes through boundary_grid
    so the exact pre-change key order is preserved. The remaining prefixes are
    enumerated by iterating sorted(hs.keys()) and filtering by prefix (stable,
    deterministic). This change is ADDITIVE — the set of keys returned is a
    superset of the pre-change set, and the order of the pre-existing subset
    is byte-identical."""

def compute_bpd_sweep(run_data: Dict) -> Dict[str, Dict]:
    """Unchanged signature, unchanged return shape. Internal loop now uses
    _enumerate_conditions instead of the hardcoded two-prefix nested loop.
    For conditions that do not carry a boundary_grid-style `b` suffix, `b` is
    resolved via a new helper _infer_b_for_condition (reads Phase A
    grid YAML keys for fixed_w4_*/fixed_b8_*, reads Phase B
    compose_meta.b==8 for patch_*/corrupt_*). Falls back to run_data.t_fixed
    semantics unchanged."""
```

No change to `compute_bpd`, `compute_recovery_metrics`, or any other public function in `post_analysis.py`. The two other hardcoded occurrences at lines 443 and 486-552 (inside report-printing paths that iterate `boundary_grid` by design and are specific to Stage-2 tables) are **not** touched — they remain Phase-A/Stage-2-only. Phase C does not call into them.

## 9. Files-to-touch (exhaustive)

- `C:\Users\system1\Boundary-sensitivity\stage1\analysis\mediation.py` — module (new); symbols: `ConditionCorrectness`, `CLAIM_ELIGIBLE_CONDITIONS`, `EPSILON_DENOM`, `load_condition_correctness`, `align_by_sample_id`, `_paired_bootstrap`, `restoration_effect`, `residual_effect`, `restoration_proportion`, `compute_decomposition_table` — **add**. Rationale: Phase C core analysis.
- `C:\Users\system1\Boundary-sensitivity\stage1\run_phase_c.py` — module (new); symbols: `run_phase_c`, `_resolve_phase_b_run`, `_phase_b_outputs_dir`, `_write_decomposition_csv`, `_write_summary_json`, `_write_summary_txt`, `_copy_upstream_provenance`, `main` (CLI) — **add**. Rationale: Phase C CLI entrypoint.
- `C:\Users\system1\Boundary-sensitivity\stage1\utils\wording.py` — symbols: `FORBIDDEN_PHRASES_PHASE_B` (new), `FORBIDDEN_PHRASES_PHASE_C` (new), `check_artifacts_for_forbidden` (add `phrases` kwarg, default None). Existing `FORBIDDEN_PHRASES` alias bound to `FORBIDDEN_PHRASES_PHASE_B` — **modify**. Rationale: Phase C vocabulary carve-out without breaking Phase B gate.
- `C:\Users\system1\Boundary-sensitivity\stage1\analysis\post_analysis.py` — symbols: `CONDITION_NAME_PREFIXES` (new), `_enumerate_conditions` (new helper), `_infer_b_for_condition` (new helper), `compute_bpd_sweep` (internal edit only; public signature unchanged) — **modify**. Rationale: support Phase A grid names and Phase B patch names without changing existing Stage 2 behaviour.
- `C:\Users\system1\Boundary-sensitivity\stage1\tests\test_phase_c_mediation.py` — tests: `test_load_condition_correctness_ok`, `test_load_condition_correctness_duplicate_sample_id_raises`, `test_align_by_sample_id_drops_and_logs`, `test_restoration_effect_point_and_ci_deterministic`, `test_residual_effect_point_and_ci_deterministic`, `test_restoration_proportion_null_when_denom_below_epsilon`, `test_restoration_proportion_unstable_denominator_branch`, `test_compute_decomposition_table_best_condition_tie_break_alphabetical`, `test_phase_c_forbidden_phrases_gate_extended_phase_c_terms_flagged`, `test_phase_c_cli_sanity_end_to_end_against_fixture` — **add**. Rationale: cover every branch of §8 interfaces; pin determinism.
- `C:\Users\system1\Boundary-sensitivity\stage1\tests\test_post_analysis_condition_names.py` — tests: `test_enumerate_conditions_hard_swap_and_random_donor_byte_identical_to_legacy`, `test_enumerate_conditions_recognizes_fixed_w4`, `test_enumerate_conditions_recognizes_fixed_b8`, `test_enumerate_conditions_recognizes_random_fixed`, `test_enumerate_conditions_recognizes_patch_prefix`, `test_enumerate_conditions_recognizes_corrupt_prefix`, `test_compute_bpd_sweep_backward_compat_pinned_output` — **add**. Rationale: H3 backward-compat test, covers all five naming families plus pre-existing two.
- `C:\Users\system1\Boundary-sensitivity\stage1\tests\fixtures\phase_b_run_fixture\` — fixture directory (new); contains synthetic `phase_b_summary.json` (minimal valid schema), `results_clean_no_patch.jsonl`, `results_restoration_no_patch.jsonl`, and three `results_restoration_patch_*.jsonl` files with known per-sample correctness patterns for deterministic-CI assertions — **add**. Rationale: hermetic, no Phase B run required for Phase C tests.
- `C:\Users\system1\Boundary-sensitivity\notes\specs\phase_c_mediation.md` — this spec — **add** (already the write target for spec-planner).

Do-NOT-touch (enforced by project memory + agent constraints):
- `stage1/inference/runner.py`
- `stage1/inference/parser.py`
- `stage1/analysis/bds.py` (reference only, for cosine_distance if ever needed — Phase C does not compute new distances)
- `stage1/analysis/evaluator.py`
- `stage1/data/loader.py`
- `stage1/run_phase_a.py` (Phase A frozen)
- `stage1/run_phase_b.py` (Phase B frozen post-rewrite)
- `stage1/intervention/patcher.py` (Phase B frozen)
- `stage1/intervention/__init__.py`
- `stage1/models/composer.py`
- `stage1/configs/stage2_confound.yaml`
- `stage1/tests/test_phase_b_patcher.py` (Phase B's test file — additive only via new files)

Deletions: none. Phase C is purely additive + two surgical edits to `wording.py` and `post_analysis.py::compute_bpd_sweep`.

## 10. Test plan

Unit tests — file `stage1/tests/test_phase_c_mediation.py`, run via `pytest -q stage1/tests/test_phase_c_mediation.py`:

1. `test_load_condition_correctness_ok` — write a 3-line temp JSONL with `sample_id` and `correct`; assert loader returns the right tuple lengths and types.
2. `test_load_condition_correctness_duplicate_sample_id_raises` — duplicate `sample_id` must raise `ValueError`.
3. `test_align_by_sample_id_drops_and_logs` — two fixtures sharing ids {a,b,c} and {b,c,d}; expect aligned_ids == [b, c] and a caplog WARNING containing `"dropped"`.
4. `test_restoration_effect_point_and_ci_deterministic` — pinned 8-sample fixture (correct patterns `0b11110000` vs `0b10101010`); assert point estimate, ci_lo, ci_hi match 6-decimal pinned values across two invocations.
5. `test_residual_effect_point_and_ci_deterministic` — same pinned-value requirement.
6. `test_restoration_proportion_null_when_denom_below_epsilon` — construct `acc_clean - acc_no_patch = 0.002 < 0.005`; expect `point is None`, `ci_reason == "denominator_below_epsilon"`.
7. `test_restoration_proportion_unstable_denominator_branch` — construct a fixture where >5% of bootstrap resamples hit near-zero denominator; expect `ci_lo is None`, `ci_hi is None`, `ci_reason == "unstable_denominator"`; `point` remains a float.
8. `test_compute_decomposition_table_best_condition_tie_break_alphabetical` — two conditions with identical restoration_effect; assert `best_condition == "patch_boundary_local"` (alphabetically first in the claim-eligible set).
9. `test_phase_c_forbidden_phrases_gate_extended_phase_c_terms_flagged` — write a temp file containing each Phase-C-specific forbidden phrase; assert `check_artifacts_for_forbidden([path], phrases=FORBIDDEN_PHRASES_PHASE_C)` returns `len(FORBIDDEN_PHRASES_PHASE_C)` violations.
10. `test_phase_c_cli_sanity_end_to_end_against_fixture` — point CLI at the `stage1/tests/fixtures/phase_b_run_fixture/` directory via `--phase-b-run <path> --sanity --seed 0`; assert exit code 0, all three artifacts present, JSON contains the verbatim caveat, gate returns `[]`, CSV has exactly one row per restoration condition present in the fixture.

Unit tests — file `stage1/tests/test_post_analysis_condition_names.py`, run via `pytest -q stage1/tests/test_post_analysis_condition_names.py`:

1. `test_enumerate_conditions_hard_swap_and_random_donor_byte_identical_to_legacy` — synthetic `hs` dict with both legacy prefixes; assert `_enumerate_conditions` output list equals the hand-rolled legacy list in pre-change order.
2. `test_enumerate_conditions_recognizes_fixed_w4` — dict with `fixed_w4_pos1..pos4`; expect all four recognized and sorted deterministically.
3. `test_enumerate_conditions_recognizes_fixed_b8` — same for `fixed_b8_w2..w8`.
4. `test_enumerate_conditions_recognizes_random_fixed` — same for `random_fixed_w4_pos1`, `random_fixed_b8_w4`.
5. `test_enumerate_conditions_recognizes_patch_prefix` — dict with `patch_boundary_local`, `patch_recovery_full`; both recognized.
6. `test_enumerate_conditions_recognizes_corrupt_prefix` — dict with `corrupt_recovery_full`; recognized.
7. `test_compute_bpd_sweep_backward_compat_pinned_output` — load the existing Phase A fixture (or a trimmed synthetic replica) with `no_swap`, `hard_swap_b8`, `random_donor_b8`; assert `bpd_mean` values for each condition match 6-decimal pinned references. Proves H3.

Smoke test — CLI (integration, non-sanity):
- `python -m stage1.run_phase_c --phase-b-run stage1/outputs/phase_b/run_<latest> --seed 0 --bootstrap-n 1000` MUST complete in under 60 seconds on CPU (no GPU needed — analysis-only), produce the three canonical artifacts, and exit 0 with the wording gate returning `[]`.

Eval-sanity inline checks emitted by `run_phase_c`:
- `acc_no_patch` and `acc_clean_no_patch` read from Phase B JSONLs match the values recorded in Phase B's `phase_b_summary.json` within absolute tolerance `1e-6` (pure re-computation sanity; flushes a stale JSONL if mismatched).
- `n_aligned` for every (patched, no_patch) pair equals `len(clean_no_patch_ids)` when Phase B ran on the full 250 samples with no per-condition sample skips; under `--sanity` the expected count is 5.
- `best_condition ∈ CLAIM_ELIGIBLE_CONDITIONS`.
- The mandated caveat string is present verbatim in both `phase_c_summary.json.caveat` and `phase_c_summary.txt`.

Any single failure is a hard RuntimeError; run exits non-zero.

## 11. Acceptance criteria

All thresholds are numeric and checked against artifacts under `RD = stage1/outputs/phase_c/run_<timestamp>/`.

1. `pytest -q stage1/tests/test_phase_c_mediation.py stage1/tests/test_post_analysis_condition_names.py` exits 0; all 17 tests pass (10 Phase C + 7 post_analysis).
2. `RD/phase_c_decomposition_table.csv` exists, is UTF-8, has exactly the columns (in this order): `condition, restoration_effect, restoration_effect_ci_lo, restoration_effect_ci_hi, residual_effect, residual_effect_ci_lo, residual_effect_ci_hi, restoration_proportion, restoration_proportion_ci_lo, restoration_proportion_ci_hi, n_aligned, is_best_condition, methodology`. `residual_effect` and `restoration_proportion` columns are populated only on the `is_best_condition == True` row; other rows have empty cells (not `null` string, not `NaN`). Row count = number of restoration conditions present in the Phase B run (5 or 6 full run, 2 sanity).
3. `RD/phase_c_summary.json` exists, is valid UTF-8 JSON, and contains keys: `phase == "C"`, `caveat` == the verbatim mandated caveat string (byte-exact), `decomposition_table` (list mirroring CSV rows), `best_condition` (str in CLAIM_ELIGIBLE_CONDITIONS), `sample_pairing` (dict), `bootstrap` (dict with `n`, `seed`, `ci`), `environment`, `upstream_provenance` (dict with `phase_b_run_path`, `phase_b_summary_sha256`, `environment` block copied verbatim), `dataset` (copied from Phase B), `forbidden_phrases_gate` (list; MUST be `[]`), `methodology` (string containing the substring `"prompt-side restoration intervention"`).
4. `RD/phase_c_summary.txt` exists, is UTF-8, and contains the verbatim caveat sentence on its own line. It explicitly uses the allowed Phase C vocabulary (`"restoration effect"`, `"residual effect"`, `"restoration proportion"`) in the decomposition paragraph; it MUST NOT contain any substring in `FORBIDDEN_PHRASES_PHASE_C` (case-insensitive).
5. `check_artifacts_for_forbidden([summary.txt, summary.json, decomposition_table.csv], phrases=FORBIDDEN_PHRASES_PHASE_C)` returns `[]`.
6. Determinism: two consecutive invocations of `python -m stage1.run_phase_c --phase-b-run <same> --seed 0 --bootstrap-n 1000` produce `phase_c_decomposition_table.csv` files that are bytewise equal (compared via SHA-256).
7. Backward-compat: on the Phase A fixture, `compute_bpd_sweep` returns a dict whose keys are a **superset** of the pre-change key set; the intersection subset has `bpd_mean` values matching pre-change 6-decimal pinned references exactly.
8. `best_condition` recorded in `phase_c_summary.json` equals the alphabetically-first condition among those tied at the maximum `restoration_effect` within the claim-eligible set (verified by `test_compute_decomposition_table_best_condition_tie_break_alphabetical`).
9. `restoration_proportion` null-handling: when `|acc(clean) − acc(no_patch)| < 0.005`, the corresponding CSV cell is empty string and the JSON field is `null` with `ci_reason == "denominator_below_epsilon"`.
10. `RD/phase_c_summary.txt` header contains the literal line `"Phase C — mediation-style decomposition of prompt-side restoration intervention (not a formal NIE/NDE decomposition)"`.
11. No occurrence of any substring in `FORBIDDEN_PHRASES_PHASE_C` anywhere in `RD/`.
12. `phase_c_summary.json.upstream_provenance.phase_b_summary_sha256` matches the SHA-256 of the consumed `phase_b_summary.json` at read time.

## 12. Risks and ablations

Risks:
- R1 — Phase B JSONL schema drift: if a future Phase B edit renames `sample_id` or `correct`, the loader will raise. Mitigation: `load_condition_correctness` fails fast with a clear message naming the missing field; unit test covers the happy path. No silent fallback.
- R2 — `no_patch` naming ambiguity: Phase B writes `results_restoration_no_patch.jsonl` (prefixed with `restoration_`) for the composed-model no-intervention baseline, and a separate `results_clean_no_patch.jsonl` for the recipient-model baseline. Phase C must consume BOTH; the loader uses explicit filename constants, not prefix matching. Test 10 (CLI end-to-end) exercises the correct distinction.
- R3 — Denominator instability when Phase B happens to produce `acc_clean ≈ acc_no_patch`: `restoration_proportion` reports `null` with a reason string rather than raising. Both branches are unit-tested (tests 6 and 7). Users consuming the CSV must check `is_best_condition` and the empty-cell convention.
- R4 — Bootstrap resample seed vs. Phase B bootstrap seed collision: Phase B uses seed=0 for its comparative-sentence CI. Phase C also uses seed=0. This is **intentional** for cross-phase determinism, but it means Phase C's CI is NOT statistically independent of Phase B's. Documented; not a scientific claim issue because Phase C reports descriptive CIs, not a hypothesis test.
- R5 — `compute_bpd_sweep` key-order drift: if `_enumerate_conditions` returns the legacy subset in a different order (e.g., dict-insertion order on a non-3.7 Python), downstream Stage-2 printouts could reorder. Mitigation: the helper explicitly iterates `boundary_grid` for the two legacy prefixes FIRST (preserving pre-change order), then appends new-family matches via `sorted(hs.keys())`. Test 1 of `test_post_analysis_condition_names.py` pins this.
- R6 — FORBIDDEN_PHRASES gate false positives on prose: Phase C text legitimately uses "restoration" and "residual" as bare words (allowed); only the three multi-word phrases are gated in Phase B, and the Phase C gate does NOT include them. The Phase C gate's Greek-letter-less phrases are narrow enough to avoid collisions with ordinary English.
- R7 — Eval contamination: none. Phase C reads only Phase B JSONLs; no new data ingress.
- R8 — Seed coupling: single-seed Phase C per project memory rule; multi-seed variance is not in scope.
- R9 — Windows / cp1252: all file I/O must use `encoding="utf-8"` per Phase B's lesson learned. Enforced by §11.2 / §11.3 via explicit byte-check tests.
- R10 — Empty `claim_eligible_set` in sanity mode: under `--sanity` Phase B runs only `patch_recovery_full`, so `C_best = patch_recovery_full` (sole eligible member). Report this explicitly; do NOT raise. When zero claim-eligible conditions are present (which would imply Phase B ran with a non-standard condition set), raise RuntimeError.

Ablations (listed; not in rewrite scope):
- A1 — `EPSILON_DENOM` default 0.005: fixed as module constant; spec-level review bump would edit the constant + one test (`test_restoration_proportion_null_when_denom_below_epsilon`).
- A2 — Best-condition tie-break policy: alphabetical; an alternative (e.g., prefer the narrower layer set) is explicitly out of scope.
- A3 — Full-sequence (generation-side) mediation — reserved for a future phase beyond Phase C; Phase C's caveat string exists precisely to guard against premature claims here.
- A4 — Corruption-arm decomposition: Phase C does NOT decompose corruption deltas. If that becomes desirable, it is a Phase-C.1 spec, not a Phase C extension.

## 13. Compute budget

- Analysis-only; no GPU required. Phase C runs CPU-only.
- Per-invocation cost: loading ≤ 7 JSONLs × 250 rows × ~2 KB ≈ 3.5 MB; bootstrap = 1000 resamples × O(n) = O(2.5 × 10^5) boolean ops per metric × ~10 metrics ≈ 2.5 × 10^6 ops total. Wall-clock target: ≤ 60 s on a standard dev CPU; ≤ 5 s under `--sanity` (5 samples).
- GPU-hours: 0.
- Storage: per run_dir ≤ 100 KB (three small artifacts); no hidden_states persisted.
- CPU RAM peak: < 100 MB (numpy arrays of booleans + small python dicts).
- No multi-node, no multi-GPU.

## 14. Rollback

Phase C is four additive files + two surgical edits. No schema migrations, no data regeneration, no Phase A/B artifact changes.

Rollback procedure:
1. `git rm stage1/analysis/mediation.py stage1/run_phase_c.py stage1/tests/test_phase_c_mediation.py stage1/tests/test_post_analysis_condition_names.py` to delete the additions.
2. `git rm -r stage1/tests/fixtures/phase_b_run_fixture` to delete the test fixture.
3. `git restore stage1/utils/wording.py stage1/analysis/post_analysis.py` to return those files to their Phase B / Phase A states.
4. Delete any `stage1/outputs/phase_c/run_<timestamp>/` directories produced during Phase C development — safe; no downstream job depends on them (Phase D does not yet exist per memory).
5. Phase A and Phase B artifacts are untouched by construction (do-not-modify list); verify with `git status stage1/run_phase_a.py stage1/run_phase_b.py stage1/intervention/patcher.py` showing no changes.
6. No HF hub state, no dataset artifacts, no caches to roll back.

Rollback is idempotent and completes in under 30 seconds. No data loss beyond Phase C run outputs, which are reproducible from the pre-rollback code at any time (analysis is deterministic under fixed seed).
