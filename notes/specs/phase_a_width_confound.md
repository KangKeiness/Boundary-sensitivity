# phase_a_width_confound

## 1. Goal
Separate boundary-position effects from swap-width effects in the Qwen2.5-1.5B layer-swapping diagnostic by running two orthogonal grids (fixed width=4 varying position; fixed boundary b=8 varying width) with width-matched random-donor controls, and reporting position-effect and width-effect tables on MGSM-zh using primary metrics that remain comparable when t varies.

## 2. Hypothesis and falsification
H1 (position): With swap width held at 4, moving the donor block across the stack changes accuracy-degradation and final-layer divergence (FLD) by more than the random-donor noise floor at matched width.

H2 (width): With lower boundary held at b=8, increasing the donor block width from 2 to 8 changes degradation and FLD monotonically (not necessarily linearly).

Falsification:
- H1 is falsified if the range of degradation across fixed_w4_pos{1..4} is within the range of degradation across random_fixed_w4_pos{1..4} AND the same is true for fld_cos.
- H2 is falsified if fixed_b8_w2..fixed_b8_w8 shows no monotone (or near-monotone) trend in either degradation or fld_cos AND the width range lies inside the random-donor width range at matched width.

Both outcomes are publishable; the spec commits to report them either way.

## 3. Prior art and delta
- Stage 1 of this repository (commits ab721ba, df60525, 9802f64) — hard-swap sweep over b in {2..18} with t=20 fixed, random-donor control at each b. Limitation: b and width (t-b) are perfectly confounded.
- notes/specs/recovery_metrics.md — prior work adding RD/FLD/CKA/RSA recovery-zone metrics to post_analysis.py. Phase A reuses the FLD computation but demotes recovery-zone-mean metrics to SECONDARY because the recovery window length L - t now varies across Phase A conditions.
- Bandarkar et al., "Layer Swapping for Zero-Shot Cross-Lingual Transfer" (arXiv:2410.01335) — the donor-layer substitution paradigm. Delta: this study is diagnostic, not transfer-optimizing; we do NOT claim causal mechanism identification, only intervention-based width-vs-position separation.

No external citation is required for Phase A beyond the above. WebFetch not used; all references resolve to artifacts already in this repo.

## 4. Datasets
- name: MGSM Chinese (mgsm zh, split=test)
- version: per stage1/configs/stage2_confound.yaml — pinned to the same mgsm_zh snapshot used by Stage 1.
- hash: must be recorded in notes/data_changelog.md before run; writer MUST add a dated entry mgsm_zh@<ver> with sha256 of the raw jsonl if one is not already present. Current data_changelog.md only contains mgsm_te@1.0; Phase A uses zh, so an entry is required.
- license: per upstream MGSM (CC-BY-SA 4.0, juletxara/mgsm on HF).
- language coverage: Chinese only for Phase A.
- N samples: 250 (full), 5 (sanity). debug_n controls this.

## 5. Models and tokenizers
- recipient: Qwen/Qwen2.5-1.5B-Instruct, HF revision pinned via models.recipient_revision.
- donor: Qwen/Qwen2.5-1.5B, pinned via models.donor_revision.
- tokenizer: loaded from recipient name, same revision as recipient.
- architecture verified at load time by _validate_architecture in stage1/models/composer.py (num_hidden_layers=28, hidden_size=1536, num_attention_heads match).
- dtype for model weights: torch.float16 (matches Stage 1).
- dtype for distance computations: torch.float32 (cast inside compute_fld and any recovery helpers; enforced by master prompt section 3.C).

## 6. Training config
No training. Phase A is inference-only intervention. Record these determinism and generation invariants in the run manifest:
- seed policy: single seed_base=42; per-condition random-donor seed derived via the mandatory formula seed = seed_base * 1000 + b * 100 + t.
- RNG: Python random.Random(seed) only inside compose_model. No numpy or torch global seeds are modified by Phase A.
- optimizer / schedule / precision / batch / grad accumulation: N/A.
- determinism flags: temperature = 0.0, do_sample = False, max_new_tokens = 512, all tensors in analysis paths cast to float32 before distance.

## 7. Evaluation protocol

### 7.1 Metrics — primary vs secondary (MANDATORY classification)

PRIMARY metrics (directly comparable across every Phase A condition regardless of where t lands):
1. Accuracy on MGSM-zh exact match via the unchanged parser.
2. Degradation = max(0.0, baseline_accuracy - condition_accuracy) where baseline is no_swap. Mirrors existing evaluator.py H1 convention.
3. FLD_cos — cosine distance between composed and recipient hidden states at layer L-1 (last transformer block output), averaged over samples. Uses stage1/analysis/bds.cosine_distance on float32 tensors.
4. FLD_l2 — L2 distance at layer L-1, same aggregation.

SECONDARY / EXPLORATORY metrics (comparability compromised because recovery-zone length L - t differs across Phase A conditions; report but do not use for H1/H2 conclusions):
- Recovery-zone mean drift RD_cos, RD_l2
- Recovery CKA, Recovery RSA
- BPD_mean, EBPD_mean (downstream-layer window also differs)

Rationale (from master prompt section 4.2): recovery-zone length varies in Phase A, so primary comparable metrics must be those that remain directly comparable across conditions. Use accuracy/degradation and FLD, plus final-layer geometry metrics if consistently implemented, as PRIMARY. Treat recovery-zone mean metrics as SECONDARY/EXPLORATORY when recovery-zone length differs.

Final-layer geometry metrics (master prompt 4.2 bullet 3) are OPTIONAL. Shipping only FLD_cos + FLD_l2 + accuracy/degradation satisfies the primary-metric requirement.

### 7.2 Reduction axes
- Per condition (full list in master prompt 4.1).
- Grid 1 position-effect table: fixed_w4_pos{1..4} + random_fixed_w4_pos{1..4}.
- Grid 2 width-effect table: fixed_b8_w{2,4,6,8} + random_fixed_b8_w{2,4,6,8}.
- No per-language breakdown (zh-only).

### 7.3 Baselines
- no_swap (recipient-only forward pass) — used for baseline_accuracy and as the reference hidden-state bundle for FLD.
- Width-matched random_* controls — one per hard-swap condition; noise floor; matching random seeds.
- Config hash of baseline run committed alongside Phase A run directory.

### 7.4 Statistical test
- For H1: bootstrap the per-sample degradation and FLD over samples (1000 resamples, 95% CI per condition); report whether fixed-width grid CIs separate from random-donor width-matched CIs. Primary test is CI-based; no formal p-value unless reviewer requests.
- For H2: Spearman rank correlation between width (t-b) and each of degradation and fld_cos over the four Grid 2 hard-swap conditions, alpha = 0.05. Caveat: n=4 is underpowered; report rho and caveat, do not claim significance.
- Seeds: single seed_base = 42 for Phase A. Multi-seed replication deferred; acknowledged as limitation in section 12.

## 8. Interfaces to add/change

```
# stage1/models/composer.py
PHASE_A_GRID: Dict[str, Tuple[int, int]]
FIXED_W4_GRID: Dict[str, Tuple[int, int]]
FIXED_B8_GRID: Dict[str, Tuple[int, int]]
RANDOM_FIXED_W4_GRID: Dict[str, Tuple[int, int]]
RANDOM_FIXED_B8_GRID: Dict[str, Tuple[int, int]]

def parse_condition_bt(condition_name: str, config=None) -> Tuple[str, Optional[int], Optional[int]]:
    """Return (cond_key in {no_swap, hard_swap, random_donor}, b, t)."""

def compute_random_donor_seed(seed_base: int, b: int, t: int) -> int:
    """Return seed_base*1000 + b*100 + t. Single source of truth."""

def compose_model(
    recipient, donor, b: int, t: int,
    condition: str = "hard_swap",
    seed: int = 42,   # MUST be used verbatim as RNG seed; no further math
) -> Tuple[AutoModelForCausalLM, Dict]:
    """For condition == 'random_donor':
        rng = random.Random(seed)
        block_width = t - b
        max_start = num_layers - block_width
        source_start = rng.randint(0, max_start)
    """

# stage1/run_phase_a.py
def build_phase_a_conditions(sanity: bool = False) -> List[Tuple[str, Optional[int], Optional[int]]]:
    """
    Full order:
      [("no_swap", None, None)] + Grid 1 hard (4) + Grid 1 random (4)
      + Grid 2 hard (4) + Grid 2 random (4)
    In sanity mode: exactly 2 conditions total
      [("no_swap", None, None), ("fixed_w4_pos2", 8, 12)]
    (2 conditions x 5 samples — master prompt 4.4.D.)
    """

def compute_fld(hs_recipient, hs_composed, n_layers: int = 28) -> Dict[str, float]:
    """Returns {fld_cos, fld_l2} at layer L-1, float32."""

def run_phase_a(config_path: str, sanity: bool = False) -> str:
    """CLI entry. Returns absolute run_dir path."""

# stage1/analysis/post_analysis.py (additive)
def load_phase_a_run(run_dir: str) -> Dict: ...
def compute_phase_a_primary_table(run_data: Dict, grid: str) -> List[Dict]: ...
def print_phase_a_summary(run_dir: str) -> None: ...
```

Existing Stage 1 print_summary / compute_recovery_sweep continue to work.

## 9. Files-to-touch (exhaustive)

### MUST create
- stage1/configs/stage2_confound.yaml — new. Required keys: models.recipient, models.donor, models.recipient_revision, models.donor_revision, boundary_grid: [8] (satisfies existing Stage1Config.validate; Phase A ignores it), t_fixed: 20 (legacy fallback), reference.b_ref: 8, reference.t_ref: 20, hidden_state.pooling: last_token, random_donor.seed: 42 (NOT seed_base — draft is buggy; see D2), random_donor.mode: same_width_random_source, dataset.name: mgsm, dataset.lang: zh, dataset.split: test, dataset.debug_n: null, generation.do_sample: false, generation.temperature: 0.0, generation.max_new_tokens: 512, evaluation.bootstrap_n: 1000, evaluation.bootstrap_ci: 0.95, evaluation.criteria_threshold: 2.

- stage1/run_phase_a.py — new CLI. Load config + data + models, iterate Phase A conditions, run inference (reusing unchanged run_inference), save parsed results + hidden states, compute PRIMARY metrics, write grid1_position_effect.csv, grid2_width_effect.csv, phase_a_all_conditions.csv, phase_a_summary.json, phase_a_summary.txt, manifest.json. Imports MUST be absolute (from stage1.models.composer import ...). See D5.

### MUST modify
- stage1/models/composer.py
  - Add PHASE_A_GRID / FIXED_W4_GRID / FIXED_B8_GRID / RANDOM_FIXED_W4_GRID / RANDOM_FIXED_B8_GRID constants.
  - Add parse_condition_bt.
  - Add compute_random_donor_seed.
  - FIX D1: rng = random.Random(seed) — use caller's seed verbatim.
  - Metadata returned for random_donor MUST include {seed, source_start, b, t}.

- stage1/analysis/post_analysis.py
  - Add load_phase_a_run, compute_phase_a_primary_table, print_phase_a_summary.
  - Every printed Phase A table must carry a header listing PRIMARY vs SECONDARY metrics using the wording in 7.1.
  - Do NOT touch existing Stage 1 entry points except additively.

### MUST NOT create or modify (forbidden — master prompt section 7)
- stage1/inference/runner.py
- stage1/inference/parser.py
- stage1/analysis/bds.py
- stage1/analysis/evaluator.py
- stage1/data/loader.py
- The prompt template and "Solution:" prefix.
- Any file under paper/, eval/, runs/.

Writer MUST fail loudly if any forbidden file is touched.

### MUST NOT touch for Phase A (deferred to Phase B/C)
- stage1/intervention/__init__.py
- stage1/intervention/patcher.py
- stage1/run_phase_b.py
- stage1/analysis/mediation.py
- stage1/run_phase_c.py

## 10. Test plan

### 10.1 Unit tests (stage1/tests/, pytest)
- test_parse_condition_bt.py: every grid mapping; unknown raises ValueError.
  - no_swap -> (no_swap, None, None)
  - hard_swap_b8 (t_fixed=20) -> (hard_swap, 8, 20)
  - random_donor_b8 (t_fixed=20) -> (random_donor, 8, 20)
  - fixed_w4_pos1 -> (hard_swap, 4, 8)
  - fixed_w4_pos2 -> (hard_swap, 8, 12)
  - fixed_w4_pos3 -> (hard_swap, 12, 16)
  - fixed_w4_pos4 -> (hard_swap, 16, 20)
  - fixed_b8_w2 -> (hard_swap, 8, 10)
  - fixed_b8_w4 -> (hard_swap, 8, 12)
  - fixed_b8_w6 -> (hard_swap, 8, 14)
  - fixed_b8_w8 -> (hard_swap, 8, 16)
  - random_fixed_w4_pos3 -> (random_donor, 12, 16)
  - random_fixed_b8_w6 -> (random_donor, 8, 14)
- test_compute_random_donor_seed.py:
  - (42, 8, 12) == 42812
  - (42, 12, 16) == 43216
  - (42, 16, 20) == 43620
- test_compose_model_random_seed.py:
  - compose_model(random_donor, seed=42812) deterministic across 2 calls.
  - Different seeds produce different source_starts (or different windows).

### 10.2 Smoke test (master prompt 4.4.D)
python -m stage1.run_phase_a --config stage1/configs/stage2_confound.yaml --sanity
Must complete under 10 min on 1 GPU, exactly 2 conditions (no_swap + fixed_w4_pos2), 5 samples each, produce non-empty phase_a_summary.txt + phase_a_all_conditions.csv. Sanity must NOT run all 17 conditions.

### 10.3 Evaluation sanity
- manifest.json has random_donor_seeds dict mapping every random_* condition to its computed seed.
- phase_a_summary.txt contains literal headers "PRIMARY metrics" and "SECONDARY/EXPLORATORY".
- No NaN in any numeric field of phase_a_all_conditions.csv.

### 10.4 Full run (post-green-light)
python -m stage1.run_phase_a --config stage1/configs/stage2_confound.yaml
17 conditions × 250 samples. Wall-clock in section 13.

## 11. Acceptance criteria

A1. stage1/configs/stage2_confound.yaml exists and yaml.safe_load(...)["generation"] equals {do_sample: False, temperature: 0.0, max_new_tokens: 512}. Parser/generation invariant; failure blocks merge.

A2. stage1/inference/parser.py byte-identical to pre-Phase-A version. git diff HEAD -- stage1/inference/parser.py empty after merge.

A3. Sanity run produces <run_dir>/phase_a_all_conditions.csv with exactly 1 data row (only fixed_w4_pos2; no_swap is baseline row excluded from per-condition table).

A4. Full run produces <run_dir>/grid1_position_effect.csv with 8 data rows (4 fixed_w4_pos* + 4 random_fixed_w4_pos*) and <run_dir>/grid2_width_effect.csv with 8 data rows (4 fixed_b8_w* + 4 random_fixed_b8_w*). Column order MUST be condition,b,t,width,accuracy,degradation,fld_cos,fld_l2.

A5. <run_dir>/phase_a_summary.json contains key primary_metrics_note whose string value contains both substrings "PRIMARY" and "SECONDARY" and names the four primary metrics (accuracy, degradation, fld_cos, fld_l2).

A6. <run_dir>/manifest.json contains random_donor_seeds mapping; for every random_* condition seed == 42*1000 + b*100 + t. Automated helper recomputes and diffs.

A7. No condition row in any CSV has fld_cos == NaN or fld_l2 == NaN.

A8. grid1_position_effect.csv and grid2_width_effect.csv contain at least one numerically non-trivial primary metric — degradation column is not uniformly zero and not uniformly equal to the baseline.

A9. Human-readable phase_a_summary.txt contains a banner flagging recovery-zone metrics as NOT directly comparable across Grid 1 conditions, with PRIMARY/SECONDARY classification included.

A10. Conservative-wording gate. No summary artifact in <run_dir> contains any of these forbidden phrases:
  - proves the mechanism
  - causal proof
  - identifies the true cause
  - fully explains
  Automated grep over phase_a_summary.txt, phase_a_summary.json, any .md artifact the writer produces. Writer MUST include this grep in the sanity-check block and fail the run on any match.

A11. Sanity mode finishes in under 10 minutes on a single A100/H100 (or equivalent). Regression otherwise.

A12. All unit tests in 10.1 pass.

A13. Forbidden files list (section 9) is byte-identical pre- and post-Phase-A. Enforced via git diff --stat HEAD restricted to the forbidden paths.

## 12. Risks and ablations

- Confound residuals. Grid 1 still varies bottom-block depth b, top-block depth L-t, and absolute layer index of the donor block. Phase A disentangles only the position-vs-width axis. State explicitly in summary.
- Recovery-zone comparability. Writer must not promote RD/CKA/RSA to primary in any CSV/JSON column.
- Random-donor seed leakage. If compose_model is not fixed per D1, random-donor source windows silently depend on (seed_base*1000+b*100+t)*1000+b, and manifest's declared seed does NOT match actual RNG stream. Must be fixed.
- Eval contamination. MGSM-zh already used in Stage 1; no unseen-data leakage. No training, so transfer leakage does not apply.
- n=4 per grid. H2 Spearman over 4 points is underpowered; report rho with caveat.
- Single seed_base. Multi-seed replication deferred; Phase A results framed as preliminary single-seed separation evidence.
- Hidden-state memory. All 17 conditions × 250 samples × 28 layers × 1536 fp32 ≈ 14 GB total. Writer should stream per-condition: hold no_swap + current only; free after FLD.
- Ablation: "does the result hold without random-donor controls?" — writer can report hard rows alone from same CSV without re-running.

## 13. Compute budget
- Sanity: 2 conditions × 5 samples; ~10 GPU-minutes, 1 GB disk.
- Full: 17 conditions × 250 samples × ~512 greedy tokens on Qwen2.5-1.5B; Stage 1 parallels ran 3-4 GPU-hours on A100. Budget 6 GPU-hours wall-clock (2x safety), 20 GB disk (~1.1 GB fp16 × 17).
- Memory: 24 GB VRAM for recipient + donor in fp16 simultaneously.
- No training compute.

## 14. Rollback
Phase A is additive.
- git revert <phase_a_commit>
- rm -rf stage1/outputs/phase_a
Stage 1 entry points (run.py, post_analysis.print_summary) untouched. No data migration. No paper claims gated on Phase A, so rollback does not require claim-skeptic re-validation. post_analysis.py additive functions (load_phase_a_run, compute_phase_a_primary_table, print_phase_a_summary) can be deleted in isolation.

---

## Appendix — Draft audit findings (writer-visible)

Files audited in draft:
- stage1/configs/stage2_confound.yaml (untracked)
- stage1/run_phase_a.py (untracked)
- stage1/models/composer.py (modified, unstaged)
- stage1/analysis/post_analysis.py (modified, staged)

### D1 (CONFIRMED, BLOCKING) — double-seed bug in compose_model
stage1/models/composer.py ~line 201:
  rng = random.Random(seed * 1000 + b)
Callers already pass seed = seed_base*1000 + b*100 + t via compute_random_donor_seed. This double-encodes the seed; the true RNG state does NOT match the declared formula.
Fix: rng = random.Random(seed). Caller owns the formula. Verify via test_compose_model_random_seed.py.

### D2 (CONFIRMED, BLOCKING) — yaml field name mismatch
stage1/configs/stage2_confound.yaml:26 uses seed_base: 42. stage1/utils/config.py:41-43 defines RandomDonorConfig with only seed: int. load_config raises TypeError: unexpected keyword argument 'seed_base'.
Fix: rename yaml key to seed: 42. Do not rename dataclass field (Stage 1 reads config.random_donor.seed).

### D3 (CONFIRMED, BLOCKING) — missing boundary_grid in Phase A yaml
Stage1Config requires boundary_grid: List[int] and validate() iterates it. Draft yaml has no boundary_grid key.
Fix: add boundary_grid: [8] (harmless non-empty; Phase A iteration comes from build_phase_a_conditions).

### D4 (CONFIRMED, BLOCKING) — sanity mode wrong condition count
stage1/run_phase_a.py::build_phase_a_conditions lines 64-69. Current sanity keeps one condition from each of four groups + no_swap = 5 conditions. Master prompt 4.4.D demands "2 conditions x 5 samples".
Fix: sanity mode returns exactly [("no_swap", None, None), ("fixed_w4_pos2", 8, 12)] — total 2.

### D5 (CONFIRMED, BLOCKING) — broken imports in run_phase_a.py
Lines 25-41 use bare module paths (from utils.config import load_config, from models.composer import ...). Repo convention (stage1/analysis/post_analysis.py:29) is absolute stage1.*. python stage1/run_phase_a.py raises ModuleNotFoundError.
Fix: absolute imports; invoke via python -m stage1.run_phase_a --sanity.

### D6 (CONFIRMED, BLOCKING) — no_swap seed misleading in manifest
run_phase_a.py:185:
  rd_seed = compute_random_donor_seed(seed_base, b, t) if (b is not None and t is not None) else seed_base
For no_swap, rd_seed is set to seed_base (=42) and passed to get_condition_model which ignores it for no_swap. Manifest's rd_seed for no_swap is misleadingly 42.
Fix: store None for no_swap.

### D7 (NOTE, non-blocking) — redundant FLD implementation
run_phase_a.py::compute_fld duplicates logic in post_analysis.compute_recovery_metrics. Acceptable because Phase A needs lightweight primary-only path without CKA/RSA.

### D8 (CONFIRMED, BLOCKING) — post_analysis.py has no Phase A path
stage1/analysis/post_analysis.py (modified, staged) only iterates boundary_grid with hard_swap_b{X}/random_donor_b{X}. No concept of fixed_w4_pos*, no primary/secondary flag, no Phase A report path.
Fix: add three new functions per section 8 so python -m stage1.analysis.post_analysis --run_dir <phase_a_run> produces PRIMARY two-grid tables (A4/A5/A9).

### D9 (NOTE, non-blocking) — conservative-wording not enforced
Draft summary says "Recovery-zone mean metrics are NOT directly comparable" (good), but no automated check that forbidden phrases are absent from artifacts. A10 makes this a required acceptance criterion.

### D10 (NOTE, non-blocking) — in-memory hidden states
run_phase_a.py accumulates all_hs across all 17 conditions. Writer should stream/free per-condition after FLD; keep only no_swap + current in RAM.

### D11 (CONFIRMED, BLOCKING) — sanity check weakened by D6
Draft sanity check (line 401) verifies "rd_seed" in all_metadata.get(c,{}), passes trivially due to D6 storing seed_base for every condition. After D6 fix, sanity check must only demand seeds for random_* conditions.

### Writer closure checklist
- [ ] D1 fixed, unit tested
- [ ] D2 fixed, load_config succeeds on new yaml
- [ ] D3 fixed, validate() passes
- [ ] D4 fixed, sanity returns exactly 2 conditions
- [ ] D5 fixed, python -m stage1.run_phase_a --sanity runs
- [ ] D6 fixed, manifest does not claim a seed for no_swap
- [ ] D8 fixed, post_analysis produces Phase A tables
- [ ] D9 grep added to sanity block
- [ ] D11 sanity check tightened
- [ ] Forbidden-file list untouched (section 9 / A13)

---

## Addendum B — Grid-intersection reality (added after first watcher review)

The two Phase A grids deliberately intersect at (b=8, t=12): `fixed_w4_pos2` (Grid 1) and `fixed_b8_w4` (Grid 2) are the same design point. The master prompt §4.1 specifies both grids verbatim, so this intersection is intentional, not a spec bug.

Consequences:
- For `hard_swap`, both conditions run the same forward pass and yield identical primary metrics — useful as a self-consistency sanity check.
- For random_donor, the formula `seed = seed_base*1000 + b*100 + t` produces `42812` for both. Data-auditor F2 correctly flags this: across the 8 random_* conditions only 7 distinct seeds exist.

Writer responsibilities (amendment):
1. Still build all 8 random_* conditions as distinct table rows (keeps Grid 1 and Grid 2 tables complete).
2. Record both labels in the manifest under `grid_intersection_notes` with the shared `(b,t)` and shared `seed`.
3. Add an explicit one-line note to `phase_a_summary.txt` stating the two conditions share (8,12) by construction and should yield identical hard-swap metrics and identical random-donor draws — this is a sanity check, not independent evidence.
4. Do NOT add a grid-index to the seed formula. The formula is fixed by spec §6 and master prompt §4.1.

Acceptance criterion amendment:
- A6 still stands but interpret "seeds for every random_* condition" as "8 entries, 7 distinct values". Automated check: `len(random_donor_seeds) == 8` AND `len(set(random_donor_seeds.values())) == 7` AND the one collision is between `random_fixed_w4_pos2` and `random_fixed_b8_w4` both mapped to `42812`.
