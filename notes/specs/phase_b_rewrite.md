# phase_b_rewrite

Scope note: This spec is a full rewrite of the existing internal Phase B (prompt-side restoration intervention) driven by 14 blocker findings from informal review. It addresses correctness (KV-cache continuation, Qwen2 forward plumbing), reproducibility (determinism, encoding, versions), and conservative-wording discipline (FORBIDDEN_PHRASES gate, removal of spurious recovery-% claim, effect-size-gated comparatives). Phase A is frozen; Phase C terminology is reserved and MUST NOT appear in Phase B artifacts.

Methodological caveat to echo in every summary artifact: "Patching applies only to prompt-side hidden-state processing. Clean hidden states are available for prompt tokens only. This is prompt-side restoration intervention, not full-sequence causal intervention."

## 1. Goal

Rewrite `stage1/intervention/patcher.py` and `stage1/run_phase_b.py` so that prompt-side hidden-state patching (a) preserves the patch effect across all autoregressively generated tokens via a `DynamicCache` built from a manual per-layer Qwen2 prompt forward with correct causal mask and RoPE, and (b) emits summary artifacts that pass an expanded, shared FORBIDDEN_PHRASES gate identical in semantics to Phase A's `run_phase_a.py:859-883`.

Testable: a single `pytest -q stage1/tests/test_phase_b_patcher.py` run plus one short real-data `--sanity` CLI run must both pass the acceptance criteria in §11. Failure of any single acceptance check fails the rewrite.

## 2. Hypothesis and falsification

Hypothesis H1 (engineering correctness): the rewritten patched-inference path is bytewise equivalent to `model.generate(input_ids)` when `patch_layers=[]`, and equivalent to the unpatched HuggingFace forward when `patch_states={}` under `forward_with_patches` (final-logits max-abs-diff < 1e-4 in fp16, < 1e-5 in fp32 on a CPU identity check).

Falsification H1: any test in §10 producing a mismatch above tolerance, OR the first generated token of a non-empty patch condition matching no_patch for >95% of samples (indicating the patch never propagated past embeddings).

Hypothesis H2 (methodology): under the conservative-wording gate, the Phase B summary artifacts never contain any phrase in the expanded FORBIDDEN_PHRASES list (§8), and any comparative claim between `patch_recovery_*` and `patch_boundary_local` is emitted only when both deltas are positive and their difference exceeds `epsilon_delta` (configurable, default 0.02 absolute accuracy).

Falsification H2: grep of `phase_b_summary.txt` / `phase_b_summary.json` / `restoration_table.csv` / `corruption_table.csv` under any sanity or full run produces a non-empty violation list; or the comparative sentence fires when its gate conditions are not met.

Neither hypothesis asserts a scientific conclusion about the recipient model — both are engineering/reporting-discipline hypotheses. Scientific restoration claims are scoped per §11 Acceptance.

## 3. Prior art and delta

We cite two primary sources for activation-patching / causal tracing methodology (verified as resolvable arxiv IDs; no invented citations):

- **Meng et al. 2022, "Locating and Editing Factual Associations in GPT" (ROME, causal tracing), arXiv:2202.05262.** We borrow the activation-patching primitive: inject donor hidden states at specified layers into a target forward. Delta vs ROME: (a) we patch full per-token prompt hidden states (not restricted to a subject-token window) and we patch residual-stream outputs after the entire decoder block, not MLP-only; (b) our "donor" and "target" are two variants of the same model compiled by two-cut swap (recipient vs. hard_swap_b8 composed), not "clean" vs. "noised embedding" corruptions; (c) we explicitly scope restoration claims to prompt tokens and reserve full-sequence causal claims (Phase C).
- **Vig et al. 2020, "Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias," arXiv:2004.12265.** We borrow the mediation framing (natural direct vs. natural indirect effect via activation substitution) but we do not yet estimate full NDE/NIE — we report marginal restoration deltas only. Phase C will extend to the proportion decomposition; Phase B MUST NOT preempt that terminology.

Delta of THIS rewrite vs. the existing internal Phase B code:
1. Replace post-prompt `model.generate(current_ids, ...)` continuation with a `DynamicCache`-backed autoregressive loop seeded by per-layer patched `past_key_values`, so patches propagate to all generated tokens.
2. Wire `attention_mask`, `position_ids`, `position_embeddings=(cos, sin)`, and a causal 4D mask into every per-layer call, as required by current `transformers` Qwen2 layer signature.
3. Add clean_baseline (no_patch on recipient) to populate `delta_from_clean_baseline`.
4. Switch all file writes to `encoding="utf-8"`.
5. Remove the spurious "% recovery" print; reserve Phase C terminology.
6. Factor FORBIDDEN_PHRASES into a shared util and gate Phase B artifacts.
7. Gate the "more causally relevant" comparative sentence by effect-size epsilon.
8. Replace vacuous sanity checks with real equivalence/hash/accuracy-recovery checks.
9. Log `t=20` explicitly; add `methodology` column to CSV (or sibling README).
10. Mirror corruption arm granularity to match restoration (default (a) per prompt).
11. Enforce determinism flags and log environment versions + git_sha.
12. Collapse O(n²) gold lookup; drop unused imports; free donor after `compose_model`.

## 4. Datasets

n/a — inference-only intervention. No training data is touched.

Fixed evaluation subset (ground truth; unchanged from Phase A):
- name: MGSM
- language: Chinese (zh)
- split: test
- n_samples: 250 (full); 5 under `--sanity`
- loader: `stage1/data/loader.py::load_mgsm(config)` — DO NOT modify; this spec only *reads* from it.
- hash/version pointer: `notes/data_changelog.md` entry for MGSM-zh as referenced by Phase A manifest (`run_phase_a.py` emits it). Phase B manifest MUST copy that entry verbatim into `phase_b_summary.json.dataset` so the two phases are pinned to the same loader state.
- license: MGSM inherits from original GSM8K (MIT) + Google translations (Apache-2.0 per Shi et al. 2023, arXiv:2210.03057). No new license obligations introduced by this rewrite.

## 5. Models and tokenizers

Fixed; copied from `stage1/configs/stage2_confound.yaml`:
- recipient: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- donor: `Qwen/Qwen2.5-1.5B`, revision `8faed761d45a263340a0528343f099c05c9a4323`
- tokenizer: from recipient, same revision
- composed model: `compose_model(recipient, donor, b=8, t=20, condition="hard_swap")` (treatment `hard_swap_b8`).
- Architecture assertions: `num_hidden_layers == 28`, `hidden_size == 1536`. Fail-fast if violated.
- dtype: weights loaded in `float16` (matches Phase A); analysis paths (logit comparisons, hash checks) MUST cast to `float32` before reduction.

No new models. No fine-tuning. No tokenizer patches.

## 6. Training config

n/a — inference-only intervention. No optimizer, no schedule, no gradient accumulation, no batch training.

Determinism and seed policy (still applies to inference):
- `seed = 42` (config.random_donor.seed) wired into:
  - `torch.manual_seed(seed)`
  - `torch.cuda.manual_seed_all(seed)` when CUDA available
  - `transformers.set_seed(seed)`
  - `random.seed(seed)`, `numpy.random.seed(seed)` (used by any bootstrapping in §7)
- `torch.use_deterministic_algorithms(True, warn_only=True)`
- Env: `os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")` set BEFORE any CUDA op
- Generation: `do_sample=False`, `temperature=0.0`, `max_new_tokens=512` (greedy; copied from config)
- Precision: fp16 weights, fp32 for comparisons and hashing

Logged into `phase_b_summary.json.environment`:
- `torch.__version__`
- `transformers.__version__`
- `device` (e.g. `cuda:0`, with `torch.cuda.get_device_name(0)` when available)
- `git_sha` (from `subprocess.check_output(["git","rev-parse","HEAD"])`, strip; fail-soft to string `"unknown"` if not a git checkout)
- `cublas_workspace_config`
- `deterministic_algorithms_enabled`: True
- `seed`: 42

## 7. Evaluation protocol

Primary metric (single):
- `accuracy` = mean of `exact_match(gold_answer, parsed_normalized_answer)` over the 250-sample MGSM-zh subset, computed via **unchanged** `analysis/evaluator.py::exact_match` and `inference/parser.py::parse_answer`. DO NOT modify either file.

Reduction axes:
- per-condition accuracy (one row per patch condition)
- per-sample `correct` (bool) retained in JSONL for audit

Per-language breakdown: n/a (single language, zh).

Conditions (full run, 10 total):
- Restoration on composed (hard_swap_b8):
  - `no_patch` (composed, no intervention) — yields `no_patch_accuracy`
  - `patch_boundary_local` layers [7,8,9]
  - `patch_recovery_early` layers [20,21,22]
  - `patch_recovery_full` layers [20..27]
  - `patch_final_only` layers [27]
  - `patch_all_downstream` layers [8..27]
- Clean baseline on recipient:
  - `clean_no_patch` (recipient, no intervention) — yields `clean_baseline_accuracy`
- Reverse corruption on recipient (mirrored to match restoration granularity — Finding #11 default (a)):
  - `corrupt_boundary_local` layers [7,8,9]
  - `corrupt_recovery_early` layers [20,21,22]
  - `corrupt_recovery_full` layers [20..27]
  - `corrupt_final_only` layers [27]

Sanity-mode (`--sanity`) conditions: `{no_patch, patch_recovery_full, clean_no_patch, corrupt_recovery_full}` × 5 samples.

Baselines (matched-config):
- For restoration deltas: `no_patch` (composed model on same 250 samples, same seed, same generation config).
- For corruption deltas: `clean_no_patch` (recipient model on same 250 samples, same seed, same generation config).
- Cross-phase check: `no_patch_accuracy` MUST equal the Phase A `hard_swap_b8` accuracy within |Δ| ≤ 0.008 (2/250 samples) tolerance. If it does not, fail the run with a hard error pointing to a generation-config or determinism drift.

Statistical test:
- For the single comparative claim (best restoration vs. `patch_boundary_local`): paired bootstrap over sample-level `correct` booleans, 1000 resamples, seed=0, 95% percentile CI of `(delta_best - delta_boundary_local)`. Comparative sentence emitted only if: both deltas > 0 AND point estimate of the difference > `epsilon_delta` (default 0.02) AND 95% CI lower bound > 0. Otherwise emit neutral language.
- α = 0.05 (one-sided lower bound)

No per-layer ANOVA, no multi-seed variance reporting in Phase B (single-seed per memory rule; Phase C handles multi-seed).

## 8. Interfaces to add/change

All paths absolute under `C:\Users\system1\Boundary-sensitivity\`. Function signatures are Python type-hinted.

New / modified in `stage1/intervention/patcher.py`:

```python
# New — Qwen2 plumbing helpers
def _build_prompt_inputs(model, input_ids: torch.Tensor) -> dict:
    """Return {'hidden', 'attention_mask_4d', 'position_ids', 'position_embeddings'}.
    Uses model.model._update_causal_mask (or equivalent for this transformers version)
    and model.model.rotary_emb to precompute RoPE (cos, sin) at prompt scale."""

# Modified — returns a DynamicCache usable by generate()
def forward_with_patches(
    model: "AutoModelForCausalLM",
    input_ids: torch.Tensor,
    patch_states: Dict[int, torch.Tensor],
    *,
    return_cache: bool = True,
) -> Tuple[torch.Tensor, List[torch.Tensor], "DynamicCache"]:
    """Per-layer manual prompt forward with correct attention_mask / position_ids /
    position_embeddings. Captures per-layer key/value via DynamicCache (populated
    by passing past_key_value=cache and use_cache=True to each layer call).
    Patches hidden states AFTER each specified layer, then that patched hidden
    is passed to the next layer (so downstream KV cache entries are computed
    from the patched residual stream)."""

# Modified — uses DynamicCache continuation instead of re-running generate on prompt
def run_patched_inference_single(
    model, tokenizer, prompt: str,
    patch_config: PatchConfig,
    clean_layer_states: Optional[List[torch.Tensor]] = None,
    corrupt_layer_states: Optional[List[torch.Tensor]] = None,
    generation_config: Optional[dict] = None,
    device=None,
) -> Dict:
    """Now: (1) call forward_with_patches to obtain (final_hidden, _, cache);
    (2) emit first token from lm_head(final_hidden[:, -1, :]);
    (3) call model.generate(
           input_ids=<first_token_as_1x1>,
           past_key_values=cache,
           use_cache=True,
           max_new_tokens=max_new_tokens - 1,
           do_sample=False)
       OR equivalent manual greedy loop — whichever the installed transformers
       version supports. Writer chooses; spec accepts either if §10 tests pass."""

# Unchanged signature
def extract_all_layer_hidden_states(...) -> List[torch.Tensor]: ...

# Unchanged signature, internals fix: O(n) gold lookup, explicit model.eval()
def run_patched_inference(...) -> List[Dict]: ...
```

New in `stage1/utils/wording.py` (NEW FILE — shared conservative-wording gate):

```python
FORBIDDEN_PHRASES: Tuple[str, ...] = (
    "proves the mechanism",
    "proves mechanism",
    "causal proof",
    "identifies the true cause",
    "fully explains",
    "demonstrates causation",
    "restoration effect",       # Phase C reserved
    "residual effect",          # Phase C reserved
    "restoration proportion",   # Phase C reserved
)

def check_artifacts_for_forbidden(paths: Sequence[str]) -> List[str]:
    """Return list of violation strings ('<path>: found forbidden phrase <p>').
    Reads each path with encoding='utf-8'; missing paths skipped silently."""
```

Phase A (`run_phase_a.py`) is DO-NOT-MODIFY, but its local `FORBIDDEN_PHRASES` list can be left in place; this new util is ADDITIVE and used only by Phase B. (The spec does NOT refactor Phase A — Finding #6's "factor into a shared util" is satisfied by creating the util now and leaving Phase A's inline list untouched; any future Phase A refactor is out of scope.)

New / modified in `stage1/run_phase_b.py`:

```python
def run_phase_b(config_path: str, sanity: bool = False) -> str:
    """(1) apply determinism flags, (2) load data + models once,
    (3) compose hard_swap_b8 model, (4) free donor, (5) run clean_no_patch on
    recipient, (6) run all restoration conditions on composed, (7) run all
    corruption conditions on recipient, (8) build tables with both
    delta_from_no_patch and delta_from_clean_baseline populated, (9) write
    CSVs + JSON + TXT with encoding='utf-8' and methodology column,
    (10) apply conservative-wording gate post-write, (11) run real sanity
    checks (§10), (12) return run_dir."""
```

## 9. Files-to-touch (exhaustive)

- `C:\Users\system1\Boundary-sensitivity\stage1\intervention\patcher.py` — `forward_with_patches`, `run_patched_inference_single`, `_build_prompt_inputs` (new), `_get_model_components` — **modify**. Rationale: Findings #1, #2, #10 (per-condition methodology string emitted as part of result dict).
- `C:\Users\system1\Boundary-sensitivity\stage1\run_phase_b.py` — `run_phase_b`, `_save_condition_results`, `_write_csv`, sanity-check block, interpretation block — **modify**. Rationale: Findings #3, #4, #5, #6, #7, #8, #9, #10, #11, #12, #13.
- `C:\Users\system1\Boundary-sensitivity\stage1\utils\wording.py` — `FORBIDDEN_PHRASES`, `check_artifacts_for_forbidden` — **add**. Rationale: Finding #6 shared util.
- `C:\Users\system1\Boundary-sensitivity\stage1\tests\test_phase_b_patcher.py` — `test_identity_patch_equivalence`, `test_empty_patch_generate_bytewise_equal`, `test_all_clean_patch_matches_recipient`, `test_state_dict_hash_stable`, `test_forbidden_phrases_gate` — **add**. Rationale: Findings #2 equivalence, #8 real sanity, #14 KV-cache unit test.
- `C:\Users\system1\Boundary-sensitivity\stage1\intervention\__init__.py` — re-export updated `forward_with_patches` signature — **modify (import list only)**. Rationale: downstream imports.
- `C:\Users\system1\Boundary-sensitivity\notes\specs\phase_b_rewrite.md` — this spec — **add** (already the write target).

Do-NOT-touch (enforced by memory + agent constraints):
- `stage1/inference/runner.py`
- `stage1/inference/parser.py`
- `stage1/analysis/bds.py`
- `stage1/analysis/evaluator.py`
- `stage1/data/loader.py`
- `stage1/run_phase_a.py`
- `stage1/models/composer.py` (read for donor-free verification; no edits)
- `stage1/configs/stage2_confound.yaml`

Deletions:
- Inside `run_phase_b.py`: remove unused imports `copy`, `compute_accuracy`. Remove the line that computes `% of performance (rough estimate)`. Remove hardcoded `("Parser not modified", True)` / `("Methodological constraint documented", True)` checks.

## 10. Test plan

Unit tests — file `stage1/tests/test_phase_b_patcher.py`, runnable via `pytest -q stage1/tests/test_phase_b_patcher.py`:

1. `test_identity_patch_equivalence` (covers Finding #2, #14): run `forward_with_patches(model, input_ids, patch_states={})` on a tiny fixed input; final-layer logits MUST equal `model(input_ids).logits` within `atol=1e-4` (fp16) / `1e-5` (fp32).
2. `test_empty_patch_generate_bytewise_equal` (covers Finding #14): `run_patched_inference_single(..., patch_config=PatchConfig("no_patch", [], "restoration"))` output token IDs MUST be byte-identical to `model.generate(input_ids, do_sample=False, max_new_tokens=32)[0, prompt_len:]` for at least one held-out fixture prompt. Tolerance: 0 (exact match on token IDs).
3. `test_all_clean_patch_matches_recipient` (covers Finding #8): on composed model, patch all 28 layers with clean recipient states; final hidden state at prompt-last position MUST equal the recipient's final hidden state at the same position within `atol=1e-3` (fp16). Generated token sequence SHOULD equal recipient's generated sequence on ≥ 95% of 5 fixture prompts (allows rare greedy tie-break drift).
4. `test_state_dict_hash_stable` (covers Finding #8, #13): SHA-256 of concatenated `composed.state_dict()` tensor bytes before and after a full `run_patched_inference` pass MUST be identical. No in-place mutation of composed weights.
5. `test_forbidden_phrases_gate` (covers Finding #6): write a temp file containing each phrase in `FORBIDDEN_PHRASES`; `check_artifacts_for_forbidden` MUST return a non-empty violation list whose length equals len(FORBIDDEN_PHRASES).

Smoke test — CLI:
- `python -m stage1.run_phase_b --config stage1/configs/stage2_confound.yaml --sanity` MUST complete end-to-end in under 10 min on the current dev GPU, produce `phase_b_summary.txt`, `phase_b_summary.json`, `restoration_table.csv`, `corruption_table.csv`, and exit 0.

Eval-sanity checks emitted inline by `run_phase_b` (real, not vacuous):
- `no_patch_accuracy` within 0.008 of the recorded Phase A `hard_swap_b8` accuracy (loaded from the latest `stage1/outputs/phase_a/run_*/phase_a_summary.json`).
- `clean_baseline_accuracy` within 0.008 of the recorded Phase A `no_swap` accuracy.
- No NaN in any accuracy or delta cell.
- `t=20` appears verbatim in `phase_b_summary.json` under `t_fixed`.
- Conservative-wording gate: `check_artifacts_for_forbidden([summary.txt, summary.json, restoration_table.csv, corruption_table.csv])` returns `[]`.
- Composed-model `state_dict` SHA-256 recorded at start of run equals the SHA-256 recorded at end of run.

Any single failure is a hard RuntimeError; run exits non-zero.

## 11. Acceptance criteria

All criteria are checked against artifacts under a single run directory `RD = stage1/outputs/phase_b/run_<timestamp>`. All thresholds are numeric and auditable.

1. `pytest -q stage1/tests/test_phase_b_patcher.py` exits 0; all 5 tests pass.
2. `RD/phase_b_summary.json` exists, is valid UTF-8 JSON, and contains keys: `phase=="B"`, `t_fixed==20`, `no_patch_accuracy`, `clean_baseline_accuracy` (non-null float), `restoration_table` (len==6 full / len==2 sanity), `corruption_table` (len==4 full / len==1 sanity), `methodological_constraint` (non-empty string containing the phrase "prompt-side"), `environment.git_sha`, `environment.torch_version`, `environment.transformers_version`, `environment.seed==42`, `compose_meta`, `dataset` block mirroring Phase A manifest.
3. `RD/restoration_table.csv` and `RD/corruption_table.csv` exist, are UTF-8, and each has a `methodology` column whose cell value for every row equals the exact string: `"prompt-side patching; patch at prompt tokens only; continuation via DynamicCache"`.
4. `delta_from_clean_baseline` is a non-null float for every row in `corruption_table.csv`.
5. Conservative-wording gate: `check_artifacts_for_forbidden` returns `[]` over `{summary.txt, summary.json, restoration_table.csv, corruption_table.csv}`.
6. First-token-KV-cache equivalence (test #2): passes for the ≥1 fixture prompt.
7. `|no_patch_accuracy − phase_a.hard_swap_b8_accuracy| ≤ 0.008` AND `|clean_baseline_accuracy − phase_a.no_swap_accuracy| ≤ 0.008`.
8. Composed state_dict SHA-256 unchanged across the run (recorded as `compose_meta.state_dict_sha256_before == ..._after`).
9. No occurrence of the literal substring "of performance (rough estimate)" anywhere in `RD/`.
10. If the comparative-sentence gate fires (both deltas > 0 AND difference > 0.02 AND bootstrap CI lower bound > 0), `phase_b_summary.txt` contains exactly one sentence beginning with "Recovery-side intervention"; otherwise `phase_b_summary.txt` contains exactly one sentence beginning with "Restoration deltas do not meet the effect-size threshold for a directional claim".
11. `RD/phase_b_summary.txt` header contains the literal line `"recovery-zone layers [20..27] defined at fixed t=20"`.

Restoration/causal-relevance claims in the TXT summary are restricted to the 4 corruption-mirrored conditions (`patch_boundary_local`, `patch_recovery_early`, `patch_recovery_full`, `patch_final_only`) per Finding #11(a); claims about `patch_all_downstream` are reported as accuracy only, with no comparative adjective.

## 12. Risks and ablations

Risks:
- R1 — `transformers` version drift: `_update_causal_mask` and `rotary_emb` APIs have moved between 4.38 and 4.45. Mitigation: pin the call via `try/except AttributeError` against both known paths; log the chosen path in `compose_meta`. Fail-fast with a clear message if neither path resolves.
- R2 — `DynamicCache` ownership: if writer's transformers version lacks `DynamicCache.from_legacy_cache`, fall back to a legacy tuple-of-tuples cache. Test #2 (byte-equality) catches any silent drift between the two paths.
- R3 — Determinism flag side-effects: `use_deterministic_algorithms(True)` can raise on some CUDA ops. We set `warn_only=True` and log any warnings under `environment.determinism_warnings`.
- R4 — UTF-8 on Windows: Windows default is cp1252; any `open("w")` without explicit encoding crashes on MGSM-zh. Blanket enforcement in §11.3 plus a lint grep in CI (out of scope here — call out as TODO in Phase C spec).
- R5 — Cache-warmup cost: building full clean/corrupt per-layer states for every sample doubles memory traffic vs. the buggy path. Mitigation: extract states once per sample per direction, free immediately after use; do NOT cache across samples (would explode GPU mem).
- R6 — Donor residency on GPU: memory rule says free `donor` after `compose_model`. First verify in `stage1/models/composer.py` (read-only) that `copy.deepcopy(recipient)` genuinely copies rather than aliases layer params, then writer adds `del donor; gc.collect(); torch.cuda.empty_cache()` immediately after compose.
- R7 — Eval contamination: none added. MGSM-zh subset is identical to Phase A's; no new data source.
- R8 — Seed coupling: single-seed per phase memory rule. Any multi-seed sweep is Phase C territory.
- R9 — Premature-Phase-C-terminology leak: the FORBIDDEN_PHRASES list explicitly blocks "restoration effect", "residual effect", "restoration proportion" so reviewer language cannot slip into Phase B output.

Ablations (not required in this rewrite, listed for completeness):
- A1 — Corruption-arm default (a) mirror vs. (b) scoped-claims: user-overridable via `--corruption-scope {mirror,scoped-claims-only}`; default `mirror`.
- A2 — `epsilon_delta` default 0.02: the spec fixes 0.02 but makes the value a module-level constant `EPSILON_DELTA` in `run_phase_b.py` so a future review can bump it without spec churn.

## 13. Compute budget

- Inference-only. No training.
- Per-sample cost: 1 forward for clean-state extraction + 1 patched forward + up-to-512-token greedy continuation = ~1.5× a plain generate.
- Conditions × samples: (6 restoration + 1 clean_no_patch + 4 corruption) × 250 = 2,750 sample-runs + 2,500 extra state-extraction forwards = ≈ 5,250 prompt-forwards + 2,750 generate-continuations.
- Wall-clock target on current single-GPU dev box: ≤ 6 h full run; ≤ 10 min `--sanity`.
- GPU-hours: ≤ 6 H100-equivalent hours (or ≤ 12 A6000 hours) for the full 250-sample, 10-condition run.
- Storage: per-sample JSONL ≈ 2 KB × 2,500 rows ≈ 5 MB; two CSVs < 10 KB; summary JSON+TXT < 50 KB; no hidden_states persisted (Phase C may opt in). Total ≤ 10 MB per run_dir.
- CPU RAM peak: one recipient + one composed resident simultaneously (donor freed post-compose) ≈ 2 × 3 GB fp16 = 6 GB weights + per-sample prompt states ≈ 28 layers × 1 × seq × 1536 × 2 bytes ≈ 1–3 MB per sample, released immediately.

No multi-node, no multi-GPU.

## 14. Rollback

The rewrite is contained in 4 files (patcher.py, run_phase_b.py, intervention/__init__.py, utils/wording.py) + 1 new test file. No schema migrations, no data regeneration, no checkpoint changes, no config changes.

Rollback procedure:
1. `git restore stage1/intervention/patcher.py stage1/run_phase_b.py stage1/intervention/__init__.py` to return those files to the pre-rewrite state.
2. `git rm stage1/utils/wording.py stage1/tests/test_phase_b_patcher.py notes/specs/phase_b_rewrite.md` to delete the additions.
3. Delete any `stage1/outputs/phase_b/run_<timestamp>/` directories produced during the rewrite (safe — no downstream job depends on them; Phase C has not yet started per memory).
4. No caches, no HF hub state, no dataset artifacts to roll back.
5. Phase A artifacts are untouched by construction (do-not-modify list); verify with `git status stage1/run_phase_a.py` showing no changes.

Rollback is idempotent and can be executed in under 30 seconds. No data loss beyond Phase B run outputs, which are reproducible from the pre-rewrite code if ever needed.
