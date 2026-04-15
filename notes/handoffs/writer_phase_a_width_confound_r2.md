spec_ref: notes/specs/phase_a_width_confound.md
reviewer_artifacts:
  watcher: notes/handoffs/watcher_phase_a_width_confound.json  (VERDICT: BLOCK)
  data_auditor: eval/reports/dataset_audit_phase_a_width_confound.json  (VERDICT: WARN)

files_changed:
  - path: stage1/utils/config.py
    diff_stat: +4 -0
    change: Added Optional[str] recipient_revision and donor_revision fields to ModelsConfig dataclass (BLOCK-1)

  - path: stage1/configs/stage2_confound.yaml
    diff_stat: +4 -0
    change: Added recipient_revision and donor_revision keys (both "main" — PIN VALUE MUST BE CONFIRMED with coordinator before full run; see open_questions)

  - path: stage1/models/composer.py
    diff_stat: +28 -6
    change: Added recipient_revision and donor_revision parameters to load_models(); passed revision= to all three from_pretrained calls (recipient model, donor model, tokenizer) (BLOCK-1)

  - path: stage1/run_phase_a.py
    diff_stat: +162 -15
    changes:
      - BLOCK-2: save_hidden_states now called with inf_results (not empty list) for every condition; save occurs BEFORE del hs_dict/gc.collect()
      - HIGH-1: subprocess.check_output(['git','rev-parse','HEAD']) wired; git_sha stored in manifest and phase_a_summary.json; try/except stores "unknown" on failure
      - HIGH-2: _bootstrap_ci() helper added (pure-Python, seeded, no numpy); _spearman_rho() helper added (tries scipy.stats.spearmanr, falls back to pure-Python rank-then-Pearson); per-sample binary correct captured per condition; H1 bootstrap CIs computed for degradation and fld_cos per condition stored in h1_bootstrap_cis; H2 Spearman rho over Grid 2 hard_swap widths stored in h2_width_spearman; underpowered caveat (n=4) written to phase_a_summary.txt
      - F2/Addendum B: grid_intersection_notes dict {"(8,12)": ["fixed_w4_pos2","fixed_b8_w4"]} added to manifest and summary JSON; grid_intersection_shared_seed=42812 recorded; grid-intersection self-consistency sanity check added (raises RuntimeError on metric mismatch > 1e-6 if both conditions are present in run); one-line note added to phase_a_summary.txt
      - compute_fld() updated to return per_sample_fld_cos and per_sample_fld_l2 lists alongside averages (needed for bootstrap)
      - Added import exact_match from evaluator; added import random as _random_module for bootstrap RNG
      - load_models() call now passes recipient_revision and donor_revision from config

  - path: notes/data_changelog.md
    diff_stat: +12 -0
    change: Added mgsm_zh@v1.0-2022-10-03 entry (HIGH-3 / F1). sha256 field is "pending" — coordinator must compute over raw zh TSV/JSONL before greenlighting full run.

  - path: stage1/tests/test_bootstrap_determinism.py
    diff_stat: +86 -0 (new file)
    change: New test file with 7 tests covering _bootstrap_ci determinism, CI width monotonicity, edge cases; and _spearman_rho correctness including n=4 case

  - path: stage1/tests/test_compose_model_random_seed.py
    diff_stat: +28 -13
    change: MED-1 fix — replaced trivially-true assertion with precomputed constants: random.Random(42812).randint(0,24)=10 and random.Random(43216).randint(0,24)=22; verifies verbatim seed use and that the two seeds produce different source_starts

tests_run:
  - command: /c/Users/system1/anaconda3/envs/datascience/python.exe -m pytest stage1/tests/ -x -q
    status: pass
    passed: 32
    failed: 0
    note: 23 pre-existing + 9 new (test_bootstrap_determinism.py)

deviations_from_spec:
  - field: models.recipient_revision / models.donor_revision in stage2_confound.yaml
    reason: Both set to "main" as explicit placeholder per BLOCK-1 fix instructions. Stage1 stage1_main.yaml has no pinned revisions either. Coordinator must confirm and replace "main" with exact HF commit SHAs before full 17-condition run. This is noted as a TODO in the yaml and flagged in open_questions.

  - field: notes/data_changelog.md mgsm_zh sha256
    reason: sha256 field is "pending" — the raw zh JSONL/TSV is not present in the sandbox so the hash cannot be computed at write time. Spec §4 and HIGH-3 require this before the full run. Flagged as TODO in the changelog entry.

  - field: H1 bootstrap seed choice
    reason: Bootstrap uses a per-condition seed derived from (rd_seed + 1) for degradation and (rd_seed + 2) for fld_cos, ensuring stable reproducibility per condition. Spec did not prescribe a specific bootstrap seed; this is a defensible implementation choice documented here.

dependencies_added:
  - scipy (optional): _spearman_rho() attempts `from scipy.stats import spearmanr` at call time; falls back to pure-Python manual Spearman if unavailable. scipy is already present in the datascience conda env used for tests. It is NOT a new hard dependency — the code functions without it.

reproducibility_audit:
  seed_wired: true  (run_phase_a accepts --seed; config.random_donor.seed used for per-condition formula; bootstrap seeded per-condition; Spearman is deterministic)
  config_logged: true  (config_path, git_sha, seed, run_name all in manifest.json)
  run_name_required: false  (optional CLI arg per spec; argparse default=None, no fail on absence)
  determinism_flags: true  (temperature=0.0, do_sample=False, max_new_tokens=512 in config; float32 cast in compute_fld)

fix_checklist:
  BLOCK-1: CLOSED — ModelsConfig extended; yaml updated; all 3 from_pretrained calls get revision=
  BLOCK-2: CLOSED — save_hidden_states called with inf_results for every condition before del hs_dict
  HIGH-1: CLOSED — git_sha wired via subprocess; manifest and summary JSON contain git_sha key
  HIGH-2: CLOSED — _bootstrap_ci() and _spearman_rho() implemented; H1 CIs in phase_a_summary.json["h1_bootstrap_cis"]; H2 rho in ["h2_width_spearman"]; underpowered caveat in phase_a_summary.txt
  HIGH-3/F1: CLOSED — mgsm_zh@v1.0-2022-10-03 entry added to notes/data_changelog.md (sha256: pending)
  F2/Addendum B: CLOSED — grid_intersection_notes in manifest + summary JSON; grid_intersection_shared_seed=42812; self-consistency sanity check in run_phase_a raises RuntimeError on mismatch; one-line note in phase_a_summary.txt
  MED-1: CLOSED — test_different_seeds_produce_different_source_starts uses precomputed constants (expected 10 and 22) and asserts equality + distinct source_starts
  NIT-1: CLOSED (auto-closed by HIGH-1 wiring subprocess)

open_questions_for_watcher:
  - CONFIRM HF REVISION PINS: stage2_confound.yaml has recipient_revision:"main" and donor_revision:"main" as explicit placeholders. Coordinator must replace with pinned commit SHAs before the full 17-condition run (spec §5 requires pinned revisions for reproducibility). Stage1 stage1_main.yaml also has no pinned revisions — coordinator should decide whether to backfill.
  - mgsm_zh sha256: notes/data_changelog.md entry has sha256:"pending". Must be filled by coordinator before greenlighting the full run (spec §4).
  - bootstrap seed policy: bootstrap uses rd_seed+1 per condition. If the watcher wants a single global seed or a different derivation, flag as a revision request.
  - scipy import: _spearman_rho does a lazy import of scipy.stats.spearmanr at call time. If the production environment does not have scipy, the pure-Python fallback activates silently. Watcher should confirm this is acceptable or require scipy as an explicit dependency.
