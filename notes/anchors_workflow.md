# Phase B Anchor Generation Workflow

YELLOW-LIGHT v3 PRIORITY 2.

This document is the **operational contract** for producing the parity-compatible
anchor runs that Phase B's cross-check requires. If Phase B fails with
`Cross-check FAILED: missing anchor(s) [...]`, this is the recipe to follow.

## 1. What the gate checks

Every full Phase B run (`--sanity` not set) must reproduce two anchor accuracies
within `PHASE_A_CROSS_CHECK_TOL = 0.008` (≈ 2 / 250 samples):

| Anchor          | Source                                       | Used to verify           |
|-----------------|----------------------------------------------|--------------------------|
| `hard_swap_b8`  | latest **Stage 1** sweep `evaluation.json`   | `no_patch` accuracy      |
| `no_swap`       | latest **Phase A** `phase_a_summary.json` (Stage 1 fallback) | `clean_baseline` accuracy |

Phase A's grid does **not** contain `hard_swap_b8 (b=8, t=20)` — by construction
it uses `fixed_w4_posN` / `fixed_b8_wN`. The hard_swap_b8 anchor therefore must
come from a Stage 1 run.

Both anchors must be present AND both within tolerance. Missing either → hard
fail. Either outside tolerance → hard fail.

## 2. Parity contract (what makes a run "compatible")

Anchor candidates are filtered by `stage1.utils.manifest_parity.check_manifest_parity`
against the current Phase B config's parity block. The fields that must match
exactly are:

- `models.recipient` and `models.donor` (full HF identifier)
- `models.recipient_revision` and `models.donor_revision` (pinned SHA)
- `dataset.name`, `dataset.lang`, `dataset.split`
- `generation.do_sample`, `generation.temperature`, `generation.max_new_tokens`
- `hidden_state.pooling`

A run with a missing or absent `manifest.json` is rejected — the anchor cannot
be proven compatible.

### The most common footgun: `max_new_tokens`

`stage1/configs/stage1_main.yaml` defaults to `max_new_tokens: 256`.
`stage1/configs/stage2_confound.yaml` (Phase A / Phase B) uses `max_new_tokens: 512`.

A Stage 1 sweep produced with `stage1_main.yaml` will be **rejected** by the
Phase B parity filter. To produce a Stage 1 anchor that Phase B can reuse, the
Stage 1 run must be configured with the **same** generation block as Phase B.

## 3. Recipe — generate a parity-compatible anchor set from scratch

Run all three commands from the repo root with the **same** YAML and seed:

```bash
# (1) Stage 1 sweep — provides the hard_swap_b8 anchor.
#     Use stage2_confound.yaml so generation.max_new_tokens matches Phase B.
python -m stage1.run --config stage1/configs/stage2_confound.yaml

# (2) Phase A — provides the no_swap anchor (and the rest of the grid).
python -m stage1.run_phase_a --config stage1/configs/stage2_confound.yaml --seed 42

# (3) Phase B — cross-checks against (1) and (2).
python -m stage1.run_phase_b --config stage1/configs/stage2_confound.yaml --seed 42
```

If you already have one of the prior runs and only need the other, run only the
missing one, but ensure all three share the same config + revision pins.

## 4. How Phase B picks an anchor

For each candidate run directory (newest first by timestamp):

1. Load `manifest.json` (skip if missing).
2. Run `check_manifest_parity` against current Phase B parity (skip on mismatch).
3. Extract the anchor accuracy.
4. First match wins.

Anchor source precedence:

- `no_swap` → Phase A first, Stage 1 fallback (Phase A always emits it).
- `hard_swap_b8` → Phase A first (won't be there in normal grids), Stage 1 fallback.

The selected sources are recorded in `phase_b_summary.json` under
`phase_a_cross_check.anchor_hard_swap_source` and `anchor_no_swap_source`
(`"phase_a"` or `"stage1"`).

## 5. Reading failure messages

| Failure message                                                                | What it means                                                                              | Resolution                                               |
|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|----------------------------------------------------------|
| `Cross-check FAILED: missing anchor(s) [hard_swap_b8]`                         | No parity-compatible Stage 1 run with hard_swap_b8 was found.                              | Run step 1 above.                                        |
| `Cross-check FAILED: missing anchor(s) [no_swap]`                              | No parity-compatible Phase A or Stage 1 run with no_swap was found.                        | Run step 2 above.                                        |
| `Cross-check FAILED: missing anchor(s) [hard_swap_b8, no_swap]`                | Neither anchor available.                                                                  | Run steps 1 and 2.                                       |
| `... rejected — manifest parity mismatch: generation max_new_tokens (...)`     | Anchor exists but was decoded under a different `max_new_tokens`.                          | Re-run anchor with the Phase B config (likely 512).      |
| `... rejected — manifest parity mismatch: recipient model revision (...)`      | Anchor used a different model revision SHA.                                                | Re-run anchor with the pinned revisions in Phase B YAML. |
| `Cross-check FAILED: hard_swap_b8: |no_patch(...) - anchor(...)| = ... > tol`  | Anchors present and parity-compatible, but the new run's accuracy drifted beyond 0.008.    | Investigate seed, decode, or model-revision drift.       |

The failure path also writes `phase_b_summary.json` with `run_status: "failed"`
and a `failure_reason:` field, plus a single-token `RUN_STATUS.txt` sentinel
(YELLOW-LIGHT v3 P3) — do not interpret a failed run dir as completed.

## 6. Sanity mode

`python -m stage1.run_phase_b --config <yaml> --sanity` relaxes the gate to
"best effort" — any anchors that happen to be present are checked, but the run
will not hard-fail on missing anchors. Sanity mode is for development only and
is never used to produce results for review.

## 7. Where the gate logic lives

- Decision logic: `stage1/utils/anchor_gate.py` (torch-free, fully unit-tested).
- Parity check: `stage1/utils/manifest_parity.py`.
- Run-status writer: `stage1/utils/run_status.py`.
- Phase B integration: `stage1/run_phase_b.py` step (12) and step (18-20).
- Tests: `stage1/tests/test_phase_b_anchor_gate_integration.py`,
  `stage1/tests/test_phase_b_run_status.py`,
  `stage1/tests/test_red_light_final_regressions.py`.
