spec_ref: notes/specs/phase_a_width_confound.md

files_changed:
  - path: stage1/configs/stage2_confound.yaml
    diff_stat: +2 -2
    description: >
      Replaced placeholder "main" values on lines 7-8 with pinned HF commit SHAs.
      recipient_revision: "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"  (Qwen/Qwen2.5-1.5B-Instruct)
      donor_revision:    "8faed761d45a263340a0528343f099c05c9a4323"   (Qwen/Qwen2.5-1.5B)
      No other lines touched.

tests_run:
  - command: /c/Users/system1/anaconda3/envs/datascience/python.exe -m pytest stage1/tests/ -x -q
    status: pass
    failing_ids: []
    output: "36 passed in 2.51s"
  - command: >
      /c/Users/system1/anaconda3/envs/datascience/python.exe -c
      "from stage1.utils.config import load_config; c = load_config('stage1/configs/stage2_confound.yaml');
       print(c.models.recipient_revision, c.models.donor_revision)"
    status: pass
    output: "989aa7980e4cf806f80c7fef2b1adb7bc71aa306 8faed761d45a263340a0528343f099c05c9a4323"

deviations_from_spec: []

dependencies_added: []

reproducibility_audit:
  seed_wired: true
    note: yaml-only change; no entrypoint touched. Seed wiring unchanged from r4.
  config_logged: true
  run_name_required: true
  determinism_flags: true

fix_checklist:
  HF_REVISION_PIN_GATE: CLOSED
    - [x] recipient_revision set to "989aa7980e4cf806f80c7fef2b1adb7bc71aa306" (Qwen/Qwen2.5-1.5B-Instruct)
    - [x] donor_revision set to "8faed761d45a263340a0528343f099c05c9a4323" (Qwen/Qwen2.5-1.5B)
    - [x] TODO comments removed from both lines
    - [x] No other lines in yaml modified
    - [x] No other files touched
    - [x] load_config confirms both SHAs round-trip correctly
    - [x] 36/36 tests pass (yaml-only change, no test impact)

open_questions_for_watcher: []
