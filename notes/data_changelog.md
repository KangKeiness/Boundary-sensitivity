# Data changelog

<!-- entries below -->

## 2026-04-10T12:00:00Z — mgsm_te@1.0
- file: data/raw/mgsm_te.jsonl   sha256: abc123def456
reason: Initial dataset load
author: KangKeiness

## 2026-04-11T00:00:00Z — mgsm_zh@v1.0-2022-10-03
- dataset: mgsm (juletxara/mgsm on HuggingFace, Chinese split)
- source: HuggingFace datasets — juletxara/mgsm, language=zh, split=test
- version: v1.0-2022-10-03 (default release matching juletxara/mgsm)
- sha256: b2fa63151022370a0de1f4211c8c284eae74b0f5a3b003b1d5982c0d4a73f661
- row_count: 250 (test split)
- license: CC-BY-SA 4.0
reason: Phase A width-confound separation grid uses mgsm_zh (dataset.lang=zh in stage2_confound.yaml)
author: KangKeiness
