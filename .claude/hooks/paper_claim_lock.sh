#!/usr/bin/env bash
# PreToolUse hook for Write|Edit. Blocks edits under paper/ unless a fresh
# claim-skeptic PASS exists with an evidence_bundle_hash recorded in
# notes/paper_locks.md whose timestamp is within the last 24 hours and whose
# hash matches the latest claim_audit artifact.
set -euo pipefail

payload=$(cat)
file_path=$(printf '%s' "$payload" | python -c "import sys, json; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null || echo "")

if [ -z "$file_path" ]; then exit 0; fi

norm=$(printf '%s' "$file_path" | tr '\\' '/')
case "/$norm/" in
  */paper/*) ;;
  *) exit 0 ;;
esac

lock_file="notes/paper_locks.md"
if [ ! -f "$lock_file" ]; then
  cat <<JSON
{"decision":"block","reason":"paper_claim_lock: no notes/paper_locks.md found. Run claim-skeptic and record evidence_bundle_hash before editing paper/."}
JSON
  exit 2
fi

# Find most recent claim_audit artifact
latest=$(ls -t notes/handoffs/claim_audit_*.md 2>/dev/null | head -n1 || true)
if [ -z "$latest" ]; then
  cat <<JSON
{"decision":"block","reason":"paper_claim_lock: no claim_audit artifact under notes/handoffs/. Run claim-skeptic first."}
JSON
  exit 2
fi

# Verify OVERALL: PASS in the artifact
if ! grep -q "^OVERALL: PASS" "$latest"; then
  cat <<JSON
{"decision":"block","reason":"paper_claim_lock: latest claim_audit artifact '$latest' is not OVERALL: PASS."}
JSON
  exit 2
fi

# Extract evidence_bundle_hash from the artifact
hash_line=$(grep "^evidence_bundle_hash:" "$latest" | head -n1 | awk '{print $2}' || true)
if [ -z "$hash_line" ]; then
  cat <<JSON
{"decision":"block","reason":"paper_claim_lock: artifact '$latest' missing evidence_bundle_hash."}
JSON
  exit 2
fi

# Verify it appears in paper_locks.md
if ! grep -q "$hash_line" "$lock_file"; then
  cat <<JSON
{"decision":"block","reason":"paper_claim_lock: evidence_bundle_hash '$hash_line' not recorded in notes/paper_locks.md. Manager must record it after verifying claim-skeptic PASS."}
JSON
  exit 2
fi

# Freshness: last mtime of artifact within 24h
now=$(date +%s)
mtime=$(python -c "import os,sys; print(int(os.path.getmtime('$latest')))" 2>/dev/null || echo 0)
age=$(( now - mtime ))
if [ "$age" -gt 86400 ]; then
  cat <<JSON
{"decision":"block","reason":"paper_claim_lock: claim_audit artifact is stale (>24h old). Re-run claim-skeptic."}
JSON
  exit 2
fi

exit 0
