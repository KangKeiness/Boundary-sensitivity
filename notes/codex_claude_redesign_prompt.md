# Claude Code용 팀 재설계 프롬프트 (Codex 중심 + 이중 리뷰 유지)

아래 프롬프트는 **Codex 중심으로 재설계하되, Claude 리뷰어와 Codex 리뷰어를 교차검증용으로 의도적으로 둘 다 유지**하는 방향을 반영한 **Claude Code용 전체 수정 프롬프트**입니다.

```md
You are a senior research engineering systems architect.

Your task is to redesign the current multi-agent Claude Code repository into a leaner, faster, more accountable research-engineering workflow centered on Codex execution, while intentionally preserving dual review lanes for cross-validation between Claude Code and Codex.

You must think and operate like a top-tier ACL / EMNLP / ICLR research engineering lead:
- reduce coordination overhead
- maximize artifact quality
- preserve reproducibility and scientific discipline
- prevent claim overreach
- make implementation ownership explicit
- keep review orthogonal rather than redundant
- optimize for research-grade execution, not agent theater

---

# 0. Context

Current repository structure:

research-engineering-system/
├── CLAUDE.md
├── dashboard.py
├── .claude/
│   ├── settings.json
│   ├── agents/
│   │   ├── spec-planner.md
│   │   ├── writer.md
│   │   ├── watcher.md
│   │   ├── data-auditor.md
│   │   ├── claim-skeptic.md
│   │   └── codex-reviewer.md
│   ├── skills/
│   │   ├── papergrade-experiment/
│   │   ├── repo-map/
│   │   ├── result-validation/
│   │   ├── dataset-audit/
│   │   ├── claim-audit/
│   │   ├── implementation-rules/
│   │   └── test-triage/
│   └── hooks/
│       ├── block_sensitive_paths.sh
│       ├── secret_scan.sh
│       ├── bash_allowlist.sh
│       ├── paper_claim_lock.sh
│       ├── training_guard.sh
│       └── format_and_lint.sh
└── notes/
    ├── workflows.md
    ├── paper_locks.md
    └── data_changelog.md

Current workflow:
User request
→ main thread coordinator in CLAUDE.md
→ spec-planner
→ user approval
→ writer
→ watcher || data-auditor
→ claim-skeptic (optional)
→ codex-reviewer (2-pass)

---

# 1. Redesign goal

Redesign this system so that it becomes a Codex-centered workflow rather than a conversation-heavy multi-agent team.

However, do NOT collapse the review system into a single reviewer.

The system intentionally preserves two review lanes for cross-validation:
1. a Codex-side engineering review lane
2. a Claude-side research/claim review lane

Your redesign must therefore achieve BOTH:
- lower redundancy
- preserved dual-review rigor

The goal is not “fewer reviewers at all costs.”
The goal is:
- fewer overlapping roles
- clearer contracts
- orthogonal review axes
- stronger artifact discipline
- faster execution with less handoff waste

---

# 2. Core operating philosophy

The redesigned system must follow these principles:

## Principle 1: single implementation owner
There must be exactly one primary writer for implementation changes.
No second agent should casually edit the same code path.

## Principle 2: dual review, but orthogonal
The two review lanes must remain, but they must NOT perform the same generic review.

They must be redesigned into different validators with different contracts:

- Codex-side reviewer:
  - implementation validity
  - diff quality
  - test adequacy
  - repo consistency
  - reproducibility
  - artifact traceability

- Claude-side reviewer:
  - research logic
  - claim-evidence alignment
  - ablation sufficiency
  - confound detection
  - missing controls
  - reviewer-style skepticism
  - overclaim detection

If two reviewer roles overlap substantially, narrow them until they become orthogonal.
Do not keep two generic reviewers.

## Principle 3: every stage must produce an artifact
Every stage must emit a concrete artifact, such as:
- plan
- patch
- test report
- repro report
- claim audit
- lock record
- dataset integrity record

No stage should exist as “just discussion.”

## Principle 4: prefer skills over agent proliferation
If a responsibility can be expressed as a deterministic procedure, checklist, or validation protocol, prefer a skill or hook over a full agent.

## Principle 5: preserve research-grade safeguards
Do not remove critical controls related to:
- claim locking
- seed/config enforcement
- secret scanning
- dangerous command blocking
- sensitive path protection
- dataset provenance
- result-to-artifact traceability

You may redesign where they live and how they are triggered.

## Principle 6: optimize for accountable execution
When forced to choose, prefer:
- fewer handoffs
- clearer pass/fail criteria
- narrower responsibilities
- stronger contracts
over
- more discussion
- broader reviewer personas
- vague oversight

---

# 3. What you must do

Inspect the current architecture and produce a redesign proposal.

Your proposal must include all of the following.

## A. Executive diagnosis
Diagnose the current system with technical specificity.
Identify:
- redundant agents
- overlapping reviewer scopes
- likely latency bottlenecks
- unnecessary approval or handoff steps
- roles that should become skills or hooks
- places where execution responsibility is ambiguous
- places where review cost is too high relative to marginal quality gain
- places where the current structure risks “commentary without accountable ownership”

Do not be polite or vague. Be precise.

## B. Proposed target architecture
Design a new architecture with a smaller and cleaner role set, centered on Codex execution while preserving dual review lanes.

Preferred high-level structure:
- Coordinator / Orchestrator
- Spec Planner
- Primary Executor
- Codex Engineering Verifier
- Claude Research Claim Skeptic

You may rename these roles, but the resulting system must satisfy:
- one main writer
- one explicit Codex-side engineering reviewer
- one explicit Claude-side research/claim reviewer
- minimal duplication
- explicit artifact flow between stages

You must explicitly decide:
- which current agents remain
- which current agents are merged
- which are removed
- which become skills
- which become hooks
- which tasks Codex should own
- which tasks Claude Code should still own

## C. File-level migration plan
Propose exact repository changes:
- files to delete
- files to merge
- files to rename
- new files to create
- how CLAUDE.md should change
- how settings.json should change
- how notes/workflows.md should change
- whether hooks need to be simplified, strengthened, or retargeted

Use explicit paths and filenames.

## D. Workflow redesign
Define the new end-to-end workflow for at least these cases:
1. feature implementation
2. experiment implementation
3. reproducibility audit
4. paper claim validation
5. dataset change
6. failed test triage

Each workflow must specify:
- entry condition
- responsible role
- invoked skill/hook if any
- output artifact
- pass/fail condition
- next stage

## E. Rewrite-ready operational content
Draft replacement content that is concrete enough to commit.

You must draft:
1. CLAUDE.md
2. reduced .claude/agents/ set
3. reorganized .claude/skills/ set
4. updated notes/workflows.md
5. any new ledger or notes file that becomes necessary

Do not give vague summaries.
Write operational text.

---

# 4. Strong design constraints

You must obey these constraints.

## Constraint 1: single writer rule
Implementation changes must be owned by one primary executor.
Other agents may validate, but must not compete for write authority.

## Constraint 2: two reviewers must remain
Do not collapse the system into a single reviewer.
The dual-review design is intentional for cross-validation between Claude Code and Codex.

## Constraint 3: review axes must differ
The two reviewers must have non-overlapping review contracts.

Bad design:
- Claude reviewer: “general review”
- Codex reviewer: “general review”

Good design:
- Codex reviewer: engineering validity, testability, reproducibility
- Claude reviewer: scientific logic, claims, confounds, ablations

## Constraint 4: broad reviewers should be narrowed
If a reviewer is too broad, narrow it into a strict validator with pass/fail criteria.
If still too broad, convert part of the role into a skill or hook.

## Constraint 5: preserve hooks that enforce discipline
Keep or improve protections for:
- paper/ claim gating
- raw data/sensitive path protection
- seed/config enforcement for training
- command risk blocking
- lint/format after edits
- secret scanning

Do not casually delete these.

## Constraint 6: claims must not be optional in the wrong places
If claim review is relevant to paper/, result summaries, discussion, or research conclusions, it should not remain loosely optional.
You must propose a stricter and more defensible trigger policy.

## Constraint 7: dataset handling must stay auditable
Dataset changes must remain traceable through hashes, changelogs, or equivalent integrity records.
If data-auditor is removed as an agent, dataset integrity must still be enforced elsewhere.

## Constraint 8: optimize for serious research code
Assume this repository supports paper-grade experimentation.
Prioritize:
- reproducibility
- evidence-backed claims
- clean separation of implementation and interpretation
- narrow diffs
- deterministic logs
- safe experimental workflow

---

# 5. Preferred design direction

You are encouraged to move toward a structure like this if justified:

- CLAUDE.md becomes a thin orchestrator, not a verbose central brain
- one primary executor owns implementation
- Codex becomes the main execution and engineering verification engine
- Claude reviewer becomes a specialized research skeptic rather than a generic code reviewer
- dataset auditing becomes a narrower validation path if possible
- more responsibilities move from broad agents into narrow skills and hooks
- notes/ becomes more ledger-oriented and less informal
- workflows are artifact-driven and contract-based

But do not copy this blindly.
First diagnose, then justify.

---

# 6. Required output format

Your response must contain these exact sections:

1. Executive diagnosis
2. Main inefficiencies in current architecture
3. Why dual review should be preserved
4. Proposed target architecture
5. Agent-by-agent migration table
6. Skill-by-skill migration table
7. Hook review and redesign
8. New workflow definitions
9. File tree of the redesigned repository
10. Draft CLAUDE.md
11. Draft agent files
12. Draft skill files
13. Draft notes/workflows.md
14. Migration sequence in safe order
15. Risks and tradeoffs

Use tables where they help.
Be explicit.
Do not stay at the level of slogans.

---

# 7. Additional correction rule

Assume the current system has become:
- too expensive in tokens
- too slow in handoffs
- too soft in execution accountability

Your redesign must reduce real coordination overhead.
Do not merely rename components.
If a role does not have a unique and necessary artifact, remove or merge it.
If a workflow stage cannot fail objectively, redefine it until it can.

---

# 8. Final objective

The final architecture should make the following division crystal clear:

- Planner:
  defines what should be built and how success will be judged

- Executor:
  is the only primary implementation owner

- Codex-side reviewer:
  answers “Can this be trusted as engineering?”

- Claude-side reviewer:
  answers “Can this be claimed as research?”

Now perform the redesign in full detail.
```

## 추가 권장 문장

아래 한 줄을 **프롬프트 맨 아래에 추가**하시면 더 공격적으로 재설계하게 만들 수 있습니다.

```md
Be especially suspicious of any role that mainly produces commentary rather than a merge decision, a validation record, or a blocking verdict.
```

## 사용 팁

실전적으로는 이 전체 프롬프트를 먼저 넣은 다음, 다음 턴에서 아래처럼 섹션별로 쪼개서 받는 방식이 가장 안정적입니다.

- `이제 10번 Draft CLAUDE.md만 실제 커밋 가능한 수준으로 써라`
- `이제 11번 Draft agent files만 작성해라`
- `이제 12번 Draft skill files만 작성해라`
- `이제 13번 Draft notes/workflows.md만 작성해라`
