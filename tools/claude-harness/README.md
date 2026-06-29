# claude-harness — the `rigor` method spine (version-controlled copy)

A general research/SWE rigor harness for Claude Code, authored for this project and kept here so it is
backed up + reviewable in git. The live copies run from `~/.claude/`; this folder is the source of truth
to re-install or edit from.

## What it is
One method spine, two modes (`research` | `build`), applied the same way to an experiment or a code change:
ground-truth-first → question the requirement → standard-over-homegrown → seal a falsifiable claim before
acting → execute one variable at a time → adversarially self-refute (CI, not point estimate) → report
honestly (negatives as headlines, relative gap). It composes the existing atoms (`pre-register`, `falsify`,
`feature-factory`, `deep-research-plan`) rather than replacing them.

Motivating lesson (baked into the spine): a homegrown `_roc_auc` that mishandled ties silently deflated a
real result (0.598 read as 0.259) and produced a wrong conclusion — **use tested libraries for
metrics/stats; if you DIY, unit-test against the standard on tie/edge cases.** The harness proved itself
the day it shipped — a `rigorist` subagent falsified its own author's sealed pre-reg premise instead of
inflating.

## Files
| file | install location | role |
|---|---|---|
| `skills/rigor/SKILL.md` | `~/.claude/skills/rigor/SKILL.md` | the spine; `/rigor research\|build <task>` |
| `agents/rigorist.md` | `~/.claude/agents/rigorist.md` | default worker; preloads the `rigor` skill into every spawn (use as `subagent_type` / Workflow `agentType`) |
| `workflows/rigor.js` | `~/.claude/workflows/rigor.js` | deterministic scan→seal→execute→verify→report; `Workflow({name:'rigor', args:{mode,task,config}})` |
| `hooks/rigor-claim-gate.sh` | `~/.claude/hooks/rigor-claim-gate.sh` | non-blocking PreToolUse reminder on git-commit result/fix claims |
| `hooks/settings.PreToolUse.snippet.json` | merge into `~/.claude/settings.json` `hooks.PreToolUse` | registers the claim-gate hook |

## Install / re-install
```sh
mkdir -p ~/.claude/skills/rigor ~/.claude/agents ~/.claude/workflows ~/.claude/hooks
cp tools/claude-harness/skills/rigor/SKILL.md  ~/.claude/skills/rigor/SKILL.md
cp tools/claude-harness/agents/rigorist.md     ~/.claude/agents/rigorist.md
cp tools/claude-harness/workflows/rigor.js     ~/.claude/workflows/rigor.js
cp tools/claude-harness/hooks/rigor-claim-gate.sh ~/.claude/hooks/rigor-claim-gate.sh && chmod +x ~/.claude/hooks/rigor-claim-gate.sh
# then merge hooks/settings.PreToolUse.snippet.json into ~/.claude/settings.json (hooks.PreToolUse[])
```
The hook is the only deterministic enforcement (non-blocking reminder); the skill + agent are strong bias.
For MUST-be-deterministic steps (scan-first, pre-reg-first) use the workflow / a blocking hook.

## Deterministic vs bias
- **Deterministic lock**: the workflow script (control flow) + hooks (gates). The model cannot skip these.
- **Strong bias**: the skill (auto-loaded by description) + the `rigorist` agent (preloads the spine). Followed, not guaranteed.
Use both: the workflow/hook for the load-bearing steps, the skill/agent for the methodology everywhere.

Edit here, then re-run the install copy. Keep this in sync if you change the live `~/.claude/` copies.
