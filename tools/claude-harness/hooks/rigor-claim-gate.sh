#!/bin/bash
# rigor claim-gate (PreToolUse / Bash). NON-BLOCKING reminder: when a git commit looks like it claims a
# result / fix / "works / done", nudge the rigor spine (pre-reg-or-spec sealed first, adversarial verify
# with a CI, standard-not-homegrown metric, honest negatives). Never blocks — exit 0 always.
input=$(cat 2>/dev/null)
cmd=$(printf '%s' "$input" | python3 -c "import json,sys
try: print(json.load(sys.stdin).get('tool_input',{}).get('command',''))
except Exception: print('')" 2>/dev/null)

# only react to git commits
case "$cmd" in
  *"git commit"*) ;;
  *) exit 0 ;;
esac

# only when the commit smells like a result / done / fix claim
if printf '%s' "$cmd" | grep -qiE "result|verdict|works|passes|fixed|fix:|done|benchmark|F1|AUC|mIoU|RayIoU|IoU|summary|feat:|perf:"; then
  {
    echo "[rigor] claim-gate — before committing a result/fix, confirm the spine held:"
    echo "  1. sealed FIRST?  research: a committed pre-registration before the data;  build: spec + failing test before the code."
    echo "  2. adversarially verified?  a confidence interval (not a point estimate); checked vs the PRIMARY source by content (code+data+behavior)."
    echo "  3. standard metric/stat (tested library, not homegrown)?  negatives reported as headlines, relative gap not absolute cutoff?"
    echo "  (reminder only — not a block.)"
  } >&2
fi
exit 0
