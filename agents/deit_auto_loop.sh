#!/usr/bin/env bash
set -euo pipefail

ROOT="/kmh-nfs-ssd-us-mount/code/qiao/work/DeiT"
TPU_PY="/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_manager/tpu.py"
SEE_LOG="/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_manager/see_log.py"
LOG_DIR="$ROOT/agents"
LOOP_LOG="$LOG_DIR/auto_loop.log"
STATUS_LOG="$LOG_DIR/auto_loop_status.md"

mkdir -p "$LOG_DIR"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] DEIT_AUTO_LOOP started" >> "$LOOP_LOG"

while true; do
  TS="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"

  # 1) Full memory refresh: read every agents file each loop.
  : > /tmp/deit_agents_last_read.txt
  for f in "$ROOT"/agents/*; do
    if [[ -f "$f" ]]; then
      {
        echo "===== $f ====="
        cat "$f"
        echo
      } >> /tmp/deit_agents_last_read.txt
    fi
  done

  # 2) Check jobs.
  python "$TPU_PY" check sqa user=sqa > /tmp/deit_check_latest.txt 2>&1 || true

  # 3) Extract DeiT status snapshot.
  python - <<'PY' > /tmp/deit_snapshot.txt
import re
raw=open('/tmp/deit_check_latest.txt','r',errors='ignore').read()
rows=[]
for b in raw.split('----------------------------------------'):
    if 'DIR: DeiT' not in b:
        continue
    w=re.search(r'Window\s+(\d+)', b)
    t=re.search(r'TPU:\s*(.*)', b)
    s=re.search(r'Status:\s*(.*)', b)
    tag=re.search(r'Window\s+\d+\s*\(tag:(.*)\)', b)
    rows.append((int(w.group(1)) if w else -1,
                 (tag.group(1).strip() if tag else 'no-tag')[:140],
                 (t.group(1).strip() if t else ''),
                 (s.group(1).strip() if s else '')))
rows=sorted(rows)
print(f"deit_windows={len(rows)}")
for r in rows:
    print(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}")
PY

  # 4) Pull latest eval lines from each active DeiT logdir.
  : > /tmp/deit_eval_snapshot.txt
  while IFS=$'\t' read -r window tag tpu status; do
    [[ "$window" =~ ^[0-9]+$ ]] || continue
    logdir="$(python "$SEE_LOG" "$window" 2>/dev/null | tail -1 || true)"
    echo "### window=$window tpu=$tpu" >> /tmp/deit_eval_snapshot.txt
    echo "tag=$tag" >> /tmp/deit_eval_snapshot.txt
    echo "status=$status" >> /tmp/deit_eval_snapshot.txt
    echo "logdir=$logdir" >> /tmp/deit_eval_snapshot.txt
    if [[ -n "$logdir" && -f "$logdir/output.log" ]]; then
      grep -E "eval epoch:|eval_loss=|Traceback|Error" "$logdir/output.log" | tail -8 >> /tmp/deit_eval_snapshot.txt || true
    else
      echo "no output.log found yet" >> /tmp/deit_eval_snapshot.txt
    fi
    echo >> /tmp/deit_eval_snapshot.txt
  done < <(tail -n +2 /tmp/deit_snapshot.txt)

  # 5) Record status and conservative decision. Do not auto-kill or auto-launch here.
  {
    echo "## $TS"
    cat /tmp/deit_snapshot.txt
    echo
    echo "### Latest eval snapshot"
    cat /tmp/deit_eval_snapshot.txt
    echo "### Auto decision"
    echo "No automatic kill/launch in this loop. External MONITOR.py owns preemption resume; if DeiT windows < 8 or real code errors appear, next human/agent loop should inspect logs and decide."
    echo
  } >> "$STATUS_LOG"

  echo "[$TS] loop done" >> "$LOOP_LOG"

  sleep 1800
 done
