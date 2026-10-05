#!/usr/bin/env bash
# Start the multi-seed check with several parallel workers (default 6 workers x 3 threads).
#   bash revision_r8/run_parallel.sh            # 6 workers
#   WORKERS=8 THREADS=2 bash revision_r8/run_parallel.sh
# Re-running the command resumes: finished (seed, dataset) jobs and methods are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
WORKERS=${WORKERS:-6}
THREADS=${THREADS:-3}
PY=${PY:-.venv_ms/bin/python}
[ -x "$PY" ] || { echo "Python not found at $PY; run: bash revision_r8/setup_env.sh"; exit 1; }
mkdir -p results/multiseed/logs
if pgrep -f revision_r8/multiseed_worker.py >/dev/null; then
  echo "workers are already running; stop them first: pkill -f revision_r8/multiseed_worker.py"; exit 1
fi
rm -f results/multiseed/claims/*.claim      # stale claims from an earlier run
"$PY" revision_r8/preflight.py || exit 1
# a claim left by a killed run is taken over automatically (the owner pid is checked)
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS
export PYTHONUNBUFFERED=1
for i in $(seq 1 "$WORKERS"); do
  nohup "$PY" revision_r8/multiseed_worker.py --worker "$i" > "results/multiseed/logs/worker_$i.log" 2>&1 &
  echo "worker $i started (pid $!)"
  sleep 2
done
echo
echo "Progress:   $PY revision_r8/status.py"
echo "Live log:   tail -f results/multiseed/logs/worker_*.log"
echo "Stop all:   pkill -f revision_r8/multiseed_worker.py"
