#!/bin/bash
# Round 5 rerun campaign: replays every experiment for the three methods now
# routed through official libraries (borutashap, powershap, shap_select).
# Run sequentially (each script's XGBoost calls use n_jobs=-1, so running
# these in parallel would just contend for the same 20 cores).
set -e
cd "$(dirname "$0")/.."
LOG=logs/round5_rerun
mkdir -p "$LOG"
METHODS="borutashap powershap shap_select"

echo "[1/7] Experiment 1 -- leaky arm"           | tee -a "$LOG/progress.log"
python3 experiments/exp1_performance.py --methods $METHODS --resume >> "$LOG/exp1_leaky.log" 2>&1

echo "[2/7] Experiment 1 -- nested arm (+ exp5 nested byproduct)" | tee -a "$LOG/progress.log"
python3 experiments/run_nested_exp1.py --methods $METHODS >> "$LOG/exp1_nested.log" 2>&1

echo "[3/7] Experiment 6 -- native threshold, leaky"  | tee -a "$LOG/progress.log"
python3 experiments/exp6_native.py >> "$LOG/exp6_leaky.log" 2>&1

echo "[4/7] Experiment 6 -- native threshold, nested" | tee -a "$LOG/progress.log"
python3 experiments/run_nested_exp6.py >> "$LOG/exp6_nested.log" 2>&1

echo "[5/7] Experiment 3 -- noise robustness (Madelon ground truth + 3 methods fixed)" | tee -a "$LOG/progress.log"
python3 experiments/exp3_noise.py --methods $METHODS --resume >> "$LOG/exp3_noise.log" 2>&1

echo "[6/7] Experiment 4 -- transfer, leaky + nested" | tee -a "$LOG/progress.log"
python3 experiments/exp4_transfer.py --methods $METHODS --resume >> "$LOG/exp4_leaky.log" 2>&1
python3 experiments/run_nested_exp4.py --methods $METHODS >> "$LOG/exp4_nested.log" 2>&1

echo "[7/7] Experiment 2 -- bootstrap stability (most expensive: 10 datasets x 50 iters x 3 methods)" | tee -a "$LOG/progress.log"
python3 experiments/exp2_stability.py --methods $METHODS --resume >> "$LOG/exp2_stability.log" 2>&1

echo "ALL DONE" | tee -a "$LOG/progress.log"
