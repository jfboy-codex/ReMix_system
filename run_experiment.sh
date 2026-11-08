#!/usr/bin/env bash
# Run the fixed 2-dataset x 2-load x 2-profile-mode experiment once.
set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && $1 != --resume ) ]]; then
  echo "Usage: bash run_experiment.sh [--resume]" >&2
  exit 2
fi
resume=0
if [[ ${1:-} == --resume ]]; then resume=1; fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"
config="${REMIX_CONFIG:-config/server.yaml}"
run_root="${REMIX_RUN_ROOT:-outputs/experiment}"
trace_root="${REMIX_TRACE_ROOT:-traces}"
mkdir -p "$run_root" "$trace_root"

for dataset in coco diffusiondb; do
  for load in low high; do
    trace="$trace_root/${dataset}_${load}.jsonl"
    "$python_bin" prepare_trace.py --config "$config" \
      --dataset "$dataset" --load "$load" --output "$trace"

    for mode in no-profile idle-profile; do
      run_dir="$run_root/${dataset}_${load}_${mode}"
      mkdir -p "$run_dir"
      echo "[$dataset/$load/$mode] Starting"

      if [[ -f "$run_dir/summary.json" ]]; then
        if (( ! resume )); then
          echo "Existing result: $run_dir/summary.json; use --resume to keep it." >&2
          exit 1
        fi
        "$python_bin" - "$trace" "$run_dir/summary.json" <<'PY'
import hashlib, json, pathlib, sys
trace, summary = map(pathlib.Path, sys.argv[1:])
expected = json.loads(summary.read_text(encoding="utf-8"))["trace_sha256"]
actual = hashlib.sha256(trace.read_bytes()).hexdigest()
if expected != actual:
    raise SystemExit(f"Trace changed since completed run: {summary}")
PY
        echo "[$dataset/$load/$mode] Serving result already complete"
      else
        "$python_bin" run_service.py --config "$config" --trace "$trace" \
          --mode "$mode" --output "$run_dir" 2>&1 | tee -i "$run_dir/run.log"
      fi

      if [[ -f "$run_dir/pickscore_summary.json" ]]; then
        if (( ! resume )); then
          echo "Existing PickScore result: $run_dir/pickscore_summary.json" >&2
          exit 1
        fi
        echo "[$dataset/$load/$mode] PickScore already complete"
      else
        "$python_bin" evaluate_pickscore.py --config "$config" \
          --output "$run_dir" 2>&1 | tee "$run_dir/pickscore.log"
      fi
      echo "[$dataset/$load/$mode] Complete"
    done
  done
done

echo "All eight runs completed. Results: $run_root"
