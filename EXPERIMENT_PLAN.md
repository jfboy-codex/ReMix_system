# ReMix background profiling experiment

## Question and comparison

Measure whether online idle-time profiling changes serving throughput, tail latency, and hardware utilization under the same two-GPU budget. First audit the paper's original run settings separately: `expriment_aaai/config.yaml` uses precomputed misses, a warmed cache, and `cache_update: false`. Results from that setup must not be labeled as online-profile-inclusive.

The paired comparison uses the same model weights, 40 inference steps, GPU0 for SD3.5, GPU1 for SD1.5, CPU CLIP, threshold 0.26, an initially empty cache, FIFO capacity 200, and real miss generation. `no-profile` suppresses latent capture/profile jobs and uses fallback start step 20 on hits. `idle-profile` captures latents and schedules one VAE-decode chunk at a time on whichever of the same two GPUs is idle; CPU CLIP scores each decoded image. No extra GPU is used.

## Fixed workload

For each of COCO and DiffusionDB, read the first 400 prompts using the old Parquet conventions. The 800-request trace contains 200 initial prompts, 200 later new prompts, 200 exact repeats of initial prompts, and 200 deterministic edits of initial prompts. The last 600 are shuffled with seed 42. Exact and edited repeats are *offered* reuse opportunities; the actual cache hit rate is measured, not assumed. FIFO eviction is deliberately exercised. One trace is generated for each dataset and load and reused unchanged in both modes.

Two Gamma(shape=1) arrival loads are specified in `config/server.yaml`: mean intervals 10 s (low) and 5 s (high). These are starting points based on the one-request smoke test; inspect a short pilot for saturation before the full matrix, then freeze the rates and regenerate traces if needed. Each of the 2 datasets × 2 loads × 2 modes is run **once**. There is no three-run repetition.

```bash
python -m pip install -e '.[server]'
for dataset in coco diffusiondb; do
  for load in low high; do
    python prepare_trace.py --dataset "$dataset" --load "$load" \
      --output "traces/${dataset}_${load}.jsonl"
    for mode in no-profile idle-profile; do
      python run_service.py --trace "traces/${dataset}_${load}.jsonl" \
        --mode "$mode" --output "outputs/experiment/${dataset}_${load}_${mode}"
      python evaluate_pickscore.py \
        --output "outputs/experiment/${dataset}_${load}_${mode}"
    done
  done
done
```

The trace SHA-256 in each run's `summary.json` must match between paired modes. Run each mode in a fresh process so the cache begins empty. Do not use `--skip-profile-drain` for the main experiment. Keep the original trace files with results.

## Records and reported results

Each output directory contains `summary.json`, `requests.jsonl` (path, response latency, offline output CLIP score, trace index/kind), `events.jsonl` (request routing and foreground timing, cache insert/hit/FIFO eviction, profile enqueue/decode/CPU score/completion/cancel/drop), `cache_entries.jsonl` (per-entry reuse before eviction or run end, including reuse after profile completion), `gpu_samples.jsonl` (NVML utilization and memory on each GPU), and generated images.

Report achieved throughput, p50/p95/p99 response and queue delay, hit/miss counts, mean output CLIP score and offline PickScore, profile completion/drop/cancel counts, profile chunk wall time by GPU, CPU scoring wall time, fraction of profile decoding concurrent with foreground work on the other GPU, GPU utilization, and memory peak/mean. The per-entry reuse distribution includes zero-reuse fraction, median and p90; distinguish profiled from pending entries. NVML utilization and memory are sampled every 0.5 s, so peaks shorter than the sampling interval can be missed. Profile chunk duration is a wall-clock occupancy proxy, not CUDA kernel time. `evaluate_pickscore.py` runs after the serving measurement and writes `pickscore.jsonl` plus `pickscore_summary.json`; its time is excluded from serving latency and utilization numbers.

The experiment does **not** estimate rental or total GPU cost and does **not** calculate a cache-hit break-even point. Interpret quality and performance together; a faster run with materially lower output CLIP score is not a comparable gain. The two loads are one fixed-workload observation each, not confidence intervals or a saturation curve.
