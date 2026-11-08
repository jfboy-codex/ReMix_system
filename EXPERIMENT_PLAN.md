# ReMix background profiling experiment

## Question and comparison

Measure whether online idle-time profiling changes serving throughput, tail latency, and hardware utilization under the same two-GPU budget. First audit the paper's original run settings separately: `expriment_aaai/config.yaml` uses precomputed misses, a warmed cache, and `cache_update: false`. Results from that setup must not be labeled as online-profile-inclusive.

The paired comparison uses the same model weights, 40 inference steps, GPU0 for SD3.5, GPU1 for SD1.5 and CLIP, threshold 0.26, an initially empty cache, FIFO capacity 200, and real miss generation. `no-profile` suppresses latent capture/profile jobs and uses fallback start step 20 on hits. `idle-profile` captures latents and schedules one VAE-decode chunk at a time on whichever of the same two GPUs is idle; GPU1 CLIP scores each decoded image. GPU1 CLIP scoring, SD1.5 editing, and VAE decoding are serialized with a shared lock. No extra GPU is used.

### Adaptive CLIP checkpoint sampling

`serving.profile.clip_stride` (default 4) controls coarse checkpoints from the noise-derived minimum step; the final denoising step is always included. After scoring all coarse checkpoints, compute each adjacent pair's CLIP slope as score change divided by step distance. If neighboring slopes differ by at least `slope_jump_threshold` (default 0.006 per step), fill every interior step in **both** neighboring intervals. Also fill an interval whose absolute slope reaches `steep_slope_threshold` (default 0.01 per step). Duplicate checkpoints are scored once. GPU VAE decodes remain interruptible between steps, including refinement steps. The profile stores points in step order and chooses its stable maximum from points whose remaining scores stay within `stable_tolerance` of the final source-image score.

The event log records coarse and fine step indices and counts; compare them with the original every-step CLIP cost. Coarse endpoints cannot reveal a jump that reverses completely inside one interval, so report stride and thresholds alongside results and inspect a small set of full profiles when tuning them. Setting `clip_stride: 1` restores every-step scoring.

## Fixed workload

For each of COCO and DiffusionDB, read the first 400 usable prompts in dataset order using the old Parquet conventions. `experiment.request_count=400` sets the number of requests. The trace does not add repeated or edited prompts or shuffle dataset rows; reuse occurs only when ordinary dataset requests match a cached image under the CLIP threshold. The actual cache hit rate and FIFO evictions are measured, not assumed. One trace is generated for each dataset and load and reused unchanged in both modes.

Two Gamma(shape=1) arrival loads are specified in `config/server.yaml`: mean intervals 5 s (low, 0.2 QPS) and 2 s (high, 0.5 QPS). Each of the 2 datasets × 2 loads × 2 modes is run **once**. There is no three-run repetition.

```bash
python -m pip install -e '.[server]'
bash run_experiment.sh
```

The script generates four traces, runs both modes for each dataset/load combination in separate Python processes, and evaluates PickScore after each complete serving run. It writes console logs into each run directory and stops on errors. Completed images and metadata are checkpointed as requests finish. Ctrl+C writes a partial summary and exits; `bash run_experiment.sh --resume` preserves completed runs after checking trace hashes and reruns an interrupted run from an empty cache. Previous partial checkpoints remain in the run directory. Set `PYTHON_BIN`, `REMIX_CONFIG`, `REMIX_TRACE_ROOT`, or `REMIX_RUN_ROOT` in the environment to override the defaults. Run the long experiment inside `tmux` or an equivalent persistent session.

The trace SHA-256 in each run's `summary.json` must match between paired modes. Run each mode in a fresh process so the cache begins empty. Do not use `--skip-profile-drain` for the main experiment. Keep the original trace files with results.

## Records and reported results

Each completed output directory contains `summary.json`, `requests.jsonl` (path, response latency, output CLIP score, trace index), `events.jsonl` (request routing and foreground timing, cache insert/hit/FIFO eviction, profile enqueue/decode/GPU1 CLIP score/completion/cancel/drop), `cache_entries.jsonl` (per-entry reuse before eviction or run end, including reuse after profile completion), `gpu_samples.jsonl` (NVML utilization and memory on each GPU), and generated images. During serving, `requests.partial.<attempt>.jsonl` and the images grow as responses finish. Interrupted attempts retain `partial_summary.<attempt>.json` and the checkpoint, while `summary.json` is reserved for a complete run.

Report achieved throughput, p50/p95/p99 response and queue delay, hit/miss counts, mean output CLIP score and offline PickScore, profile completion/drop/cancel counts, coarse/fine CLIP score counts, profile chunk wall time by GPU, GPU1 CLIP scoring wall time, fraction of profile decoding concurrent with foreground work on the other GPU, GPU utilization, and memory peak/mean. The per-entry reuse distribution includes zero-reuse fraction, median and p90; distinguish profiled from pending entries. NVML utilization and memory are sampled every 0.5 s, so peaks shorter than the sampling interval can be missed. Profile chunk duration is a wall-clock occupancy proxy, not CUDA kernel time. `evaluate_pickscore.py` runs after the serving measurement and writes `pickscore.jsonl` plus `pickscore_summary.json`; its time is excluded from serving latency and utilization numbers.

The experiment does **not** estimate rental or total GPU cost and does **not** calculate a cache-hit break-even point. Interpret quality and performance together; a faster run with materially lower output CLIP score is not a comparable gain. The two loads are one fixed-workload observation each, not confidence intervals or a saturation curve.
