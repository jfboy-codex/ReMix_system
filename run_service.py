"""Run the modular ReMix service with the existing patched Diffusers pipelines.

Input: JSONL records with a `prompt` field, or one prompt per line.
This runner uses real generation and editing, never precomputed miss images.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import signal
import threading
import time
from pathlib import Path

from remix_system import ReMixRuntime
from remix_system.data import dataset_prompts
from remix_system.checkpoint import RequestRecorder
from remix_system.events import EventLog
from remix_system.experiment import read_trace, summarize, write_jsonl
from remix_system.hardware import GpuSampler
from remix_system.policy import ReMixPolicy
from remix_system.diffusers_adapters import (
    CLIPScorer, SD15Backend, SD35Backend, SD35VAEDecoder,
)


def prompts_from(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line) if line.lstrip().startswith("{") else line
        yield value["prompt"] if isinstance(value, dict) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/server.yaml"))
    parser.add_argument("--sd35", help="override local SD3.5 Medium model directory")
    parser.add_argument("--sd15", help="override local SD1.5 model directory")
    parser.add_argument("--clip", help="override local CLIP model directory")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompts", type=Path, help="JSONL or plain-text prompts")
    source.add_argument("--dataset", choices=("coco", "diffusiondb"))
    source.add_argument("--trace", type=Path, help="fixed JSONL trace with prompt and arrival_s")
    parser.add_argument("--mode", choices=("idle-profile", "no-profile"),
                        default="idle-profile")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, help="override both model step counts")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--interarrival-ms", type=float,
                        help="fixed interval override; 0 submits all requests immediately")
    parser.add_argument("--skip-profile-drain", action="store_true")
    args = parser.parse_args()

    import yaml
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    large_cfg = config["models"]["sd35"]
    small_cfg = config["models"]["sd15"]
    serving_cfg = config["serving"]
    large_path = args.sd35 or large_cfg["path"]
    small_path = args.sd15 or small_cfg["path"]
    clip_cfg = config["models"]["clip"]
    clip_path = args.clip or clip_cfg["path"]
    clip_device = clip_cfg["device"]
    if clip_device != small_cfg["device"]:
        raise ValueError("CLIP must share the SD1.5 GPU for serialized execution")
    large_steps = args.steps or large_cfg["steps"]
    small_steps = args.steps or small_cfg["steps"]
    threshold = (args.threshold if args.threshold is not None else
                 serving_cfg["hit_threshold"].get(args.dataset, 0.26))
    capacity = args.capacity or serving_cfg["cache_capacity"]
    output = args.output or Path(config["output"]["root"]) / (args.dataset or "prompts")
    arrival_cfg = serving_cfg["arrival"]
    if args.interarrival_ms is not None and args.interarrival_ms < 0:
        parser.error("--interarrival-ms must be nonnegative")
    arrival_rng = random.Random(serving_cfg["seed"])
    if serving_cfg.get("precompute"):
        raise ValueError("server runner requires real miss generation; set precompute: false")
    if args.trace and args.interarrival_ms is not None:
        parser.error("--interarrival-ms cannot be used with --trace")
    profile_enabled = args.mode == "idle-profile"

    # The patched SD1.5 pipeline imports this only on its first soft-edit hit.
    # Check before the long run so a missing installation fails immediately.
    from src.mask_algo import ExpandingMask  # noqa: F401

    import torch
    from diffusers import DDIMScheduler, StableDiffusion3Pipeline, StableDiffusionImg2ImgPipeline
    from tqdm import tqdm

    sd35 = StableDiffusion3Pipeline.from_pretrained(
        large_path, torch_dtype=torch.float16, local_files_only=True).to(large_cfg["device"])
    sd15 = StableDiffusionImg2ImgPipeline.from_pretrained(
        small_path, torch_dtype=torch.float16, local_files_only=True).to(small_cfg["device"])
    sd35.set_progress_bar_config(disable=True)
    sd15.set_progress_bar_config(disable=True)
    sd15.scheduler = DDIMScheduler.from_config(sd15.scheduler.config)
    if profile_enabled and not hasattr(sd35, "generate_with_profile"):
        raise RuntimeError("SD3.5 pipeline lacks the experiment's generate_with_profile patch")
    if not hasattr(sd15, "diffedit_rewrite_soft"):
        raise RuntimeError("SD1.5 pipeline lacks the experiment's diffedit_rewrite_soft patch")
    vae_gpu1 = copy.deepcopy(sd35.vae).to(small_cfg["device"])
    gpu1_lock = threading.Lock()
    clip = CLIPScorer(clip_path, device=clip_device, device_lock=gpu1_lock)
    events = EventLog()
    service = ReMixRuntime(
        SD35Backend(sd35, large_cfg["device"], steps=large_steps,
                    height=large_cfg["height"], width=large_cfg["width"],
                    guidance=large_cfg["guidance_scale"],
                    seed=serving_cfg["seed"], negative_prompt=large_cfg["negative_prompt"],
                    capture_profile=profile_enabled),
        SD15Backend(sd15, small_cfg["device"], steps=small_steps,
                    guidance=small_cfg["guidance_scale"],
                    seed=serving_cfg["seed"], negative_prompt=small_cfg["negative_prompt"],
                    device_lock=gpu1_lock),
        SD35VAEDecoder(sd35.vae, large_cfg["device"]),
        SD35VAEDecoder(vae_gpu1, small_cfg["device"], device_lock=gpu1_lock),
        cache_score=clip, profile_score=clip,
        threshold=threshold, capacity=capacity,
        max_pending_profiles=serving_cfg["max_pending_profiles"],
        profile_clip_stride=serving_cfg["profile"]["clip_stride"],
        profile_stable_tolerance=serving_cfg["profile"]["stable_tolerance"],
        profile_slope_jump_threshold=serving_cfg["profile"]["slope_jump_threshold"],
        profile_steep_slope_threshold=serving_cfg["profile"]["steep_slope_threshold"],
        profile_enabled=profile_enabled, events=events,
        policy=ReMixPolicy(total_steps=small_steps,
                           fallback_start_step=serving_cfg["fallback_start_step"],
                           hit_threshold=threshold))

    trace = read_trace(args.trace) if args.trace else None
    source_prompts = (None if trace is not None else list(
        prompts_from(args.prompts) if args.prompts else dataset_prompts(config, args.dataset)))
    total_requests = len(trace) if trace is not None else len(source_prompts)

    output.mkdir(parents=True, exist_ok=True)
    attempt_id = time.strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
    recorder = RequestRecorder(output, attempt_id)
    sampler = GpuSampler((int(large_cfg["device"].split(":")[1]),
                          int(small_cfg["device"].split(":")[1])),
                         config["experiment"]["gpu_sample_interval_s"])
    sampler.start()
    futures = []
    started = time.perf_counter()
    progress = tqdm(total=total_requests, desc="Requests", unit="req",
                    dynamic_ncols=True, mininterval=1.0)
    progress_lock = threading.Lock()
    last_refresh = 0.0
    failed_requests = 0
    cancelled_requests = 0

    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def request_stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_stop)

    def update_progress(finished: object = None) -> None:
        nonlocal last_refresh, failed_requests, cancelled_requests
        with progress_lock:
            if finished is not None:
                if finished.cancelled():
                    cancelled_requests += 1
                elif finished.exception() is None:
                    progress.update(1)
                else:
                    failed_requests += 1
            progress.set_postfix_str(
                f"sd35_q={service.large.queued_foreground} "
                f"sd15_q={service.small.queued_foreground} "
                f"cache={service.cache.size}/{capacity}"
                + (f" failed={failed_requests}" if failed_requests else "")
                + (f" cancelled={cancelled_requests}" if cancelled_requests else ""),
                refresh=False)
            now = time.perf_counter()
            if now - last_refresh >= 1.0:
                progress.refresh()
                last_refresh = now

    update_progress()
    interrupted = False
    run_error = None
    try:
        if trace is not None:
            for row in trace:
                delay = started + float(row["arrival_s"]) - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                future = service.submit(row["prompt"])
                future.add_done_callback(update_progress)
                future.add_done_callback(
                    lambda done, prompt=row["prompt"], meta=row:
                    recorder.record(prompt, done, meta))
                futures.append((row["prompt"], future, row))
                update_progress()
        else:
            for prompt in source_prompts:
                if args.interarrival_ms is not None:
                    delay = args.interarrival_ms / 1000
                elif arrival_cfg["distribution"] == "gamma":
                    delay = arrival_rng.gammavariate(arrival_cfg["shape"], arrival_cfg["scale_s"])
                else:
                    raise ValueError(f"unsupported arrival distribution: {arrival_cfg['distribution']}")
                if delay:
                    time.sleep(delay)
                future = service.submit(prompt)
                future.add_done_callback(update_progress)
                future.add_done_callback(
                    lambda done, prompt=prompt: recorder.record(prompt, done, None))
                futures.append((prompt, future, None))
                update_progress()
        for _, future, _ in futures:
            future.result()
        responses_finished = time.perf_counter()
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:
        run_error = str(exc)
    finally:
        drained = service.close(drain_profiles=not args.skip_profile_drain and
                                not (interrupted or run_error),
                                cancel_pending=bool(interrupted or run_error))
        sampler.stop()
        progress.close()
        recorder.close()
        signal.signal(signal.SIGTERM, previous_sigterm)
    all_work_finished = time.perf_counter()
    if interrupted or run_error:
        responses_finished = all_work_finished

    from PIL import Image
    clip_scores = []
    ordered_rows = sorted(recorder.rows, key=lambda row: (
        row["trace_index"] if row["trace_index"] is not None else float("inf")))
    with (output / "requests.jsonl").open("w", encoding="utf-8") as out:
        for row in ordered_rows:
            with Image.open(recorder.images / f"{row['request_id']}.png") as image:
                score = clip(image.convert("RGB"), row["prompt"])
            clip_scores.append(score)
            out.write(json.dumps({**row, "clip_score": score}, ensure_ascii=False) + "\n")
    events.save(output / "events.jsonl")
    write_jsonl(output / "gpu_samples.jsonl", sampler.samples)
    measured, entries = summarize(events.snapshot(), sampler.samples)
    write_jsonl(output / "cache_entries.jsonl", entries)
    complete = (not interrupted and run_error is None and
                not recorder.errors and len(recorder.rows) == total_requests)
    summary = {
        "run_status": "complete" if complete else "partial",
        "attempt_id": attempt_id,
        "checkpoint_path": str(recorder.path),
        "requested_count": total_requests,
        "submitted_count": len(futures),
        "request_errors": recorder.errors,
        "run_error": run_error,
        "mode": args.mode,
        "trace": str(args.trace) if args.trace else None,
        "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest() if args.trace else None,
        "cache_capacity": capacity,
        "replacement": "fifo",
        "clip_device": clip_device,
        "mean_output_clip_score": sum(clip_scores) / len(clip_scores) if clip_scores else None,
        "count": len(recorder.rows),
        "response_window_s": responses_finished - started,
        "throughput_responses_per_s": (len(recorder.rows) / (responses_finished - started)
                                       if responses_finished > started else 0),
        "all_work_window_s": all_work_finished - started,
        "profiles_drained": drained,
        "profiles_pending": service.profiles.pending,
        "profiles_dropped": service.profiles.dropped,
        "profile_errors": [str(error) for _, error in service.profiles.errors],
        **measured,
    }
    summary_path = output / ("summary.json" if complete else
                             f"partial_summary.{attempt_id}.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not complete:
        raise SystemExit(130 if interrupted else 1)


if __name__ == "__main__":
    main()
