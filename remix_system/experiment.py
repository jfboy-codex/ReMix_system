"""Deterministic request workload and compact event summaries."""
from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from .data import dataset_prompts


def make_trace(config: dict, dataset: str, load: str) -> list[dict]:
    spec = config["experiment"]
    if spec["replacement"].lower() != "fifo":
        raise ValueError("only FIFO is supported")
    request_count = spec["request_count"]
    if not isinstance(request_count, int) or request_count <= 0:
        raise ValueError("experiment.request_count must be a positive integer")
    prompts = list(dataset_prompts(config, dataset))
    if len(prompts) < request_count:
        raise ValueError(f"need {request_count} dataset prompts, got {len(prompts)}")
    rng = random.Random(config["serving"]["seed"])
    arrival = config["experiment"]["loads"][load]
    if arrival["gamma_shape"] <= 0 or arrival["gamma_scale_s"] <= 0:
        raise ValueError("Gamma shape and scale must be positive")
    elapsed = 0.0
    trace = []
    for index, prompt in enumerate(prompts[:request_count]):
        elapsed += rng.gammavariate(arrival["gamma_shape"], arrival["gamma_scale_s"])
        trace.append({"index": index, "arrival_s": elapsed, "prompt": prompt})
    return trace


def read_trace(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any("prompt" not in r or "arrival_s" not in r for r in rows):
        raise ValueError("trace needs prompt and arrival_s on every row")
    if any(float(b["arrival_s"]) < float(a["arrival_s"]) for a, b in zip(rows, rows[1:])):
        raise ValueError("trace arrival_s must be nondecreasing")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p / 100
    lo = math.floor(pos)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (pos - lo)


def summarize(events: list[dict], samples: list[dict]) -> tuple[dict, list[dict]]:
    entries: dict[str, dict] = {}
    arrivals, starts, dones, routes = {}, {}, {}, {}
    profile_gpu = defaultdict(float)
    profile_cpu = 0.0
    profile_counts = defaultdict(int)
    coarse_scores = 0
    fine_scores = 0
    refined_entries = 0
    active_profiles = {}
    active_foreground = {}
    profile_intervals = []
    foreground_intervals = []
    for event in events:
        kind = event["kind"]
        eid = event.get("entry_id")
        rid = event.get("request_id")
        if kind == "cache_insert":
            entries[eid] = {"entry_id": eid, "hits": 0, "hits_after_profile": 0,
                            "profiled": False, "evicted": False}
        elif kind == "cache_hit" and eid in entries:
            entries[eid]["hits"] += 1
            entries[eid]["hits_after_profile"] += int(event["profiled"])
        elif kind == "cache_evict" and eid in entries:
            entries[eid]["evicted"] = True
        elif kind == "profile_publish" and eid in entries:
            entries[eid]["profiled"] = True
        elif kind == "request_arrive":
            arrivals[rid] = event["t"]
        elif kind == "foreground_start":
            starts[rid] = event["t"]
            active_foreground[(event["device"], rid)] = event["t"]
        elif kind == "foreground_end":
            key = (event["device"], rid)
            if key in active_foreground:
                foreground_intervals.append((event["device"], active_foreground.pop(key), event["t"]))
        elif kind == "request_done":
            dones[rid] = event["t"]
        elif kind == "request_route":
            routes[rid] = event
        elif kind == "profile_decode_end":
            profile_gpu[event["device"]] += event["duration_s"]
            key = (event["device"], event["job_id"])
            if key in active_profiles:
                profile_intervals.append((event["device"], active_profiles.pop(key), event["t"]))
        elif kind == "profile_decode_start":
            active_profiles[(event["device"], event["job_id"])] = event["t"]
        elif kind == "profile_score":
            profile_cpu += event["duration_s"]
        if kind in ("profile_enqueue", "profile_complete", "profile_cancel", "profile_drop", "profile_error"):
            profile_counts[kind] += 1
        if kind == "profile_complete":
            coarse_scores += event.get("coarse_steps", 0)
            fine_scores += event.get("fine_steps", 0)
            refined_entries += int(event.get("fine_steps", 0) > 0)
    latency = [dones[r] - arrivals[r] for r in dones if r in arrivals]
    queue = [starts[r] - arrivals[r] for r in starts if r in arrivals]
    hits = sum(e.get("path") == "hit" for e in routes.values())
    reuses = [e["hits"] for e in entries.values()]
    profiled_reuses = [e["hits_after_profile"] for e in entries.values() if e["profiled"]]
    overlap = 0.0
    for device, p0, p1 in profile_intervals:
        for other_device, f0, f1 in foreground_intervals:
            if other_device != device:
                overlap += max(0.0, min(p1, f1) - max(p0, f0))
    total_profile_window = sum(b-a for _, a, b in profile_intervals)
    gpu = {}
    for index in sorted({s["gpu_index"] for s in samples}):
        selected = [s for s in samples if s["gpu_index"] == index]
        gpu[str(index)] = {
            "mean_utilization_pct": statistics.mean(s["utilization_pct"] for s in selected),
            "max_memory_used_mib": max(s["memory_used_mib"] for s in selected),
            "mean_memory_used_mib": statistics.mean(s["memory_used_mib"] for s in selected),
            "samples": len(selected),
        }
    summary = {
        "completed": len(dones), "hit_count": hits,
        "hit_rate": hits / len(routes) if routes else 0,
        "throughput_per_s": (len(dones) / (max(dones.values()) - min(arrivals.values()))
                             if dones and arrivals and max(dones.values()) > min(arrivals.values()) else 0),
        "latency_s": {"p50": percentile(latency, 50), "p95": percentile(latency, 95),
                      "p99": percentile(latency, 99)},
        "queue_s": {"p50": percentile(queue, 50), "p95": percentile(queue, 95),
                    "p99": percentile(queue, 99)},
        "cache_entries": len(entries),
        "reuse_before_eviction_or_end": {
            "zero_fraction": (sum(n == 0 for n in reuses) / len(reuses) if reuses else None),
            "p50": percentile(reuses, 50), "p90": percentile(reuses, 90),
        },
        "reuse_after_profile_before_eviction_or_end": {
            "entries": len(profiled_reuses),
            "zero_fraction": (sum(n == 0 for n in profiled_reuses) / len(profiled_reuses)
                              if profiled_reuses else None),
            "p50": percentile(profiled_reuses, 50),
            "p90": percentile(profiled_reuses, 90),
        },
        "profile_events": dict(profile_counts),
        "profile_clip_coarse_scores": coarse_scores,
        "profile_clip_fine_scores": fine_scores,
        "profile_refined_entries": refined_entries,
        "profile_decode_wall_s_by_device": dict(profile_gpu),
        "profile_decode_chunks": len(profile_intervals),
        "profile_overlap_other_gpu_foreground_s": overlap,
        "profile_overlap_fraction": (overlap / total_profile_window
                                     if total_profile_window else None),
        "profile_cpu_score_wall_s": profile_cpu,
        "gpu": gpu,
    }
    return summary, list(entries.values())
