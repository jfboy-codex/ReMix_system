"""Print four compact tables from ReMix experiment run directories.

Usage: python summarize_experiment.py --root outputs/experiment
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def number(value, digits: int = 2, scale: float = 1) -> str:
    if value is None:
        return "-"
    return f"{float(value) * scale:.{digits}f}"


def integer(value) -> str:
    return "-" if value is None else str(int(value))


def mean(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else None


def table(title: str, columns: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(str(value)) for value in [name] + [row[i] for row in rows])
              for i, name in enumerate(columns)]
    print(f"\n{title}")
    print("  ".join(name.ljust(widths[i]) if i == 0 else name.rjust(widths[i])
                    for i, name in enumerate(columns)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[i]) if i == 0 else value.rjust(widths[i])
                        for i, value in enumerate(row)))


def load_runs(root: Path) -> list[dict]:
    if not root.is_dir():
        raise SystemExit(f"Run directory does not exist: {root}")
    runs = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        summary_path = folder / "summary.json"
        if not summary_path.exists():
            partials = sorted(folder.glob("partial_summary.*.json"))
            if not partials:
                print(f"Skipping {folder.name}: no summary yet", file=sys.stderr)
                continue
            summary_path = partials[-1]
        summary = read_json(summary_path)
        requests_path = folder / "requests.jsonl"
        if not requests_path.exists():
            partials = sorted(folder.glob("requests.partial.*.jsonl"))
            requests_path = partials[-1] if partials else requests_path
        runs.append({
            "name": folder.name,
            "summary": summary,
            "requests": read_jsonl(requests_path),
            "entries": read_jsonl(folder / "cache_entries.jsonl"),
            "pickscore": read_jsonl(folder / "pickscore.jsonl"),
            "pick_summary": (read_json(folder / "pickscore_summary.json")
                             if (folder / "pickscore_summary.json").exists() else {}),
        })
    if not runs:
        raise SystemExit(f"No completed or partial runs found in {root}")
    dataset_order = {"coco": 0, "diffusiondb": 1}
    load_order = {"low": 0, "high": 1}
    mode_order = {"no-profile": 0, "idle-profile": 1}

    def sort_key(run):
        parts = run["name"].rsplit("_", 2)
        if len(parts) != 3:
            return (99, 99, 99, run["name"])
        dataset, load, mode = parts
        return (dataset_order.get(dataset, 99), load_order.get(load, 99),
                mode_order.get(mode, 99), run["name"])

    return sorted(runs, key=sort_key)


def print_cache(runs: list[dict]) -> None:
    rows = []
    for run in runs:
        s, entries = run["summary"], run["entries"]
        reuse = s.get("reuse_before_eviction_or_end", {})
        rows.append([
            run["name"], s.get("run_status", "complete"),
            integer(s.get("hit_count")), number(s.get("hit_rate"), 1, 100),
            integer(s.get("cache_entries")),
            integer(sum(not e.get("evicted", False) for e in entries) if entries else None),
            integer(sum(bool(e.get("evicted")) for e in entries) if entries else None),
            number(reuse.get("zero_fraction"), 1, 100), number(reuse.get("p90"), 1),
            integer(sum(e.get("hits_after_profile", 0) for e in entries) if entries else None),
        ])
    table("CACHE  (inserted = cumulative; resident <= capacity)",
          ["run", "status", "hits", "hit%", "inserted", "resident", "evicted",
           "zeroReuse%", "reuseP90", "hitsAfterProf"], rows)


def print_profile(runs: list[dict]) -> None:
    rows = []
    for run in runs:
        s = run["summary"]
        events = s.get("profile_events", {})
        decode = s.get("profile_decode_wall_s_by_device", {})
        completed = events.get("profile_complete", 0)
        refined = s.get("profile_refined_entries", 0)
        score_wall = s.get("profile_score_wall_s",
                           s.get("profile_cpu_score_wall_s"))
        rows.append([
            run["name"], integer(events.get("profile_enqueue", 0)),
            integer(completed), integer(events.get("profile_drop", 0)),
            integer(events.get("profile_cancel", 0)),
            integer(s.get("profile_clip_coarse_scores")),
            integer(s.get("profile_clip_fine_scores")),
            number(refined / completed if completed else None, 1, 100),
            number(decode.get("gpu-large", 0)), number(decode.get("gpu-small", 0)),
            number(score_wall), number(s.get("profile_overlap_fraction"), 1, 100),
        ])
    table("PROFILE  (wall seconds include scheduling/lock wait)",
          ["run", "jobs", "done", "drop", "cancel", "coarse", "fine", "refined%",
           "GPU0dec_s", "GPU1dec_s", "CLIPwall_s", "overlap%"], rows)


def print_quality(runs: list[dict]) -> None:
    rows = []
    for run in runs:
        requests = run["requests"]
        pick_by_id = {row["request_id"]: row["pickscore"]
                      for row in run["pickscore"] if "request_id" in row and "pickscore" in row}
        scored = [{**row, "pickscore": pick_by_id.get(row.get("request_id"))}
                  for row in requests]
        hits = [row for row in scored if row.get("path") == "hit"]
        misses = [row for row in scored if row.get("path") == "miss"]
        pick_all = mean(scored, "pickscore")
        if (pick_all is None and not run["pickscore"] and
                run["pick_summary"].get("count") == len(requests)):
            pick_all = run["pick_summary"].get("mean_pickscore")
        clip_all = mean(scored, "clip_score")
        if clip_all is None:
            clip_all = run["summary"].get("mean_output_clip_score")
        rows.append([
            run["name"], integer(len(scored)),
            number(clip_all, 4),
            number(mean(hits, "clip_score"), 4),
            number(mean(misses, "clip_score"), 4),
            number(pick_all, 4), number(mean(hits, "pickscore"), 4),
            number(mean(misses, "pickscore"), 4),
        ])
    table("QUALITY  (scores are higher when better aligned to the prompt)",
          ["run", "images", "CLIPall", "CLIPhit", "CLIPmiss",
           "Pickall", "Pickhit", "Pickmiss"], rows)


def print_efficiency(runs: list[dict]) -> None:
    rows = []
    for run in runs:
        s = run["summary"]
        lat, wait, gpu = s.get("latency_s", {}), s.get("queue_s", {}), s.get("gpu", {})
        gpu0, gpu1 = gpu.get("0", {}), gpu.get("1", {})
        rows.append([
            run["name"], integer(s.get("completed", s.get("count"))),
            number(s.get("throughput_per_s", s.get("throughput_responses_per_s")), 3),
            number(lat.get("p50"), 1), number(lat.get("p95"), 1),
            number(lat.get("p99"), 1), number(wait.get("p95"), 1),
            number(gpu0.get("mean_utilization_pct"), 1),
            number(gpu1.get("mean_utilization_pct"), 1),
            number(gpu0.get("max_memory_used_mib"), 1, 1 / 1024),
            number(gpu1.get("max_memory_used_mib"), 1, 1 / 1024),
        ])
    table("EFFICIENCY  (latency/queue in seconds; memory in GiB)",
          ["run", "done", "req/s", "latP50", "latP95", "latP99", "queueP95",
           "GPU0%", "GPU1%", "GPU0GiB", "GPU1GiB"], rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/experiment"),
                        help="directory containing the eight run folders")
    args = parser.parse_args()
    runs = load_runs(args.root)
    print(f"Loaded {len(runs)} runs from {args.root}")
    print_cache(runs)
    print_profile(runs)
    print_quality(runs)
    print_efficiency(runs)


if __name__ == "__main__":
    main()
