import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from summarize_experiment import (load_runs, print_cache, print_efficiency,
                                  print_profile, print_quality)


class SummaryScriptTests(unittest.TestCase):
    def test_four_tables_join_request_quality_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "coco_low_idle-profile"
            folder.mkdir()
            summary = {
                "run_status": "complete", "count": 2, "completed": 2,
                "hit_count": 1, "hit_rate": 0.5, "cache_entries": 1,
                "reuse_before_eviction_or_end": {"zero_fraction": 0.0, "p90": 1},
                "profile_events": {"profile_enqueue": 1, "profile_complete": 1},
                "profile_clip_coarse_scores": 2, "profile_clip_fine_scores": 1,
                "profile_refined_entries": 1,
                "profile_decode_wall_s_by_device": {"gpu-small": 0.5},
                "profile_score_wall_s": 0.2, "profile_overlap_fraction": 0.5,
                "throughput_per_s": 0.25,
                "latency_s": {"p50": 2, "p95": 3, "p99": 4},
                "queue_s": {"p95": 1},
                "gpu": {"0": {"mean_utilization_pct": 90,
                                "max_memory_used_mib": 1024},
                        "1": {"mean_utilization_pct": 30,
                                "max_memory_used_mib": 2048}},
            }
            (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            requests = [
                {"request_id": "a", "path": "miss", "clip_score": 0.3},
                {"request_id": "b", "path": "hit", "clip_score": 0.5},
            ]
            (folder / "requests.jsonl").write_text(
                "\n".join(json.dumps(row) for row in requests) + "\n", encoding="utf-8")
            (folder / "pickscore.jsonl").write_text(
                '{"request_id":"a","pickscore":1.0}\n'
                '{"request_id":"b","pickscore":2.0}\n', encoding="utf-8")
            (folder / "cache_entries.jsonl").write_text(
                '{"entry_id":"x","hits":1,"hits_after_profile":1,'
                '"profiled":true,"evicted":false}\n', encoding="utf-8")

            runs = load_runs(Path(temp))
            output = StringIO()
            with redirect_stdout(output):
                print_cache(runs)
                print_profile(runs)
                print_quality(runs)
                print_efficiency(runs)
            result = output.getvalue()
            for heading in ("CACHE", "PROFILE", "QUALITY", "EFFICIENCY"):
                self.assertIn(heading, result)
            self.assertIn("0.4000", result)  # mean output CLIP
            self.assertIn("1.5000", result)  # mean PickScore
            self.assertIn("hitsAfterProf", result)
