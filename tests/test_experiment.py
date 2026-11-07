import unittest
from unittest.mock import patch

from remix_system.experiment import make_trace, summarize


class ExperimentTests(unittest.TestCase):
    def test_trace_is_repeatable_and_has_800_requests(self):
        config = {
            "datasets": {}, "serving": {"seed": 42},
            "experiment": {"replacement": "fifo", "request_count": 800,
                           "unique_pool": 400, "exact_reuses": 200,
                           "edited_reuses": 200,
                           "loads": {"low": {"gamma_shape": 1, "gamma_scale_s": 10}}},
        }
        with patch("remix_system.experiment.dataset_prompts",
                   return_value=iter(f"prompt {i}" for i in range(400))):
            first = make_trace(config, "coco", "low")
        with patch("remix_system.experiment.dataset_prompts",
                   return_value=iter(f"prompt {i}" for i in range(400))):
            second = make_trace(config, "coco", "low")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 800)
        self.assertEqual({k: sum(r["kind"] == k for r in first)
                          for k in ("initial", "new", "exact", "edited")},
                         {"initial": 200, "new": 200, "exact": 200, "edited": 200})
        self.assertTrue(all(b["arrival_s"] > a["arrival_s"]
                            for a, b in zip(first, first[1:])))

    def test_summary_counts_reuse_and_profile_overlap(self):
        events = [
            {"t": 0.0, "kind": "cache_insert", "entry_id": "a"},
            {"t": 1.0, "kind": "cache_hit", "entry_id": "a", "profiled": False},
            {"t": 2.0, "kind": "profile_publish", "entry_id": "a"},
            {"t": 3.0, "kind": "cache_hit", "entry_id": "a", "profiled": True},
            {"t": 4.0, "kind": "cache_evict", "entry_id": "a"},
            {"t": 0.5, "kind": "foreground_start", "device": "gpu-small", "request_id": "r"},
            {"t": 2.5, "kind": "foreground_end", "device": "gpu-small", "request_id": "r"},
            {"t": 1.0, "kind": "profile_decode_start", "device": "gpu-large", "job_id": "j"},
            {"t": 2.0, "kind": "profile_decode_end", "device": "gpu-large", "job_id": "j", "duration_s": 1.0},
        ]
        summary, entries = summarize(events, [])
        self.assertEqual(entries[0]["hits"], 2)
        self.assertEqual(entries[0]["hits_after_profile"], 1)
        self.assertTrue(entries[0]["evicted"])
        self.assertEqual(summary["profile_overlap_other_gpu_foreground_s"], 1.0)


if __name__ == "__main__":
    unittest.main()
