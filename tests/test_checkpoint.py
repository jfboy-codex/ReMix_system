import json
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path

from remix_system.checkpoint import RequestRecorder
from remix_system.types import Result


class FakeImage:
    def save(self, path):
        Path(path).write_bytes(b"image")


class CheckpointTests(unittest.TestCase):
    def test_completed_response_is_persisted_before_run_finishes(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            recorder = RequestRecorder(output, "test")
            future = Future()
            future.set_result(Result("request-1", FakeImage(), "miss", 1.0, 2.0))
            recorder.record("a fox", future, {"index": 3})
            recorder._queue.join()
            rows = [json.loads(line) for line in recorder.path.read_text().splitlines()]
            self.assertEqual(rows[0]["trace_index"], 3)
            self.assertEqual(rows[0]["end_to_end_s"], 1.0)
            self.assertEqual((output / "images" / "request-1.png").read_bytes(), b"image")
            recorder.close()

    def test_cancelled_request_does_not_create_a_result(self):
        with tempfile.TemporaryDirectory() as temp:
            recorder = RequestRecorder(Path(temp), "test")
            future = Future()
            future.cancel()
            recorder.record("cancelled", future, None)
            recorder.close()
            self.assertEqual(recorder.rows, [])
            self.assertEqual(recorder.path.read_text(), "")
