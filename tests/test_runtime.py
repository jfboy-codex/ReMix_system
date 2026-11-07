import threading
import unittest

from remix_system import Generated, Profile, ReMixRuntime, Trace
from remix_system.cache import CacheEntry, CacheManager
from remix_system.events import EventLog
from remix_system.device import DeviceManager
from remix_system.policy import ReMixPolicy
from remix_system.profiling import ProfileCoordinator
from remix_system.workers import ProfileDecoder


class FakeLarge:
    def generate(self, prompt):
        return Generated(prompt, Trace(tuple(range(6)), (0, 6, 5, 4, 3, 2)))


class FakeSmall:
    def edit(self, plan):
        return (plan.source_image, plan.target_prompt, plan.start_step)


class FakeDecoder:
    def decode(self, latent):
        return latent


def cache_score(image, prompt):
    return 0.9 if image == "source" and prompt == "edit" else 0.0


def profile_score(image, prompt):
    return 0.9 if image == "source" else 0.1 + float(image) * 0.1


class RuntimeTests(unittest.TestCase):
    def test_miss_profiles_and_hit_uses_ready_profile(self):
        runtime = ReMixRuntime(FakeLarge(), FakeSmall(), FakeDecoder(), FakeDecoder(),
                               cache_score, profile_score, threshold=0.5, capacity=2)
        try:
            miss = runtime.submit("source").result(timeout=2)
            self.assertEqual(miss.path, "miss")
            self.assertTrue(runtime.profiles.wait_empty(2))
            entry = runtime.cache.snapshot()[0]
            self.assertIsNotNone(entry.profile)
            hit = runtime.submit("edit").result(timeout=2)
            self.assertEqual(hit.path, "hit")
            self.assertEqual(hit.image[1], "edit")
            self.assertGreaterEqual(hit.end_to_end_seconds, 0)
        finally:
            runtime.close(timeout=2)

    def test_evicted_entry_does_not_receive_late_profile(self):
        cache = CacheManager(lambda image, prompt: 0, threshold=1, capacity=1)
        profiler = ProfileCoordinator(cache, profile_score)
        first = cache.insert("old", "source")
        profiler.enqueue(first, Trace((1, 2, 3), (0, 3, 2)))
        cache.insert("new", "new-image")
        self.assertIsNone(profiler.claim())
        self.assertTrue(profiler.wait_empty(1))
        self.assertIsNone(cache.snapshot()[0].profile)
        profiler.close()

    def test_foreground_runs_before_next_profile_chunk(self):
        started = threading.Event()
        release = threading.Event()
        events = []

        class BlockingDecoder:
            def decode(self, latent):
                events.append(f"decode-{latent}")
                started.set()
                release.wait(2)
                return latent

        cache = CacheManager(lambda image, prompt: 0, threshold=1, capacity=2)
        profiler = ProfileCoordinator(cache, profile_score)
        entry = cache.insert("source", "source")
        profiler.enqueue(entry, Trace((0, 1, 2), (0, 3, 2)))
        device = DeviceManager("test-device", lambda x: events.append("front") or x,
                               ProfileDecoder(BlockingDecoder()), profiler)
        try:
            self.assertTrue(started.wait(2))
            foreground = device.submit("request", lambda x: x)
            release.set()
            self.assertEqual(foreground.result(timeout=2), "request")
            self.assertEqual(events[:2], ["decode-0", "front"])
            self.assertTrue(profiler.wait_empty(2))
        finally:
            release.set()
            device.close()
            profiler.close()

    def test_step_selection_responds_to_query_gap(self):
        entry = CacheEntry("id", "v1", "source", "image",
                           Profile(0, 9, 0.9, ((0, 0.2), (5, 0.5), (9, 0.9))))
        policy = ReMixPolicy(total_steps=10, hit_threshold=0.2)
        near = policy.plan(entry, 0.9, "near")
        far = policy.plan(entry, 0.2, "far")
        self.assertEqual((near.start_step, far.start_step), (9, 0))
        self.assertGreater(far.strength, near.strength)

    def test_pending_limit_discards_extra_analysis(self):
        cache = CacheManager(lambda image, prompt: 0, threshold=1, capacity=2)
        profiler = ProfileCoordinator(cache, profile_score, max_pending=1)
        try:
            first = cache.insert("first", "source")
            second = cache.insert("second", "second")
            trace = Trace((1, 2), (0, 1))
            self.assertIsNotNone(profiler.enqueue(first, trace))
            self.assertIsNone(profiler.enqueue(second, trace))
            self.assertEqual(profiler.dropped, 1)
        finally:
            profiler.close()

    def test_no_profile_mode_still_caches_and_reuses(self):
        events = EventLog()
        runtime = ReMixRuntime(FakeLarge(), FakeSmall(), FakeDecoder(), FakeDecoder(),
                               cache_score, profile_score, threshold=0.5, capacity=2,
                               profile_enabled=False, events=events)
        try:
            self.assertEqual(runtime.submit("source").result(timeout=2).path, "miss")
            self.assertEqual(runtime.submit("edit").result(timeout=2).path, "hit")
            self.assertIsNone(runtime.cache.snapshot()[0].profile)
            self.assertFalse(any(e["kind"] == "profile_enqueue" for e in events.snapshot()))
        finally:
            runtime.close(timeout=2)


if __name__ == "__main__":
    unittest.main()
