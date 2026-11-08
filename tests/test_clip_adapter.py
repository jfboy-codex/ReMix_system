"""Check device placement without requiring GPU libraries on the development host."""
import sys
import types
import unittest
from contextlib import nullcontext
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from remix_system.diffusers_adapters import CLIPScorer, SD35Backend


class FakeTensor:
    placements = []

    def __init__(self, values=(0.6, 0.8)):
        self.values = values

    def to(self, device):
        self.placements.append(device)
        return self

    def norm(self, dim=-1, keepdim=True):
        return 1

    def __truediv__(self, other):
        return self

    def __getitem__(self, index):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class FakeModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def get_text_features(self, **kwargs):
        return FakeTensor()

    def get_image_features(self, **kwargs):
        return FakeTensor()


class FakeProcessor:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, **kwargs):
        return {"input_ids": FakeTensor()}


class CLIPAdapterTests(unittest.TestCase):
    def test_hds_timing_line_is_filtered_without_muting_other_output(self):
        namespace = {}
        exec("def generate_with_profile():\n"
             "    print('[HDS inference] timing')\n"
             "    print('other diagnostic')\n", namespace)
        pipeline = types.SimpleNamespace(generate_with_profile=namespace["generate_with_profile"])
        SD35Backend(pipeline, "cuda:0")
        output = StringIO()
        with redirect_stdout(output):
            pipeline.generate_with_profile()
        self.assertEqual(output.getvalue(), "other diagnostic\n")

    def test_model_and_inputs_use_configured_gpu(self):
        FakeTensor.placements = []
        modules = {
            "torch": types.SimpleNamespace(Tensor=FakeTensor, inference_mode=nullcontext),
            "transformers": types.SimpleNamespace(CLIPModel=FakeModel,
                                                   CLIPProcessor=FakeProcessor),
        }
        with patch.dict(sys.modules, modules):
            scorer = CLIPScorer("model", device="cuda:1")
            self.assertAlmostEqual(scorer("image", "prompt"), 1.0)
        self.assertEqual(scorer.model.device, "cuda:1")
        self.assertEqual(FakeTensor.placements, ["cuda:1", "cuda:1"])
