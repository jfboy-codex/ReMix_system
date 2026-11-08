"""Read prompt traces from the server's COCO/DiffusionDB Parquet layout."""
from __future__ import annotations

import random
from pathlib import Path


def dataset_prompts(config: dict, name: str):
    import pandas as pd

    spec = config["datasets"][name]
    limit = config["experiment"]["workload"]["source_prompts"]
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("experiment.workload.source_prompts must be a positive integer")
    data_dir = Path(config["datasets"]["root"]) / spec["path"] / "data"
    pattern = f'{spec.get("split", "val")}-*.parquet' if name == "coco" else "*.parquet"
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {name} Parquet shards found in {data_dir}")
    rng = random.Random(spec.get("seed", 42))
    emitted = 0
    for shard in files:
        frame = pd.read_parquet(shard)
        for _, row in frame.iterrows():
            if name == "coco":
                captions = row.get("answer")
                if hasattr(captions, "tolist"):
                    captions = captions.tolist()
                if not isinstance(captions, list):
                    captions = [captions] if captions is not None else []
                captions = [str(x).strip() for x in captions
                            if x is not None and str(x).strip().lower() not in ("", "none")]
                if not captions:
                    continue
                prompt = rng.choice(captions)
            else:
                prompt = str(row.get("prompt", "")).strip()
                if not prompt:
                    continue
            yield prompt
            emitted += 1
            if emitted >= limit:
                return
