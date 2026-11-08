"""Create one fixed request trace to share across profiling modes."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from remix_system.experiment import make_trace, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/server.yaml"))
    parser.add_argument("--dataset", choices=("coco", "diffusiondb"), required=True)
    parser.add_argument("--load", choices=("low", "high"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    trace = make_trace(config, args.dataset, args.load)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, trace)
    print(f"wrote {len(trace)} requests to {args.output}")


if __name__ == "__main__":
    main()
