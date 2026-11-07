"""Compute PickScore after serving so quality evaluation does not affect latency."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/server.yaml"))
    parser.add_argument("--output", type=Path, required=True,
                        help="a run directory containing requests.jsonl and images")
    args = parser.parse_args()
    import torch
    import yaml
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_path = config["models"]["pickscore"]["path"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True,
                                      torch_dtype=dtype, low_cpu_mem_usage=False).eval().to(device)
    requests = [json.loads(line) for line in (args.output / "requests.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    scores = []
    with (args.output / "pickscore.jsonl").open("w", encoding="utf-8") as out:
        for row in requests:
            with Image.open(args.output / "images" / f"{row['request_id']}.png") as image:
                inputs = processor(images=image.convert("RGB"), text=row["prompt"],
                                   return_tensors="pt", padding=True,
                                   truncation=True).to(device)
                with torch.inference_mode():
                    score = float(model(**inputs).logits_per_image.item())
            scores.append(score)
            out.write(json.dumps({"request_id": row["request_id"], "pickscore": score}) + "\n")
    summary = {"count": len(scores), "mean_pickscore": sum(scores) / len(scores) if scores else None}
    (args.output / "pickscore_summary.json").write_text(json.dumps(summary, indent=2),
                                                        encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
