# ReMix diffusion system

This repository implements the request-serving architecture discussed for ReMix. One `DeviceManager` runs on each GPU. Each manager handles foreground requests first and, when idle, claims one latent-decoding chunk from a shared profile coordinator. The CPU scorer checks whether the per-entry CLIP curve has stabilized and either requeues the next chunk or publishes a complete profile. A foreground arrival waits for the already-started decode chunk; GPU kernels are not forcibly preempted.

## Components

| Module | Responsibility |
| --- | --- |
| `remix_system.runtime` | Route hit/miss requests and own lifecycle |
| `remix_system.cache` | Search, FIFO eviction, versioned atomic profile publication |
| `remix_system.device` | One per GPU; foreground-first scheduling |
| `remix_system.profiling` | Single owner of profile progress, CPU scoring and resumption |
| `remix_system.policy` | Per-entry step selection and repaint plan |
| `remix_system.workers` | Thin large/small model and decoder boundaries |
| `remix_system.diffusers_adapters` | Optional adapters for the existing patched diffusion pipelines |

The core package is pure Python; `torch`, `diffusers`, `transformers`, and `Pillow` are imported only by the optional adapters. The backend passed to each device remains resident, including its SD3.5 VAE decoder. Profile jobs carry CPU latent data and entry/version IDs, not live model instances. A bounded profile queue drops new analysis jobs when full; requests can still use pending entries through the policy's fallback step.

## Use with the existing models

Construct `ReMixRuntime` with `SD35Backend`, `SD15Backend`, two `SD35VAEDecoder` instances, and a shared `CLIPScorer`. The SD3.5 pipeline must provide `generate_with_profile`; the SD1.5 pipeline must provide `diffedit_rewrite_soft`, as in the existing experiment's patched Diffusers pipelines. Both decoders require **SD3.5 VAE weights**, one resident on each GPU. Pass the same `CLIPScorer` as `cache_score` and `profile_score`.

```python
from remix_system import ReMixRuntime
from remix_system.diffusers_adapters import (
    CLIPScorer, SD15Backend, SD35Backend, SD35VAEDecoder,
)

# sd35 and sd15 are already-loaded patched pipelines on cuda:0 and cuda:1.
# sd35_vae_gpu1 is a separate SD3.5 VAE copy resident on cuda:1.
clip = CLIPScorer("/path/to/clip-vit-base-patch32")
service = ReMixRuntime(
    SD35Backend(sd35, "cuda:0"),
    SD15Backend(sd15, "cuda:1"),
    SD35VAEDecoder(sd35.vae, "cuda:0"),
    SD35VAEDecoder(sd35_vae_gpu1, "cuda:1"),
    cache_score=clip, profile_score=clip,
)
try:
    result = service.submit("a blue flower in a vase").result()
finally:
    drained = service.close(drain_profiles=True)
```

The two workers must not share one VAE object across devices. Set `policy=ReMixPolicy(...)` to configure step and repaint decisions. The default policy uses a fixed fallback start step while an entry's profile is pending, then inverts that entry's monotone CLIP curve using the new prompt's score. `repaint_mode="full"` selects standard image-to-image repainting; `"graduated"` selects the patched ReMix method.

## Verification

For the reviewer-facing paired profiling experiment (800 requests, FIFO capacity 200, one run per configuration), see [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md). Generate a fixed trace with `prepare_trace.py`, then pass it to `run_service.py --trace ... --mode no-profile|idle-profile`. The driver records request, profile, cache, GPU utilization, memory, and output CLIP data in the output directory.
Run `evaluate_pickscore.py --output <run-directory>` after serving to score generated images with the old experiment's local PickScore model without affecting latency measurements.

Run `python -B -m unittest discover -s tests -v`. The tests use deterministic fake backends to check miss-to-profile-to-hit flow, invalidation after cache eviction, foreground priority between profile chunks, and gap-responsive step selection. They do not measure image quality or GPU performance.

`config/server.yaml` carries the validated server paths and generation settings from `expriment_aaai/config.yaml`: SD3.5 Medium at `/home/dataset-assist-0/models/stable-diffusion-3.5-medium` on `cuda:0`, SD1.5 at `/home/dataset-assist-0/models/stable-diffusion-v1-5` on `cuda:1`, and CLIP ViT-B/32 at `/home/dataset-assist-0/models/clip-vit-base-patch32` on CPU. Both models use 40 steps; the guidance scales are 5.0 and 7.5. COCO captions come from `/home/dataset-assist-0/data/coco_caption/data/val-*.parquet`, and DiffusionDB prompts from `/home/dataset-assist-0/data/diffusiondb_10k/data/*.parquet`. Each source reads the first 400 usable rows; COCO caption selection uses seed 42. The reviewer experiment uses FIFO capacity 200 and threshold 0.26. The old `hds_soft_edit` baseline used capacity 100 and `alpha=1`; the threshold is therefore applied to cached-image versus query-text CLIP cosine similarity, as in this implementation.

On the server, install this package's server dependencies and apply the existing experiment's patched Diffusers pipelines (including its `src.mask_algo` module on `PYTHONPATH`). The patches are supplied by `expriment_aaai/install_hds_pipeline.sh`; the model and dataset weights are not part of this repository. Then run from this repository root:

```bash
python -m pip install -e '.[server]'
python run_service.py --dataset coco --output outputs/coco
python run_service.py --dataset diffusiondb --output outputs/diffusiondb
```

`--prompts prompts.jsonl` accepts records with a `prompt` field, or one prompt per line. `--sd35`, `--sd15`, `--clip`, `--steps`, `--threshold`, and `--capacity` override the YAML defaults. The default arrival intervals follow the old configuration's Gamma(shape=1, scale=1 second) distribution, sampled with a local seed; `--interarrival-ms 0` submits requests as fast as possible. The driver writes generated images, per-request response latency, response throughput, and time to drain profiling. It is a smoke and measurement entry point, not a controlled multi-baseline evaluation protocol.

The old `hds_soft_edit` experiment warms a shared cache and sets `cache_update: false`. This live serving configuration starts with an empty cache, inserts generated miss images, and profiles them during GPU idle time; it disables precomputed miss images. These workload differences change hit rate and latency, so compare results only under matched arrival traces and cache initialization.

## Remaining integration work

This is a working scheduling core and adapter layer, not a reproduced GPU result. The patched pipelines and model files are not present in this repository, so GPU inference has not been executed here. Before using it for paper measurements, add a controlled arrival trace and common multi-baseline metrics; validate profile timing, CLIP score conventions, noise schedule coordinates, and image quality against the old pipeline. Cache search currently uses exact embedding comparison over entries; switch to a vector index if cache size makes that a bottleneck. Baseline variants beyond ReMix and full-image repainting should be added as policy/adapters rather than copies of the runtime.
