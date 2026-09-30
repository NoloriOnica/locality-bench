# Locality Bench 0.1

Evaluate how much an image edit changes the target and its surroundings. Supply your
own source, output and target mask. The CPU package reports regional MSE, PSNR, SSIM
and changed-background fraction; semantic success comes from separately supplied labels.

## Quickstart

Python 3.10 or later:

```sh
python -m pip install ./dist/locality_bench-0.1.0-py3-none-any.whl
locality-bench demo demo-data
locality-bench validate demo-data/manifest.json
locality-bench score demo-data/manifest.json --out demo-report
```

Open `demo-report/report.html`. JSON and CSV retain per-output metrics, hashes,
geometry, failures, eligibility and separate denominators. The generated demo uses
synthetic pixels only. It supplies no human or judge labels, so semantic success is
reported as `not_evaluated`.

```python
from locality_bench import evaluate_pair, evaluate_manifest, aggregate

result = evaluate_pair("source.png", "edited.png", "mask.png")
rows = evaluate_manifest("manifest.json")
summary = aggregate(rows, bootstrap_samples=2000, seed=20260930)
```

## Manifest

Use a JSON array or CSV. Paths resolve relative to the manifest. `sample_id` identifies
one output and must be unique; `source_id` groups outputs sharing a source for uncertainty.
The demo writes a complete example. Required fields:

| Field | Meaning |
|---|---|
| `sample_id`, `source_id`, `model_id` | Stable output, source and model identifiers |
| `instruction` | Actual requested edit |
| `source`, `mask`, `output` | Local paths; output required only for produced images |
| `generation_status` | `produced`, `failed`, `refused`, `not_run`, or `degenerate` |
| `prompt_valid` | Explicit JSON boolean or CSV `true`/`false` |
| `locality_disposition` | `keep`, `fidelity_only`, or `exclude`; gates locality only |
| `model_revision`, `recipe`, `seed` | Generation provenance; use `unrecorded` and null explicitly if unknown |

Optional `source_sha256`, `mask_sha256`, `output_sha256` are checked before scoring.
Computed hashes are always returned for scored pairs. Optional `semantic_success` is
a boolean with mandatory `semantic_source` and `semantic_rubric` identities. Threshold
raw judge scores before import using a declared rule. The package does not invent a
judge result, download weights, or infer semantic correctness from pixel preservation.

## Protocol and interpretation

`locality_cpu_v1` is a package protocol, distinct from the historical `repaired_v1`
analysis policy. Its default numerical definitions match production: float32 RGB in
[0,1], edited image resampled to source dimensions with LANCZOS, mask >127 with NEAREST
resampling, Euclidean disk dilation of eight pixels, RMS-change threshold 0.05 and
PSNR capped at 100 dB. A seven-pixel SSIM window uses scikit-image defaults. LPIPS is
explicitly not evaluated. No model weights are required.

Empty target masks fail validation during scoring. An empty protected region returns
null metrics and is ineligible; it never receives a perfect score. Images smaller
than seven pixels on either axis cannot be scored. Readable all-black outputs retain
pixel diagnostics but count as generation failures and are excluded from locality.
This is a declared benchmark failure policy, not a universal rule for arbitrary tasks.

Batch summaries retain valid attempted tasks, including failures and unrun jobs.
Unobserved semantic labels remain unknown; `success_rate` is null unless every valid
attempt has a known outcome. `success_rate_lower_bound` and label coverage expose
incomplete evidence. Conditional preservation and coverage use known successes only.
Mixed metric configurations or semantic instruments are rejected. Geometry distortion
is reported; both 4% and 2% clean-subset counts are shown without excluding other
measurable outputs from the main mean.

Intervals resample source IDs with replacement and carry eligibility with each source.
They describe the observed sources, not seed or rater variability. Single-source
examples have no interval. Models are summarised separately: this is not a paired
significance test or a universal quality ranking.

## HTTP interface

Install `python -m pip install '.[http]'` from the source directory, or install the
wheel with the `[http]` extra. Register a trusted local manifest, then bind to loopback:

```powershell
$env:LOCALITY_BENCH_MANIFEST = (Resolve-Path demo-data/manifest.json).Path
python -m uvicorn locality_bench.http:create_app --factory --host 127.0.0.1 --port 8000
```

On Linux/macOS use `export LOCALITY_BENCH_MANIFEST=/absolute/path/manifest.json`.
`GET /health`, `GET /protocol`, `POST /pair` and `POST /batch` share the Python core.
Interactive documentation is at `/docs`; machine-readable OpenAPI is at `/openapi.json`.

```json
{"sample_id": "spill", "band_px": 8, "leak_threshold": 0.05}
```

The batch body uses `sample_ids` instead. Requests are limited to 16 KiB and 32 samples.
Clients can reference registered IDs only; paths and URLs are not accepted. Only the
operator registers image files. This is a synchronous single-machine service; use a
private authenticated proxy before exposing it beyond loopback. It is not a public
image-hosting or generation endpoint.

## Custom editors and development

`locality_bench.adapters.Editor` defines `generate(source, instruction, destination,
seed=...) -> GeneratedOutput`. The output carries a path, revision, recipe and seed.
`IdentityExample` is an executable no-change control. Add its returned path/provenance
to a manifest; semantic labels remain separate.

```sh
python -m pip install -e '.[http,test]'
python -m pytest tests/package -q
python scripts/package_verify.py
python -m build
```

The parity suite compares the CPU core against the unchanged production function.
The synthetic verification records use production disk dilation and live separately
from the historical four-neighbour stress suite. `docs/package/PROVENANCE.md` records
scope and third-party dependencies. Author: Kaliraj Santosh. This is a public research
preview; no open-source license has been assigned. Copyright and third-party rights
are retained. The GitHub release supplies an installable wheel and source archive.
