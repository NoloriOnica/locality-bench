# Locality Bench 0.1.0 - research preview

Regional image-editing evaluation by Kaliraj Santosh.

This release provides a CPU Python API, command-line reports and a self-hostable HTTP
API over one scoring core. It reports background/target preservation separately from
supplied semantic success, retaining failures, missing labels and coverage.

Included assets: installable wheel, source archive and SHA256 checksums. Install the
wheel, then run `locality-bench demo demo-data` and
`locality-bench score demo-data/manifest.json --out demo-report`.

Validation: 14 tests passed against installed wheels in clean Windows and Ubuntu
environments. Tests cover production arithmetic parity, geometry, failures, hashes,
denominators, missing labels, reports and API parity. Twelve synthetic transformations
were also scored with production disk dilation. No human-validation results are claimed.

The release contains no benchmark images, model weights or participant data. LPIPS
and visual judging are not bundled. This is a research preview with no open-source
license grant; third-party dependencies retain their own terms. It is published on
GitHub, not PyPI. See the repository README for the full input contract and limitations.
