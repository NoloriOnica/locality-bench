# Package provenance and release boundary

Version 0.1.0, prepared 30 September 2026 by Codex for Kaliraj Santosh.

The CPU metric arithmetic was extracted from this project's
`scripts/phase5_locality_metrics.py` without changing the active production runner.
`scripts/phase5_common.py` supplies the historical resizing and threshold rules.
The package tests compare both implementations directly at multiple band widths.
This temporary duplication protects ongoing model jobs; later consolidation
must retain parity and preserve the production reference.

The package is not the full historical analysis pipeline: LPIPS and visual judging
are not bundled, and the caller must provide prompt/mask eligibility and semantic
labels. `locality_cpu_v1` identifies that scope. `repaired_v1` remains unchanged.
No dataset images, restricted masks, model weights, API keys or participant data
are included. Demo fixtures are generated procedurally.

Runtime dependencies are NumPy, Pillow and scikit-image. The optional HTTP surface
uses FastAPI, Pydantic, Starlette and Uvicorn; tests use pytest and HTTPX. Dependencies
retain their own licenses and are installed separately. Packaging follows
[setuptools pyproject configuration](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html);
HTTP schemas follow [FastAPI request models](https://fastapi.tiangolo.com/tutorial/body/).

The owner authorised publication under Kaliraj Santosh at
`https://github.com/NoloriOnica/locality-bench`, with the research repository private.
No open-source license is assigned on the owner's behalf. The public research preview
retains copyright; third-party dependencies retain their own licenses. Public source
visibility is not a claim of an open-source license or publication on PyPI.

## Release notes

0.1.0 adds pair and manifest scoring; explicit generation, geometry and semantic
states; source-cluster intervals; JSON/CSV/HTML exports; synthetic demo generation;
an editor adapter contract; and a self-hostable API using registered sample IDs.
