"""Single-machine HTTP wrapper using operator-registered sample IDs only."""
from __future__ import annotations
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from starlette.responses import JSONResponse

from . import __version__
from .batch import aggregate, evaluate_manifest, validate_manifest
from .core import MetricConfig, PROTOCOL_VERSION


class PairRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_id: str = Field(min_length=1, max_length=200)
    band_px: StrictInt = Field(default=8, ge=0, le=128)
    leak_threshold: float = Field(default=0.05, ge=0, le=1, allow_inf_nan=False)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_ids: list[str] = Field(min_length=1, max_length=32)
    band_px: StrictInt = Field(default=8, ge=0, le=128)
    leak_threshold: float = Field(default=0.05, ge=0, le=1, allow_inf_nan=False)


class BodyLimit:
    """Bound bytes before JSON parsing, including chunked requests."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            total += len(message.get("body", b""))
            if total > 16384:
                return await JSONResponse({"detail": "Request exceeds 16 KiB"}, status_code=413)(scope, receive, send)
            chunks.append(message)
            if not message.get("more_body", False):
                break
        async def replay():
            return chunks.pop(0) if chunks else await receive()
        await self.app(scope, replay, send)


def create_app(manifest=None):
    """Register a trusted local manifest at startup; clients cannot choose paths/URLs."""
    manifest = manifest or os.environ.get("LOCALITY_BENCH_MANIFEST")
    records = validate_manifest(manifest) if manifest else []
    registry = {r["sample_id"]: r for r in records}
    app = FastAPI(title="Locality Bench", version=__version__, description="Evaluate operator-registered samples. Bind to loopback or an authenticated private proxy.")
    app.add_middleware(BodyLimit)

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "registered_samples": len(registry)}

    @app.get("/protocol")
    def protocol():
        return {"protocol_version": PROTOCOL_VERSION, "defaults": MetricConfig().metadata(),
                "semantic_policy": "supplied labels only; absent is not_evaluated",
                "max_batch": 32, "max_body_bytes": 16384}

    def run(ids, band, threshold):
        if len(ids) != len(set(ids)):
            raise HTTPException(422, "Duplicate sample IDs")
        if any(key not in registry for key in ids):
            raise HTTPException(404, "Unknown sample ID")
        with TemporaryDirectory(prefix="locality-bench-") as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps([registry[key] for key in ids]), encoding="utf-8")
            try:
                rows = evaluate_manifest(path, config=MetricConfig(band, threshold))
            except (ValueError, OSError):
                raise HTTPException(422, "Registered data failed validation") from None
        # Return provenance and hashes without exposing local filesystem paths.
        for row in rows:
            for key in ("source", "mask", "output", "scoring_error"):
                row.pop(key, None)
        return rows

    @app.post("/pair")
    def pair(request: PairRequest):
        return run([request.sample_id], request.band_px, request.leak_threshold)[0]

    @app.post("/batch")
    def batch(request: BatchRequest):
        rows = run(request.sample_ids, request.band_px, request.leak_threshold)
        try:
            summary = aggregate(rows)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        return {"results": rows, "summary": summary}

    return app
