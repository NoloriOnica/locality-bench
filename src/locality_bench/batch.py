"""Manifest validation and denominator-preserving aggregation."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import re
import numpy as np

from .core import MetricConfig, PROTOCOL_VERSION, SCHEMA_VERSION, evaluate_pair, sha256

STATUSES = {"produced", "failed", "refused", "not_run", "degenerate"}
REQUIRED = {"sample_id", "source_id", "model_id", "instruction", "source", "mask", "generation_status",
            "prompt_valid", "locality_disposition", "model_revision", "recipe", "seed"}


def load_manifest(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = json.load(stream) if path.suffix.lower() == ".json" else list(csv.DictReader(stream))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must contain a nonempty list of records")
    return rows


def _bool(value):
    if type(value) is bool:
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError("Boolean must be true or false")


def validate_manifest(path, *, check_files=True):
    rows = load_manifest(path)
    base = Path(path).resolve().parent
    seen = set()
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict) or REQUIRED - row.keys():
            raise ValueError(f"Row {i}: missing required fields {sorted(REQUIRED - row.keys()) if isinstance(row, dict) else ''}")
        for key in ("sample_id", "source_id", "model_id", "instruction", "model_revision", "recipe"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"Row {i}: {key} must be a nonempty string (use 'unrecorded' explicitly)")
        identity = row["sample_id"]
        if identity in seen:
            raise ValueError(f"Duplicate sample_id: {identity}")
        seen.add(identity)
        row["prompt_valid"] = _bool(row["prompt_valid"])
        if row["generation_status"] not in STATUSES or row["locality_disposition"] not in {"keep", "fidelity_only", "exclude"}:
            raise ValueError(f"Row {i}: invalid generation status or mask disposition")
        semantic = row.get("semantic_success")
        if semantic not in (None, ""):
            row["semantic_success"] = _bool(semantic)
            if not row.get("semantic_source") or not row.get("semantic_rubric"):
                raise ValueError(f"Row {i}: semantic labels require source and rubric identities")
        else:
            row["semantic_success"] = None
        if row["generation_status"] == "not_run" and row["semantic_success"] is not None:
            raise ValueError(f"Row {i}: unrun jobs cannot have semantic labels")
        if row["seed"] not in (None, "", "unrecorded"):
            if isinstance(row["seed"], bool) or not re.fullmatch(r"-?\d+", str(row["seed"])):
                raise ValueError(f"Row {i}: seed must be an integer or explicitly unrecorded")
        if row["generation_status"] != "produced" and row["semantic_success"] is True:
            raise ValueError(f"Row {i}: failed/unrun generation cannot be a semantic success")
        for key in ("source", "mask", "output"):
            value = row.get(key)
            needed = key != "output" or row["generation_status"] == "produced"
            if not value:
                if needed:
                    raise ValueError(f"Row {i}: missing {key} path")
                continue
            if not isinstance(value, str):
                raise ValueError(f"Row {i}: {key} path must be a string")
            resolved = (base / value).resolve()
            row[key] = str(resolved)
            expected = row.get(f"{key}_sha256")
            if expected and not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
                raise ValueError(f"Row {i}: malformed {key} SHA256")
            if check_files and needed:
                if not resolved.is_file():
                    raise ValueError(f"Row {i}: missing {key} file: {resolved}")
                if expected and sha256(resolved).lower() != expected.lower():
                    raise ValueError(f"Row {i}: {key} SHA256 mismatch")
    return rows


def evaluate_manifest(path, *, config=MetricConfig()):
    from . import __version__
    rows = validate_manifest(path)
    results = []
    for row in rows:
        rec = dict(row, schema_version=SCHEMA_VERSION, package_version=__version__, protocol_version=PROTOCOL_VERSION,
                   metric_config=config.metadata(), scoring_status="not_evaluated", metrics={}, geometry={},
                   locality_eligible=False, eligibility_reasons=[], semantic_status="not_evaluated")
        if row["generation_status"] == "produced":
            try:
                rec.update(evaluate_pair(row["source"], row["output"], row["mask"], config=config))
            except (ValueError, OSError) as exc:
                rec.update(scoring_status="failed", scoring_error=str(exc), eligibility_reasons=["scoring_failed"])
        else:
            rec["eligibility_reasons"].append("generation_" + row["generation_status"])
        if not row["prompt_valid"]:
            rec["eligibility_reasons"].append("invalid_prompt")
        if row["locality_disposition"] != "keep":
            rec["eligibility_reasons"].append("mask_" + row["locality_disposition"])
        rec["locality_eligible"] = rec["scoring_status"] == "ok" and not rec["eligibility_reasons"]
        rec["semantic_success"] = row["semantic_success"]
        if rec["generation_status"] in {"failed", "refused", "degenerate"}:
            rec.update(semantic_success=False, semantic_status="generation_failure")
        elif row["semantic_success"] is not None:
            rec["semantic_status"] = "supplied"
        results.append(rec)
    return results


def aggregate(rows, *, bootstrap_samples=2000, seed=20260930):
    """Group by model; bootstrap source IDs, retaining all rows for each source.

    Unknown semantic labels remain unknown. A lower bound is reported separately
    when attempt success is not completely observed. Invalid prompts never count.
    """
    if type(bootstrap_samples) is not int or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer")
    if not rows:
        return []
    settings = {(r.get("protocol_version"), json.dumps(r.get("metric_config"), sort_keys=True)) for r in rows}
    if len(settings) != 1:
        raise ValueError("Mixed metric/protocol settings cannot be pooled")
    labels = {(r.get("semantic_source"), r.get("semantic_rubric")) for r in rows if r.get("semantic_status") == "supplied"}
    if len(labels) > 1:
        raise ValueError("Mixed semantic sources/rubrics cannot be pooled")
    identities = [r["sample_id"] for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate sample IDs")
    summaries = []
    for model in sorted({r["model_id"] for r in rows}):
        all_model = [r for r in rows if r["model_id"] == model]
        valid = [r for r in all_model if r["prompt_valid"]]
        locality = [r for r in valid if r["locality_eligible"]]
        conditional = [r for r in locality if r["semantic_success"] is True]
        n = len(valid)
        known = sum(r["semantic_success"] is not None for r in valid)
        success = sum(r["semantic_success"] is True for r in valid)
        def mean(group):
            return float(np.mean([r["metrics"]["bgb_psnr"] for r in group])) if group else None
        def interval(group):
            if not group or not bootstrap_samples:
                return [None, None]
            # Use the entire valid source population so eligibility/missingness travels
            # with each resampled source. Only finite replicate means define this CI.
            ids = sorted({r["source_id"] for r in valid})
            if len(ids) < 2:
                return [None, None]
            groups = {key: [r["metrics"]["bgb_psnr"] for r in group if r["source_id"] == key] for key in ids}
            sums = np.array([sum(groups[key]) for key in ids])
            counts = np.array([len(groups[key]) for key in ids])
            rng = np.random.default_rng(seed)
            means = []
            for _ in range(bootstrap_samples):
                sample = rng.integers(0, len(ids), len(ids))
                denominator = counts[sample].sum()
                if denominator:
                    means.append(sums[sample].sum() / denominator)
            return np.percentile(means, [2.5, 97.5]).tolist() if means else [None, None]
        summaries.append({"model_id": model, "protocol_version": rows[0]["protocol_version"],
                          "metric_config": rows[0]["metric_config"], "n_rows": len(all_model),
                          "n_valid_attempts": n, "n_invalid_prompts": len(all_model) - n,
                          "n_locality": len(locality), "n_locality_clean_4pct": sum(r["geometry"]["relative_aspect_distortion"] <= 0.04 for r in locality),
                          "n_locality_clean_2pct": sum(r["geometry"]["relative_aspect_distortion"] <= 0.02 for r in locality),
                          "generation_counts": {s: sum(r["generation_status"] == s for r in valid) for s in sorted(STATUSES)},
                          "n_scoring_failed": sum(r["scoring_status"] == "failed" for r in valid),
                          "n_semantic_known": known, "n_semantic_unknown": n - known, "n_success": success,
                          "success_rate": success / n if n and known == n else None,
                          "success_rate_lower_bound": success / n if n else None,
                          "semantic_label_coverage": known / n if n else None,
                          "locality_coverage": len(locality) / n if n else None,
                          "conditional_coverage": len(conditional) / n if n else None,
                          "bgb_psnr_mean": mean(locality), "bgb_psnr_ci95": interval(locality),
                          "bgb_psnr_given_success": mean(conditional), "conditional_ci95": interval(conditional),
                          "uncertainty": {"method": "source_cluster_percentile", "resamples": bootstrap_samples, "seed": seed}})
    return summaries
