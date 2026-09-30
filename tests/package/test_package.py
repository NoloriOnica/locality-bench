import copy
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from skimage.metrics import structural_similarity

from locality_bench import MetricConfig, aggregate, evaluate_manifest, evaluate_pair, validate_manifest
from locality_bench.core import compute_metrics
from locality_bench.demo import create_demo
from locality_bench.http import create_app
from locality_bench.cli import main


@pytest.fixture
def demo(tmp_path):
    return create_demo(tmp_path / "data")


@pytest.mark.parametrize("band", [0, 1, 8, 12])
def test_production_parity(band):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from phase5_locality_metrics import compute_row
    rng = np.random.default_rng(40)
    orig = rng.random((48, 64, 3), dtype=np.float32)
    edit = np.clip(orig + rng.normal(0, 0.08, orig.shape), 0, 1).astype(np.float32)
    mask = np.zeros(orig.shape[:2], dtype=bool)
    mask[15:30, 18:38] = True
    _, ssim = structural_similarity(orig, edit, channel_axis=2, data_range=1, full=True)
    ssim = ssim.mean(axis=2)
    old = compute_row(orig, edit, mask, ssim, np.zeros(mask.shape), 0.05, band)
    new = compute_metrics(orig, edit, mask, MetricConfig(band), ssim_map=ssim)
    assert {key: new[key] for key in old if not key.endswith("lpips")} == {key: value for key, value in old.items() if not key.endswith("lpips")}


def test_spatial_controls(demo):
    rows = evaluate_manifest(demo)
    assert rows[0]["metrics"]["fg_mse"] == 0
    assert rows[0]["metrics"]["bgb_psnr"] == rows[1]["metrics"]["bgb_psnr"] == 100
    assert rows[1]["metrics"]["fg_mse"] > 0
    assert rows[2]["metrics"]["bgb_psnr"] < 100
    assert rows[2]["metrics"]["bgb_leak_frac"] > 0
    assert all(row["semantic_success"] is None for row in rows)


def test_empty_region_never_perfect():
    image = np.ones((16, 16, 3), dtype=np.float32) / 2
    values = compute_metrics(image, image, np.ones((16, 16), dtype=bool))
    assert values["bgb_psnr"] is None
    with pytest.raises(ValueError, match="Empty"):
        compute_metrics(image, image, np.zeros((16, 16), dtype=bool))


def test_geometry_and_black_output(demo):
    directory = demo.parent
    Image.new("RGB", (32, 16)).save(directory / "black.png")
    row = evaluate_pair(directory / "source.png", directory / "black.png", directory / "mask.png")
    assert row["generation_status"] == "degenerate"
    assert row["scoring_status"] == "ok"
    assert not row["locality_eligible"]
    assert row["geometry"]["relative_aspect_distortion"] == 1
    assert row["geometry"]["resized"]


def test_hash_and_duplicates(demo):
    rows = json.loads(demo.read_text())
    rows[0]["source_sha256"] = "0" * 64
    demo.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="mismatch"):
        validate_manifest(demo)
    rows[0].pop("source_sha256")
    rows.append(rows[0])
    demo.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="Duplicate"):
        validate_manifest(demo)


def test_failed_missing_invalid_and_conditioned_denominators(demo):
    rows = json.loads(demo.read_text())
    for row in rows:
        row["model_id"] = "same"
    rows[0].update(semantic_success=True, semantic_source="test-labels", semantic_rubric="binary-v1")
    rows[1]["generation_status"] = "refused"
    extra = dict(rows[0], sample_id="invalid", prompt_valid=False)
    rows.append(extra)
    demo.write_text(json.dumps(rows))
    scored = evaluate_manifest(demo)
    summary = aggregate(scored, bootstrap_samples=20)[0]
    assert summary["n_valid_attempts"] == 3
    assert summary["n_semantic_known"] == 2
    assert summary["success_rate"] is None
    assert summary["success_rate_lower_bound"] == pytest.approx(1 / 3)
    assert summary["conditional_coverage"] == pytest.approx(1 / 3)
    assert summary["bgb_psnr_given_success"] == 100
    assert aggregate(scored, bootstrap_samples=20) == aggregate(scored, bootstrap_samples=20)
    altered = copy.deepcopy(scored)
    altered[0]["metric_config"]["band_px"] = 4
    with pytest.raises(ValueError, match="Mixed"):
        aggregate(altered)


def test_missing_mask_fails_validation(demo):
    (demo.parent / "mask.png").unlink()
    with pytest.raises(ValueError, match="missing mask"):
        validate_manifest(demo)


def test_http_parity_and_path_boundary(demo):
    client = TestClient(create_app(demo))
    response = client.post("/pair", json={"sample_id": "spill"})
    assert response.status_code == 200
    assert response.json()["metrics"] == evaluate_manifest(demo)[2]["metrics"]
    assert "source" not in response.json()
    assert client.post("/pair", json={"sample_id": "../../private"}).status_code == 404
    assert client.post("/pair", json={"sample_id": "spill", "source": "C:/secret"}).status_code == 422
    assert client.post("/batch", json={"sample_ids": ["spill", "spill"]}).status_code == 422
    assert client.post("/batch", json={"sample_ids": ["spill"] * 33}).status_code == 422
    assert client.post("/pair", content=b" " * 17000).status_code == 413
    assert client.get("/openapi.json").status_code == 200
    assert client.post("/batch", json={"sample_ids": ["identity", "target"]}).status_code == 200


def test_cli_reports(demo, tmp_path):
    out = tmp_path / "report"
    assert main(["validate", str(demo)]) == 0
    assert main(["score", str(demo), "--out", str(out)]) == 0
    assert len(json.loads((out / "results.json").read_text())) == 3
    assert "not evaluated" in (out / "report.html").read_text(encoding="utf-8")
    assert (out / "summary.csv").is_file()


def test_unrun_is_unknown_and_mixed_rubric_api_error(demo):
    rows = json.loads(demo.read_text())
    rows[0].update(generation_status="not_run", semantic_success=False, semantic_source="test", semantic_rubric="v1")
    demo.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="unrun"):
        validate_manifest(demo)
    rows[0].update(generation_status="produced")
    rows[1].update(semantic_success=True, semantic_source="test", semantic_rubric="v2")
    demo.write_text(json.dumps(rows))
    client = TestClient(create_app(demo))
    assert client.post("/batch", json={"sample_ids": ["identity", "target"]}).status_code == 422


def test_cluster_interval_and_adapter(demo, tmp_path):
    from locality_bench.adapters import IdentityExample
    generated = IdentityExample().generate(demo.parent / "source.png", "Keep unchanged", tmp_path / "copy.png", seed=1)
    assert generated.output.read_bytes() == (demo.parent / "source.png").read_bytes()
    rows = evaluate_manifest(demo)
    for i, row in enumerate(rows):
        row.update(model_id="one", source_id=f"source-{i}")
    first = aggregate(rows, bootstrap_samples=100)
    assert first == aggregate(rows, bootstrap_samples=100)
    lo, hi = first[0]["bgb_psnr_ci95"]
    assert lo < first[0]["bgb_psnr_mean"] < hi
