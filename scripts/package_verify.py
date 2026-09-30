"""Record production-disk synthetic diagnostics without changing frozen artifacts."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from locality_bench.core import compute_metrics, MetricConfig, PROTOCOL_VERSION
from repair_stress_suite import build_cases


def main():
    destination = ROOT / "evaluations/package_v0_1"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in build_cases(48):
        values = compute_metrics(case["orig"], case["edited"], case["mask"], MetricConfig())
        rows.append(dict(case=case["case"], protocol_version=PROTOCOL_VERSION, **values))
    with (destination / "synthetic_disk.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by = {r["case"]: r for r in rows}
    assert by["identity"]["bg_mse"] == by["exact_target"]["bg_mse"] == 0
    assert by["wrong_region"]["bg_mse"] > 0
    assert by["black_frame"]["bg_mse"] > 0
    (destination / "verification.json").write_text(json.dumps({"protocol": PROTOCOL_VERSION, "cases": len(rows),
        "settings": MetricConfig().metadata(), "assertions": "identity/target preserve background; wrong-region and black-frame change it",
        "scope": "Production disk diagnostics, separate from historical Manhattan-dilation suite. No semantic validation."}, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(rows)} synthetic disk cases: {destination}")


if __name__ == "__main__":
    main()
