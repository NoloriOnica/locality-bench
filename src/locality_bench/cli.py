"""Command-line interface and portable JSON/CSV/HTML reports."""
import argparse
import csv
import html
import json
from pathlib import Path
import sys

from . import __version__
from .core import MetricConfig, PROTOCOL_VERSION, evaluate_pair
from .batch import aggregate, evaluate_manifest, validate_manifest


def write_report(rows, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    summary = aggregate(rows)
    for name, value in (("results", rows), ("summary", summary)):
        (destination / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        flat = [{k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in value]
        keys = list(dict.fromkeys(k for r in flat for k in r))
        with (destination / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(flat)
    columns = ["model_id", "n_valid_attempts", "n_locality", "n_semantic_known", "success_rate", "bgb_psnr_mean", "bgb_psnr_given_success", "conditional_coverage"]
    def cell(value):
        return html.escape("not evaluated" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value))
    body = "".join("<tr>" + "".join(f"<td>{cell(r[k])}</td>" for k in columns) + "</tr>" for r in summary)
    page = "<!doctype html><html lang='en'><meta charset='utf-8'><title>Locality benchmark</title><style>body{font:16px system-ui;margin:3rem;max-width:1200px;color:#183044}table{border-collapse:collapse}td,th{padding:.7rem;border-bottom:1px solid #ccd;text-align:left}th{background:#eef3f6}</style><h1>Locality benchmark</h1>"
    page += f"<p>Package {__version__} · Protocol {PROTOCOL_VERSION}</p><p>Preservation describes pixel similarity. Semantic success requires supplied labels. Missing labels are not failures; compare coverage alongside conditional scores.</p>"
    page += "<table><thead><tr>" + "".join(f"<th>{html.escape(k.replace('_', ' '))}</th>" for k in columns) + "</tr></thead><tbody>" + body + "</tbody></table><p>Source-cluster bootstrap intervals and full provenance are in summary.json and results.json. LPIPS is not evaluated.</p></html>"
    (destination / "report.html").write_text(page, encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate regional preservation with explicit semantic coverage")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Validate manifest schema, paths and supplied hashes")
    validate.add_argument("manifest")
    score = sub.add_parser("score", help="Score manifest and export JSON, CSV and HTML")
    score.add_argument("manifest")
    score.add_argument("--out", required=True)
    pair = sub.add_parser("pair", help="Score a source/output/mask triplet")
    for key in ("source", "output", "mask"):
        pair.add_argument(key)
    for command in (score, pair):
        command.add_argument("--band-px", type=int, default=8)
        command.add_argument("--leak-threshold", type=float, default=0.05)
    demo = sub.add_parser("demo", help="Write a synthetic example manifest and images")
    demo.add_argument("destination")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            print(f"Valid: {len(validate_manifest(args.manifest))} records")
        elif args.command == "demo":
            from .demo import create_demo
            print(create_demo(args.destination))
        elif args.command == "pair":
            print(json.dumps(evaluate_pair(args.source, args.output, args.mask, config=MetricConfig(args.band_px, args.leak_threshold)), indent=2, allow_nan=False))
        else:
            rows = evaluate_manifest(args.manifest, config=MetricConfig(args.band_px, args.leak_threshold))
            write_report(rows, args.out)
            print(f"Scored {len(rows)} records; report: {Path(args.out) / 'report.html'}")
            if any(r["scoring_status"] == "failed" for r in rows):
                return 2
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
