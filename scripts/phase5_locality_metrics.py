"""Phase 5 · Milestone 1 — locality / background-preservation metrics.

For every Phase-4 result (model x image) this computes how well the edit stayed *local*:
how much the **background** (everything outside the target mask) was preserved, plus how
much actually changed **inside** the mask (a crude did-an-edit-happen signal). This is the
thesis core — "did only the intended object change?"

Per (model, image) it writes one row to `evaluations/metrics_by_image.csv` with:

  provenance : native_w/h, orig_w/h, resized, aspect_changed, mask_area_ratio
  background (locality, computed over pixels OUTSIDE the mask; higher PSNR/SSIM = better
             preservation, lower MSE/LPIPS/leakage = better):
      bg_mse, bg_psnr, bg_ssim, bg_lpips, bg_leak_frac
  background, boundary-robust (same, but excluding a dilation band around the object so
             boundary-blending isn't scored as leakage):
      bgb_mse, bgb_psnr, bgb_ssim, bgb_lpips, bgb_leak_frac
  foreground (inside the mask; change magnitude — higher = more edit applied):
      fg_mse, fg_ssim

Definitions
-----------
Images are float RGB in [0, 1] at the source resolution (edited is LANCZOS-resampled to it).
* MSE      : mean of (orig - edit)^2 over the selected pixels and all 3 channels.
* PSNR     : 10 * log10(1 / MSE)  (MAX = 1.0); capped at 100 dB when MSE -> 0.
* SSIM     : per-pixel SSIM map (skimage, data_range=1) averaged over the selected pixels.
* LPIPS    : per-pixel LPIPS map (lpips, net=alex, spatial=True) averaged over the pixels.
* leak_frac: fraction of selected (background) pixels whose per-pixel RMS change exceeds
             --leak-threshold — an interpretable "how much of the background visibly moved".

Run on the GPU server (needs torch + lpips). See requirements-phase5.txt.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
from skimage.metrics import structural_similarity

from phase5_common import (
    load_mask,
    load_rgb,
    merge_on_write,
    model_class,
    normalize_edited,
    read_csv,
    resolve_paths,
)

FIELDNAMES = [
    "job_id", "model_id", "model_class", "image_id", "benchmark_category",
    "main_object", "edit_type", "prompt_source",
    "native_w", "native_h", "orig_w", "orig_h", "resized", "aspect_changed",
    "mask_area_ratio",
    "bg_mse", "bg_psnr", "bg_ssim", "bg_lpips", "bg_leak_frac",
    "bgb_mse", "bgb_psnr", "bgb_ssim", "bgb_lpips", "bgb_leak_frac",
    "fg_mse", "fg_ssim",
    "status", "error",
    # The metric configuration this row was computed under, so an analysis can refuse to
    # pool rows scored with different settings (repair plan A8). Rows written before
    # 2026-09-28 lack these columns; they are "unrecorded", not assumed to match.
    "band_px", "leak_threshold", "lpips_net",
]


def _psnr(mse: float) -> float:
    if mse <= 1e-12:
        return 100.0
    return min(100.0, 10.0 * math.log10(1.0 / mse))


def _masked_mean(per_pixel: np.ndarray, sel: np.ndarray) -> float:
    """Mean of an H x W per-pixel map over boolean selection `sel`."""
    if not sel.any():
        return float("nan")
    return float(per_pixel[sel].mean())


def _rms_change(orig: np.ndarray, edit: np.ndarray) -> np.ndarray:
    """Per-pixel RMS change across channels -> H x W in [0, 1]."""
    return np.sqrt(((orig - edit) ** 2).mean(axis=2))


def compute_row(
    orig: np.ndarray,
    edit: np.ndarray,
    mask: np.ndarray,
    ssim_map: np.ndarray,
    lpips_map: np.ndarray,
    leak_threshold: float,
    band_px: int,
) -> dict[str, Any]:
    from skimage.morphology import dilation, disk

    inside = mask
    background = ~mask
    # boundary-robust background: exclude a dilation band around the object
    if band_px > 0:
        protected = ~dilation(mask, disk(band_px)).astype(bool)
    else:
        protected = background

    sq = (orig - edit) ** 2            # H x W x 3
    se_map = sq.mean(axis=2)           # H x W  per-pixel MSE (across channels)
    rms = _rms_change(orig, edit)      # H x W

    def bg_block(sel: np.ndarray, prefix: str) -> dict[str, Any]:
        mse = _masked_mean(se_map, sel)
        leak = float((rms[sel] > leak_threshold).mean()) if sel.any() else float("nan")
        return {
            f"{prefix}_mse": round(mse, 8),
            f"{prefix}_psnr": round(_psnr(mse), 4),
            f"{prefix}_ssim": round(_masked_mean(ssim_map, sel), 6),
            f"{prefix}_lpips": round(_masked_mean(lpips_map, sel), 6),
            f"{prefix}_leak_frac": round(leak, 6),
        }

    out: dict[str, Any] = {}
    out.update(bg_block(background, "bg"))
    out.update(bg_block(protected, "bgb"))
    out["fg_mse"] = round(_masked_mean(se_map, inside), 8)
    out["fg_ssim"] = round(_masked_mean(ssim_map, inside), 6)
    out["mask_area_ratio"] = round(float(inside.mean()), 6)
    return out


class LpipsScorer:
    """Lazy LPIPS(net, spatial=True) wrapper returning an H x W per-pixel map."""

    def __init__(self, net: str, device: str) -> None:
        import lpips
        import torch

        self.torch = torch
        self.device = device
        self.model = lpips.LPIPS(net=net, spatial=True).to(device)
        self.model.eval()

    def map(self, orig: np.ndarray, edit: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            def to_t(a: np.ndarray):
                t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(self.device)
                return t * 2.0 - 1.0  # [0,1] -> [-1,1]
            d = self.model(to_t(orig), to_t(edit))  # (1, 1, H, W)
        return d[0, 0].detach().cpu().numpy().astype(np.float32)


def run(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    rows = read_csv(root / args.results)
    if args.model_id:
        rows = [r for r in rows if r["model_id"] in set(args.model_id)]
    if args.limit:
        rows = rows[: args.limit]

    scorer = LpipsScorer(args.lpips_net, args.device)
    results: list[dict[str, Any]] = []
    n = len(rows)
    for i, row in enumerate(rows, 1):
        rec: dict[str, Any] = {
            "job_id": row.get("job_id", ""),
            "model_id": row["model_id"],
            "model_class": model_class(row["model_id"]),
            "image_id": row["image_id"],
            "benchmark_category": row.get("benchmark_category", ""),
            "main_object": row.get("main_object", ""),
            "edit_type": row.get("edit_type", ""),
            "prompt_source": row.get("prompt_source", ""),
            "status": "ok",
            "error": "",
            "band_px": args.band_px,
            "leak_threshold": args.leak_threshold,
            "lpips_net": args.lpips_net,
        }
        try:
            paths = resolve_paths(row, root, args.mask_dir)
            orig = load_rgb(paths["original"])
            orig_size = (orig.shape[1], orig.shape[0])  # (w, h)
            edit, flags = normalize_edited(paths["edited"], orig_size)
            mask = load_mask(paths["mask"], orig_size)
            rec.update(flags)

            ssim_mean, ssim_full = structural_similarity(
                orig, edit, channel_axis=2, data_range=1.0, full=True
            )
            ssim_map = ssim_full.mean(axis=2)  # H x W
            lpips_map = scorer.map(orig, edit)  # H x W

            rec.update(
                compute_row(orig, edit, mask, ssim_map, lpips_map, args.leak_threshold, args.band_px)
            )
        except Exception as exc:  # keep the batch alive; log the failure per row
            rec["status"] = "failed"
            rec["error"] = f"{type(exc).__name__}: {exc}"

        results.append(rec)
        if i % 20 == 0 or i == n:
            print(f"  [{i}/{n}] {row['model_id']}/{row['image_id']} status={rec['status']}", flush=True)

    out_path = root / args.output
    # Merge, don't clobber: a --model-id run must not wipe every other model's rows.
    stats = merge_on_write(out_path, results, FIELDNAMES, overwrite=args.overwrite)
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nWrote {out_path}  ({ok}/{len(results)} ok this run)")
    print(f"  {stats['written']} rows total "
          f"({stats['replaced']} replaced, {stats['added']} added, {stats['kept']} kept from before)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5 M1: masked-background locality metrics.")
    p.add_argument("--root", default=".")
    p.add_argument("--mask-dir", default="",
                   help="directory of <image_id>.png masks; defaults to "
                        "benchmark_data/masks. Used by the external transfer study "
                        "so the protocol can score other data unchanged.")
    p.add_argument("--results", default="manifests/phase4_generation_results.csv")
    p.add_argument("--output", default="evaluations/metrics_by_image.csv")
    p.add_argument("--model-id", nargs="*", help="restrict to these model ids")
    p.add_argument("--limit", type=int, default=0, help="only the first N rows (debug)")
    p.add_argument("--device", default="cuda", help="cuda | cpu")
    p.add_argument("--lpips-net", default="alex", help="alex | vgg | squeeze")
    p.add_argument("--band-px", type=int, default=8,
                   help="dilation band (px) excluded from the boundary-robust background")
    p.add_argument("--leak-threshold", type=float, default=0.05,
                   help="per-pixel RMS change above which a background pixel counts as leaked")
    p.add_argument("--overwrite", action="store_true",
                   help="replace the output file wholesale instead of merging into it. "
                        "Without this, rows for job_ids not computed in this run are kept.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
