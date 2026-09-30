"""Production-compatible pixel metrics; missing regions are explicitly unavailable.

The arithmetic follows scripts/phase5_locality_metrics.py, frozen 2026-09-30.
The original runner remains untouched while the M01 workstream uses it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from skimage.morphology import dilation, disk

PROTOCOL_VERSION = "locality_cpu_v1"
SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class MetricConfig:
    band_px: int = 8
    leak_threshold: float = 0.05

    def __post_init__(self):
        if type(self.band_px) is not int or not 0 <= self.band_px <= 128:
            raise ValueError("band_px must be an integer between 0 and 128")
        if isinstance(self.leak_threshold, bool) or not math.isfinite(self.leak_threshold) or not 0 <= self.leak_threshold <= 1:
            raise ValueError("leak_threshold must be finite and between 0 and 1")

    def metadata(self):
        return dict(asdict(self), dilation="euclidean_disk", mask_threshold=127,
                    mask_resample="NEAREST", image_resample="LANCZOS",
                    psnr_cap_db=100, lpips="not_evaluated", ssim="skimage_default_7px")


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash_stream(stream)


def _hash_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def compute_metrics(orig, edit, mask, config=MetricConfig(), *, ssim_map=None):
    """Score aligned float32 RGB arrays in [0,1] and a boolean target mask."""
    orig, edit = np.asarray(orig, dtype=np.float32), np.asarray(edit, dtype=np.float32)
    mask = np.asarray(mask)
    if orig.ndim != 3 or orig.shape[-1] != 3 or edit.shape != orig.shape or mask.shape != orig.shape[:2]:
        raise ValueError("Expected equal HxWx3 images and an HxW mask")
    if mask.dtype != bool or min(orig.shape[:2]) < 7:
        raise ValueError("Mask must be boolean and image dimensions at least 7 pixels")
    if not np.isfinite(orig).all() or not np.isfinite(edit).all() or min(orig.min(), edit.min()) < 0 or max(orig.max(), edit.max()) > 1:
        raise ValueError("Images must be finite RGB values in [0,1]")
    if not mask.any():
        raise ValueError("Empty target mask")
    if ssim_map is None:
        _, full = structural_similarity(orig, edit, channel_axis=2, data_range=1.0, full=True)
        ssim_map = full.mean(axis=2)
    if ssim_map.shape != mask.shape or not np.isfinite(ssim_map).all():
        raise ValueError("Invalid SSIM map")
    se = ((orig - edit) ** 2).mean(axis=2)
    rms = np.sqrt(se)
    protected = ~dilation(mask, disk(config.band_px)).astype(bool) if config.band_px else ~mask
    result = {"mask_area_ratio": round(float(mask.mean()), 6), "fg_mse": round(float(se[mask].mean()), 8),
              "fg_ssim": round(float(ssim_map[mask].mean()), 6)}
    for prefix, selection in (("bg", ~mask), ("bgb", protected)):
        result[f"{prefix}_pixels"] = int(selection.sum())
        if not selection.any():
            result.update({f"{prefix}_{key}": None for key in ("mse", "psnr", "ssim", "leak_frac")})
            continue
        mse = float(se[selection].mean())
        result.update({f"{prefix}_mse": round(mse, 8),
                       f"{prefix}_psnr": round(100.0 if mse <= 1e-12 else min(100.0, 10 * math.log10(1 / mse)), 4),
                       f"{prefix}_ssim": round(float(ssim_map[selection].mean()), 6),
                       f"{prefix}_leak_frac": round(float((rms[selection] > config.leak_threshold).mean()), 6)})
    return result


def evaluate_pair(source, edited, mask, *, config=MetricConfig()):
    """Evaluate local image paths. Semantic success requires separate evidence."""
    from . import __version__
    paths = {"source": Path(source), "output": Path(edited), "mask": Path(mask)}
    with Image.open(paths["source"]) as im:
        orig_im = im.convert("RGB")
    with Image.open(paths["output"]) as im:
        edit_im = im.convert("RGB")
    with Image.open(paths["mask"]) as im:
        mask_im = im.convert("L")
    w, h = orig_im.size
    nw, nh = edit_im.size
    native_black = edit_im.getbbox() is None
    geometry = {"orig_w": w, "orig_h": h, "native_w": nw, "native_h": nh,
                "resized": edit_im.size != orig_im.size, "mask_resized": mask_im.size != orig_im.size,
                "relative_aspect_distortion": abs((nw / nh) / (w / h) - 1)}
    orig = np.asarray(orig_im, dtype=np.float32) / 255.0
    edit = np.asarray(edit_im.resize((w, h), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    inside = np.asarray(mask_im.resize((w, h), Image.Resampling.NEAREST), dtype=np.uint8) > 127
    values = compute_metrics(orig, edit, inside, config)
    reasons = []
    if native_black:
        reasons.append("all_black_output")
    if values["bgb_pixels"] == 0:
        reasons.append("no_protected_background")
    return {"schema_version": SCHEMA_VERSION, "package_version": __version__,
            "protocol_version": PROTOCOL_VERSION, "metric_config": config.metadata(),
            "hashes": {key: sha256(path) for key, path in paths.items()},
            "geometry": geometry, "metrics": values, "scoring_status": "ok",
            "generation_status": "degenerate" if native_black else "produced",
            "locality_eligible": not reasons, "eligibility_reasons": reasons,
            "semantic_status": "not_evaluated", "semantic_success": None}
