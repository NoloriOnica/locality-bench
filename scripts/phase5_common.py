"""Shared helpers for Phase 5 (evaluation).

Phase 5 turns the Phase-4 edited images into locality / fidelity numbers. The two
non-obvious facts this module encodes, both surfaced by the 2026-07-15 pipeline audit:

1. **Geometry mismatch (P0).** Edited outputs come out at each model's native
   resolution (e.g. 1024x1024) which differs from the source + mask frame for ~110/140
   images per model. Every masked pixel metric requires original, edited and mask to be
   pixel-aligned, so we resample the edited image back to the *exact source H x W* before
   any metric. We never resize the source to meet the edit (that would move the mask).

2. **Masks are keyed by image_id, not the results CSV.** `phase4_generation_results.csv`
   only fills `mask_path` for the 2 inpaint models; the 5 instruction models leave it
   blank. The canonical mask for any image lives at
   `benchmark_data/masks/<image_id>.png`, so we always resolve the mask from image_id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# numpy/PIL are only needed by the image helpers (M1/M2a). Import lazily so the pure-CSV
# aggregation/plot scripts can import this module (for read_csv/write_csv/model_class) on a
# machine without numpy (e.g. the Windows dev box).
try:
    import numpy as np
    from PIL import Image
except ImportError:  # pragma: no cover
    np = None
    Image = None

from phase4_common import read_csv, write_csv  # noqa: F401  (re-exported for phase5 scripts)


# The only two mask-conditioned (inpaint) models in the accepted set. Everything else is
# instruction-only. Kept explicit so aggregates never pool the two classes together.
MASK_CONDITIONED_MODELS = {"sdxl_inpaint", "qwen_image_edit_inpaint"}


def model_class(model_id: str) -> str:
    """`mask_conditioned` (given the target mask) vs `instruction` (mask-blind)."""
    if model_id in MASK_CONDITIONED_MODELS or "inpaint" in model_id:
        return "mask_conditioned"
    return "instruction"


def row_key(row: dict[str, Any]) -> str:
    """Stable identity for a per-(model, image) metric row.

    `job_id` is the real key everywhere in this project. The fallback exists only for
    older/hand-made CSVs that predate the column.
    """
    job_id = str(row.get("job_id", "")).strip()
    if job_id:
        return job_id
    return f"{row.get('model_id', '')}__{row.get('image_id', '')}"


def merge_on_write(
    out_path: Path,
    fresh: list[dict[str, Any]],
    fieldnames: list[str],
    *,
    overwrite: bool = False,
    label: str = "rows",
) -> dict[str, int]:
    """Write `fresh` to `out_path` WITHOUT discarding rows this run did not recompute.

    WHY THIS EXISTS
    ---------------
    `phase5_locality_metrics.py` and `phase5_feature_metrics.py` used to call
    `write_csv(out_path, results, FIELDNAMES)` directly, which replaces the entire file
    with whatever the current invocation happened to compute. Combined with `--model-id`
    that is a silent data-loss trap: scoring one new model wipes every other model's rows,
    and the only way to add a model was to re-run all 980. With three more models queued
    that would have destroyed the existing metrics the first time someone scored one of
    them. `phase5_vlm_fidelity.py` already merged on `job_id`; this is the same contract,
    factored out so all three behave alike.

    CONTRACT
    --------
    * Rows are keyed by `job_id` (see `row_key`). A fresh row REPLACES the prior row with
      the same key; prior rows with no fresh counterpart are KEPT verbatim.
    * Prior order is preserved and genuinely new keys are appended, so re-running the
      exact same subset reproduces the file byte-for-byte.
    * The output can never silently shrink. If a merge would somehow yield fewer rows than
      the file already had, it aborts and tells the caller to be explicit instead.
    * Columns present in the existing file but absent from `fieldnames` abort too, rather
      than being quietly dropped — that is a schema change and deserves a human.
    * `overwrite=True` is the explicit escape hatch that replaces the file wholesale.

    Returns a small stats dict for the caller to print.
    """
    if overwrite or not out_path.exists():
        write_csv(out_path, fresh, fieldnames)
        return {"written": len(fresh), "replaced": 0, "added": len(fresh), "kept": 0}

    prior = read_csv(out_path)

    unknown = {k for row in prior for k in row} - set(fieldnames)
    if unknown:
        raise SystemExit(
            f"\nERROR: {out_path} has columns this script does not write: {sorted(unknown)}.\n"
            "Merging would drop them. Either point --output at a new file, or pass\n"
            "--overwrite if you really mean to replace the file (and its schema) wholesale.\n"
        )

    fresh_by_key = {row_key(r): r for r in fresh}
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    replaced = 0
    duplicates = 0

    for r in prior:                      # prior order preserved
        key = row_key(r)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        if key in fresh_by_key:
            merged.append(fresh_by_key[key])
            replaced += 1
        else:
            merged.append(r)

    added = 0
    for r in fresh:                      # genuinely new keys appended in computed order
        key = row_key(r)
        if key not in seen:
            seen.add(key)
            merged.append(r)
            added += 1

    if len(merged) < len(prior):
        raise SystemExit(
            f"\nERROR: refusing to shrink {out_path} ({len(prior)} -> {len(merged)} {label}).\n"
            f"{duplicates} duplicate job_id(s) in the existing file would be collapsed.\n"
            "That is a repair, not a metrics update, so it needs to be deliberate:\n"
            "re-run with --overwrite once you have checked which rows are duplicated.\n"
        )

    write_csv(out_path, merged, fieldnames)
    return {
        "written": len(merged),
        "replaced": replaced,
        "added": added,
        "kept": len(merged) - replaced - added,
    }


def resolve_paths(
    row: dict[str, str], root: Path, mask_dir: Path | str | None = None
) -> dict[str, Path]:
    """Original image, edited output, and canonical mask for one results row.

    `mask_path` in the CSV is unreliable (blank for instruction models), so the mask is
    taken from a directory keyed on `image_id` rather than from the row.

    `mask_dir` defaults to `benchmark_data/masks`, which keeps every existing caller
    byte-identical. It exists so an external benchmark can be scored by this same code
    without editing it: the transfer study (configs/transfer_giebench_v1.json) supplies its
    own mask directory, and applying the protocol *unchanged* to external data is the whole
    point of that study.
    """
    image_id = row["image_id"]
    masks = Path(mask_dir) if mask_dir else root / "benchmark_data" / "masks"
    return {
        "original": root / row["benchmark_image_path"],
        "edited": root / row["edited_output_path"],
        "mask": masks / f"{image_id}.png",
    }


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """Load an image as float32 RGB in [0, 1]. Optionally resample to `size` (w, h).

    Uses LANCZOS for high-quality resampling — this is the geometry-normalization step
    for the edited image.
    """
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    """Load a mask as a boolean array at `size` (w, h). True = target object (inside).

    NEAREST resampling so the mask stays binary. Any non-zero pixel is treated as object.
    """
    mask = Image.open(path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    return np.asarray(mask, dtype=np.uint8) > 127


def mask_bbox(mask: np.ndarray, pad_frac: float = 0.10) -> tuple[int, int, int, int]:
    """Bounding box (x0, y0, x1, y1) of the object mask, padded by `pad_frac` of its size.

    Returns the full-image box if the mask is empty. Used to crop the object region for
    in-region CLIP edit-success and object-level DINOv2 identity.
    """
    h, w = mask.shape
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0, 0, w, h
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad_x = int((x1 - x0) * pad_frac)
    pad_y = int((y1 - y0) * pad_frac)
    x0 = max(0, x0 - pad_x); y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x); y1 = min(h, y1 + pad_y)
    return x0, y0, x1, y1


def normalize_edited(edited_path: Path, original_size: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (edited RGB resampled to original_size, provenance flags).

    `original_size` is (w, h) of the source image. Provenance records the model's native
    output size and whether resampling / an aspect-ratio change occurred, so the write-up
    can flag which comparisons were geometry-normalized.

    ⚠️ `aspect_changed` is DIAGNOSTIC ONLY — it is **not** the criterion behind any reported
    number. It uses a 1e-3 *absolute* AR difference, which is so sensitive that it flags
    visually irrelevant rounding. The "clean subset" that feeds every published locality
    figure is computed independently in `phase5_aggregate_all.py` as a 0.02 *relative*
    distortion (|native_AR / source_AR - 1|), recomputed from the raw w/h columns and
    ignoring this flag entirely. Do not filter on `aspect_changed`, and do not quote it as
    "the number of warped images" — the two criteria disagree by design. Retire the column
    on the next M1 re-run (it cannot be dropped without regenerating metrics_by_image.csv).
    """
    img = Image.open(edited_path).convert("RGB")
    native_w, native_h = img.size
    orig_w, orig_h = original_size
    resized = (native_w, native_h) != (orig_w, orig_h)
    native_ar = native_w / native_h if native_h else 0.0
    orig_ar = orig_w / orig_h if orig_h else 0.0
    aspect_changed = resized and abs(native_ar - orig_ar) > 1e-3
    if resized:
        img = img.resize(original_size, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    flags = {
        "native_w": native_w,
        "native_h": native_h,
        "orig_w": orig_w,
        "orig_h": orig_h,
        "resized": int(resized),
        "aspect_changed": int(aspect_changed),
    }
    return arr, flags
