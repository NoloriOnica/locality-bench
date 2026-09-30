from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_registry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    models = registry.get("models", [])
    registry["models_by_id"] = {model["model_id"]: model for model in models}
    return registry


# --- Degenerate-output detection ------------------------------------------------
#
# WHY THIS EXISTS
# ---------------
# Six of the 140 `instruct_pix2pix` outputs are entirely black (every pixel zero) and
# Phase 4 recorded all six as `status=success`, because the adapter only ever asked
# "did the pipeline hand back an image?", never "is there anything in it?". They went
# undetected for six weeks. Five of the six are faces
# (celeb_face_008 / _021 / _053 / _191 / celeb_face_clothing_069, plus
# coco_object_dining_table_013), which is the signature of a safety filter blanking
# the frame rather than a decode fault.
#
# THE STATISTIC
# -------------
# The test is the MAXIMUM PER-CHANNEL standard deviation, not the standard deviation
# over the pooled RGB array. A frame that is a single flat colour — say pure blue,
# (0, 0, 255) everywhere — has a large *pooled* std purely because the channel means
# differ, yet it carries no image content at all. Taking the max over the three
# per-channel stds asks the right question: is EVERY channel flat? One flat channel in
# an otherwise varied frame (a saturated sky, a blown-out highlight) is normal and is
# correctly left alone.
#
# THE THRESHOLD
# -------------
# Chosen empirically against the frozen 980-output matrix rather than guessed:
#
#   * the six known-bad frames    -> max per-channel std == 0.0 exactly
#   * the lowest LEGITIMATE output -> 31.92 (sdxl_inpaint__coco_animal_bird_009, a bird
#                                    on a near-uniform pale sky — the hardest real case)
#   * 1st percentile of the legitimate 974 -> 36.70
#
# So the observed gap between "degenerate" and "real, but very low contrast" spans more
# than thirty standard-deviation units with nothing whatsoever inside it. DEGENERATE_STD
# is set at 1.0: ~32x below the lowest real output, which leaves no plausible route to a
# false positive, while still catching a frame that is uniform-plus-dither rather than
# exactly constant (mid-grey fills, faint-noise blanks) that an `== 0` test would miss.
# Verified: at 1.0 the sweep flags exactly those six of 980 and nothing else.
DEGENERATE_STD = 1.0


def check_degenerate(path: Path, std_threshold: float = DEGENERATE_STD) -> tuple[str, str]:
    """Test one rendered frame for degenerate (contentless) output.

    Returns ``(reason, detail)``. ``reason`` is the empty string when the frame looks
    like real content; otherwise it is one of ``all_black``, ``all_white``,
    ``near_uniform`` or ``unreadable``, and ``detail`` carries the measured numbers so a
    reviewer can second-guess the threshold without re-opening the image.

    numpy and PIL are imported lazily so that merely importing ``phase4_common`` keeps
    working in the lighter environments used by the job-preparation scripts.
    """
    import numpy as np
    from PIL import Image

    try:
        with Image.open(path) as handle:
            arr = np.asarray(handle.convert("RGB"), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - an unreadable frame is a finding, not a crash.
        return "unreadable", f"{type(exc).__name__}: {exc}"

    if arr.size == 0:
        return "unreadable", "zero-sized image"

    lo = float(arr.min())
    hi = float(arr.max())
    std_ch_max = float(max(arr[..., c].std() for c in range(arr.shape[2])))
    detail = f"min={lo:.0f} max={hi:.0f} std_ch_max={std_ch_max:.4f}"

    # Ordered most-specific first, so the reason names the actual failure mode.
    if hi == 0.0:
        return "all_black", detail
    if lo == 255.0:
        return "all_white", detail
    if std_ch_max < std_threshold:
        return "near_uniform", detail
    return "", detail


# --- Resolution policy ------------------------------------------------------------
#
# WHY THIS EXISTS
# ---------------
# The registry has always DECLARED `resolution_policy`, but no code read it, so every
# model generated at whatever size its pipeline defaulted to. The cost is visible in the
# results: `sdxl_inpaint` emits 1024x1024 for every input regardless of the source aspect,
# which distorts its background under comparison and collapsed its usable clean-locality
# subset to 21 of 119 samples. FLUX Kontext would do exactly the same thing by default.
#
# OPT-IN, AND OFF BY DEFAULT — THIS IS LOAD-BEARING
# --------------------------------------------------
# The seven models in the frozen 980-edit matrix must keep generating byte-identically.
# So the policy is read from the PER-MODEL entry, the fallback is `native`, and `native`
# returns an empty dict — the caller's kwargs are then untouched, not merely equivalent.
# The registry's `default_generation.resolution_policy` is the fallback for models that do
# not name one, and it is set to `native` precisely so that adding this feature cannot
# retroactively change a model that already ran. Turning it on is a per-model decision.
#
# POLICIES
# --------
#   native         (default) do nothing. The pipeline picks its own size, as before.
#   source_aspect  generate at the source image's aspect ratio, at approximately
#                  `resolution_target_area` pixels, with both sides snapped to a multiple
#                  of `resolution_multiple` (VAE/patch stride). Returns height + width, and
#                  `max_area` as well when the model declares the pipeline accepts it
#                  (FLUX Kontext takes `max_area` and does its own internal rounding).
#
# Per-model keys (all optional; only `resolution_policy` switches behaviour on):
#   "resolution_policy": "source_aspect"
#   "resolution_target_area": 1048576      # default: 1024*1024, i.e. SDXL's native budget
#   "resolution_multiple": 16              # default: 16, safe for SD/SDXL (8) and DiT (16)
#   "resolution_supports_max_area": true   # also pass max_area=<target area>
RESOLUTION_POLICIES = ("native", "source_aspect")
DEFAULT_TARGET_AREA = 1024 * 1024
DEFAULT_MULTIPLE = 16
MIN_SIDE = 256


def _snap(value: float, multiple: int) -> int:
    """Round to the nearest positive multiple, never below MIN_SIDE."""
    snapped = int(round(value / multiple)) * multiple
    return max(snapped, ((MIN_SIDE + multiple - 1) // multiple) * multiple)


def resolution_kwargs(
    model: dict[str, Any],
    source_size: tuple[int, int],
    default_policy: str = "native",
) -> dict[str, Any]:
    """Extra pipeline kwargs implementing the model's resolution policy.

    ``source_size`` is ``(width, height)``, matching ``PIL.Image.size``.

    Returns ``{}`` for the default `native` policy, so a caller doing
    ``kwargs.update(resolution_kwargs(...))`` is a strict no-op for every model that has
    not opted in. That is what keeps the frozen seven byte-identical.
    """
    policy = str(model.get("resolution_policy") or default_policy or "native").strip()
    if policy == "native":
        return {}
    if policy not in RESOLUTION_POLICIES:
        raise ValueError(
            f"Unknown resolution_policy {policy!r} for model "
            f"{model.get('model_id', '<unknown>')}; expected one of {RESOLUTION_POLICIES}."
        )

    width, height = source_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Bad source size {source_size!r}")

    area = int(model.get("resolution_target_area") or DEFAULT_TARGET_AREA)
    multiple = int(model.get("resolution_multiple") or DEFAULT_MULTIPLE)
    aspect = width / height

    # Solve W*H = area and W/H = aspect, then snap both sides to the stride.
    target_h = (area / aspect) ** 0.5
    target_w = target_h * aspect
    out: dict[str, Any] = {
        "height": _snap(target_h, multiple),
        "width": _snap(target_w, multiple),
    }
    if bool_value(model.get("resolution_supports_max_area", False)):
        out["max_area"] = area
    return out


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalized_model_ids(registry: dict[str, Any], requested: list[str] | None, include_optional_paid: bool) -> list[str]:
    if requested:
        return requested
    if include_optional_paid:
        return [model["model_id"] for model in registry["models"]]
    return list(registry.get("core_model_ids", []))
