#!/usr/bin/env python3
"""B1 -- controlled evaluator stress suite with known ground truth.

    python scripts/repair_stress_suite.py

Every case is a synthetic (original, edited, mask) triple built by a known pixel
transformation, so the correct answer is arithmetic rather than opinion. The expected
property of each case is declared in `CASES` **before** any metric runs, and the suite
reports where the existing locality metrics agree with it and where they do not.

What this can and cannot establish
----------------------------------
It establishes exact spatial facts: whether a metric's "background" error responds only to
change outside the declared region, whether a boundary band suppresses a spill of a known
width, whether a leak fraction tracks a known leaked area. Those are pixel guarantees.

It establishes nothing about semantic alignment with human preference. A synthetic edit is
not a natural one, and no judge is run here. Conclusions about `edit_success` remain
provisional and are labelled so.

The honesty rule that shapes the design
---------------------------------------
Several cases are true *by construction*: a hard composite has zero outside-mask error
because it was built by pasting. A protocol that "detects" this has not been validated -- it
has restated its own construction. So each expectation is marked `by_construction` or
`held_out`, and only the held-out ones are evidence that a metric works. Held-out severities
(spill widths and leak magnitudes the band/threshold were not tuned to) carry the weight.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "evaluations", "repaired_v1", "stress_suite")

RNG_SEED = 20260919
SIZE = (256, 256)          # (h, w)

# Metric configuration under test -- the same values the real protocol declares.
BAND_PX = 8
LEAK_THRESHOLD = 0.05


# ── synthetic scene ───────────────────────────────────────────────────────────────────
def make_scene(seed):
    """A textured image. Flat images make every metric look perfect, which proves nothing."""
    rng = np.random.default_rng(seed)
    h, w = SIZE
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (
        0.45
        + 0.18 * np.sin(xx / 13.0)
        + 0.14 * np.cos(yy / 9.0)
        + 0.10 * np.sin((xx + yy) / 21.0)
    )
    img = np.stack([base, base * 0.92 + 0.05, base * 0.86 + 0.09], axis=-1)
    img = img + rng.normal(0, 0.02, img.shape).astype(np.float32)   # fine grain
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def disc_mask(cy, cx, radius):
    h, w = SIZE
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2


def dilate(mask, px):
    """Square-structuring-element dilation. Pure numpy: no skimage on this machine."""
    if px <= 0:
        return mask.copy()
    out = mask.copy()
    for _ in range(px):
        p = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
        )
    return out


# ── metrics under test ────────────────────────────────────────────────────────────────
def psnr(mse):
    return float("inf") if mse <= 0 else float(10.0 * np.log10(1.0 / mse))


def measure(orig, edit, mask, band_px=BAND_PX, leak_threshold=LEAK_THRESHOLD):
    """The locality family exactly as phase5_locality_metrics computes it."""
    se = ((orig - edit) ** 2).mean(axis=2)
    rms = np.sqrt(se)
    inside = mask
    background = ~mask
    protected = ~dilate(mask, band_px)

    def block(sel):
        if not sel.any():
            return {"mse": float("nan"), "psnr": float("nan"), "leak_frac": float("nan")}
        mse = float(se[sel].mean())
        return {"mse": mse, "psnr": psnr(mse),
                "leak_frac": float((rms[sel] > leak_threshold).mean())}

    bg, bgb = block(background), block(protected)
    return {
        "total_se": float(se.sum()),
        "bg_mse": bg["mse"], "bg_psnr": bg["psnr"], "bg_leak_frac": bg["leak_frac"],
        "bgb_mse": bgb["mse"], "bgb_psnr": bgb["psnr"], "bgb_leak_frac": bgb["leak_frac"],
        "fg_mse": float(se[inside].mean()) if inside.any() else float("nan"),
        "mask_area_ratio": float(inside.mean()),
    }


# ── cases: expectations declared BEFORE measurement ───────────────────────────────────
def build_cases(mask_radius):
    """-> list of dicts with original, edited, mask, and the property each must satisfy."""
    h, w = SIZE
    orig = make_scene(RNG_SEED)
    mask = disc_mask(h // 2, w // 2, mask_radius)
    # An equal-area region well away from the mask, for the location test.
    other = disc_mask(h // 5, w // 5, mask_radius)
    rng = np.random.default_rng(RNG_SEED + 1)
    patch = np.clip(orig + rng.normal(0, 0.25, orig.shape).astype(np.float32), 0, 1)

    cases = []

    def add(name, edited, expect, evidence, note):
        cases.append({"case": name, "orig": orig, "edited": edited, "mask": mask,
                      "expect": expect, "evidence": evidence, "note": note})

    # 1. identity copy
    add("identity", orig.copy(),
        {"bg_mse": ("==", 0.0), "fg_mse": ("==", 0.0), "bg_leak_frac": ("==", 0.0)},
        "by_construction",
        "Zero pixel change. It also cannot satisfy any nontrivial edit request, which the "
        "locality family alone cannot detect -- that needs a success measure.")

    # 2. exact target-region transformation
    e = orig.copy()
    e[mask] = patch[mask]
    add("exact_target", e,
        {"bg_mse": ("==", 0.0), "fg_mse": (">", 0.0)},
        "by_construction",
        "Change restricted to the declared region.")

    # 3. equivalent WRONG-region transformation, matched in magnitude
    e = orig.copy()
    e[other] = patch[other]
    add("wrong_region", e,
        {"fg_mse": ("==", 0.0), "bg_mse": (">", 0.0)},
        "held_out",
        "Same transformation, same area, wrong place. The magnitude is matched to "
        "`exact_target`, so any metric that scores them alike is location-blind.")

    # 4. boundary spill at several widths
    for width in (2, 4, 8, 16, 32):
        ring = dilate(mask, width) & ~mask
        e = orig.copy()
        e[mask] = patch[mask]
        e[ring] = patch[ring]
        add(f"boundary_spill_{width}px", e,
            {"bg_mse": (">", 0.0),
             # The 8px band is the protocol's declared value. Spills at or below it should be
             # suppressed by bgb; spills wider than it must NOT be.
             "bgb_mse": ("==", 0.0) if width <= BAND_PX else (">", 0.0)},
            "by_construction" if width in (BAND_PX,) else "held_out",
            f"Spill of exactly {width}px outside the region, against an {BAND_PX}px band.")

    # 5. global photometric change
    add("global_brightness", np.clip(orig + 0.08, 0, 1),
        {"bg_mse": (">", 0.0), "bg_leak_frac": (">", 0.9)},
        "held_out",
        "A whole-frame photometric shift. Large background error here is NOT evidence that "
        "the model regenerated content; it is evidence that the frame moved.")

    # 6. black / constant frame
    add("black_frame", np.zeros_like(orig),
        {"bg_mse": (">", 0.05), "fg_mse": (">", 0.05)},
        "held_out",
        "An invalid output. It must score badly, and must never be silently dropped: under "
        "the repaired protocol it is a generation failure that is still scorable.")

    add("constant_grey", np.full_like(orig, 0.5),
        {"bg_mse": (">", 0.0)},
        "held_out",
        "Constant but not black. A detector keyed only on max==0 misses this.")

    # 7. hard pixel composite
    e = patch.copy()
    e[~mask] = orig[~mask]
    add("hard_composite", e,
        {"bg_mse": ("==", 0.0), "bgb_mse": ("==", 0.0)},
        "by_construction",
        "Outside-mask error is zero because it was pasted. A protocol that reports this as "
        "excellent preservation has measured its own construction, not the editor.")

    return cases


def satisfied(value, op, target, tol=1e-12):
    if value != value:                      # NaN
        return False
    if op == "==":
        return abs(value - target) <= tol
    if op == ">":
        return value > target + tol
    if op == "<":
        return value < target - tol
    raise ValueError(op)


# ── sensitivity sweeps ────────────────────────────────────────────────────────────────
def sweep_rows(mask_radius):
    """Vary mask size, band width and leak threshold; record how the verdicts move."""
    rows = []
    for radius in (16, 32, 64, 96):
        cases = build_cases(radius)
        for band in (0, 4, 8, 16):
            for thr in (0.01, 0.05, 0.10):
                for c in cases:
                    m = measure(c["orig"], c["edited"], c["mask"], band, thr)
                    rows.append({
                        "case": c["case"], "mask_radius": radius,
                        "mask_area_pct": round(100 * m["mask_area_ratio"], 3),
                        "band_px": band, "leak_threshold": thr,
                        "bg_mse": round(m["bg_mse"], 10),
                        "bgb_mse": round(m["bgb_mse"], 10),
                        "bg_psnr": round(m["bg_psnr"], 4) if m["bg_psnr"] != float("inf") else "inf",
                        "bgb_psnr": round(m["bgb_psnr"], 4) if m["bgb_psnr"] != float("inf") else "inf",
                        "bg_leak_frac": round(m["bg_leak_frac"], 6),
                        "fg_mse": round(m["fg_mse"], 10),
                    })
    return rows


def main(args):
    os.makedirs(OUT, exist_ok=True)
    cases = build_cases(args.radius)

    results, failures = [], []
    for c in cases:
        m = measure(c["orig"], c["edited"], c["mask"])
        verdicts = {}
        for metric, (op, target) in c["expect"].items():
            ok = satisfied(m[metric], op, target)
            verdicts[metric] = ok
            if not ok:
                failures.append((c["case"], metric, op, target, m[metric]))
        results.append({
            "case": c["case"],
            "evidence": c["evidence"],
            "expectation": "; ".join(f"{k} {v[0]} {v[1]}" for k, v in c["expect"].items()),
            "all_expectations_met": int(all(verdicts.values())),
            "mask_area_pct": round(100 * m["mask_area_ratio"], 3),
            "bg_mse": round(m["bg_mse"], 10),
            "bg_psnr": "inf" if m["bg_psnr"] == float("inf") else round(m["bg_psnr"], 4),
            "bg_leak_frac": round(m["bg_leak_frac"], 6),
            "bgb_mse": round(m["bgb_mse"], 10),
            "bgb_psnr": "inf" if m["bgb_psnr"] == float("inf") else round(m["bgb_psnr"], 4),
            "fg_mse": round(m["fg_mse"], 10),
            "total_se": round(m["total_se"], 4),
            "note": c["note"],
        })

    with open(os.path.join(OUT, "cases.csv"), "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(results[0]))
        wtr.writeheader()
        wtr.writerows(results)

    sweeps = sweep_rows(args.radius)
    with open(os.path.join(OUT, "sensitivity.csv"), "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(sweeps[0]))
        wtr.writeheader()
        wtr.writerows(sweeps)

    write_report(results, sweeps, failures, args.radius)

    print(f"{len(results)} cases, mask radius {args.radius}px "
          f"({results[0]['mask_area_pct']}% of frame)")
    print(f"{'case':22s} {'evidence':16s} {'met':>4s} {'bg_mse':>12s} {'bgb_mse':>12s} {'fg_mse':>10s}")
    for r in results:
        print(f"{r['case']:22s} {r['evidence']:16s} "
              f"{'yes' if r['all_expectations_met'] else 'NO':>4s} "
              f"{r['bg_mse']:12.8f} {r['bgb_mse']:12.8f} {r['fg_mse']:10.6f}")
    print()
    if failures:
        print(f"{len(failures)} declared expectation(s) NOT met:")
        for case, metric, op, target, got in failures:
            print(f"  {case}: expected {metric} {op} {target}, got {got}")
    else:
        print("every declared expectation was met")
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}/{{cases,sensitivity}}.csv "
          f"and reports/stress_suite.md")
    return 0


def write_report(results, sweeps, failures, radius):
    L = ["# Controlled evaluator stress suite (B1)", ""]
    L.append("Generated by `python scripts/repair_stress_suite.py`. Each case is a synthetic "
             "pixel transformation with a known correct answer, and the expected property of "
             "each was declared in the source **before** any metric was computed.")
    L.append("")
    L.append(f"Frame {SIZE[1]}x{SIZE[0]}, mask radius {radius}px, boundary band {BAND_PX}px, "
             f"leak threshold {LEAK_THRESHOLD}. The scene is textured on purpose: a flat image "
             "makes every locality metric look perfect and proves nothing.")
    L.append("")
    L.append("`evidence = by_construction` means the case is true because of how it was built, "
             "so a metric agreeing with it has restated the construction. Only `held_out` rows "
             "are evidence that a metric works.")
    L.append("")
    L.append("| case | evidence | expectation | met | bg MSE | bgb MSE | fg MSE | bg leak |")
    L.append("| --- | --- | --- | :---: | ---: | ---: | ---: | ---: |")
    for r in results:
        L.append(f"| `{r['case']}` | {r['evidence']} | {r['expectation']} | "
                 f"{'yes' if r['all_expectations_met'] else '**NO**'} | {r['bg_mse']} | "
                 f"{r['bgb_mse']} | {r['fg_mse']} | {r['bg_leak_frac']} |")
    L.append("")

    L.append("## What each case shows")
    L.append("")
    for r in results:
        L.append(f"- **`{r['case']}`** — {r['note']}")
    L.append("")

    L.append("## The location test")
    L.append("")
    exact = next(r for r in results if r["case"] == "exact_target")
    wrong = next(r for r in results if r["case"] == "wrong_region")
    L.append("`exact_target` and `wrong_region` apply the *same* transformation to the *same* "
             "number of pixels; only the location differs.")
    L.append("")
    L.append("| case | bg MSE | fg MSE | total squared error over the frame |")
    L.append("| --- | ---: | ---: | ---: |")
    L.append(f"| `exact_target` | {exact['bg_mse']} | {exact['fg_mse']} | {exact['total_se']} |")
    L.append(f"| `wrong_region` | {wrong['bg_mse']} | {wrong['fg_mse']} | {wrong['total_se']} |")
    L.append("")
    L.append("The roles swap: each case puts all of its error in the region where the other has "
             "none. The per-pixel means are not numerically equal because they are averaged over "
             "different denominators — the mask is "
             f"{exact['mask_area_pct']}% of the frame and the background is the rest — so the "
             "column to compare is the total squared error, which matches to within the "
             "randomness of the perturbation.")
    L.append("")
    L.append("A single whole-frame error measure scores these two outputs nearly identically, "
             "while one performs the requested edit and the other damages an untouched object "
             "and leaves the target alone. That is the concrete failure the separated axes "
             "prevent, and it is a held-out result: nothing in the metric definition was tuned "
             "to it.")
    L.append("")

    L.append("## Boundary band: held-out spill widths")
    L.append("")
    L.append("The band is 8px. Spills of 2, 4, 16 and 32px are held out — the band was not "
             "chosen with them in view.")
    L.append("")
    L.append("| spill width | bg MSE | bgb MSE | bgb suppresses it? |")
    L.append("| ---: | ---: | ---: | :---: |")
    for width in (2, 4, 8, 16, 32):
        r = next((x for x in results if x["case"] == f"boundary_spill_{width}px"), None)
        if r:
            L.append(f"| {width}px | {r['bg_mse']} | {r['bgb_mse']} | "
                     f"{'yes' if float(r['bgb_mse']) == 0 else 'no'} |")
    L.append("")
    L.append("The band behaves as specified: it absorbs spill up to its own width and reports "
             "everything wider. So a boundary-robust PSNR is a statement about error *beyond a "
             "declared distance*, not about error in general, and quoting it without the band "
             "width is meaningless.")
    L.append("")

    L.append("## Sensitivity")
    L.append("")
    L.append(f"{len(sweeps)} rows in `evaluations/repaired_v1/stress_suite/sensitivity.csv` "
             "sweep mask radius (16/32/64/96px), band width (0/4/8/16px) and leak threshold "
             "(0.01/0.05/0.10).")
    L.append("")
    band0 = [s for s in sweeps if s["band_px"] == 0 and s["case"] == "boundary_spill_4px"]
    band8 = [s for s in sweeps if s["band_px"] == 8 and s["case"] == "boundary_spill_4px"]
    if band0 and band8:
        L.append(f"With no band, a 4px spill registers as background damage "
                 f"(bg MSE {band0[0]['bg_mse']}); with the 8px band it registers as zero. The "
                 "same output is 'leaky' or 'clean' depending on a configuration choice, which "
                 "is why mixed metric configurations must be rejected rather than pooled.")
    L.append("")
    L.append("Leak fraction is threshold-dependent by definition: the same output's leak "
             "fraction changes with the 0.01/0.05/0.10 cut, so a leak percentage without its "
             "threshold is not a quantity.")
    L.append("")

    L.append("## Limits")
    L.append("")
    L.append("- These are synthetic transformations, not natural edits. They establish spatial "
             "and arithmetic properties only.")
    L.append("- **No judge is run here.** Nothing in this suite validates `edit_success`, "
             "`preservation` or any semantic label, and no claim about human agreement follows "
             "from it.")
    L.append("- The `identity` and `hard_composite` cases show that the locality family alone "
             "cannot distinguish a perfect preserver from a model that did nothing. A success "
             "measure is required, which is the argument for success-conditioned reporting.")
    L.append("- The real autoencoder round-trip is **not** synthesised here. It is measured on "
             "real images in `evaluations/vae_floor_by_image.csv` and stays a descriptive "
             "reference, not an error floor.")
    if failures:
        L.append("")
        L.append("## Declared expectations not met")
        L.append("")
        for case, metric, op, target, got in failures:
            L.append(f"- `{case}`: expected `{metric} {op} {target}`, measured `{got}`")

    with open(os.path.join(ROOT, "reports", "stress_suite.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--radius", type=int, default=48, help="mask disc radius in pixels")
    sys.exit(main(p.parse_args()))
