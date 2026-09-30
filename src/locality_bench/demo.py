"""Procedural, redistributable fixtures; no benchmark images or human ratings."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .core import sha256


def create_demo(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    names = ("source.png", "mask.png", "identity.png", "target.png", "spill.png", "manifest.json")
    if any((destination / name).exists() for name in names):
        raise ValueError("Demo destination already contains generated files")
    y, x = np.mgrid[:64, :64]
    source = np.stack((60 + x * 2, 50 + y * 2, 70 + (x + y) // 2), axis=-1).astype(np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[24:40, 24:40] = True
    target = source.copy()
    target[mask] = [220, 40, 40]
    spill = target.copy()
    spill[:12, :12] = [240, 240, 240]
    for name, pixels in (("source", source), ("mask", mask.astype(np.uint8) * 255), ("identity", source), ("target", target), ("spill", spill)):
        Image.fromarray(pixels).save(destination / f"{name}.png")
    rows = []
    for name in ("identity", "target", "spill"):
        row = {"sample_id": name, "source_id": "synthetic-square", "model_id": name,
               "instruction": "Make the centre square red; preserve the surroundings.",
               "source": "source.png", "mask": "mask.png", "output": f"{name}.png",
               "generation_status": "produced", "prompt_valid": True, "locality_disposition": "keep",
               "model_revision": "procedural-v1", "recipe": "deterministic_array_edit", "seed": None,
               "semantic_success": None}
        for key in ("source", "mask", "output"):
            row[f"{key}_sha256"] = sha256(destination / row[key])
        rows.append(row)
    path = destination / "manifest.json"
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return path
