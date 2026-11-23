"""FR6 — Dataset Mode: run the pipeline on every Kaggle image.

Split: a fixed hash of the file path puts ~50% in "dev" (where thresholds are
read off) and ~50% in "test" (reported accuracy), so the score is not tuned on
the images it is measured on.

usage: python scripts/eval_dataset.py           -> out/eval.json
       python scripts/eval_dataset.py --stats   -> per-class feature percentiles on dev
"""
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solarguard.analyze import DIRTY, KAGGLE_TO_LABEL, LABELS, analyze  # noqa: E402
from solarguard.panel import load  # noqa: E402

ROOT = Path("data/raw/Faulty_solar_panel")
EXTS = {".jpg", ".jpeg", ".png"}


def split(path):
    return "dev" if zlib.crc32(str(path).encode()) % 2 == 0 else "test"


def confusion(rows):
    m = [[0] * len(LABELS) for _ in LABELS]
    for r in rows:
        m[LABELS.index(r["true"])][LABELS.index(r["pred"])] += 1
    return m


def main():
    rows, t0 = [], time.perf_counter()
    for cls in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        for f in sorted(cls.rglob("*")):
            if f.suffix.lower() not in EXTS:
                continue
            res, _ = analyze(load(f))
            rows.append({"path": str(f.relative_to(ROOT)), "kaggle": cls.name, "split": split(f.relative_to(ROOT)),
                         "true": KAGGLE_TO_LABEL[cls.name], "pred": res["label"], "reason": res["reason"],
                         "features": res["features"]})
    ms = (time.perf_counter() - t0) / len(rows) * 1000

    if "--stats" in sys.argv:
        dev = [r for r in rows if r["split"] == "dev"]
        for k in dev[0]["features"]:
            print(f"\n{k}")
            for cls in sorted({r["kaggle"] for r in dev}):
                v = np.array([r["features"][k] for r in dev if r["kaggle"] == cls])
                print(f"  {cls:18s} p10={np.percentile(v,10):8.3f} p50={np.percentile(v,50):8.3f} p90={np.percentile(v,90):8.3f}")

    summary = {}
    for s in ("dev", "test"):
        part = [r for r in rows if r["split"] == s]
        # The cleaning decision only needs "dirty or not", so report that too (anomalies excluded).
        nb = [r for r in part if r["true"] != "Anomali"]
        summary[s] = {"n": len(part), "accuracy": sum(r["true"] == r["pred"] for r in part) / len(part),
                      "binary_accuracy": sum((r["pred"] in DIRTY) == (r["true"] in DIRTY) for r in nb) / len(nb),
                      "confusion": confusion(part)}
    out = {"labels": LABELS, "ms_per_image": ms, "summary": summary, "images": rows}
    Path("out").mkdir(exist_ok=True)
    Path("out/eval.json").write_text(json.dumps(out, ensure_ascii=False))
    for s, v in summary.items():
        print(f"{s}: n={v['n']} 5-class={v['accuracy']:.1%} clean-vs-dirty={v['binary_accuracy']:.1%}")
        for lab, row in zip(LABELS, v["confusion"]):
            print(f"  {lab:12s} {row}")
    print(f"{ms:.0f} ms/image")


if __name__ == "__main__":
    main()
