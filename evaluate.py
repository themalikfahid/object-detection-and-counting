"""Compare predicted vs manually verified counts on a test set.

ground_truth.csv columns: filename,class,manual_count
Usage: python evaluate.py --images samples --truth ground_truth.csv --conf 0.25
"""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd

import counter as oc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="samples")
    ap.add_argument("--truth", default="ground_truth.csv")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="evaluation_results")
    a = ap.parse_args()

    truth = pd.read_csv(a.truth)
    model = oc.load_model(a.weights)
    by_file = defaultdict(list)
    for r in truth.itertuples():
        by_file[r.filename].append((str(r._2).strip(), int(r.manual_count)))

    rows = []
    for fname, items in by_file.items():
        img = cv2.imread(str(Path(a.images) / fname))
        if img is None:
            print(f"! cannot read {fname}")
            continue
        dets = oc.detect(model, img, [c for c, _ in items], a.conf, a.iou, a.imgsz)
        pred = oc.count(dets)
        for cls, manual in items:
            p = pred.get(cls, 0)
            rows.append({"image": fname, "class": cls, "manual": manual,
                         "predicted": p, "error": p - manual,
                         "abs_error": abs(p - manual),
                         "pct_error": round(100 * abs(p - manual) / manual, 1)
                         if manual else None})
        cv2.imwrite(str(Path(a.out + "_" + Path(fname).stem + ".jpg")),
                    oc.annotate(img, dets))

    df = pd.DataFrame(rows)
    df.to_csv(a.out + ".csv", index=False)
    print(df.to_markdown(index=False) if hasattr(df, "to_markdown") else df)
    if len(df):
        print(f"\nImages/rows: {len(df)} | MAE: {df.abs_error.mean():.2f} | "
              f"Exact matches: {(df.error == 0).mean() * 100:.0f}% | "
              f"Mean signed error (bias): {df.error.mean():+.2f}")
        print("Negative bias = undercounting (occlusion/misses); positive = overcounting.")


if __name__ == "__main__":
    main()
