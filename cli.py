"""Count objects from the command line (no web page needed).

Examples:
  python cli.py samples/street1.jpg --classes person car
  python cli.py samples --classes car --conf 0.3 --out results
"""
import argparse
from pathlib import Path

import cv2

import counter as oc

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="image file or folder of images")
    ap.add_argument("--classes", nargs="*", default=None)
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    p = Path(a.path)
    files = sorted(f for f in p.iterdir() if f.suffix.lower() in EXT) if p.is_dir() else [p]
    model = oc.load_model(a.weights)
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"Cannot read {f}")
            continue
        dets = oc.detect(model, img, a.classes, a.conf, a.iou, a.imgsz)
        rec = oc.build_record(f.name, dets, a.weights, a.conf, a.iou, a.classes)
        oc.export_all(a.out, f.name, oc.annotate(img, dets), rec, dets)
        print(f"{f.name}: total={len(dets)} {dict(oc.count(dets))}")
    print(f"Saved results in: {a.out}/")


if __name__ == "__main__":
    main()
