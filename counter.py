"""Core logic: detection, class filtering, counting, annotation, export.

Uses a pretrained COCO YOLO model (Ultralytics). No face recognition or
person identification is performed: "person" is just a box class to count.
"""
from __future__ import annotations

import colorsys
import csv
import io
import json
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    cls_id: int
    label: str
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int
    track_id: int | None = None


# ---------------------------------------------------------------- model
def load_model(weights: str = "yolov8n.pt"):
    """Load a pretrained model (weights auto-download on first use)."""
    from ultralytics import YOLO
    return YOLO(weights)


def class_names(model) -> list[str]:
    return [model.names[i] for i in sorted(model.names)]


def _class_ids(model, classes):
    if not classes:
        return None
    wanted = {c.lower() for c in classes}
    return [i for i, n in model.names.items() if n.lower() in wanted]


def _to_detections(model, result) -> list[Detection]:
    dets = []
    if result.boxes is None:
        return dets
    for b in result.boxes:
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
        cid = int(b.cls[0])
        tid = int(b.id[0]) if getattr(b, "id", None) is not None else None
        dets.append(Detection(cid, model.names[cid], float(b.conf[0]),
                              x1, y1, x2, y2, tid))
    return dets


# ------------------------------------------------------------ detection
def detect(model, image_bgr, classes=None, conf=0.25, iou=0.45, imgsz=640):
    """Run inference on one BGR image, keeping only the selected classes."""
    ids = _class_ids(model, classes)
    if classes and not ids:
        return []
    res = model.predict(image_bgr, conf=conf, iou=iou, imgsz=imgsz,
                        classes=ids, verbose=False)[0]
    return _to_detections(model, res)


def count(dets: list[Detection]) -> Counter:
    return Counter(d.label for d in dets)


# ----------------------------------------------------------- annotation
def _color(cls_id: int):
    r, g, b = colorsys.hsv_to_rgb((cls_id * 0.137) % 1.0, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR


def annotate(image_bgr, dets: list[Detection], summary: bool = True):
    """Draw boxes, labels, confidences and a summary panel."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    scale = max(0.4, min(w, h) / 900)
    th = max(1, int(round(2 * scale)))
    for d in dets:
        col = _color(d.cls_id)
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), col, th)
        tag = f"{d.label} {d.conf:.2f}"
        if d.track_id is not None:
            tag = f"#{d.track_id} {tag}"
        (tw, tht), bl = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5 * scale, th)
        y0 = max(d.y1 - tht - bl - 2, 0)
        cv2.rectangle(out, (d.x1, y0), (d.x1 + tw + 4, y0 + tht + bl + 2), col, -1)
        cv2.putText(out, tag, (d.x1 + 2, y0 + tht + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255),
                    max(1, th - 1), cv2.LINE_AA)
    if summary:
        counts = count(dets)
        lines = [f"Total: {len(dets)}"] + [f"{k}: {v}" for k, v in sorted(counts.items())]
        fs, lh = 0.6 * scale, int(26 * scale)
        pw = int(max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0]
                     for l in lines) + 20)
        ov = out.copy()
        cv2.rectangle(ov, (8, 8), (8 + pw, 8 + lh * len(lines) + 10), (0, 0, 0), -1)
        out = cv2.addWeighted(ov, 0.6, out, 0.4, 0)
        for i, l in enumerate(lines):
            cv2.putText(out, l, (18, 8 + lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------- export
def build_record(source_name: str, dets: list[Detection], model_name: str,
                 conf: float, iou: float, selected: list[str] | None) -> dict:
    counts = count(dets)
    return {
        "input_filename": source_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "confidence_threshold": conf,
        "iou_threshold": iou,
        "selected_classes": selected or "all",
        "total_count": len(dets),
        "per_class_counts": dict(counts),
    }


def record_to_csv(record: dict) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["input_filename", "timestamp", "model", "confidence",
                "class", "count"])
    base = [record["input_filename"], record["timestamp"], record["model"],
            record["confidence_threshold"]]
    for k, v in sorted(record["per_class_counts"].items()):
        w.writerow(base + [k, v])
    w.writerow(base + ["TOTAL", record["total_count"]])
    return buf.getvalue().encode()


def record_to_json(record: dict, dets: list[Detection]) -> bytes:
    return json.dumps({**record, "detections": [asdict(d) for d in dets]},
                      indent=2).encode()


def export_all(out_dir: str | Path, source_name: str, annotated, record, dets):
    """Save annotated image + CSV + JSON; names carry filename and timestamp."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(source_name).stem}_{datetime.now():%Y%m%d_%H%M%S}"
    cv2.imwrite(str(out / f"{stem}_annotated.jpg"), annotated)
    (out / f"{stem}_counts.csv").write_bytes(record_to_csv(record))
    (out / f"{stem}_counts.json").write_bytes(record_to_json(record, dets))
    return out / stem


# ---------------------------------------------------------------- video
def process_video(model, in_path, out_path, classes=None, conf=0.25, iou=0.45,
                  imgsz=640, stride=1, line=None, progress_cb=None):
    """Annotate a video. `line` = ((x1,y1),(x2,y2)) in pixels enables
    line-crossing counts using tracking IDs.

    Returns dict with per-frame counts, peak/mean count and line crossings.
    """
    cap = cv2.VideoCapture(str(in_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / stride, (w, h))
    ids = _class_ids(model, classes)
    per_frame, last_side, crossed = [], {}, {}
    seen_ids = set()
    i = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % stride:
            continue
        if classes and not ids:
            dets = []
        elif line:
            res = model.track(frame, persist=True, conf=conf, iou=iou,
                              imgsz=imgsz, classes=ids, verbose=False)[0]
            dets = _to_detections(model, res)
        else:
            dets = detect(model, frame, classes, conf, iou, imgsz)
        per_frame.append(dict(count(dets), frame=i, total=len(dets)))
        if line:
            (lx1, ly1), (lx2, ly2) = line
            for d in dets:
                if d.track_id is None:
                    continue
                seen_ids.add(d.track_id)
                cx, cy = (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2
                side = np.sign((lx2 - lx1) * (cy - ly1) - (ly2 - ly1) * (cx - lx1))
                prev = last_side.get(d.track_id)
                if prev is not None and side and prev != side:
                    crossed[d.label] = crossed.get(d.label, 0) + 1
                if side:
                    last_side[d.track_id] = side
        ann = annotate(frame, dets)
        if line:
            cv2.line(ann, line[0], line[1], (0, 255, 255), 3)
            txt = "Crossed: " + (", ".join(f"{k}={v}" for k, v in crossed.items()) or "0")
            cv2.putText(ann, txt, (18, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(ann)
        if progress_cb:
            progress_cb(min((i + 1) / total, 1.0))
    cap.release()
    writer.release()
    totals = [f["total"] for f in per_frame] or [0]
    return {"frames_processed": len(per_frame), "peak_count": max(totals),
            "mean_count": float(np.mean(totals)), "line_crossings": crossed,
            "unique_tracks": len(seen_ids), "per_frame": per_frame}
