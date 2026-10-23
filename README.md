# AI Object Counting (YOLO + OpenCV + Streamlit)

Detects and counts objects (people, vehicles, boxes, products…) in images or
video, draws boxes/labels/confidence, and shows total + per-class counts.

See SETUP_GUIDE.md for the full step-by-step guide. Also: `python cli.py samples --classes person car` (no web page) and `python tests/test_counter.py` (logic tests).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py          # weights (yolov8n.pt) download on first run
```
Add permitted images/videos to `samples/` and record their licenses there.

## How it works
1. **Model:** pretrained COCO YOLOv8 (80 classes, e.g. person, car, bus, truck, bottle…).
2. **Inference:** `counter.detect()` runs the model with a confidence threshold, NMS IoU, and class filter.
3. **Counting:** detections are tallied per class (`counter.count`) → total and per-class.
4. **Output:** `counter.annotate()` draws boxes, `label conf`, and a summary panel.
5. **Export:** PNG, CSV, JSON named with input filename + timestamp (CSV/JSON include filename, timestamp, model, thresholds, counts). `counter.export_all()` saves them to disk.
6. **Video (optional):** per-frame counts, peak/mean count, and line-crossing counts via ultralytics tracking.

## Evaluation (acceptance criterion)
1. Put 10–20 varied test images in `samples/` (sparse, crowded, night, occluded, small objects).
2. Count objects by hand; fill `ground_truth.csv` (`filename,class,manual_count`). Have a second person verify.
3. Run `python evaluate.py --images samples --truth ground_truth.csv`.
4. Paste the table into the report:

| Image | Class | Manual | Predicted | Error | Abs. error | Likely cause |
|-------|-------|-------:|----------:|------:|-----------:|--------------|
| | | | | | | |

Report MAE, exact-match %, and bias (signed mean error). Optionally rerun at
different `--conf`, `--weights`, `--imgsz` to compare model settings.

## Known limitations
Occlusion; overlapping/crowded objects (merged or suppressed boxes); poor lighting,
blur, glare; small/distant objects; fixed COCO classes and model bias; confidence
threshold trade-off (false positives vs misses); tracker ID switches in video;
per-frame video counts are snapshots, not unique-object totals. Counts are estimates.

## Ethics & privacy
People are counted as anonymous "person" boxes. No face recognition or identification.
Use only images you have permission to use; do not store footage of private individuals.
