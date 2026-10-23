"""Streamlit UI:  streamlit run app.py"""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import counter as oc

st.set_page_config(page_title="AI Object Counter", layout="wide")
SAMPLES = Path(__file__).parent / "samples"
IMG_EXT, VID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}, {".mp4", ".avi", ".mov"}


@st.cache_resource(show_spinner="Loading model…")
def get_model(weights):
    return oc.load_model(weights)


st.title("AI Object Counting")
st.caption("Pretrained YOLO detector · counts objects by class · no face recognition")

with st.sidebar:
    st.header("Settings")
    weights = st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
                           help="Larger = slower but usually more accurate.")
    model = get_model(weights)
    names = oc.class_names(model)
    default = [c for c in ("person", "car") if c in names]
    classes = st.multiselect("Classes to count", names, default=default)
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("NMS IoU threshold", 0.1, 0.9, 0.45, 0.05,
                    help="Lower merges overlapping boxes more aggressively.")
    imgsz = st.select_slider("Inference size", [320, 480, 640, 960, 1280], 640)

with st.expander("Known limitations (read before trusting counts)"):
    st.markdown("""
- **Occlusion:** hidden or partly visible objects are missed or split into several boxes.
- **Overlap / crowds:** touching objects may merge into one box (undercount); NMS can suppress true neighbours.
- **Lighting & quality:** low light, glare, blur, and compression reduce recall and confidence.
- **Small / distant objects:** below roughly 20–30 px they are often missed at default size; raise inference size.
- **Model accuracy:** COCO-pretrained models have fixed classes, biases, and false positives. Results are estimates; verify manually for important decisions.
- **Threshold trade-off:** low confidence → more false positives; high → more misses.
- **Video:** frame counts are per-frame snapshots; line crossings depend on tracker ID stability.
- **Privacy:** people are counted as anonymous boxes only. No identification is performed.
""")

tab_img, tab_vid = st.tabs(["Image", "Video"])

# ------------------------------------------------------------------ image
with tab_img:
    samples = sorted(p.name for p in SAMPLES.glob("*") if p.suffix.lower() in IMG_EXT)
    up = st.file_uploader("Upload an image", type=[e[1:] for e in IMG_EXT])
    pick = st.selectbox("…or choose an approved sample", ["—"] + samples)
    src_name, img = None, None
    if up:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        src_name = up.name
    elif pick != "—":
        img = cv2.imread(str(SAMPLES / pick))
        src_name = pick

    if img is not None and st.button("Count objects", type="primary", key="img"):
        if not classes:
            st.warning("Select at least one class.")
        else:
            dets = oc.detect(model, img, classes, conf, iou, imgsz)
            ann = oc.annotate(img, dets)
            rec = oc.build_record(src_name, dets, weights, conf, iou, classes)
            c1, c2 = st.columns([3, 1])
            c1.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                     caption=f"{src_name}", use_container_width=True)
            c2.metric("Total", len(dets))
            for k, v in sorted(oc.count(dets).items()):
                c2.metric(k, v)
            if dets:
                st.dataframe(pd.DataFrame([{
                    "class": d.label, "confidence": round(d.conf, 3),
                    "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2} for d in dets]),
                    use_container_width=True)
            stem = f"{Path(src_name).stem}_{rec['timestamp'].replace(':', '')}"
            d1, d2, d3 = st.columns(3)
            d1.download_button("Annotated image (PNG)",
                               cv2.imencode(".png", ann)[1].tobytes(),
                               f"{stem}_annotated.png", "image/png")
            d2.download_button("Counts (CSV)", oc.record_to_csv(rec),
                               f"{stem}_counts.csv", "text/csv")
            d3.download_button("Full results (JSON)", oc.record_to_json(rec, dets),
                               f"{stem}_counts.json", "application/json")
            st.caption("Tip: also run `python evaluate.py` to compare against manual counts.")

# ------------------------------------------------------------------ video
with tab_vid:
    vsamples = sorted(p.name for p in SAMPLES.glob("*") if p.suffix.lower() in VID_EXT)
    vup = st.file_uploader("Upload a video", type=[e[1:] for e in VID_EXT])
    vpick = st.selectbox("…or choose an approved sample video", ["—"] + vsamples)
    stride = st.slider("Process every Nth frame", 1, 10, 2)
    use_line = st.checkbox("Count objects crossing a line (uses tracking)")
    orient, pos = "Horizontal", 50
    if use_line:
        orient = st.radio("Line", ["Horizontal", "Vertical"], horizontal=True)
        pos = st.slider("Line position (%)", 5, 95, 50)

    vpath, vname = None, None
    if vup:
        tmp = Path(tempfile.mkdtemp()) / vup.name
        tmp.write_bytes(vup.read())
        vpath, vname = tmp, vup.name
    elif vpick != "—":
        vpath, vname = SAMPLES / vpick, vpick

    if vpath and st.button("Process video", type="primary", key="vid"):
        if not classes:
            st.warning("Select at least one class.")
        else:
            cap = cv2.VideoCapture(str(vpath))
            W, H = int(cap.get(3)), int(cap.get(4))
            cap.release()
            line = None
            if use_line:
                line = (((0, H * pos // 100), (W, H * pos // 100)) if orient == "Horizontal"
                        else ((W * pos // 100, 0), (W * pos // 100, H)))
            outp = Path(tempfile.mkdtemp()) / f"{Path(vname).stem}_annotated.mp4"
            bar = st.progress(0.0)
            stats = oc.process_video(model, vpath, outp, classes, conf, iou, imgsz,
                                     stride, line, bar.progress)
            m1, m2, m3 = st.columns(3)
            m1.metric("Peak count in a frame", stats["peak_count"])
            m2.metric("Mean count per frame", f"{stats['mean_count']:.1f}")
            m3.metric("Frames processed", stats["frames_processed"])
            if line:
                st.write("Line crossings:", stats["line_crossings"] or 0)
            st.video(str(outp))
            st.download_button("Annotated video (MP4)", outp.read_bytes(), outp.name,
                               "video/mp4")
            df = pd.DataFrame(stats["per_frame"]).fillna(0)
            st.line_chart(df.set_index("frame")["total"])
            st.download_button("Per-frame counts (CSV)", df.to_csv(index=False).encode(),
                               f"{Path(vname).stem}_frame_counts.csv", "text/csv")
