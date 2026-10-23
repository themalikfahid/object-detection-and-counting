"""Logic tests using a fake model (no GPU, no download needed).
Run:  python tests/test_counter.py
"""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import counter as oc


class T:  # tiny tensor-like helper
    def __init__(self, v): self.v = np.array(v)
    def __getitem__(self, i): return T(self.v[i]) if self.v.ndim > 1 else self.v[i]
    def tolist(self): return self.v.tolist()
    def __float__(self): return float(self.v)
    def __int__(self): return int(self.v)


class Box:
    def __init__(self, xyxy, cls, conf, tid=None):
        self.xyxy, self.cls, self.conf = T([xyxy]), T([cls]), T([conf])
        self.id = T([tid]) if tid is not None else None


class Res:
    def __init__(self, boxes): self.boxes = boxes


class FakeModel:
    names = {0: "person", 2: "car", 39: "bottle"}

    def __init__(self, script=None):
        self.script, self.n = script, 0
        self.last_classes = "unset"

    def _boxes(self, tracked):
        if self.script:  # per-frame scripted boxes
            b = self.script[min(self.n, len(self.script) - 1)]
            self.n += 1
            return b
        return [Box((10, 10, 50, 90), 0, .9, 1 if tracked else None),
                Box((100, 20, 180, 80), 2, .7, 2 if tracked else None),
                Box((200, 20, 240, 90), 0, .5, 3 if tracked else None)]

    def predict(self, img, classes=None, **kw):
        self.last_classes = classes
        return [Res([b for b in self._boxes(False) if classes is None or b.cls[0] in classes])]

    def track(self, img, classes=None, **kw):
        return [Res([b for b in self._boxes(True) if classes is None or b.cls[0] in classes])]


def test_detect_and_count():
    m, img = FakeModel(), np.zeros((120, 300, 3), np.uint8)
    d = oc.detect(m, img)
    assert len(d) == 3 and oc.count(d) == {"person": 2, "car": 1}
    d = oc.detect(m, img, ["car"])
    assert len(d) == 1 and d[0].label == "car"
    d = oc.detect(m, img, ["person", "car"])
    assert len(d) == 3
    assert oc.detect(m, img, ["unicorn"]) == []  # unknown class -> empty, not "all"
    print("ok detect/count/class filter")


def test_annotate_export():
    m, img = FakeModel(), np.full((120, 300, 3), 100, np.uint8)
    d = oc.detect(m, img)
    a = oc.annotate(img, d)
    assert a.shape == img.shape and (a != img).any()
    rec = oc.build_record("my photo.jpg", d, "fake", .25, .45, None)
    assert rec["total_count"] == 3 and rec["input_filename"] == "my photo.jpg"
    with tempfile.TemporaryDirectory() as t:
        stem = oc.export_all(t, "my photo.jpg", a, rec, d)
        names = sorted(p.name for p in Path(t).iterdir())
        assert len(names) == 3 and all(n.startswith("my photo_") for n in names), names
    assert b"TOTAL,3" in oc.record_to_csv(rec)
    print("ok annotate/export")


def test_empty():
    img = np.zeros((50, 50, 3), np.uint8)
    m = FakeModel(script=[[]])
    d = oc.detect(m, img)
    assert d == [] and len(oc.annotate(img, d)) == 50
    print("ok zero detections")


def test_video_line_crossing():
    # one car moves top -> bottom across a horizontal line at y=50
    script = [[Box((20, y - 10, 40, y + 10), 2, .8, 7)] for y in (10, 30, 45, 60, 80)]
    with tempfile.TemporaryDirectory() as t:
        vin, vout = Path(t) / "in.mp4", Path(t) / "out.mp4"
        w = cv2.VideoWriter(str(vin), cv2.VideoWriter_fourcc(*"mp4v"), 5, (100, 100))
        for _ in script:
            w.write(np.zeros((100, 100, 3), np.uint8))
        w.release()
        s = oc.process_video(FakeModel(script), vin, vout, ["car"],
                             line=((0, 50), (100, 50)))
        assert s["frames_processed"] == 5 and s["peak_count"] == 1
        assert s["line_crossings"] == {"car": 1}, s["line_crossings"]
        assert vout.exists() and vout.stat().st_size > 0
    print("ok video + line crossing")


if __name__ == "__main__":
    for f in (test_detect_and_count, test_annotate_export, test_empty, test_video_line_crossing):
        f()
    print("ALL TESTS PASSED")
