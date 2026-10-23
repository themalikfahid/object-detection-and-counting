# Setup Guide (Windows / Mac / Linux)

## 1. What you need
- Python 3.10, 3.11 or 3.12 (download from python.org; on Windows tick "Add Python to PATH")
- About 3 GB free disk space (PyTorch is large)
- Internet for the first run (the model file yolov8n.pt downloads once, about 6 MB)
- A normal laptop is enough. A GPU is not needed.

## 2. Install
Open a terminal inside the `object_counter` folder.

Windows:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Mac / Linux:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Check that the code works (no internet model needed)
```
python tests/test_counter.py
```
You should see `ALL TESTS PASSED`.

## 4. Add test pictures
Copy 15 to 20 pictures into the `samples` folder. Use only permitted sources
(COCO, Open Images, Pexels). Write the source and license in `samples/README.md`.

## 5. Run the web app
```
streamlit run app.py
```
A browser page opens (http://localhost:8501).
1. Upload a picture or choose one from samples.
2. Choose the classes to count.
3. Set the confidence slider (start with 0.25).
4. Click "Count objects".
5. Download the PNG / CSV / JSON results.

## 6. Run without the web page
```
python cli.py samples --classes person car --out results
```
Annotated images, CSV and JSON files are saved in `results/`.

## 7. Make the evaluation table
1. Count the objects by hand in each picture.
2. Edit `ground_truth.csv`:
```
filename,class,manual_count
street1.jpg,car,5
street1.jpg,person,3
```
3. Run:
```
python evaluate.py --images samples --truth ground_truth.csv
```
4. Copy the table into REPORT.md. Try again with `--conf 0.4` or `--weights yolov8s.pt` to compare settings.

## 8. Common problems
| Problem | Fix |
|---|---|
| `No module named ultralytics` | Activate the virtual environment, then run pip install again |
| Model does not download | Check internet. Or download yolov8n.pt manually from the Ultralytics releases page and put it in the project folder |
| Video does not play in the browser | Use the Download button, or convert with ffmpeg: `ffmpeg -i out.mp4 -vcodec libx264 fixed.mp4` |
| App is slow on video | Use yolov8n, raise "Process every Nth frame", use a short video |
| Many small objects missed | Raise inference size to 960 or 1280, lower confidence a little |
| Too many false boxes | Raise the confidence slider |
