# VisionPay — Project Handoff

Use this file as the first prompt when resuming on another computer. Paste it to Cascade and say: "continue from here."

---

## What this project is
**VisionPay** — real-time Indian currency detector for visually impaired users.
- Webcam → MobileNetV2 classifier → TTS announces the note value
- Runs offline, ~30 FPS on CPU using TFLite
- Repo path on current machine: `/Users/shabaz/CascadeProjects/currency-detector`

## Repo structure
- `capture_dataset.py` — capture training images per class via webcam
- `dataset/<class>/` — captured images (10, 20, 50, 100, 200, 500, background)
- `VisionPay.ipynb` — training notebook (MobileNetV2 transfer learning + TFLite export)
- `model/visionpay_model.keras` — trained Keras model
- `model/visionpay_model.tflite` — quantized TFLite (preferred at runtime)
- `model/class_names.json` — class labels
- `realtime.py` — webcam app with smoothing, TTS, auto-count, threshold control
- `requirements.txt` — deps
- `venv/` — local virtualenv (Python 3.13)

## Current state (as of May 27, 2026)
- **Model in use**: old model from `Curency BAckup/currency-detector.zip` (restored on May 24 because the retrained one was worse)
- **Dataset captured (current)**:
  - 10: 378, 20: 394, 50: 318, 100: 379, 200: 390, 500: 443, background: 350
  - `2000` folder removed (no ₹2000 notes available)
- **Notebook is updated** with: stronger augmentation, class weights, deeper fine-tuning (last 60 layers), 20+20 epochs, TFLite export
- **Not yet retrained** on the new dataset — the old model is currently active

## Known issue
Closed-set classifier confuses **non-note objects** (mouse, paper, electronics, bottle caps) for notes — predicts with high confidence because softmax forces a class. Even threshold + margin gates don't reject (e.g. paper notebook → ₹200 @ 97%/95% margin).

## Two paths to fix
1. **Cheap fix** — retrain with way more diverse `background/` images (mouse, hands, paper, bottles, faces, electronics, varied colors). Notebook is ready; just Run All.
2. **Proper fix (planned, not started)** — switch to **YOLOv8** for bounding-box detection. Requires labeling existing dataset with boxes (Roboflow or LabelImg). User said "lets add yolo some other day."

## Resume options (pick one when continuing)
- **A. Retrain on current dataset** — restart kernel in `VisionPay.ipynb` → Run All. Auto-exports `.keras` + `.tflite`. Then run `venv/bin/python realtime.py`.
- **B. Capture more `background/` images first** — run `python capture_dataset.py`, choose `background`, capture 200–300 images of random objects, then retrain.
- **C. Start YOLOv8 work** — label dataset in Roboflow, train `yolov8n`, integrate into `realtime.py`. Heavy but the right long-term answer.

## How to run
```bash
cd /path/to/currency-detector
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train (only needed if retraining)
jupyter notebook VisionPay.ipynb   # Run All

# Real-time detector
python realtime.py
```

Hotkeys in `realtime.py`: `q` quit, `r` reset, `s` speak total, `a` auto-count, `m` mute, `+/-` adjust threshold.

## Tech stack
TensorFlow/Keras, MobileNetV2 (transfer learning), TFLite, OpenCV, macOS `say` / Linux `espeak` for TTS.

## Important notes for the next session
- Old model is at `model/visionpay_model.keras` — works decently for notes but bad on non-note objects
- The poorly-performing newer model is backed up in `model_new_*/`
- Full project zip: `~/CascadeProjects/Curency BAckup/currency-detector*.zip`
- `realtime.py` auto-prefers TFLite if `model/visionpay_model.tflite` exists, falls back to `.keras`
- TF 2.21, Python 3.13, no GPU
