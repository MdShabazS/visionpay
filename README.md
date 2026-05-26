<div align="center">

# VisionPay

### Real-Time Indian Currency Detector for the Visually Impaired

Point a webcam at a banknote and the app speaks the denomination out loud.
Runs fully offline on a regular laptop — no GPU, no cloud, no special hardware.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![TFLite](https://img.shields.io/badge/TFLite-quantized-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

</div>

---

## Overview

**VisionPay** is a lightweight desktop tool that helps blind and visually
impaired users identify Indian banknotes. A webcam frame is classified by a
fine-tuned **MobileNetV2** model, smoothed across consecutive frames, and the
result is announced through the system's text-to-speech engine. The whole
pipeline runs at ~30 FPS on a regular CPU using a quantized TFLite model.

Currently supports: ₹10, ₹20, ₹50, ₹100, ₹200, ₹500.

## Highlights

- **Real-time webcam classification** — ~30 FPS on CPU with TFLite
- **Offline & private** — no internet, no telemetry, no cloud APIs
- **Spoken feedback** — uses `say` on macOS, `espeak` on Linux
- **Stable predictions** — confidence + margin gates with majority vote across frames
- **Auto-count mode** — adds up notes shown one after another and announces the total
- **Adjustable threshold** at runtime so it can be tuned in the field
- **Background class** rejects non-note frames

## Project Layout

```
.
├── capture_dataset.py      # webcam tool to record training images per class
├── VisionPay.ipynb         # training notebook (MobileNetV2 + TFLite export)
├── realtime.py             # webcam loop, smoothing, TTS announcer
├── realtime.command        # macOS double-click launcher
├── model/
│   ├── visionpay_model.keras    # gitignored
│   ├── visionpay_model.tflite   # gitignored, preferred at runtime
│   └── class_names.json
├── dataset/                # gitignored, class-folder images
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/MdShabazS/visionpay.git
cd visionpay
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Tested on Python 3.13, TensorFlow 2.21, macOS.

## Usage

### 1. Capture a dataset

```bash
python capture_dataset.py
```

Keys: `SPACE` save · `n`/`p` switch class · `a` auto-capture · `q` quit.

Aim for ~300 images per denomination with variation in lighting, angle,
distance, and front/back. The `background` class needs at least as much
variety as everything else combined — anything that isn't a note (hands,
paper, electronics, plain surfaces) belongs there.

### 2. Train

Open `VisionPay.ipynb` and run all cells. The notebook saves both `.keras`
and `.tflite` models plus `class_names.json` into `model/`.

### 3. Run

```bash
python realtime.py
```

| Key   | Action                       |
|-------|------------------------------|
| `q`   | quit                         |
| `r`   | reset running total          |
| `s`   | speak the running total      |
| `a`   | toggle auto-count mode       |
| `m`   | mute / unmute                |
| `+/-` | raise / lower the threshold  |

## Model

- **Backbone** — MobileNetV2, 224×224 input, ImageNet weights
- **Head** — GAP → BatchNorm → Dropout → Dense(256) → BatchNorm → Dropout → Softmax
- **Training** — ~20 epochs head only, then ~20 epochs fine-tuning the last 60 layers
- **Augmentation** — horizontal flip, ±25% rotation/zoom, ±15% translation, ±30% brightness/contrast
- **Class weighting** to compensate for dataset imbalance
- **Validation accuracy** — ~93% with ~300 images per class

The Keras model is converted to a quantized TFLite model at the end of
training. `realtime.py` prefers the TFLite file when present and falls
back to Keras otherwise.

## Known Limitation

This is a closed-set classifier — softmax always returns one of the
trained classes, so unfamiliar objects (a green bottle cap, a notebook
page, a circuit board) can be confidently misclassified as a note.

The current mitigation is a large, varied `background` class plus
confidence and margin thresholds. The proper fix is to switch to a
detector (YOLOv8) that produces bounding boxes and can return zero
detections — that's on the roadmap.

## Roadmap

- [ ] YOLOv8-based detector to replace the closed-set classifier
- [ ] Multi-note counting in a single frame
- [ ] Hindi TTS toggle
- [ ] TFLite deployment to Android / Raspberry Pi
- [ ] Counterfeit-feature checks (security thread, watermark)

## License

Released under the [MIT License](LICENSE).
