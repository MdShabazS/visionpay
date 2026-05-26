# VisionPay

A real-time Indian currency note classifier built to assist visually impaired users.
Point a webcam at a banknote, and the app speaks the denomination out loud.

It runs fully offline on a regular laptop — no GPU, no cloud, no special hardware.

Currently supports: ₹10, ₹20, ₹50, ₹100, ₹200, ₹500.

## How it works

A MobileNetV2 model (ImageNet-pretrained) is fine-tuned on a custom dataset of
currency images captured through a webcam. At runtime, the model is loaded as a
TFLite interpreter for speed (~30 FPS on a MacBook CPU), and predictions across
consecutive frames are aggregated using a majority vote with confidence and
margin gates so the announcement only fires once the prediction is stable.

The dataset includes a `background` class containing non-note images (hands,
papers, surfaces, random objects). This is what the model uses as the
"no note in frame" answer.

## Project layout

```
.
├── capture_dataset.py      # webcam tool to record training images per class
├── VisionPay.ipynb         # training notebook (MobileNetV2 + TFLite export)
├── realtime.py             # the actual detector — webcam loop, smoothing, TTS
├── realtime.command        # macOS double-click launcher
├── model/
│   ├── visionpay_model.keras    # (gitignored)
│   ├── visionpay_model.tflite   # (gitignored, preferred at runtime)
│   └── class_names.json
├── dataset/                # (gitignored) class-folder images
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

Tested on Python 3.13 / TensorFlow 2.21 / macOS.

## Usage

### 1. Capture a dataset

```bash
python capture_dataset.py
```

Keys: `SPACE` save, `n`/`p` switch class, `a` auto-capture, `q` quit.

Aim for ~300 images per denomination, with variation in lighting, angle,
distance, and front/back. The `background` class needs at least as much
variety as everything else combined — anything that isn't a note (hands,
paper, electronics, plain surfaces) belongs here.

### 2. Train

Open `VisionPay.ipynb` and run all cells. It saves both `.keras` and `.tflite`
models plus `class_names.json` into `model/`.

### 3. Run

```bash
python realtime.py
```

Hotkeys:

| Key   | Action                       |
|-------|------------------------------|
| `q`   | quit                         |
| `r`   | reset running total          |
| `s`   | speak the running total      |
| `a`   | toggle auto-count mode       |
| `m`   | mute / unmute                |
| `+/-` | raise / lower the threshold  |

## Model

- Backbone: MobileNetV2, 224×224 input, ImageNet weights
- Head: GAP → BatchNorm → Dropout → Dense(256) → BatchNorm → Dropout → Softmax
- Two-phase training: ~20 epochs head only, then ~20 epochs fine-tuning the
  last 60 layers of the backbone
- Augmentation: horizontal flip, ±25% rotation/zoom, ±15% translation,
  ±30% brightness/contrast
- Class weights to compensate for dataset imbalance
- Validation accuracy: ~93% with ~300 images/class

The Keras model is converted to a quantized TFLite model at the end of
training. `realtime.py` prefers the TFLite file when available and falls
back to the Keras model otherwise.

## Known limitation

This is a closed-set classifier — softmax always returns one of the trained
classes, so unfamiliar objects (a green bottle cap, a notebook page, a
circuit board) can be confidently misclassified as a note.

The current mitigation is a large, varied `background` class plus a
confidence + margin threshold. The proper fix is to switch to a detector
(YOLOv8) that produces bounding boxes and can return zero detections —
that's on the roadmap.

## Roadmap

- YOLOv8-based detector to replace the closed-set classifier
- Multi-note counting in a single frame
- Hindi TTS toggle
- TFLite deployment to Android / Raspberry Pi
- Counterfeit-feature checks (security thread, watermark)

## License

MIT
