# VisionPay 👁️💸
### Real-Time Indian Currency Detector for the Visually Impaired

A lightweight, offline desktop tool that uses a **webcam** and **MobileNetV2** to identify Indian banknotes in real time and **announce them aloud**. Built to assist blind and visually impaired users with confident cash handling — no internet, no special hardware.

---

## ✨ Features

- 📷 **Real-time webcam detection** at ~30 FPS (TFLite)
- 🔊 **Text-to-Speech** announces each detected note (`say` on macOS, `espeak` on Linux)
- 🧠 **MobileNetV2 transfer learning** — small, fast, accurate
- 🪙 **Auto-count mode** — sums up notes shown one after another
- 🎯 **Temporal smoothing** — confidence + margin + majority-vote across frames prevents flicker / false reads
- 🎚️ **Adjustable confidence threshold** at runtime (`+` / `-` keys)
- 🔇 **Mute toggle** for silent mode
- ⚡ **Runs on CPU** — no GPU required
- 📦 **TFLite export** — 3–4× faster inference than raw Keras

---

## 📁 Project structure

```
currency-detector/
├── VisionPay.ipynb        # Training notebook (MobileNetV2 + TFLite export)
├── capture_dataset.py     # Webcam tool to build the dataset
├── realtime.py            # Real-time webcam classifier with TTS
├── realtime.command       # macOS double-click launcher
├── dataset/               # (gitignored) 10/, 20/, 50/, 100/, 200/, 500/, background/
├── model/                 # class_names.json + (gitignored) .keras / .tflite
├── requirements.txt
├── HANDOFF.md             # Project state notes for resuming work
└── README.md
```

---

## 🚀 Quickstart

### 1. Install
```bash
git clone https://github.com/MdShabazS/visionpay.git
cd visionpay
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Capture your dataset
```bash
python capture_dataset.py
```
- `SPACE` save • `n`/`p` next/prev class • `a` auto-capture • `q` quit
- **300+ images per denomination** with varied lighting, angles, distances, front/back
- Capture **lots of variety in `background`** — hands, papers, electronics, random objects, plain surfaces. This prevents the model from misclassifying non-note objects.

### 3. Train the model
Open `VisionPay.ipynb` in VS Code or Jupyter → **Run All**.
Outputs:
- `model/visionpay_model.keras`
- `model/visionpay_model.tflite` (preferred at runtime)
- `model/class_names.json`

### 4. Run real-time detector
```bash
python realtime.py
```

---

## 🎮 Hotkeys (in `realtime.py`)

| Key | Action |
|---|---|
| `q` | Quit |
| `r` | Reset total |
| `s` | Speak total |
| `a` | Toggle auto-count mode |
| `m` | Mute / unmute TTS |
| `+` / `-` | Increase / decrease confidence threshold |

---

## 🧠 Model details

- **Backbone:** MobileNetV2 (ImageNet pretrained)
- **Head:** GAP → BatchNorm → Dropout → Dense(256) → BatchNorm → Dropout → Softmax
- **Training:** ~20 epochs frozen base + ~20 epochs fine-tuning last 60 layers
- **Input:** 224×224 RGB, MobileNetV2 preprocessing (`[-1, 1]`)
- **Augmentation:** flip, rotation (0.25), zoom (0.25), translation, brightness, contrast
- **Class weighting** to handle dataset imbalance
- **Classes:** `10, 20, 50, 100, 200, 500, background`

**Validation accuracy:** ~93% with ~300 images per class.

---

## ⚠️ Known limitation

The model is a **closed-set classifier** — softmax always picks one of the trained classes. So unfamiliar objects (a mouse, a paper notebook, a green bottle cap) can be confidently misclassified as a note.

**Mitigation:** large, varied `background/` class.
**Long-term fix:** switch to **YOLOv8** for bounding-box detection so the model can say *"no note in frame"*. Planned, not yet implemented.

---

## 🔧 Roadmap

- [ ] **YOLOv8** object detection (bounding boxes) to eliminate non-note false positives
- [ ] **Multi-note counting** in a single frame
- [ ] **Counterfeit feature** detection (security thread / watermark)
- [ ] **Bilingual TTS** — English + Hindi toggle
- [ ] **Mobile app** port (TFLite already ready)
- [ ] **Raspberry Pi** edge deployment

---

## 🧑‍🦯 Built for accessibility

- Pure audio output — no need to look at the screen
- Keyboard-only operation
- Works fully offline
- Free & open-source

---

## 📝 License
MIT — built for educational and assistive use.
