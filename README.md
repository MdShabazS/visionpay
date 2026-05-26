# VisionPay 👁️💸
### Indian Currency Detector for the Visually Impaired

An end-to-end, industrial-ready system that helps blind and visually impaired users identify Indian currency notes (₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000) using their **voice** or a **single large button**, with a webcam and **spoken feedback**.

---

## ✨ Features

- 🎙️ **Voice Assistant** — say *"scan"*, *"continuous"*, *"total"*, *"reset"*, *"help"* (Web Speech API)
- 🟢 **Big accessible DETECT button** + keyboard `SPACE`/`V`/`C` shortcuts
- 📷 **Live webcam** with on-screen scan overlay
- 🔊 **Text-to-Speech** announces each detected note
- 🪙 **Session wallet** — adds up detected notes, breakdown by denomination
- 🔁 **Continuous mode** — auto-detect every 2 s
- 🌐 **Bilingual** — English + Hindi (TTS + voice commands)
- ♿ **Accessibility** — large fonts, high-contrast toggle, haptic vibration
- 🧠 **Custom-trained MobileNetV2** with transfer learning + fine-tuning
- 🔒 **Confidence threshold** + `background` class so it never lies
- ⚡ Lightweight — runs on CPU

---

## 📁 Project structure

```
currency-detector/
├── VisionPay.ipynb        # Training notebook (MobileNetV2 transfer learning)
├── capture_dataset.py     # Webcam tool to build the dataset
├── app.py                 # Flask backend + /predict
├── templates/index.html   # Pro UI (Tailwind, accessible)
├── static/js/app.js       # Camera + voice + TTS client
├── dataset/               # Created by capture script: 10/, 20/, ... 2000/, background/
├── model/                 # Saved model + class_names.json
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### 1. Install
```bash
cd currency-detector
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Capture your dataset
```bash
python capture_dataset.py
```
- `SPACE` save • `n`/`p` next/prev class • `a` auto-capture • `q` quit
- Capture **80–150 images per denomination**, varying lighting, angle, distance, front/back, and backgrounds
- Don't forget the `background` class (no note in frame) — this prevents false positives

### 3. Train the model
```bash
jupyter notebook VisionPay.ipynb
```
Run all cells. Saves `model/visionpay_model.keras` + `model/class_names.json`.

### 4. Run the web app
```bash
python app.py
# open http://localhost:5050
```

> Use **Chrome / Edge** for full Web Speech API (voice) support. Allow camera + microphone permissions.

---

## 🎮 Usage

| Action | Voice | Keyboard | Button |
|---|---|---|---|
| Detect note | "scan" / "detect" / "what note" | `SPACE` | DETECT |
| Toggle voice listening | — | `V` | Voice Assistant |
| Continuous auto-scan | "continuous" / "stop" | `C` | Continuous Mode |
| Speak total | "total" | — | — |
| Reset total | "reset" | — | Reset Total |

---

## 🧠 Model details

- **Backbone:** MobileNetV2 (ImageNet pretrained)
- **Head:** GAP → Dropout → Dense(128) → Dropout → Softmax
- **Training:** 12 epochs frozen base + 8 epochs fine-tuning last 30 layers
- **Input:** 224×224 RGB, MobileNetV2 preprocessing (`[-1, 1]`)
- **Augmentation:** flip, rotation, zoom, translation, brightness, contrast
- **Classes:** `10, 20, 50, 100, 200, 500, 2000, background`

Expected accuracy: **95%+** with ~100 well-varied images per class.

---

## 🔧 Enhancement ideas (already wired or easy to add)

- ✅ Background class to reject non-currency frames
- ✅ Top-3 predictions returned for debugging
- ✅ Confidence threshold control
- ✅ Continuous mode + session totals
- 🔜 ONNX / TFLite export for Raspberry Pi or Android
- 🔜 Counterfeit-note check via UV/IR pattern detection
- 🔜 Multi-note counting in a single frame (object detection w/ YOLO)
- 🔜 Offline desktop wrapper (PyInstaller + Tkinter)

---

## 🧑‍🦯 Designed for accessibility

- High-contrast dark UI with adjustable contrast
- All actions reachable by single key or single voice command
- Large primary button (160×160 px) with focus ring
- Audio + haptic feedback on every detection
- Bilingual support (English / Hindi)
- Works on phones (rear camera supported via Flip button)

---

## 📝 License
MIT — built for educational and assistive use.
