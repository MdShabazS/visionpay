"""
VisionPay - Real-Time Currency Detection
========================================
Headless real-time detector. Uses the locally trained MobileNetV2 model
(model/visionpay_model.keras) for ~10-15 FPS inference, fully offline.

Run:
    python realtime.py

Controls (focus on the camera window):
    q  -> quit
    r  -> reset session total
    s  -> speak current total
    a  -> toggle auto-count (every new stable detection is added to total)
    m  -> toggle mute (voice announcements on/off)
    +/-> adjust confidence threshold

Features:
    - Frame-by-frame inference, FPS shown on overlay
    - Temporal smoothing (majority vote over last N frames) -> no jitter
    - Speaks denomination only on STABLE NEW detections (no chatter)
    - Per-frame confidence + margin shown
    - Session total + per-denomination breakdown
    - Background class explicitly suppressed
"""

import os
import json
import time
import subprocess
import platform
from collections import deque, Counter

import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
TFLITE_PATH = os.path.join(MODEL_DIR, "visionpay_model.tflite")
KERAS_PATH = os.path.join(MODEL_DIR, "visionpay_model.keras")
CLASSES_PATH = os.path.join(MODEL_DIR, "class_names.json")
IMG_SIZE = 224

# Tunables
SMOOTH_WINDOW = 10          # frames to consider for majority vote
STABLE_VOTES = 6            # need >= this many matching votes to commit
CONF_THRESHOLD = 0.65       # min top-1 softmax to count as a vote
MARGIN_THRESHOLD = 0.20     # min (top1 - top2) to count as a vote
SPEAK_COOLDOWN_S = 1.5      # min seconds between announcements of the SAME note
RESET_AFTER_S = 1.2         # seconds of no-note before allowing same note to re-announce


# ----------------- TTS (cross-platform, non-blocking) -----------------
class Speaker:
    def __init__(self):
        self.muted = False
        self.proc = None
        self.system = platform.system()

    def say(self, text):
        if self.muted:
            return
        # kill any in-flight speech so we never queue up
        if self.proc and self.proc.poll() is None:
            try: self.proc.terminate()
            except Exception: pass
        try:
            if self.system == "Darwin":
                self.proc = subprocess.Popen(["say", text])
            elif self.system == "Linux":
                self.proc = subprocess.Popen(["espeak", text])
            else:  # Windows fallback
                self.proc = subprocess.Popen(
                    ["powershell", "-Command",
                     f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"])
        except Exception as e:
            print(f"[tts] {e}")


# ----------------- Model -----------------
class TFLiteWrapper:
    """Make a TFLite interpreter behave like a keras model for our use."""
    def __init__(self, path):
        import tensorflow as tf
        self.interp = tf.lite.Interpreter(model_path=path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

    def predict(self, x, verbose=0):
        # x: (1, 224, 224, 3) float32 already preprocessed
        self.interp.set_tensor(self.inp["index"], x.astype(np.float32))
        self.interp.invoke()
        return self.interp.get_tensor(self.out["index"])


def load_classifier():
    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(
            f"class_names.json not found in {MODEL_DIR}. Train the model first."
        )
    with open(CLASSES_PATH) as f:
        classes = json.load(f)

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    if os.path.exists(TFLITE_PATH):
        print("[init] loading TFLite model (fast)...")
        model = TFLiteWrapper(TFLITE_PATH)
    elif os.path.exists(KERAS_PATH):
        print("[init] loading Keras model (no TFLite found)...")
        from tensorflow.keras.models import load_model
        model = load_model(KERAS_PATH)
    else:
        raise FileNotFoundError(
            f"No trained model found in {MODEL_DIR}. "
            f"Run VisionPay.ipynb to train."
        )

    print(f"[init] classes: {classes}")
    return model, classes, preprocess_input


def preprocess_frame(frame, preprocess_fn):
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = frame[y0:y0 + side, x0:x0 + side]
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.expand_dims(preprocess_fn(crop), 0), (x0, y0, side)


# ----------------- Main loop -----------------
def main():
    model, classes, preprocess_fn = load_classifier()
    speaker = Speaker()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    votes = deque(maxlen=SMOOTH_WINDOW)
    last_announced = None
    last_announce_t = 0.0
    last_seen_t = 0.0
    total = 0
    count = 0
    breakdown = {}
    auto_count = False
    conf_thr = CONF_THRESHOLD

    fps_t = time.time(); fps_n = 0; fps = 0.0

    print("[run] real-time detection started. q=quit  r=reset  s=total  a=autocount  m=mute")
    speaker.say("Vision Pay real time mode ready.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        x, (x0, y0, side) = preprocess_frame(frame, preprocess_fn)
        preds = model.predict(x, verbose=0)[0]

        order = np.argsort(preds)[::-1]
        top_idx = int(order[0])
        top_conf = float(preds[top_idx])
        margin = float(preds[order[0]] - preds[order[1]])
        label = classes[top_idx]

        # Vote (ignore uncertain frames)
        if top_conf >= conf_thr and margin >= MARGIN_THRESHOLD:
            votes.append(label)
        else:
            votes.append("__low__")

        smoothed, votes_count = Counter(votes).most_common(1)[0]
        now = time.time()

        # Stable detection => maybe speak / count
        committed = None
        if (smoothed not in ("__low__", "background")
                and votes_count >= STABLE_VOTES):
            committed = smoothed
            last_seen_t = now

            should_speak = (
                committed != last_announced
                or (now - last_announce_t) > SPEAK_COOLDOWN_S
            )
            if should_speak and committed != last_announced:
                speaker.say(f"{committed} rupees")
                last_announce_t = now
                last_announced = committed
                if auto_count:
                    value = int(committed)
                    total += value
                    count += 1
                    breakdown[value] = breakdown.get(value, 0) + 1

        # If no note has been seen for a while, allow the same note to be re-announced
        if (now - last_seen_t) > RESET_AFTER_S:
            last_announced = None

        # FPS
        fps_n += 1
        if fps_n >= 10:
            fps = fps_n / (time.time() - fps_t)
            fps_t = time.time(); fps_n = 0

        # ---------- overlay ----------
        h, w = frame.shape[:2]
        # capture region box
        cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), (0, 255, 0), 2)

        # top bar
        cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
        if committed:
            big = f"Rs. {committed}"
            color = (60, 220, 255)
        elif smoothed == "background":
            big = "No note in frame"
            color = (180, 180, 180)
        else:
            big = "Analyzing..."
            color = (180, 180, 180)
        cv2.putText(frame, big, (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        info = (f"raw={label} {top_conf*100:.0f}%  margin={margin*100:.0f}%  "
                f"votes={votes_count}/{SMOOTH_WINDOW}  fps={fps:.1f}  thr={conf_thr*100:.0f}%")
        cv2.putText(frame, info, (18, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        # bottom bar
        cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
        bd_str = "  ".join(f"Rs{k}x{v}" for k, v in sorted(breakdown.items(), reverse=True))
        line1 = f"Total: Rs.{total}   Notes: {count}   AutoCount:{'ON' if auto_count else 'OFF'}   Mute:{'ON' if speaker.muted else 'OFF'}"
        cv2.putText(frame, line1, (18, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1)
        cv2.putText(frame, "[q]uit  [r]eset  [s]peak-total  [a]uto-count  [m]ute  [+/-]thr",
                    (18, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if bd_str:
            cv2.putText(frame, bd_str, (w - 360, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)

        cv2.imshow("VisionPay - Real-Time", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            total = 0; count = 0; breakdown.clear()
            speaker.say("Total cleared")
        elif key == ord('s'):
            speaker.say(f"Total {total} rupees from {count} notes" if count else "No notes counted yet")
        elif key == ord('a'):
            auto_count = not auto_count
            speaker.say("Auto count on" if auto_count else "Auto count off")
        elif key == ord('m'):
            speaker.muted = not speaker.muted
            print(f"[ctrl] muted={speaker.muted}")
        elif key in (ord('+'), ord('=')):
            conf_thr = min(0.95, conf_thr + 0.05)
        elif key in (ord('-'), ord('_')):
            conf_thr = max(0.30, conf_thr - 0.05)

    cap.release()
    cv2.destroyAllWindows()
    speaker.say("Vision Pay closed")
    print(f"\n[done] Final total: Rs.{total} from {count} notes")
    for k, v in sorted(breakdown.items(), reverse=True):
        print(f"   Rs{k} x {v}")


if __name__ == "__main__":
    main()
