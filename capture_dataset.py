"""
VisionPay - Dataset Capture Tool
================================
Capture webcam images for each Indian currency denomination to build
your own training dataset.

Usage:
    python capture_dataset.py

Controls (per session):
    SPACE  -> save current frame to current class folder
    n      -> next denomination
    p      -> previous denomination
    a      -> toggle auto-capture (saves 1 frame every 0.4s)
    q      -> quit

Recommendation: capture 80-150 images per denomination, varying:
    - lighting (bright, dim, shadow)
    - angle (flat, tilted, rotated)
    - distance (near, far)
    - backgrounds
    - front and back of note
"""
import os
import time
import cv2

DENOMINATIONS = ["10", "20", "50", "100", "200", "500", "2000", "background"]
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
IMG_SIZE = 224


def ensure_dirs():
    for d in DENOMINATIONS:
        os.makedirs(os.path.join(DATASET_DIR, d), exist_ok=True)


def count_images(label):
    p = os.path.join(DATASET_DIR, label)
    return len([f for f in os.listdir(p) if f.lower().endswith((".jpg", ".png"))])


def save_frame(frame, label):
    folder = os.path.join(DATASET_DIR, label)
    idx = count_images(label)
    fname = f"{label}_{idx:04d}_{int(time.time()*1000)}.jpg"
    path = os.path.join(folder, fname)
    # Save resized 224x224 center crop for consistency
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = frame[y0:y0 + side, x0:x0 + side]
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    cv2.imwrite(path, crop)
    return path


def overlay(frame, label, count, auto):
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    # Capture region box
    cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), (0, 255, 0), 2)

    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.6, frame, 0.4, 0, frame)

    txt1 = f"Class: {label}   Saved: {count}"
    txt2 = f"[SPACE] save  [n] next  [p] prev  [a] auto:{ 'ON' if auto else 'OFF' }  [q] quit"
    cv2.putText(frame, txt1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, txt2, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame


def main():
    ensure_dirs()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    idx = 0
    auto = False
    last_auto = 0.0

    print("VisionPay dataset capture started.")
    print("Classes:", DENOMINATIONS)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        label = DENOMINATIONS[idx]
        cnt = count_images(label)

        display = overlay(frame.copy(), label, cnt, auto)
        cv2.imshow("VisionPay - Dataset Capture", display)

        now = time.time()
        if auto and now - last_auto > 0.4:
            save_frame(frame, label)
            last_auto = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            p = save_frame(frame, label)
            print("saved:", p)
        elif key == ord('n'):
            idx = (idx + 1) % len(DENOMINATIONS)
        elif key == ord('p'):
            idx = (idx - 1) % len(DENOMINATIONS)
        elif key == ord('a'):
            auto = not auto

    cap.release()
    cv2.destroyAllWindows()
    print("\nDataset summary:")
    for d in DENOMINATIONS:
        print(f"  {d:>10}: {count_images(d)} images")


if __name__ == "__main__":
    main()
