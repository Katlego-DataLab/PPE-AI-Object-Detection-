"""
Construction Site Safety Monitoring System
Using Computer Vision and Deep Learning (YOLOv8)

Author  : Katlego Mathebula
Project : AI-Powered PPE Detection System 
Classes : Gloves, No-Gloves, Goggles, No-Goggles,
          Helmet, No-Helmet, Mask, No-Mask,
          Safety-Vest, No-Safety-Vest

Run from project root:
    python SRC/ppe_detection.py --source Images/test.jpg
    python SRC/ppe_detection.py --source Videos/site.mp4
    python SRC/ppe_detection.py --source 0          (webcam)
"""

import cv2
import os
import datetime
import argparse
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────

MODEL_PATH    = 'Models/ppe_8class_best.pt'
OUTPUT_FOLDER = 'Output'
CONFIDENCE    = 0.40

# ── Must match your Roboflow class names EXACTLY ──────────
PPE_CLASSES = [
    'Gloves',       'No-Gloves',
    'Goggles',      'No-Goggles',
    'Helmet',       'No-Helmet',
    'Mask',         'No-Mask',
    'Safety-Vest',  'No-Safety-Vest',
]

COMPLIANT_ITEMS = ['Gloves', 'Goggles', 'Helmet', 'Mask', 'Safety-Vest']
VIOLATION_ITEMS = ['No-Gloves', 'No-Goggles', 'No-Helmet', 'No-Mask', 'No-Safety-Vest']

COLOURS = {
    'compliant'  : (0,   255,  0),    # green
    'violation'  : (0,   0,   255),   # red
    'background' : (30,  30,  30),
}


# ─────────────────────────────────────────────────────────
# 2. LOAD MODEL
# ─────────────────────────────────────────────────────────

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Train your model first using ppe_training.ipynb\n"
            f"then make sure ppe_8class_best.pt is in your Models/ folder."
        )
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)
    print(f"[INFO] Model loaded! Classes: {list(model.names.values())}")
    return model


# ─────────────────────────────────────────────────────────
# 3. DRAW DETECTIONS ON FRAME
# ─────────────────────────────────────────────────────────

def draw_detections(frame, results, model_names):
    violations = 0
    compliant  = 0

    for r in results:
        for box in r.boxes:
            class_id   = int(box.cls)
            class_name = model_names[class_id]
            confidence = float(box.conf)

            if class_name not in PPE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if class_name in VIOLATION_ITEMS:
                colour = COLOURS['violation']
                violations += 1
            else:
                colour = COLOURS['compliant']
                compliant += 1

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            # Draw label background + text
            label = f"{class_name} {confidence*100:.0f}%"
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_y = max(y1 - 10, label_size[1] + 10)
            cv2.rectangle(
                frame,
                (x1, label_y - label_size[1] - 6),
                (x1 + label_size[0] + 4, label_y + 2),
                colour, -1
            )
            cv2.putText(
                frame, label,
                (x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1
            )

    return frame, violations, compliant


# ─────────────────────────────────────────────────────────
# 4. DRAW SUMMARY BANNER
# ─────────────────────────────────────────────────────────

def draw_banner(frame, violations, compliant):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 52), (30, 30, 30), -1)
    cv2.putText(frame, "PPE Detection System — Construction Site Safety",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1)
    timestamp   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stats       = (f"Compliant: {compliant}  |  "
                   f"Violations: {violations}  |  {timestamp}")
    stats_colour = (0, 0, 255) if violations > 0 else (0, 255, 0)
    cv2.putText(frame, stats, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, stats_colour, 1)
    return frame


# ─────────────────────────────────────────────────────────
# 5. SAVE VIOLATION SCREENSHOT
# ─────────────────────────────────────────────────────────

def save_violation(frame, output_folder=OUTPUT_FOLDER):
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"VIOLATION_{timestamp}.jpg"
    filepath  = os.path.join(output_folder, filename)
    cv2.imwrite(filepath, frame)
    print(f"[ALERT] Violation screenshot saved → {filepath}")
    return filepath


# ─────────────────────────────────────────────────────────
# 6. RUN ON IMAGE
# ─────────────────────────────────────────────────────────

def run_on_image(image_path, model):
    print(f"[INFO] Processing image: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    frame   = cv2.imread(image_path)
    results = model.predict(source=frame, conf=CONFIDENCE, verbose=False)

    frame, violations, compliant = draw_detections(frame, results, model.names)
    frame = draw_banner(frame, violations, compliant)

    print("\n" + "="*50)
    print("  PPE COMPLIANCE REPORT")
    print("="*50)
    print(f"  Compliant items   : {compliant}")
    print(f"  Violations found  : {violations}")
    if violations == 0:
        print("  All PPE compliant!")
    else:
        print(f"  {violations} violation(s) detected!")
    print("="*50)

    if violations > 0:
        save_violation(frame)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    out_path = os.path.join(OUTPUT_FOLDER, f"RESULT_{os.path.basename(image_path)}")
    cv2.imwrite(out_path, frame)
    print(f"[INFO] Result saved → {out_path}")

    cv2.imshow("PPE Detection — Construction Site Safety", frame)
    print("[INFO] Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────
# 7. RUN ON VIDEO OR WEBCAM
# ─────────────────────────────────────────────────────────

def run_on_video(source, model):
    is_webcam = (source == '0' or source == 0)

    if is_webcam:
        print("[INFO] Starting webcam... Press Q to quit.")
        cap = cv2.VideoCapture(0)
    else:
        print(f"[INFO] Processing video: {source}")
        if not os.path.exists(source):
            raise FileNotFoundError(f"Video not found: {source}")
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError("Could not open video source.")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path   = os.path.join(OUTPUT_FOLDER, f"PPE_OUTPUT_{timestamp}.mp4")
    fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
    fps        = cap.get(cv2.CAP_PROP_FPS) or 25
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_count      = 0
    total_violations = 0
    last_save_time   = datetime.datetime.now()

    print("[INFO] Running... Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video.")
            break

        frame_count += 1
        results = model.predict(source=frame, conf=CONFIDENCE, verbose=False)
        frame, violations, compliant = draw_detections(frame, results, model.names)
        frame = draw_banner(frame, violations, compliant)

        if violations > 0:
            total_violations += violations
            now = datetime.datetime.now()
            if (now - last_save_time).seconds >= 10:
                save_violation(frame)
                last_save_time = now

        out_writer.write(frame)
        cv2.imshow("PPE Detection — Construction Site Safety", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Stopped by user.")
            break

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()

    print("\n" + "="*50)
    print("  SESSION SUMMARY")
    print("="*50)
    print(f"  Frames processed  : {frame_count}")
    print(f"  Total violations  : {total_violations}")
    print(f"  Output saved to   : {out_path}")
    print("="*50)


# ─────────────────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PPE Detection — Construction Site Safety"
    )
    parser.add_argument(
        '--source',
        type    = str,
        default = 'Images/test_image.jpg',
        help    = "Image path, video path, or '0' for webcam"
    )
    args  = parser.parse_args()
    model = load_model(MODEL_PATH)

    source = args.source
    if source == '0' or source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        run_on_video(source, model)
    elif source.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tiff','.webp')):
        run_on_image(source, model)
    else:
        print(f"[ERROR] Unrecognised source: {source}")
        print("Usage: --source Images/test.jpg | --source Videos/site.mp4 | --source 0")
