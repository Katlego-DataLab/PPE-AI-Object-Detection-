from ultralytics import YOLO
import cv2

trained_model = YOLO(r"C:\Users\User\OneDrive\DecodeLabs Internship\Project 4 -PPE_dectection\best.pt")

PPE_CLASSES = ['Gloves', 'Goggles', 'Helmet', 'Mask', 'No-Gloves',
               'No-Goggles', 'No-Helmet', 'No-Mask', 'No-Safety-Vest', 'Safety-Vest']
COMPLIANT = {'Gloves', 'Goggles', 'Helmet', 'Mask', 'Safety-Vest'}
VIOLATION = {'No-Gloves', 'No-Goggles', 'No-Helmet', 'No-Mask', 'No-Safety-Vest'}

# Config
SOURCE = 0  # 0 = webcam | or swap to a video path, e.g. 'Videos/site.mp4'
CONF   = 0.40

COLOUR_MAP = {
    cls: (0, 255, 0) if cls in COMPLIANT else (0, 0, 255)
    for cls in PPE_CLASSES
}

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print("ERROR: could not open webcam. Check that no other app (Zoom/Teams/Camera app) is holding it.")
else:
    print('Live PPE detection running. Press Q to quit')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results    = trained_model.predict(frame, conf=CONF, verbose=False)[0]
        violations = 0

        for box in results.boxes:
            cls_name = PPE_CLASSES[int(box.cls)]
            conf_val = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            colour = COLOUR_MAP.get(cls_name, (255, 255, 255))

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            label = f'{cls_name} {conf_val*100:.0f}%'
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

            if cls_name in VIOLATION:
                violations += 1

        status_text   = f'VIOLATIONS: {violations}' if violations > 0 else 'ALL PPE COMPLIANT'
        status_colour = (0, 0, 255) if violations > 0 else (0, 255, 0)
        cv2.putText(frame, status_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_colour, 3)

        cv2.imshow('PPE Detection — Live', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Detection stopped.')