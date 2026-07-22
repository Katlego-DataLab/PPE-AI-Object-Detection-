"""
PPE Violation Detection
------------------------
This script combines TWO models to figure out who is missing PPE:

1. person_model  -> the stock yolov8s.pt (pretrained on COCO). It only
                     tells us where PEOPLE are in the image.
2. ppe_model     ->  best.pt. It tells us where PPE items
                     (Helmet, Gloves, Goggles, Safety-Vest, and their
                     "No-" violation versions) are in the image.

The logic:
  - Find every person box.
  - For each person, check if a Helmet/Gloves/Goggles/Safety-Vest box
    overlaps with them.
  - If a required item is NOT found overlapping that person -> flag it
    as a violation for that person.

This way we are not relying only on the weaker "No-X" classes. We are
using "was the compliant item found near this person" as the main
signal, which is more reliable given your class-level results.
"""

from ultralytics import YOLO
import os

# ---------------------------------------------------------------------
# 1. LOAD BOTH MODELS
# ---------------------------------------------------------------------

# This is the stock COCO model. Do NOT point this at best.pt or last.pt.
# It already knows about 80 everyday object classes, including "person"
# which is class id 0 in COCO.
PERSON_MODEL_PATH = "c:/Users/User/OneDrive/DecodeLabs Internship/Project 4 -PPE_dectection/Models/yolov8s.pt"

# This is the trained model
# It knows the 10 PPE classes.
PPE_MODEL_PATH = "c:/Users/User/OneDrive/DecodeLabs Internship/Project 4 -PPE_dectection/Output/ppe_10class-3/weights/best.pt"

person_model = YOLO(PERSON_MODEL_PATH)
ppe_model = YOLO(PPE_MODEL_PATH)

# ---------------------------------------------------------------------
# 2. DEFINE YOUR CLASS NAMES
# ---------------------------------------------------------------------
# These must match the order your model was trained with (check your
# data.yaml "names" list to confirm the exact order/spelling).
PPE_CLASS_NAMES = [
    "Gloves",
    "Goggles",
    "Helmet",
    "Mask",
    "No-Gloves",
    "No-Goggles",
    "No-Helmet",
    "No-Mask",
    "No-Safety-Vest",
    "Safety-Vest",
]

# The PPE items we actually want to check for on every person.
# Add/remove items here depending on what your site requires.
REQUIRED_ITEMS = ["Helmet", "Goggles", "Gloves", "Safety-Vest"]

# ---------------------------------------------------------------------
# 3. HELPER FUNCTION: DOES A PPE BOX BELONG TO A GIVEN PERSON BOX?
# ---------------------------------------------------------------------
def box_center_inside(inner_box, outer_box):
    """
    Checks whether the CENTER of inner_box falls inside outer_box.
    Boxes are in (x1, y1, x2, y2) pixel format.

    We use "center inside" instead of strict IoU overlap because a
    small Gloves box or Goggles box may only slightly overlap a large
    person box, but its center will still clearly sit inside the
    person region. This is simpler and more forgiving than IoU.
    """
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box

    cx = (ix1 + ix2) / 2
    cy = (iy1 + iy2) / 2

    return (ox1 <= cx <= ox2) and (oy1 <= cy <= oy2)


# ---------------------------------------------------------------------
# 4. MAIN FUNCTION: RUN BOTH MODELS AND BUILD VIOLATION REPORT
# ---------------------------------------------------------------------
def check_ppe_violations(image_path, person_conf=0.5, ppe_conf=0.4):
    """
    Runs both models on one image and returns a list of violations,
    one entry per person, listing which required items were missing.
    """

    # --- Step A: find all people in the image ---
    person_results = person_model.predict(
        source=image_path,
        classes=[0],          # class 0 in COCO = person
        conf=person_conf,
        verbose=False,
    )

    person_boxes = []
    for box in person_results[0].boxes:
        xyxy = box.xyxy[0].tolist()   # [x1, y1, x2, y2]
        person_boxes.append(xyxy)

    if len(person_boxes) == 0:
        print(f"No people detected in {image_path}")
        return []

    # --- Step B: find all PPE items in the same image ---
    ppe_results = ppe_model.predict(
        source=image_path,
        conf=ppe_conf,
        verbose=False,
    )

    ppe_detections = []
    for box in ppe_results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = PPE_CLASS_NAMES[cls_id]
        xyxy = box.xyxy[0].tolist()
        ppe_detections.append({"class": cls_name, "box": xyxy})

    # --- Step C: for each person, check which required items are present ---
    report = []

    for i, person_box in enumerate(person_boxes):
        found_items = set()

        for det in ppe_detections:
            if det["class"] in REQUIRED_ITEMS:
                if box_center_inside(det["box"], person_box):
                    found_items.add(det["class"])

        missing_items = [item for item in REQUIRED_ITEMS if item not in found_items]

        report.append({
            "person_id": i + 1,
            "person_box": person_box,
            "found_items": list(found_items),
            "missing_items": missing_items,
            "violation": len(missing_items) > 0,
        })

    return report


# ---------------------------------------------------------------------
# 5. RUN ON A FOLDER OF TEST IMAGES AND PRINT A SUMMARY
# ---------------------------------------------------------------------
def run_on_folder(test_dir):
    image_files = [
        f for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for filename in image_files:
        image_path = os.path.join(test_dir, filename)
        print(f"\n--- {filename} ---")

        results = check_ppe_violations(image_path)

        for person in results:
            status = "VIOLATION" if person["violation"] else "COMPLIANT"
            print(f"Person {person['person_id']}: {status}")
            if person["violation"]:
                print(f"  Missing: {', '.join(person['missing_items'])}")
            print(f"  Found:   {', '.join(person['found_items']) if person['found_items'] else 'None'}")


# ---------------------------------------------------------------------
# 6. EXAMPLE USAGE
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Testing a whole folder of images
    test_folder = r"C:\Users\User\OneDrive\DecodeLabs Internship\Project 4 -PPE_dectection\Dataset\test\images"
    if os.path.exists(test_folder):
        run_on_folder(test_folder)
    else:
        print(f"Folder not found: {test_folder}")