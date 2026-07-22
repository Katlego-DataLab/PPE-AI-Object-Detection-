# PPE AI Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-YOLOv8-red)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-orange)
![Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-yellow)
![Model](https://img.shields.io/badge/Hugging%20Face-Model-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

> Note: The full labeled dataset and trained model weights are not stored in this GitHub repository, since GitHub's uploader only handles about a hundred files at a time and this project has thousands. Both are hosted on Hugging Face instead:
> * Dataset: [KatlegoMathebula/ppe_dectection](https://huggingface.co/datasets/KatlegoMathebula/ppe_dectection)
> * Model weights: [KatlegoMathebula/ppe detection model](https://huggingface.co/KatlegoMathebula/ppe-detection-model)

A YOLOv8 based computer vision system that detects Personal Protective Equipment (PPE) compliance and violations in real time. This is a personal project.

## What It Does

The model detects 10 classes covering both compliant PPE usage and violations: Gloves, No Gloves, Goggles, No Goggles, Helmet, No Helmet, Mask, No Mask, Safety Vest, and No Safety Vest.

It runs a two model pipeline:
1. A custom trained YOLOv8 model fine tuned specifically on PPE classes.
2. A COCO pretrained YOLOv8 model (`yolov8s.pt`) used for person detection, working alongside the custom model to check which required items are present on each detected person.

## Tools and Workflow

| Stage | Tool Used |
|---|---|
| Image labeling | [Roboflow](https://roboflow.com) |
| Model training | Kaggle (3 GPUs, approximately 30 hours) |
| Video output review | VS Code |
| Dataset and weights hosting | Hugging Face |

## Training Results

The model was trained over 30 epochs on Kaggle, reaching:
* mAP50: 0.83
* mAP50 95: 0.549

Weakest performing classes currently are No Goggles, No Mask, and No Gloves. These are actively being improved with additional labeled data sourced from Roboflow Universe.

## Challenges and How They Were Solved

**Labeling the dataset by hand**
Every image used to train the model was labeled manually on Roboflow, one image at a time, across all 10 PPE classes. This was one of the most time consuming parts of the project, since there was no shortcut around labeling each item individually.

**Finding the right training platform**
Training was first attempted on Google Colab. A single run took around four days to complete, only to find that the changes made to the training setup had not actually improved the model. After that experience, training was moved to Kaggle, which turned out to be a far more effective platform, running the same kind of work on 3 GPUs in about 30 hours instead.

**Debugging throughout training and inference**
Various debugging issues came up during both training and getting the inference and demo scripts working correctly, which took ongoing troubleshooting to resolve.

**Hosting a dataset too large for GitHub**
With over 5,600 image and label files, the dataset was far beyond what GitHub's uploader could handle. The solution was to host the dataset and trained model weights on Hugging Face instead, linking to both from this README.

## Repo Structure

```
├── notebooks/
│   └── ppe_training.ipynb        Training notebook
├── src/
│   ├── ppe_detection.py          Runs inference on an image, video, or webcam
│   ├── ppe_violation_detection.py  Two model pipeline: person detection + PPE check per person
│   └── BestPath.py               Simple live webcam demo
└── README.md
```

The dataset and model weights are not included here; see the Hugging Face links above.

## Live Demo

`BestPath.py` runs a live webcam demo using OpenCV and the trained model, drawing bounding boxes and flagging violations frame by frame.

`ppe_detection.py` supports images, video files, or webcam input:

```
python src/ppe_detection.py --source Images/test.jpg
python src/ppe_detection.py --source Videos/site.mp4
python src/ppe_detection.py --source 0
```

`ppe_violation_detection.py` combines the person detection model with the PPE model to check, per person, which required items are missing rather than relying only on the weaker "No item" classes directly.

## Getting Started

1. Clone this repository.
2. Download the dataset and model weights from the Hugging Face links above.
3. Install the required dependencies (`ultralytics`, `opencv-python`).
4. Update the model and dataset paths in the scripts to match where you saved them.
5. Run one of the scripts above to test detection locally.

Built with 🤍 by Katlego Mathebula
