# PPE Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-YOLOv8-red)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-orange)
![Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-yellow)
![Model](https://img.shields.io/badge/Hugging%20Face-Model-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

A YOLOv8 based computer vision system that detects Personal Protective Equipment (PPE) compliance and violations in real time. 
## What It Does

The model detects 10 classes covering both compliant PPE usage and violations, including categories like Mask, No Mask, Gloves, No Gloves, Goggles, No Goggles, and Helmet, No Helmet, Safety Vest, No safety Vest

It runs a two model pipeline:
1. A custom trained YOLOv8 model fine tuned specifically on PPE classes.
2. A COCO pretrained model used for person detection, working alongside the custom model.

## Tools and Workflow

| Stage | Tool Used |
|---|---|
| Image labeling | Roboflow |
| Model training | Kaggle (3 GPUs, approximately 30 hours) |
| Video output review | VS Code |
| Dataset and weights hosting | Hugging Face |

## Training Results

Fine tuned for 30 epochs from an existing checkpoint:
* mAP50: 0.83
* mAP50 95: 0.549

Weakest performing classes currently are No Goggles, No Mask, and No Gloves. These are actively being improved with additional labeled data sourced from Roboflow Universe.

## Dataset and Model Weights

Due to the size of the dataset, images and trained weights are hosted on Hugging Face rather than in this repo:

* Dataset: [KatlegoMathebula/ppe_dectection](https://huggingface.co/datasets/KatlegoMathebula/ppe_dectection)
* Trained model weights: [KatlegoMathebula/ppe detection model](https://huggingface.co/KatlegoMathebula/ppe-detection-model)

This repo contains the code, training scripts, and demo, not the raw data itself.

## Live Demo

A live webcam demo script (cv2 + trained_model.predict) is included for showcasing real time detection locally.

## Challenges and How They Were Solved

Building and hosting this project came with a few real obstacles along the way:

**Dataset size versus GitHub limits**
The labeled dataset contains over 5,600 image and label files, well beyond what GitHub's web uploader can handle at once. The solution was to host the full dataset and trained weights on Hugging Face instead, using its dataset and model repository types, and linking to them from this README rather than committing the raw files to GitHub.

**Outdated library and command line tooling**
Early attempts to use the Hugging Face command line tool failed because the installed version of huggingface_hub was outdated and its executable was not recognized by the system PATH. This was resolved by upgrading the package directly through the active Python interpreter and, when the command line tool still could not be found, calling the Hugging Face API directly through Python instead.

**Locating files across a deeply nested project structure**
With separate folders for the dataset, images, models, notebooks, and source code, tracking down the exact file paths needed for upload took some trial and error. Copying the exact path directly from the file explorer for each file resolved this rather than guessing folder structure.

**Authentication issues during upload**
Generating and using a Hugging Face access token took a few attempts due to invalid tokens and a locked account. This was solved by generating a fresh write access token and passing it directly into the login step rather than relying on the interactive command line prompt.

## Repo Structure

```
├── train/              Training scripts
├── demo/               Live webcam demo
├── notebooks/          Kaggle training notebooks
└── README.md
```

## Getting Started

1. Clone this repository.
2. Download the dataset and model weights from the Hugging Face links above.
3. Install the required dependencies.
4. Run the demo script to test live webcam detection locally.

Built with 🤍 by Katlego Mathebula
