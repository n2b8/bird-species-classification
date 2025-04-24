# Backyard Bird Classification & Detection (FeederWatch Automation)

![Feeder Bird](images/example_bird.jpg)

*A species classification system for backyard feeder birds.*

---

## Table of Contents
- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Data Sources](#data-sources)
  - [Dataset Details](#dataset-details)
- [Methodology](#methodology)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Modeling](#2-modeling)
  - [3. Evaluation & Interpretability](#3-evaluation--interpretability)
- [Deployment](#deployment)
- [Recommendations](#recommendations)
- [Future Work](#future-work)
- [File Structure](#file-structure)
- [Requirements](#requirements)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

This project focuses on classifying bird species from backyard feeder images using a fine-tuned deep learning model. The goal is to support citizen science efforts like **[FeederWatch](https://feederwatch.org/)** by automating species logging and improving observation accuracy. A real-time version of this pipeline runs on a Raspberry Pi 5 with a Hailo-8 AI accelerator, enabling live detection, classification, and review.

---

## Objectives

- **Primary Goal**: Classify common North American feeder birds using real-time inference.
- **Supporting Goals**:
  1. Build an accurate and efficient classifier trained on fine-grained bird species.
  2. Use interpretability tools to explain model decisions.
  3. Streamline species logging for the FeederWatch program.

---

## Data Sources

The classifier was trained using a filtered and curated subset of the NABirds dataset. The subset was chosen based on the [Cornell University Common Feeder Birds List](https://feederwatch.org/learn/common-feeder-birds/)

### Dataset Details

- **Source**: [NABirds dataset](https://dl.allaboutbirds.org/nabirds)
- **Subset Source**: [Backyard Feeder Birds](https://www.kaggle.com/datasets/jakemccaig/backyard-feeder-birds-nabirds-subset)  
- **Subset**: 134 species selected based on frequency at North American feeders
- **Image Count**: 13,252 total images across all classes  
- **Image Size**: Resized to 600×600 for model training  
- **Augmentations**: Color jitter, crop, rotation, horizontal flip  

---

## Methodology

### 1. Data Preparation
- Resized and standardized all input images to 600×600 resolution.
- Applied image augmentations to simulate natural variability.
- Used stratified 80/20 train-validation split.

### 2. Modeling
- **Baseline**: Custom CNNs trained from scratch to establish a performance floor.
- **Final Model**: EfficientNet-B7 with transfer learning (ImageNet pretrained).
- **Export Format**: ONNX, optimized for Raspberry Pi inference with ONNX Runtime.
- **Training Pipeline**:
  - ~~Weighted sampling for class balance~~
  - Early stopping based on validation accuracy
  - Mixed precision for faster training

### 3. Evaluation & Interpretability

- **Validation Accuracy**: 91.82%  
- **Macro F1 Score**: 0.913  

**Grad-CAM**

![Grad-CAM](images/grad-cam.png)

- Correct predictions: Model focused on full-body features like wings, tail, and chest.
- Misclassifications: Attention was often limited to the head or misdirected to the background.

**LIME**

![LIME](images/lime.png)

- Correct predictions: Highlighted regions aligned well with key bird anatomy.
- Misclassifications: Emphasis was placed on irrelevant parts of the image — branches, shadows, or cluttered backgrounds.

## Deployment

The final model was deployed as part of a complete AI-powered birdwatching system on a Raspberry Pi 5 with a Hailo-8 AI accelerator. The system runs continuously and processes an RTSP camera feed in real time.

The full deployment code is a work in progress and is available here:  
**[github.com/n2b8/birdwatcher](https://github.com/n2b8/birdwatcher)**

### System Architecture
- **Detection**: YOLOv8 (Hailo-optimized) runs in real time to detect birds.
- **Classification**: EfficientNet-B7 ONNX model classifies detected frames.
- **Web Dashboard**: Flask app with image logs, species charts, and a review interface.
- **Database**: SQLite stores all visit metadata and prediction confidence.
- **Notifications**: Telegram bot alerts for high-confidence visits.
- **Backups**: Daily image and database syncs to MinIO (S3-compatible).
- **Service Management**: All core components run as `systemd` services.

This system has already made my FeederWatch logging much easier and more consistent, automatically recording species, timestamps, and confidence scores.

---

## Recommendations

Based on evaluation and deployment testing, the following steps could improve system accuracy and flexibility:

1. **Add More Real-World Variability**:
   - Include birds in poor lighting, partial occlusion, and motion blur.
   - Expand to more feeder environments (urban, wooded, snowy, etc.).

2. **Enhance Classification Pipeline**:
   - Incorporate multi-frame or multi-angle classification support.
   - Explore attention-enhanced architectures for better localization.

3. **Deploy Beyond Backyard**:
   - Pilot in community parks or schools to increase usage and feedback.
   - Integrate with broader FeederWatch or iNaturalist APIs for validation.

---

## Future Work

- Add support for multi-bird classification per frame.
- Retrain on seasonal and region-specific species groups.
- Build a mobile or web app for tagging and reporting visits.
- Publicly host the model on Hugging Face or Kaggle for reproducibility.

---

## File Structure

- **[index.ipynb](index.ipynb)**: Full analysis notebook following CRISP-DM

---

## Requirements

- **Python 3.10+**
- **Core Libraries**:
  - numpy, pandas, matplotlib, seaborn
  - torch, torchvision
  - onnxruntime

---

## Acknowledgments

- Dataset: [NABirds](https://dl.allaboutbirds.org/nabirds)
- Deployment: Raspberry Pi 5 + Hailo-8 AI Hat
- Citizen science inspiration: [FeederWatch](https://feederwatch.org/)