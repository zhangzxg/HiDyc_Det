# HiDyC-Det

This repository provides the official implementation of:

**HiDyC-Det: Stage-Aware Context Aggregation and Sequential Bidirectional Compensation for Aerial Small-Object Detection**

HiDyC-Det is a YOLO11n-based detector designed for aerial small-object detection. It improves small-object representation, adjacent-scale feature interaction, and prediction reliability while maintaining a relatively compact parameter count.

## Overview

Aerial small-object detection remains challenging because tiny targets contain limited visual information and are easily affected by complex backgrounds, dense distributions, occlusion, and large scale variations.

To address these issues, HiDyC-Det introduces three main components:

- **PHCA (Progressive Hierarchical Context Aggregation):** performs stage-aware context modeling through detail, local, regional, and global contextual branches.
- **BDCF (Bidirectional Dynamic Compensation Fusion):** performs sequential bidirectional compensation between adjacent-scale features to enhance cross-scale information interaction.
- **DSC-Head (Distribution-Statistics-Calibrated Detection Head):** uses lightweight shared feature refinement and regression-distribution statistics to calibrate classification prediction.

In addition, a **P2 prediction level** is introduced to retain high-resolution information for extremely small objects.

## Architecture

The overall architecture of HiDyC-Det is shown below:

[View the framework PDF](HiDyC-Det.pdf)

## Datasets

The experiments are conducted on two public aerial/tiny-object detection datasets:

- VisDrone2019
- TinyPerson

Please download the datasets from their official repositories and organize them according to the corresponding YOLO dataset format.

## Experimental Results

HiDyC-Det is evaluated on VisDrone2019 and TinyPerson.

On **VisDrone2019**, HiDyC-Det achieves:

- mAP@0.5: **30.3%**
- mAP@0.5:0.95: **17.3%**
- Parameters: **3.3M**
- GFLOPs: **13.3G**

Compared with YOLO11n, the improvements in mAP@0.5 and mAP@0.5:0.95 are **4.1** and **2.8 percentage points**, respectively.

On **TinyPerson**, HiDyC-Det improves mAP@0.5 and mAP@0.5:0.95 by **4.5** and **1.7 percentage points**, respectively.

## Requirements

The implementation is based on PyTorch and Ultralytics YOLO.

Recommended environment:

- Python 3.10
- PyTorch 2.2.2
- CUDA 12.1
- Ultralytics YOLO11

Install the required dependencies according to your local CUDA and PyTorch environment.

## Training

Training configurations follow the settings reported in the paper.

Example:

```bash
python train.py
