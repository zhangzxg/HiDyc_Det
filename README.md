# HiDyc-Det

This repository provides the implementation of HiDyc-Det: A Hierarchical Dynamic Calibration Detector for Small Objects in Aerial Scenes.

HiDyc-Det is designed for small object detection in aerial scenes. It improves the baseline YOLO detector by enhancing multi-scale contextual representation, feature fusion, and detection calibration for dense, distant, occluded, and low-illumination aerial targets.

Overview

Small objects in aerial images often suffer from limited visual details, complex backgrounds, dense distributions, scale variation, and occlusion. To address these challenges, HiDyc-Det introduces a lightweight and effective detection framework for aerial small object detection.

The main components of HiDyc-Det include:

A hierarchical context-guided feature extraction module for enhancing small object representations.
A dynamic feature fusion strategy for improving cross-scale information interaction.
A detection calibration head for improving localization and classification reliability.
A small-object-oriented detection structure suitable for aerial image scenarios.
