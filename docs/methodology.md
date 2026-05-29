# Methodology

## Overview

This project follows a simple real-time computer vision workflow for facial emotion recognition. The system captures live video from a webcam, detects faces, predicts emotions, and displays the result on the screen.

## Step-by-Step Process

1. Webcam input is captured using OpenCV.
2. Video is processed frame by frame.
3. The system detects human faces from each frame.
4. The detected face is passed to the emotion recognition model.
5. The model predicts emotion scores for different emotions.
6. The emotion with the highest confidence score is selected.
7. The final emotion label and confidence percentage are displayed on the screen.

## Workflow

Webcam Input
↓
Frame Capture
↓
Face Detection
↓
Emotion Recognition
↓
Confidence Score Calculation
↓
Real-Time Output Display

## Emotion Classes

The system detects seven emotion classes:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

## Tools Used

* Python for programming
* OpenCV for webcam access and frame processing
* FER library for emotion detection
* TensorFlow as the deep learning backend

## Summary

The methodology is simple, lightweight, and suitable for a low-configuration laptop. Since the system uses a pre-trained model, it does not require heavy model training on the local machine.
