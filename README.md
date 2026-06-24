# Adaptive Bias-Resistant Eye Aspect Ratio (EAR) based Driver Drowsiness Detection System
A real-time AI and Computer Vision based safety application built with Python to detect driver fatigue and prevent accidents.

## About the Project
This project is an intelligent, real-time driver safety system designed to track and calculate the **Eye Aspect Ratio (EAR)** using live camera feeds. 

Unlike standard drowsiness detectors, this system features an **Adaptive Threshold Mechanism** that dynamically adjusts to environmental lighting changes and structural facial variations. This prevents false alarms caused by shadows or different facial structures, ensuring highly accurate microsleep detection and immediate alerting.

## Key Features
* **Real-Time Facial Landmark Tracking:** Powered by MediaPipe Face Mesh for lightweight and high-speed landmark localization.
* **Adaptive EAR Calculation:** Dynamic baseline shifting to resist ambient lighting bias.
* **Multi-Threading Architecture:** Audio alerts and camera processing run on separate threads to prevent UI/frame lagging.
* **Optimized Performance:** Built using NumPy arrays for ultra-fast algebraic face coordinate calculations.

## Setup Instructions

### 1. Install Dependencies
Make sure you have Python installed, then run the following command in your terminal to install the required libraries:
```bash
pip install opencv-python mediapipe numpy

### 2. Run the Application
Execute the main Python script to initialize the camera feed and start tracking:
```bash
python Drowsiness_Detection.py
