📌 Case Study 2 – Object Detection and Multi-Object Tracking (MOT17)
📖 Overview

This project focuses on understanding how deep learning–based object detection and multi-object tracking perform in real-world video sequences.

The goal is not only to achieve high metric scores, but to analyze how different detector–tracker combinations behave under varying scene conditions such as crowd density, occlusion, and motion complexity.

The experiments were conducted on selected sequences from the MOT17 dataset, and performance was evaluated using official multi-object tracking metrics.

🎯 Objectives

Compare different detector and tracker combinations

Analyze how detector quality affects MOTA

Analyze how tracker choice affects IDF1

Fine-tune tracking parameters to reduce ID switches

Perform graphical analysis using bubble charts

🧠 Models and Tracking Algorithms
🔍 Detectors

YOLOv8

YOLOv11

🎯 Trackers

ByteTrack

BotSORT

By combining each detector with each tracker, four experimental pipelines were created:

YOLOv8 + ByteTrack

YOLOv8 + BotSORT

YOLOv11 + ByteTrack

YOLOv11 + BotSORT

⚙️ Custom Tracker Configuration

Instead of using default tracker settings, custom YAML configuration files were created.

BotSORT Key Adjustments

track_high_thresh: 0.4

track_low_thresh: 0.1

match_thresh: 0.85

track_buffer: 60

ReID disabled

Stricter matching was used to reduce ID switches and improve identity consistency.

ByteTrack Adjustments

track_high_thresh: 0.25

match_thresh: 0.8

track_buffer: 60

These settings allowed better detection recovery in crowded scenes.

📊 Evaluation Metrics

Performance was evaluated using:

MOTA (Multi-Object Tracking Accuracy)
→ Measures overall tracking accuracy (false positives, missed detections, ID switches)

IDF1
→ Measures identity consistency over time

Bubble charts were created to visualize the relationship between MOTA and IDF1 across different sequences.

🎥 Dataset

The following MOT17 sequences were used:

MOT17-02

MOT17-04

MOT17-05

MOT17-09

MOT17-10

MOT17-11

MOT17-13

These sequences vary in:

Crowd density

Occlusion levels

Scene complexity

Camera motion

📈 Key Observations

YOLOv11 consistently achieves higher MOTA, indicating improved detection quality.

BotSORT generally achieves higher IDF1, showing better identity consistency.

Scene difficulty significantly affects both metrics.

The best overall balance is often achieved with YOLOv11 + BotSORT, though performance varies by sequence.

Improving detection accuracy does not automatically guarantee improved identity consistency.

🛠️ Implementation Details

Framework: Ultralytics

Language: Python

Environment: Google Colab (GPU)

Metrics: MOT evaluation toolkit

Visualization: Matplotlib bubble charts
