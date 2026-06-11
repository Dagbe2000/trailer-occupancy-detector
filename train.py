"""
YOLOv8-nano classification training for Trailer Occupancy Detector.

Dataset hosted on Roboflow. Set ROBOFLOW_API_KEY in your environment
or create a .env file (see .env.example).

Usage:
    python train.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

from roboflow import Roboflow
from ultralytics import YOLO

API_KEY      = os.environ.get("ROBOFLOW_API_KEY")
WORKSPACE    = os.environ.get("ROBOFLOW_WORKSPACE", "wilfrids-workspace-8qcnd")
PROJECT_NAME = os.environ.get("ROBOFLOW_PROJECT",   "trailer-occupancy-detector")
VERSION      = int(os.environ.get("ROBOFLOW_VERSION", "1"))

if not API_KEY:
    raise EnvironmentError(
        "ROBOFLOW_API_KEY not set. Copy .env.example to .env and fill in your key."
    )

# Download dataset
print("Downloading dataset from Roboflow...")
rf      = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT_NAME)
version = project.version(VERSION)
dataset = version.download("folder")  # classification format

print(f"Dataset downloaded to: {dataset.location}")

# Train YOLOv8-nano classification model
print("Starting training...")
model = YOLO("yolov8n-cls.pt")

results = model.train(
    data=dataset.location,
    epochs=50,
    imgsz=320,
    batch=16,
    name="occupancy_detector",
    patience=10,
    save=True,
    plots=True,
)

print("\n✅ Training complete!")
print(f"Best model saved to: runs/classify/occupancy_detector/weights/best.pt")

metrics = model.val()
print(f"\nTop-1 Accuracy : {metrics.top1:.3f}")
print(f"Top-5 Accuracy : {metrics.top5:.3f}")
