import os
import shutil

from roboflow import Roboflow
from ultralytics import YOLO

# This gets the dataset for the model yolov5
rf = Roboflow(api_key="rpgbdYo5dz77E0IPKL7O")
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(1)
dataset = version.download("yolov5")

# This moves the files to the expected directories by the model
if os.path.exists("football-players-detection-1/train"):
    shutil.move(
        "football-players-detection-1/train/",
        "football-players-detection-1/football-players-detection-1/train",
    )
    shutil.move(
        "football-players-detection-1/test/",
        "football-players-detection-1/football-players-detection-1/test",
    )
    shutil.move(
        "football-players-detection-1/valid/",
        "football-players-detection-1/football-players-detection-1/valid",
    )
else:
    print("Files already moved")

# Runs the behaviour for the model on the command line
model = YOLO("yolov5mu.pt")
results = model.train(
    data=f"{dataset.location}/data.yaml", epochs=100, imgsz=640, batch=-1
)
