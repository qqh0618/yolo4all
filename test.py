"""
2024/4/6 15:47
本文件由my_ywj首次创建编写
"""
from ultralytics import YOLO

# test file for some demo
model = YOLO(r'E:\my_pro\PyQt5-YOLOv8\pt\yolov8n.pt')  # Load a custom trained model
results = model.track(source=r"E:\code-other\yolov8-upgrade\code\videos\1.mp4", show=True, tracker="deepsort.yaml")  # Tracking with ByteTrack tracker
