from ultralytics import YOLO
from pathlib import Path

weights = Path(__file__).parent / "best.pt"
model = YOLO(str(weights))
model.export(format="onnx")
