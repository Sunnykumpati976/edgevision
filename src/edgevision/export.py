from ultralytics import YOLO

weights = Path(__file__).parent / "best.pt"
model = YOLO(str(weights))
model.export(format="onnx")
