from ultralytics import YOLO
from pathlib import Path
import numpy as np

weights_dir = Path(__file__).parent
img_path = str(weights_dir / "image.webp")

# Run both models on same image
pt_model = YOLO(str(weights_dir / "best.pt"))
onnx_model = YOLO(str(weights_dir / "best.onnx"), task="detect")

pt_results = pt_model.predict(img_path)
onnx_results = onnx_model.predict(img_path)

# Get boxes and confidence scores
pt_boxes = pt_results[0].boxes.data.cpu().numpy()
onnx_boxes = onnx_results[0].boxes.data.cpu().numpy()

# Compare
print(f"PT boxes:   {len(pt_boxes)}")
print(f"ONNX boxes: {len(onnx_boxes)}")

if len(pt_boxes) > 0 and len(onnx_boxes) > 0:
    pt_conf = pt_boxes[:, 4]
    onnx_conf = onnx_boxes[:, 4]
    print(f"\nPT   mean conf: {pt_conf.mean():.4f}, max: {pt_conf.max():.4f}")
    print(f"ONNX mean conf: {onnx_conf.mean():.4f}, max: {onnx_conf.max():.4f}")

    # Box coordinate delta (if same number of detections)
    if len(pt_boxes) == len(onnx_boxes):
        coord_delta = np.abs(pt_boxes[:, :4] - onnx_boxes[:, :4]).mean()
        print(f"\nMean coord delta (PT vs ONNX): {coord_delta:.6f}")
    else:
        print("\nDetection count mismatch — skipping coord delta")