from ultralytics import YOLO
from pathlib import Path
import bentoml
from typing import Annotated
import time
from prometheus_client import Counter, Histogram


weights_dir = Path(__file__).parent
REQUEST_COUNT= Counter("request_count", "total detection requests")
LATENCY = Histogram("inference_latency_ms", "Inference latency in ms", buckets =[5,10,25,50,100,150])
CONFIDENCE = Histogram("detection_confidence", "Confidence scores", buckets=[0.3, 0.5, 0.7, 0.8, 0.9, 1.0])

@bentoml.service
class Detection:
        def __init__(self) -> None:
            self.model = YOLO(str(weights_dir/"best.pt"))
        
        @bentoml.api
        def detect(self, frame: Annotated[Path, bentoml.validators.ContentType("image/*")]) -> dict:
              start = time.time()
              results = self.model.predict(str(frame))
              end = time.time()
              latency = (end - start) * 1000
              REQUEST_COUNT.inc()
              LATENCY.observe(latency)
              boxes = results[0].boxes.data.cpu().numpy().tolist()
              for box in boxes:
                    CONFIDENCE.observe(box[4])
              return {"detections": boxes}









