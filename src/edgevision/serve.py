from ultralytics import YOLO
from pathlib import Path
import bentoml
from PIL import Image
import numpy as np
from typing import Annotated


weights_dir = Path(__file__).parent

@bentoml.service
class Detection:
        def __init__(self) -> None:
            self.model = YOLO(str(weights_dir/"best.pt"))
        
        @bentoml.api
        def detect(self, frame: Annotated[Path, bentoml.validators.ContentType("image/*")]) -> dict:
              results = self.model.predict(str(frame))
              boxes = results[0].boxes.data.cpu().numpy().tolist()
              return {"detections": boxes}









