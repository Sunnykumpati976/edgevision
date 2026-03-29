# EdgeVision — Aerial Object Detection MLOps Pipeline

End-to-end production pipeline for aerial object detection: YOLOv8 trained on VisDrone → ONNX export → TensorRT INT8 quantization (15x speedup) → BentoML serving → Prometheus/Grafana monitoring → GitHub Actions CI/CD. Built for real-time inference on edge hardware.

---

## Benchmarks

All benchmarks measured on NVIDIA Tesla T4 (15GB VRAM), input resolution 640×640.

| Runtime | Inference | Speedup | Model Size | mAP@50 |
|---------|-----------|---------|------------|--------|
| PyTorch FP32 | 57.0 ms | 1x | 21.5 MB | 0.377 |
| ONNX Runtime | 17.3 ms | 3.3x | 42.7 MB | 0.377 |
| TensorRT FP16 | 5.5 ms | 10.4x | 23.1 MB | ~0.375 |
| **TensorRT INT8** | **3.8 ms** | **15x** | **13.4 MB** | **~0.370** |

TensorRT INT8 achieves **263 FPS** with less than 2% mAP degradation from the original PyTorch model.

---

## What This Project Demonstrates

This is not a model training project — it is an **MLOps and deployment** project. The model (YOLOv8s fine-tuned on VisDrone) is the starting point. Everything that follows is what production ML systems require and what most portfolio projects skip:

- **Model optimization:** ONNX export for portability, TensorRT compilation with FP16 and INT8 quantization for edge-grade inference speed
- **Model serving:** REST API via BentoML — send an image, receive JSON detections
- **Production monitoring:** Prometheus metrics (latency histograms, throughput counters, confidence distribution tracking) with Grafana dashboards
- **CI/CD:** GitHub Actions pipeline — lint, test, and validate on every push
- **Containerization:** Full stack launches with one command via Docker Compose

---

## Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Model | YOLOv8s (Ultralytics) | Real-time object detection |
| Dataset | VisDrone-DET | 10k+ aerial drone images, 10 object classes |
| Export | ONNX | Framework-agnostic model format |
| Optimization | TensorRT (FP16/INT8) | GPU-optimized inference compilation |
| Serving | BentoML | Production model serving with REST API |
| Monitoring | Prometheus + Grafana | Real-time metrics collection and visualization |
| CI/CD | GitHub Actions | Automated linting, testing, validation |
| Containerization | Docker + Docker Compose | Full stack orchestration |
| Language | Python 3.12 | Primary language |

---

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- NVIDIA GPU + drivers (for TensorRT inference)
- `best.pt` model weights in `src/edgevision/`

### Launch the full stack
```bash
git clone https://github.com/YOUR_USERNAME/edgevision.git
cd edgevision
docker-compose up --build
```

This starts three services:

| Service | URL | Purpose |
|---------|-----|---------|
| BentoML API | http://localhost:3000 | Model serving + Swagger UI |
| Prometheus | http://localhost:9090 | Metrics collection |
| Grafana | http://localhost:3001 | Monitoring dashboards |

### Send a detection request
```bash
curl -X POST http://localhost:3000/detect \
  -F "frame=@your_drone_image.jpg"
```

### Response
```json
{
  "detections": [
    [412, 206, 448, 228, 0.94, 0],
    [88, 178, 100, 196, 0.82, 1]
  ]
}
```

Each detection is `[x1, y1, x2, y2, confidence, class_id]` — bounding box coordinates, model confidence, and object class.

---

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  VisDrone    │────▶│  YOLOv8s    │────▶│  best.pt        │
│  Dataset     │     │  Training   │     │  (21.5 MB)      │
└─────────────┘     └─────────────┘     └────────┬────────┘
                                                  │
                                                  ▼
                                        ┌─────────────────┐
                                        │  ONNX Export     │
                                        │  best.onnx       │
                                        │  (42.7 MB)       │
                                        └────────┬────────┘
                                                  │
                                                  ▼
                                        ┌─────────────────┐
                                        │  TensorRT INT8   │
                                        │  best.engine     │
                                        │  (13.4 MB)       │
                                        │  3.8ms inference  │
                                        └────────┬────────┘
                                                  │
                    ┌─────────────────────────────┼──────────────────────────┐
                    │                             │                          │
                    ▼                             ▼                          ▼
          ┌─────────────────┐           ┌─────────────────┐       ┌─────────────────┐
          │  BentoML API    │──────────▶│  Prometheus      │──────▶│  Grafana        │
          │  REST endpoint  │  metrics  │  Scrapes /metrics│  viz  │  Dashboards     │
          │  Port 3000      │           │  Port 9090       │       │  Port 3001      │
          └─────────────────┘           └─────────────────┘       └─────────────────┘
```

### Pipeline stages

**1. Training** — YOLOv8s fine-tuned on VisDrone-DET dataset (10,000+ drone-captured images, 10 classes: pedestrian, car, van, truck, bus, bicycle, etc.). Achieved 37.7% mAP@50, which is strong for this dataset — objects are small (10-15px), densely packed, and captured from 50-150m altitude.

**2. ONNX Export** — Converted PyTorch weights to ONNX format for framework-agnostic deployment. Validated numerical parity between PyTorch and ONNX outputs (confidence scores within 0.003 tolerance).

**3. TensorRT Optimization** — Compiled ONNX model to TensorRT engines with two precision modes. FP16 reduces precision from 32-bit to 16-bit floats (10.4x speedup). INT8 quantizes to 8-bit integers using a calibration dataset from VisDrone validation images (15x speedup, 40% smaller model). Both maintain mAP within 2% of the original.

**4. Model Serving** — BentoML wraps the model as a REST API. POST an image to `/detect`, receive JSON detections with bounding boxes, confidence scores, and class IDs. Swagger UI available at the root URL for interactive testing.

**5. Monitoring** — Three Prometheus metrics instrumented on every request:
  - `request_count` (Counter) — total requests served, used to compute throughput via `rate()`
  - `inference_latency_ms` (Histogram) — latency distribution with buckets at 5/10/25/50/100/150ms, enabling P50/P95/P99 percentile tracking
  - `detection_confidence` (Histogram) — per-detection confidence score distribution with buckets at 0.3/0.5/0.7/0.8/0.9/1.0, used for drift detection

**6. CI/CD** — GitHub Actions runs on every push to main: code linting with `ruff`, ONNX export smoke test to validate the export pipeline hasn't broken.

---

## Project Structure

```
edgevision/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions: lint + test
├── monitoring/
│   └── prometheus.yml             # Prometheus scrape config
├── src/
│   └── edgevision/
│       ├── __init__.py
│       ├── serve.py               # BentoML service with Prometheus metrics
│       ├── export.py              # PyTorch → ONNX export
│       └── validate.py            # ONNX vs PyTorch output comparison
├── configs/
├── scripts/
├── tests/
├── docs/
├── Dockerfile                     # App container (Python + BentoML + Ultralytics)
├── docker-compose.yml             # Full stack: app + Prometheus + Grafana
└── README.md
```

---

## Monitoring Queries (Grafana)

After launching with `docker-compose up`, add Prometheus as a data source in Grafana (http://localhost:3001), then create panels with these queries:

| Panel | PromQL Query |
|-------|-------------|
| Request Rate | `rate(request_count_total[1m])` |
| Latency P95 | `histogram_quantile(0.95, rate(inference_latency_ms_bucket[1m]))` |
| Mean Confidence | `rate(detection_confidence_sum[1m]) / rate(detection_confidence_count[1m])` |
| Total Requests | `request_count_total` |

---

## VisDrone Dataset

[VisDrone-DET](https://github.com/VisDrone/VisDrone-Dataset) contains images captured by drone-mounted cameras over 14 different cities in China. The dataset is challenging for object detection because:

- **Small objects** — vehicles and pedestrians appear as 10-20px from typical drone altitudes
- **Dense scenes** — intersections can contain 100+ annotated objects per frame
- **Variable altitude** — images captured from 50m to 150m+ altitude
- **10 classes** — pedestrian, person, car, van, bus, truck, motor, bicycle, awning-tricycle, tricycle

State-of-the-art models typically achieve 40-45% mAP@50 on this dataset. Our 37.7% with YOLOv8s is competitive and sufficient for demonstrating the full MLOps pipeline.

---

## What I Would Add Next

- **NVIDIA Triton Inference Server** for multi-model serving and dynamic batching
- **Model versioning** with automatic rollback if confidence drift exceeds threshold
- **Jetson Orin Nano deployment** for true edge inference with power profiling
- **Kubernetes orchestration** for scaling across multiple nodes
- **Domain adaptation** for deployment in environments not represented in VisDrone (desert, forest, maritime)
