FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libxcb1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir bentoml prometheus-client ultralytics --extra-index-url https://download.pytorch.org/whl/cpu

COPY src/ src/

CMD ["bentoml", "serve", "src.edgevision.serve:Detection", "--host", "0.0.0.0", "--port", "3000"]