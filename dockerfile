# ---------- STAGE 1: Builder ----------
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential gcc g++ gfortran \
        pkg-config libjpeg-dev libpng-dev libtiff-dev libopenblas-dev \
        ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-dev

COPY main.py ./
COPY autoenconders/ ./autoenconders/
COPY metrics/ ./metrics/

CMD ["uv", "run", "python", "main.py"]
