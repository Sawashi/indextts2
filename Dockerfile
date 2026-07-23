# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.8.15@sha256:1ececcacbbde240ffca54d400df86e4fdd38f29c1a2366299279d197e92eaed3 AS uv

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@sha256:59e0e4376a0f16d10b03d3a14344b80a866a1674cb4948cb318291387ac05010

ARG INDEXTTS_REF=13495845e3028f0bb6ca1462ad22aa0e76349e40

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON=3.11.13 \
    UV_LINK_MODE=copy \
    MODEL_DIR=/runpod-volume/indextts2/checkpoints \
    CONFIG_PATH=/runpod-volume/indextts2/checkpoints/config.yaml \
    MODEL_DOWNLOAD_ON_START=true \
    USE_FP16=true \
    USE_DEEPSPEED=false \
    USE_CUDA_KERNEL=false \
    MAX_TEXT_LENGTH=5000 \
    MAX_AUDIO_BYTES=26214400 \
    DOWNLOAD_TIMEOUT_SECONDS=30

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        git \
        git-lfs \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/
ENV PATH="/app/index-tts/.venv/bin:${PATH}"

WORKDIR /app
RUN git clone --filter=blob:none https://github.com/index-tts/index-tts.git /app/index-tts \
    && git -C /app/index-tts checkout --detach "${INDEXTTS_REF}" \
    && test "$(git -C /app/index-tts rev-parse HEAD)" = "${INDEXTTS_REF}"

WORKDIR /app/index-tts
COPY requirements.txt /app/requirements.txt
RUN uv python install 3.11.13 \
    && uv sync --frozen --no-dev \
    && uv pip install --python .venv/bin/python -r /app/requirements.txt

COPY download_models.py handler.py /app/index-tts/
RUN useradd --create-home --uid 10001 worker \
    && mkdir -p /app/index-tts/indextts/utils/tagger_cache \
    && chown worker:worker /app/index-tts/indextts/utils/tagger_cache
USER worker
WORKDIR /app/index-tts

CMD ["python", "-u", "handler.py"]
