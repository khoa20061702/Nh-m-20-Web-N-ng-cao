# ---- Tầng 1: build React ----
FROM node:22-slim AS web
WORKDIR /web
COPY web/package*.json ./
RUN npm install --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---- Tầng 2: pre-build FAISS index ----
# Dùng fastembed (ONNX) để embed KB lúc BUILD, không lúc runtime.
FROM python:3.11-slim AS indexer
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir faiss-cpu fastembed numpy

COPY data/ data/
COPY config.py .
COPY core/llm.py core/llm.py
COPY core/__init__.py core/__init__.py
COPY scripts/build_index.py scripts/build_index.py

# Pre-build FAISS index với model đa ngôn ngữ nhỏ (ONNX ~30MB)
# bge-small-en-v1.5 hỗ trợ tiếng Việt ở mức cơ bản
# Nếu muốn chất lượng tốt hơn có thể đổi sang multilingual-e5-small
RUN python scripts/build_index.py --model BAAI/bge-small-en-v1.5

# ---- Tầng 3: API Python siêu nhẹ cho Render free (512 MB RAM) ----
# Không có torch, không có transformers, không có sentence-transformers.
# Startup RAM: faiss load index (~5MB) + fastapi + uvicorn ≈ ~80MB
# Query RAM: gọi OpenAI Embeddings API (0 MB local model) ← KHÔNG tốn RAM
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

COPY config.py ./
COPY core/ core/
COPY api/ api/
COPY data/ data/
COPY artifacts/ artifacts/
# Copy pre-built index từ tầng indexer
COPY --from=indexer /build/data/prebuilt_index data/prebuilt_index
COPY --from=web /web/dist web/dist

ENV APP_ROOT=/app \
    HF_HOME=/app/.cache/hf \
    LLM_BACKEND=openai_compatible \
    USE_FASTEMBED=false \
    USE_RERANKER=false \
    ENABLED_MODELS=llm

RUN useradd -m app && chown -R app /app
USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:7860/api/health', timeout=5)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
