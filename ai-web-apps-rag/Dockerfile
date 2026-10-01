# ---- Tầng 1: build React ----
FROM node:22-slim AS web
WORKDIR /web
COPY web/package*.json ./
RUN npm install --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---- Tầng 2: API Python tối giản cho Render free (512 MB RAM) ----
# KHÔNG cài torch/transformers/sentence-transformers (~400 MB) —
# dùng fastembed (ONNX runtime, ~80 MB) và pre-download model lúc build.
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Cài dependencies nhẹ (không có torch)
COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

# Pre-download fastembed model vào image lúc BUILD để không download lúc runtime
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed
RUN python -c "\
from fastembed import TextEmbedding; \
m = TextEmbedding(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_dir='/app/.cache/fastembed'); \
list(m.embed(['test'])); \
print('fastembed model pre-downloaded OK')"

COPY config.py ./
COPY core/ core/
COPY api/ api/
COPY data/ data/
COPY artifacts/ artifacts/
COPY --from=web /web/dist web/dist

ENV APP_ROOT=/app \
    HF_HOME=/app/.cache/hf \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    LLM_BACKEND=openai_compatible \
    USE_FASTEMBED=true \
    FASTEMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    USE_RERANKER=false \
    ENABLED_MODELS=llm

RUN useradd -m app && chown -R app /app
USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:7860/api/health', timeout=5)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
