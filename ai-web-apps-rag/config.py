"""Cấu hình tập trung. Mọi giá trị đều ghi đè được bằng biến môi trường."""
import os
import sys
from pathlib import Path

import torch

ROOT = Path(os.environ.get("APP_ROOT", Path(__file__).resolve().parent))
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Trên một số máy macOS, PyTorch/FAISS có thể lỗi khi hai transformer dùng số
# luồng CPU mặc định. Có thể đặt 0 để giữ mặc định của PyTorch.
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "1" if sys.platform == "darwin" else "0"))
if TORCH_NUM_THREADS > 0:
    torch.set_num_threads(TORCH_NUM_THREADS)

# Mô hình (đổi tên model = đổi biến môi trường, không sửa code)
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", str(ART_DIR / "detector" / "yolo11n.pt"))
CLIP_MODEL = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL = os.environ.get(
    "LLM_MODEL",
    "Qwen/Qwen2.5-1.5B-Instruct" if DEVICE == "cuda" else "Qwen/Qwen2.5-0.5B-Instruct",
)
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Lớp sinh có thể chạy cục bộ hoặc qua một API tương thích OpenAI. Không đặt
# khoá trong mã nguồn: xem .env.example để cấu hình khi triển khai.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "local").strip().lower()
LLM_API_KEY = os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
LLM_BASE_URL = os.environ.get(
    "LLM_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
).rstrip("/")
LLM_API_MODEL = os.environ.get("LLM_API_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
LLM_REQUEST_TIMEOUT = float(os.environ.get("LLM_REQUEST_TIMEOUT", "60"))

# Bật/tắt từng mô hình để tiết kiệm bộ nhớ, ví dụ ENABLED_MODELS="classifier,detector"
ENABLED_MODELS = {
    m.strip() for m in os.environ.get("ENABLED_MODELS", "classifier,detector,retrieval,llm").split(",") if m.strip()
}

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "8"))
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:8501").split(",")


def resolve_path(path: str) -> Path:
    """Dữ liệu lưu đường dẫn tương đối so với ROOT để mang sang máy khác (Docker, HF Spaces)."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
