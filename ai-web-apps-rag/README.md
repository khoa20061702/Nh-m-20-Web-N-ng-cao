---
title: Trợ lý học vụ RAG
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Trợ lý học vụ RAG

Chatbot tiếng Việt trả lời dựa trên tài liệu Markdown đã cung cấp. Kiến trúc tách thành `core/` (RAG) → `api/` (FastAPI) → giao diện React/Streamlit. API stream câu trả lời qua Server-Sent Events (SSE) và luôn trả về nguồn đã truy xuất.

> Hệ thống chỉ phục vụ học tập/demo, không thay thế phòng đào tạo hoặc văn bản chính thức. Xem giới hạn tại [MODEL_CARD.md](MODEL_CARD.md).

## Tính năng

- Embedding đa ngôn ngữ + FAISS, chunking theo đề mục với overlap.
- Reranker cho top-k ngữ cảnh trước khi sinh đáp án.
- Chạy Qwen cục bộ hoặc chuyển sang LLM API tương thích OpenAI chỉ bằng biến môi trường.
- Guardrail chặn các mẫu prompt injection phổ biến; có test tự động.
- React và Streamlit cùng gọi một FastAPI backend.
- Kiểm tra schema, lỗi 400/422/503, giới hạn upload, CORS và header thời gian xử lý.

## Chạy trên máy

Yêu cầu: Python 3.11+, Node 22+ (chỉ cần Node khi dùng React). Các lệnh dưới đây dùng shell zsh/bash.

```bash
git clone <GITHUB_REPOSITORY_URL>
cd ai-web-apps-rag
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt pytest psutil

# Chạy chatbot RAG cục bộ; lần đầu sẽ tải các model cần thiết.
ENABLED_MODELS=llm uvicorn api.main:app --port 8000
```

Mở `http://localhost:8000/docs` để thử API. Sau khi build React, FastAPI sẽ tự phục vụ giao diện tại `http://localhost:8000`.

```bash
cd web && npm install && npm run build
cd ..
ENABLED_MODELS=llm uvicorn api.main:app --port 8000
```

Hoặc chạy React ở chế độ phát triển (mặc định gọi API cổng 8000):

```bash
cd web && npm install && npm run dev
```

Giao diện Streamlit là lựa chọn độc lập:

```bash
API_URL=http://localhost:8000 streamlit run streamlit_app.py
```

## Cấu hình LLM

Sao chép `.env.example` thành `.env`, điền giá trị phù hợp rồi nạp biến môi trường vào shell:

```bash
cp .env.example .env
set -a; source .env; set +a
```

| Biến | Mặc định | Mô tả |
|---|---|---|
| `ENABLED_MODELS` | `classifier,detector,retrieval,llm` | Với dự án này nên đặt `llm` để tiết kiệm RAM |
| `LLM_BACKEND` | `local` | `local` hoặc `openai_compatible` |
| `LLM_MODEL` | Qwen2.5 Instruct | Mô hình sinh chạy tại server khi dùng local |
| `LLM_API_KEY` | rỗng | Secret cho backend API; không commit |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Base URL, tự bỏ dấu `/` cuối để tránh lỗi 404 |
| `LLM_API_MODEL` | `gpt-4o-mini` | Tên model của nhà cung cấp API |
| `EMBED_MODEL` / `RERANKER_MODEL` | MiniLM | Model truy xuất và xếp hạng lại |
| `TORCH_NUM_THREADS` | 1 trên macOS | Đặt `0` để dùng mặc định PyTorch; macOS nên giữ 1 nếu gặp lỗi native |
| `CORS_ORIGINS` | localhost | Danh sách origin React/Streamlit được gọi API |

Gemini OpenAI-compatible dùng `LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai` (không thêm `/` cuối). Tuy nhiên cần kiểm tra điều khoản, chi phí và cách xử lý dữ liệu của nhà cung cấp trước khi dùng.

## Kiểm thử và đánh giá

```bash
python -m pytest -q tests
python evaluate_rag.py
python evaluate_llm_answers.py
```

`rag_test_questions.json` có 30 câu kiểm thử retrieval. Kết quả Hit@3 vừa chạy là **100% (30/30)**, nhưng cần chạy lại mỗi khi đổi kho tri thức, embedding hoặc reranker. Không công bố kết quả chấm LLM nếu script trả lỗi API.

Đo p50/p95/RAM theo [BENCHMARK.md](BENCHMARK.md). CI ở `.github/workflows/tests.yml` sẽ tự chạy test khi push/pull request lên GitHub.

Để thực hiện phần nâng cao (so sánh local/API) mà không trộn kết quả:

```bash
LLM_BACKEND=local python evaluate_llm_answers.py --output artifacts/rag_local_eval.json
LLM_BACKEND=openai_compatible python evaluate_llm_answers.py --output artifacts/rag_api_eval.json
python scripts/compare_evaluations.py --local artifacts/rag_local_eval.json --api artifacts/rag_api_eval.json
```

## Docker và triển khai

```bash
docker build -t ai-web-apps-rag .
docker run --rm -p 7860:7860 \
  -e ENABLED_MODELS=llm \
  -e LLM_BACKEND=openai_compatible \
  -e LLM_API_KEY='<SECRET>' \
  -e LLM_BASE_URL='https://api.openai.com/v1' \
  -e LLM_API_MODEL='gpt-4o-mini' \
  ai-web-apps-rag
```

- Hugging Face Spaces: metadata Docker ở đầu README đã sẵn sàng; đẩy toàn bộ repo lên một Space loại Docker, rồi thêm secrets trong Settings.
- Render: dùng [`render.yaml`](render.yaml), đồng bộ `LLM_API_KEY` dưới dạng secret và đặt `CORS_ORIGINS` nếu frontend tách domain.
- Vercel/Netlify: dùng [`vercel.json`](vercel.json), đặt `VITE_API_URL=https://<backend-domain>` khi build và thêm domain Vercel vào `CORS_ORIGINS` của backend.

Xem các bước submit không cần đoán tại [DEPLOYMENT.md](DEPLOYMENT.md).

## Cấu trúc và tài liệu nộp

```text
core/llm.py                 # chunking, retrieval, reranking, guardrail và LLM
api/main.py                 # /api/chat (SSE), /api/chat/sync, /api/health
requirements-rag.txt        # dependency tối thiểu cho Docker/CI của nhóm RAG
web/                        # React/Vite client
streamlit_app.py            # Streamlit client
tests/test_api.py           # success + 400 + 422 + injection test
MODEL_CARD.md               # dữ liệu, chỉ số, rủi ro và sử dụng đúng/sai
BENCHMARK.md                # cách sinh số liệu p50/p95/RAM thật
docs/REPORT.md              # bản thảo báo cáo ≤8 trang
docs/SUBMISSION_CHECKLIST.md
```

Trước khi tag `v1.0`, thay toàn bộ ô `[ĐIỀN ...]` trong báo cáo/checklist bằng link và số liệu thật, đặt ảnh chụp giao diện trong `docs/screenshots/`, và đối chiếu nguồn tài liệu ở `data/kb/`.
