# Đo hiệu năng API

Không đưa số liệu giả vào báo cáo. Chạy các lệnh dưới đây trên đúng máy hoặc container được dùng để demo, sau khi API đã nạp xong mô hình.

```bash
# Terminal 1: khởi động API và ghi lại PID của tiến trình uvicorn.
ENABLED_MODELS=llm uvicorn api.main:app --port 8000

# Terminal 2: thay 12345 bằng PID uvicorn ở terminal 1.
python scripts/benchmark_api.py \
  --base-url http://127.0.0.1:8000 \
  --endpoint /api/chat/sync \
  --runs 20 --warmup 3 --pid 12345
```

Script tạo `artifacts/benchmark_api.json` với p50, p95, min/max, số request lỗi và RSS đỉnh. Điền nguyên kết quả vào bảng README dưới đây, kèm thông tin phần cứng.

| Ngày đo | Phần cứng | LLM backend/model | Số lần chạy | p50 | p95 | RSS đỉnh | Tỉ lệ lỗi |
|---|---|---|---:|---:|---:|---:|---:|
| Chưa đo | Chưa đo | Chưa đo | 20 | — | — | — | — |

Với backend từ xa không thể đọc RAM của máy chủ, bỏ `--pid` và ghi rõ “không quan sát được từ client” thay vì đoán số RAM.
