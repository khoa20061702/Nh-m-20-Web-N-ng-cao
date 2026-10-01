# Báo cáo — Trợ lý học vụ RAG

> Bản thảo này được thiết kế để giữ trong giới hạn 8 trang khi xuất PDF. Thay các ô `[ĐIỀN ...]` bằng số liệu/đường link đã kiểm chứng; không thay bằng số liệu ước lượng.

## 1. Bài toán và mục tiêu

Nhóm xây dựng trợ lý hỏi đáp tiếng Việt cho các quy định học vụ. Mục tiêu là giúp người học tìm đúng đoạn tài liệu nhanh hơn, đồng thời bắt buộc hiển thị nguồn để người dùng tự kiểm tra. Hệ thống không đưa ra quyết định học vụ tự động.

## 2. Dữ liệu

Kho tri thức gồm `[ĐIỀN SỐ]` tài liệu, `[ĐIỀN SỐ]` trang, có nguồn/phiên bản `[ĐIỀN]`. Tài liệu được chia theo heading `##`, tối đa 600 ký tự/chunk, overlap 150 ký tự. Những dữ liệu có thông tin cá nhân đã được loại bỏ.

## 3. Mô hình và kiến trúc

Retriever dùng multilingual MiniLM, FAISS `IndexFlatIP` và cross-encoder reranker. Generator là `[ĐIỀN MODEL/BACKEND]`. Frontend React/Streamlit gọi FastAPI; endpoint `/api/chat` stream SSE, `/api/chat/sync` phục vụ đánh giá. Cấu hình model/secret dùng biến môi trường.

```text
Người dùng → React/Streamlit → FastAPI → embedding + FAISS → reranker → LLM
                                      ↑                              ↓
                                  kho Markdown ← nguồn và câu trả lời
```

## 4. Đánh giá

Tập đánh giá có `[ĐIỀN SỐ]` câu. Hit@3: `[ĐIỀN]`. Độ chính xác câu trả lời có nguồn: `[ĐIỀN]`. Đánh giá được chạy trên `[ĐIỀN PHẦN CỨNG]` vào `[ĐIỀN NGÀY]`. Nêu ít nhất hai ví dụ lỗi và nguyên nhân.

| Endpoint | p50 (ms) | p95 (ms) | RSS đỉnh (MB) | Số lần chạy |
|---|---:|---:|---:|---:|
| `/api/chat/sync` | [ĐIỀN] | [ĐIỀN] | [ĐIỀN] | 20 |

## 5. Triển khai và kiểm thử

Backend: `[ĐIỀN URL]`. Giao diện: `[ĐIỀN URL]`. Kiểm thử API bao gồm thành công, 400, 422 và prompt injection; GitHub Actions: `[ĐIỀN URL]`. Nêu cách cấu hình CORS và cách giữ API key trong secret.

## 6. Hạn chế và trách nhiệm

Kết quả phụ thuộc vào tính mới/chính xác của tài liệu, retrieval có thể sai và LLM có thể diễn giải nhầm. Hệ thống từ chối mẫu prompt injection phổ biến nhưng không đảm bảo chặn tuyệt đối. Người dùng phải kiểm tra văn bản nguồn; không dùng chatbot để ra quyết định kỷ luật, tốt nghiệp hoặc xử lý thông tin cá nhân.
