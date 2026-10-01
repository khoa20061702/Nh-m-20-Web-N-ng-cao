# Triển khai và nộp bài

## 1. GitHub

1. Tạo repository GitHub rỗng, rồi đẩy thư mục dự án lên repository đó.
2. Vào tab **Actions**, xác nhận workflow `Test API` màu xanh.
3. Kiểm tra `.env` không xuất hiện trên GitHub; chỉ `.env.example` được commit.
4. Tạo release/tag `v1.0` sau khi các mục dưới đây hoàn thành.

## 2. Backend công khai

### Render

1. Tại Render chọn **New → Blueprint**, kết nối GitHub repository và chọn `render.yaml`.
2. Trong Environment, nhập `LLM_API_KEY`; không ghi khoá vào file hay commit.
3. Nếu dùng Gemini-compatible, đổi `LLM_BASE_URL` thành `https://generativelanguage.googleapis.com/v1beta/openai` và `LLM_API_MODEL` thành model đã được cấp quyền.
4. Sau deploy, mở `https://<render-domain>/api/health`. Kết quả cần có `"status":"ok"` và `"llm":true`.

### Hugging Face Spaces

1. Tạo Space loại **Docker**, visibility theo yêu cầu của giảng viên.
2. Đẩy toàn bộ repository. Front matter Docker trong README đã khai báo cổng 7860.
3. Trong **Settings → Variables and secrets**, thêm LLM API secret (nếu dùng API); không đưa secret vào Git.
4. Mở `<space-url>/api/health` và giao diện gốc `/` để kiểm tra.

## 3. Giao diện

Có hai phương án:

- Đơn giản nhất: dùng giao diện React đã được FastAPI phục vụ cùng domain backend. Không cần CORS.
- Tách frontend: import repo vào Vercel, build bằng `vercel.json`, đặt `VITE_API_URL=https://<backend-domain>`, sau đó thêm origin Vercel chính xác vào `CORS_ORIGINS` ở backend và redeploy.

Kiểm tra trên điện thoại: mở giao diện, gửi một câu hỏi hợp lệ, mở mục **Nguồn**, rồi thử mất mạng để xem trạng thái lỗi.

## 4. Minh chứng để nộp

- GitHub URL: `[ĐIỀN LINK]`
- Backend URL và `/api/health`: `[ĐIỀN LINK]`
- Giao diện URL: `[ĐIỀN LINK]`
- Video demo 3 phút: `[ĐIỀN LINK]`
- Tag/release `v1.0`: `[ĐIỀN LINK]`

Link public chỉ có thể được tạo từ tài khoản GitHub/Render/Hugging Face/Vercel của nhóm. Không tạo tài khoản hoặc chia sẻ API key trong repository.
