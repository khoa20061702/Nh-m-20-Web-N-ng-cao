# Checklist trước khi nộp

## Bắt buộc

- [ ] Kho tri thức có nguồn được phép, tối thiểu 20 trang; ghi ngày hiệu lực và chủ sở hữu tài liệu.
- [ ] Chạy `python -m pytest -q tests` thành công; GitHub Actions xanh.
- [ ] Chạy `python evaluate_rag.py`, lưu Hit@3 mới và kiểm tra các câu miss.
- [ ] Chạy `python evaluate_llm_answers.py` không có lỗi API; xem thủ công các câu bị chấm sai.
- [ ] Chạy benchmark 20 lần, ghi p50/p95/RAM và phần cứng vào README/BENCHMARK.md.
- [ ] Cập nhật MODEL_CARD.md khi đổi dữ liệu/model.
- [ ] Đặt ảnh giao diện và `/api/health` thật vào `docs/screenshots/`.
- [ ] Backend public trả `/api/health` là `ok`; giao diện mở được trên điện thoại.
- [ ] CORS chỉ chứa frontend domain được phép; API key nằm trong secret, không nằm trong Git.
- [ ] Đẩy tag `v1.0`, hoàn thành `docs/REPORT.md` và quay video 3 phút.

## Kịch bản video 3 phút

1. 0:00–0:25: bài toán, dữ liệu và giới hạn.
2. 0:25–1:20: đặt một câu hỏi có trong tài liệu, mở nguồn được truy xuất.
3. 1:20–1:55: đặt câu ngoài tài liệu hoặc prompt injection; nêu cách hệ thống từ chối/giới hạn.
4. 1:55–2:30: mở `/api/health`, test/CI, Hit@3 và benchmark.
5. 2:30–3:00: nêu một câu mô hình có thể sai, cách người dùng kiểm tra lại và link deploy.
