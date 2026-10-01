# Model Card — Trợ lý học vụ RAG

## Mục đích

Trợ lý trả lời câu hỏi tiếng Việt về nội dung trong kho tri thức Markdown. Hệ thống dành cho mục đích học tập/demo; không thay thế phòng đào tạo, cố vấn học tập hoặc quy định chính thức.

## Kiến trúc

- Retriever: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` + FAISS `IndexFlatIP`.
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`; có thể thay bằng biến `RERANKER_MODEL`.
- Generator: Qwen2.5 Instruct chạy cục bộ, hoặc API tương thích OpenAI qua biến môi trường.
- Chunking: chia theo heading `##`, tối đa 600 ký tự với overlap 150 ký tự; mỗi chunk giữ tên file và heading nguồn.

## Dữ liệu

Kho hiện tại là [`data/kb/so_tay_sinh_vien.md`](data/kb/so_tay_sinh_vien.md), có 20 đề mục và khoảng 1.866 từ. Trước khi nộp chính thức, nhóm phải thay/đối chiếu tài liệu này với nguồn được phép sử dụng và bổ sung đủ **ít nhất 20 trang tài liệu thật** theo yêu cầu đề bài. Không được trình bày nội dung mẫu là quy chế chính thức nếu chưa có xác nhận từ nhà trường.

Không gửi dữ liệu có thông tin cá nhân, hồ sơ điểm, số điện thoại hoặc tài liệu bị hạn chế quyền truy cập vào kho tri thức.

## Đánh giá

- `rag_test_questions.json`: 30 câu hỏi kiểm thử theo đề mục.
- Kết quả retrieval đang lưu: Hit@3 = **100%** (30/30), xem `artifacts/rag_metrics.json`.
- Đây là đánh giá nội bộ trên dữ liệu cùng miền; không chứng minh độ chính xác trên quy định mới hoặc câu hỏi ngoài tài liệu.
- Kết quả chấm đáp án sinh hiện có không hợp lệ vì lần gọi API trước bị lỗi 404. Sau khi cấu hình LLM đúng, chạy `python evaluate_llm_answers.py` và chỉ công bố kết quả mới có log tương ứng.

## Hạn chế và rủi ro

- Tài liệu có thể lỗi thời, thiếu ngữ cảnh hoặc mâu thuẫn; câu trả lời vẫn có thể sai/hallucinate.
- Reranker mặc định được huấn luyện chủ yếu trên tiếng Anh; cần đánh giá lại khi dùng hoàn toàn tiếng Việt.
- LLM/API ngoài có thể lưu hoặc xử lý dữ liệu theo chính sách của nhà cung cấp.
- Hệ thống chặn các mẫu prompt injection phổ biến, nhưng đây không phải cơ chế an toàn tuyệt đối.

## Sử dụng đúng và sai

Đúng: tra cứu thủ tục, thời hạn, điều kiện học vụ và luôn mở tài liệu nguồn khi quyết định quan trọng.

Sai: dùng để ra quyết định kỷ luật/tốt nghiệp tự động, công bố thông tin cá nhân, hoặc coi câu trả lời là văn bản pháp lý/chính thức.

## Giám sát và cập nhật

Ghi nhận câu hỏi không trả lời được, cập nhật tài liệu nguồn có phiên bản/ngày hiệu lực, chạy lại Hit@3 và kiểm thử prompt injection trước mỗi lần phát hành.
