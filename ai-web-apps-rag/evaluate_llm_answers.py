import json
from pathlib import Path
from core.llm import RAGChatbot

def evaluate_llm_answers(questions_file: str):
    print("Khởi tạo RAG Chatbot (sẽ tải LLM và Embedder/Reranker)...")
    chatbot = RAGChatbot()
    
    with open(questions_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
        
    correct_count = 0
    total = len(qa_pairs)
    
    print(f"\nBắt đầu tự động chấm điểm {total} câu hỏi...")
    results_log = []
    
    for idx, item in enumerate(qa_pairs, 1):
        question = item['question']
        expected_answer = item['expected_answer']
        
        # Chạy dự đoán
        result = chatbot.answer(question, k=3)
        generated_answer = result['answer']
        
        # Kiểm tra đúng/sai (dựa trên keyword matching & kiểm tra trích nguồn)
        is_correct = False
        expected_lower = expected_answer.lower()
        gen_lower = generated_answer.lower()
        
        # Yêu cầu phải có trích nguồn
        has_source = "[so_tay_sinh_vien.md]" in gen_lower or "so_tay_sinh_vien.md" in gen_lower
        
        # Đếm tỷ lệ từ khóa xuất hiện
        expected_words = [w for w in expected_lower.split() if len(w) >= 2]
        match_count = sum(1 for w in expected_words if w in gen_lower)
        match_ratio = match_count / max(len(expected_words), 1)
        
        # Tỷ lệ match >= 60% và có nguồn thì coi như đúng
        if match_ratio >= 0.6 and has_source:
            is_correct = True
            
        if is_correct:
            correct_count += 1
            
        results_log.append({
            "question": question,
            "expected": expected_answer,
            "generated": generated_answer,
            "correct": is_correct
        })
        
        print(f"[{idx}/{total}] {'✅ ĐÚNG' if is_correct else '❌ SAI '} - {question}")
        if not is_correct:
            print(f"   Kỳ vọng: {expected_answer}")
            print(f"   Thực tế: {generated_answer}")
            
    accuracy = correct_count / total
    print(f"\n=== KẾT QUẢ CHẤM ĐIỂM TỰ ĐỘNG ===")
    print(f"{correct_count}/{total} câu trả lời đạt yêu cầu (Accuracy = {accuracy:.2%})")
    
    # Ghi log
    out_path = Path("artifacts/rag_auto_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "accuracy": accuracy, 
        "correct_count": correct_count,
        "total": total,
        "details": results_log
    }, indent=2, ensure_ascii=False))
    print(f"Đã lưu kết quả chi tiết tại {out_path}")

if __name__ == "__main__":
    evaluate_llm_answers("rag_test_questions.json")
