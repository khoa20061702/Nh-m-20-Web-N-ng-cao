import json
from pathlib import Path
from core.llm import Retriever, load_chunks
from config import DATA_DIR

def evaluate_hit_at_k(questions_file: str, kb_dir: Path, k: int = 3):
    print(f"Loading chunks from {kb_dir}...")
    chunks = load_chunks(kb_dir)
    retriever = Retriever(chunks)
    
    with open(questions_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
        
    hits = 0
    total = len(qa_pairs)
    
    print(f"Evaluating Hit@{k} on {total} questions...")
    for idx, item in enumerate(qa_pairs, 1):
        question = item['question']
        expected_section = item['section']
        
        # Retrieve top k contexts
        results = retriever.search(question, k)
        
        # Check if the expected section is in any of the retrieved contexts
        # The section title might not perfectly match the chunk's text if chunking splits it,
        # but the chunk text should contain the section title or we can check substring.
        # Another approach: since our test questions explicitly listed "section", let's check if the section text appears in retrieved texts.
        
        hit = False
        for res in results:
            if expected_section in res['text'] or res['text'].startswith(expected_section.split('. ')[-1]):
                hit = True
                break
            
            # More flexible check: if 50% of the words in expected_answer are in the retrieved text
            expected_ans_words = item['expected_answer'].lower().split()
            retrieved_text_lower = res['text'].lower()
            match_count = sum(1 for w in expected_ans_words if w in retrieved_text_lower)
            if match_count / len(expected_ans_words) >= 0.5:
                hit = True
                break

        if hit:
            hits += 1
        else:
            print(f"Miss [{idx}]: {question}")
            print(f"  Expected: {item['expected_answer']}")
            print(f"  Retrieved top 1: {results[0]['text'][:100]}...\n")
            
    hit_rate = hits / total
    print(f"\nResult: {hits}/{total} hits -> Hit@{k} = {hit_rate:.2%}")
    
    # Save metric
    metrics = {"Hit@3": hit_rate, "total_questions": total, "dataset": "so_tay_sinh_vien"}
    metrics_path = Path("artifacts/rag_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")

if __name__ == "__main__":
    evaluate_hit_at_k("rag_test_questions.json", DATA_DIR / "kb", k=3)
