"""
Script chạy lúc Docker BUILD để:
1. Đọc toàn bộ KB chunks
2. Embed bằng OpenAI API (text-embedding-3-small, 1536 dims) hoặc fastembed local
3. Lưu FAISS index + chunks JSON ra data/prebuilt_index/

Chạy: python scripts/build_index.py [--use-api]
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.llm import load_chunks
from config import DATA_DIR

OUT_DIR = DATA_DIR / "prebuilt_index"


def build_with_api(chunks: list[dict], api_key: str, model: str = "text-embedding-3-small") -> None:
    """Dùng OpenAI Embeddings API — không cần local model."""
    import numpy as np
    import faiss
    import requests

    print(f"Embedding {len(chunks)} chunks via OpenAI API ({model})...")
    texts = [c["text"] for c in chunks]
    
    # Batch size 100
    all_vecs = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "input": batch},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda x: x["index"])
        all_vecs.extend([d["embedding"] for d in data])
        print(f"  Embedded {min(i+100, len(texts))}/{len(texts)}")

    mat = np.array(all_vecs, dtype="float32")
    # Normalize for cosine similarity
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.where(norms == 0, 1, norms)

    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OUT_DIR / "index.faiss"))
    (OUT_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "meta.json").write_text(
        json.dumps({"embed_model": model, "embed_dims": mat.shape[1], "n_chunks": len(chunks)}, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Index saved to {OUT_DIR} ({len(chunks)} chunks, dim={mat.shape[1]})")


def build_with_fastembed(chunks: list[dict], model: str = "BAAI/bge-small-en-v1.5") -> None:
    """Dùng fastembed (ONNX) — không cần OpenAI key."""
    import numpy as np
    import faiss
    from fastembed import TextEmbedding

    print(f"Embedding {len(chunks)} chunks via fastembed ({model})...")
    embedder = TextEmbedding(model_name=model)
    texts = [c["text"] for c in chunks]
    vecs = list(embedder.embed(texts))
    mat = np.stack(vecs).astype("float32")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.where(norms == 0, 1, norms)

    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OUT_DIR / "index.faiss"))
    (OUT_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "meta.json").write_text(
        json.dumps({"embed_model": model, "embed_dims": mat.shape[1], "n_chunks": len(chunks)}, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Index saved to {OUT_DIR} ({len(chunks)} chunks, dim={mat.shape[1]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-api", action="store_true", help="Dùng OpenAI API thay fastembed")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    chunks = load_chunks()
    if not chunks:
        print("ERROR: Không có chunks nào trong data/kb/")
        sys.exit(1)

    if args.use_api:
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: Cần LLM_API_KEY hoặc OPENAI_API_KEY")
            sys.exit(1)
        build_with_api(chunks, api_key, model=args.model or "text-embedding-3-small")
    else:
        build_with_fastembed(chunks, model=args.model or "BAAI/bge-small-en-v1.5")
