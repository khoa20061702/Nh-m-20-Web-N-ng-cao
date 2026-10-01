"""Tạo bảng so sánh local LLM và LLM API từ hai file evaluate_llm_answers.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_complete(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("status") != "complete":
        raise ValueError(f"{path} chưa hoàn tất, không thể dùng để so sánh.")
    return data


def row(name: str, data: dict) -> str:
    accuracy = data.get("accuracy")
    return f"| {name} | {data.get('llm_model', 'không rõ')} | {data['correct_count']}/{data['total']} | {accuracy:.2%} |"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", required=True, help="Kết quả backend local đã hoàn tất")
    parser.add_argument("--api", required=True, help="Kết quả backend API đã hoàn tất")
    parser.add_argument("--output", default="artifacts/llm_comparison.md")
    args = parser.parse_args()

    local, api = load_complete(args.local), load_complete(args.api)
    markdown = "\n".join([
        "# So sánh lớp sinh LLM",
        "",
        "| Cấu hình | Model | Đúng/tổng | Accuracy |",
        "|---|---|---:|---:|",
        row("Local", local),
        row("API", api),
        "",
        "Bổ sung p50/p95, chi phí/token và phần cứng từ các file benchmark trước khi đưa vào báo cáo.",
        "",
    ])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
