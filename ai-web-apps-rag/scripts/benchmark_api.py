"""Đo độ trễ HTTP của một endpoint API và lưu số liệu có thể đưa vào README.

Ví dụ:
  python scripts/benchmark_api.py --base-url http://127.0.0.1:8000 \
    --endpoint /api/chat/sync --payload '{"message":"CPA tối thiểu là bao nhiêu?"}'
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def percentile(values: list[float], p: float) -> float:
    """Nội suy tuyến tính để p95 vẫn có ý nghĩa khi số lần chạy nhỏ."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Không có mẫu thành công để tính phần trăm vị")
    position = (len(ordered) - 1) * p
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def rss_mb(pid: int | None) -> float | None:
    if pid is None:
        return None
    try:
        import psutil
    except ImportError:
        return None
    try:
        process = psutil.Process(pid)
        return round(process.memory_info().rss / 1024 / 1024, 1)
    except psutil.Error:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark một endpoint JSON của FastAPI")
    parser.add_argument("--base-url", required=True, help="Ví dụ: http://127.0.0.1:8000")
    parser.add_argument("--endpoint", default="/api/chat/sync")
    parser.add_argument("--payload", default='{"message":"CPA tối thiểu để nhận đồ án tốt nghiệp là bao nhiêu?"}')
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--pid", type=int, help="PID uvicorn để lấy RSS; bỏ trống nếu đo server từ xa")
    parser.add_argument("--output", default="artifacts/benchmark_api.json")
    args = parser.parse_args()

    if args.runs < 2 or args.warmup < 0:
        parser.error("--runs phải ≥ 2 và --warmup phải ≥ 0")
    try:
        payload: dict[str, Any] = json.loads(args.payload)
    except json.JSONDecodeError as exc:
        parser.error(f"--payload không phải JSON hợp lệ: {exc}")

    url = f"{args.base_url.rstrip('/')}/{args.endpoint.lstrip('/')}"
    with requests.Session() as session:
        for _ in range(args.warmup):
            response = session.post(url, json=payload, timeout=args.timeout)
            response.raise_for_status()

        latencies, failures = [], []
        peak_rss = rss_mb(args.pid)
        for run in range(1, args.runs + 1):
            started = time.perf_counter()
            try:
                response = session.post(url, json=payload, timeout=args.timeout)
                elapsed_ms = (time.perf_counter() - started) * 1000
                response.raise_for_status()
                latencies.append(elapsed_ms)
            except requests.RequestException as exc:
                failures.append({"run": run, "error": str(exc)})
            current_rss = rss_mb(args.pid)
            if current_rss is not None:
                peak_rss = max(peak_rss or 0, current_rss)

    if not latencies:
        raise SystemExit("Không có request thành công; không ghi số liệu không đúng.")
    result = {
        "endpoint": args.endpoint,
        "base_url": args.base_url,
        "runs_requested": args.runs,
        "runs_succeeded": len(latencies),
        "failures": failures,
        "latency_ms": {
            "min": round(min(latencies), 1),
            "mean": round(statistics.mean(latencies), 1),
            "p50": round(percentile(latencies, 0.50), 1),
            "p95": round(percentile(latencies, 0.95), 1),
            "max": round(max(latencies), 1),
        },
        "peak_rss_mb": peak_rss,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Đã lưu: {output}")


if __name__ == "__main__":
    main()
