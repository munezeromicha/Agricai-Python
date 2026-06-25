#!/usr/bin/env python3
"""Quick Roboflow smoke test. Usage: python scripts/test_roboflow.py path/to/leaf.jpg"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inference.roboflow import run_roboflow_detect


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_roboflow.py <image>")
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"Not found: {path}")
        sys.exit(1)
    resp = run_roboflow_detect(path.read_bytes())
    print(f"mode={resp.inference_mode} detections={len(resp.detections)} top={resp.top_class_id} {resp.top_confidence_pct}%")
    for d in resp.detections[:5]:
        print(f"  {d.class_name} {d.confidence}% @ ({d.x:.0f},{d.y:.0f})")


if __name__ == "__main__":
    main()
