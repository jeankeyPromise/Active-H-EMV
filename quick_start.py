#!/usr/bin/env python3
"""
Lightweight environment readiness check for Active-H-EMV.

This script does not run any model inference. It only checks whether the local
runtime is close to the project's expected setup.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_PYTHON = (3, 10)


def _check_python() -> bool:
    ok = sys.version_info[:2] == EXPECTED_PYTHON
    version = ".".join(map(str, sys.version_info[:3]))
    expected = ".".join(map(str, EXPECTED_PYTHON))
    print(f"[Python] current={version} expected={expected} -> {'OK' if ok else 'WARN'}")
    return ok


def _check_imports() -> bool:
    required_modules = [
        "yaml",
        "torch",
        "sentence_transformers",
        "langchain",
        "langchain_core",
        "langchain_community",
        "langchain_openai",
        "faiss",
        "nltk",
        "tiktoken",
    ]
    all_ok = True
    print("[Imports]")
    for name in required_modules:
        try:
            importlib.import_module(name)
            print(f"  - {name}: OK")
        except Exception as exc:
            all_ok = False
            print(f"  - {name}: MISSING ({exc.__class__.__name__}: {exc})")
    return all_ok


def _check_api_env() -> bool:
    key_vars = ["OPENAI_API_KEY", "QWEN_API_KEY", "KAIHONG_API_KEY"]
    base_vars = ["OPENAI_BASE_URL", "CUSTOM_API_BASE_URL", "QWEN_API_BASE_URL", "KAIHONG_API_URL"]

    found_key = next((name for name in key_vars if os.getenv(name)), None)
    found_base = next((name for name in base_vars if os.getenv(name)), None)

    print("[API]")
    print(f"  - key: {found_key or 'NOT SET'}")
    print(f"  - base_url: {found_base or 'NOT SET'}")

    if found_key and found_base:
        return True
    if found_key:
        print("  - note: key is set but base_url is not set; this is fine only for direct OpenAI usage.")
    return False


def _check_teach_layout() -> bool:
    teach_root = PROJECT_ROOT / "dataset" / "TEACh"
    required = [
        teach_root / "games",
        teach_root / "images",
    ]
    ok = all(path.exists() for path in required)
    print("[Dataset][TEACh]")
    print(f"  - root: {teach_root}")
    for path in required:
        print(f"  - {path.relative_to(PROJECT_ROOT)}: {'OK' if path.exists() else 'MISSING'}")
    return ok


def _check_ego4d_layout() -> bool:
    ego_root = PROJECT_ROOT / "dataset" / "Ego4D"
    required = [
        ego_root / "pkl",
    ]
    ok = all(path.exists() for path in required)
    print("[Dataset][Ego4D]")
    print(f"  - root: {ego_root}")
    for path in required:
        print(f"  - {path.relative_to(PROJECT_ROOT)}: {'OK' if path.exists() else 'MISSING'}")
    return ok


def main() -> int:
    print("Active-H-EMV quick start check\n")
    python_ok = _check_python()
    imports_ok = _check_imports()
    api_ok = _check_api_env()
    teach_ok = _check_teach_layout()
    ego_ok = _check_ego4d_layout()

    print("\n[Summary]")
    print(f"  - Python 3.10: {'OK' if python_ok else 'WARN'}")
    print(f"  - Core imports: {'OK' if imports_ok else 'MISSING DEPENDENCIES'}")
    print(f"  - API env: {'READY' if api_ok else 'INCOMPLETE'}")
    print(f"  - TEACh layout: {'READY' if teach_ok else 'INCOMPLETE'}")
    print(f"  - Ego4D layout: {'READY' if ego_ok else 'INCOMPLETE'}")

    if all([python_ok, imports_ok, api_ok]):
        print("\nEnvironment looks close to runnable.")
        return 0

    print("\nEnvironment is not ready yet. Fix the warnings above and run again.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
