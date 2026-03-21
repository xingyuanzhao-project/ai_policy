"""Run the local bills QA browser app.

Usage:
    python -m scripts.run_qa_app
    python -m scripts.run_qa_app --host 127.0.0.1 --port 5050
"""

from __future__ import annotations

import argparse
import atexit
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.qa import build_qa_browser_runtime, load_qa_config


def parse_args() -> argparse.Namespace:
    """Parse command-line flags for the local browser app."""

    parser = argparse.ArgumentParser(description="Run the local bills QA browser app")
    parser.add_argument("--host", default=None, help="Host interface for the Flask app")
    parser.add_argument("--port", type=int, default=None, help="Port for the Flask app")
    parser.add_argument(
        "--corpus-path",
        default=None,
        help="Optional corpus path override used for demos or alternate datasets",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory override used for demos or alternate datasets",
    )
    parser.add_argument(
        "--max-bills",
        type=int,
        default=None,
        help="Optional bill limit expected by the selected cached index",
    )
    return parser.parse_args()


def main() -> None:
    """Load the best available retrieval backend and start the local QA browser app."""

    args = parse_args()
    config = load_qa_config(_PROJECT_ROOT)

    runtime = build_qa_browser_runtime(
        _PROJECT_ROOT,
        corpus_path=args.corpus_path,
        cache_dir=args.cache_dir,
        max_bills=args.max_bills,
    )
    atexit.register(runtime.close)
    if runtime.retrieval_backend == "vector":
        print(f"Loaded persisted vector QA index with {runtime.chunk_count} chunks.")
    else:
        print(
            "Loaded lexical QA retriever over "
            f"{runtime.chunk_count} chunks because no ready vector index was available."
        )
    runtime.app.run(
        host=args.host or config.app.host,
        port=args.port or config.app.port,
        debug=False,
    )


if __name__ == "__main__":
    main()
