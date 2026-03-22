"""Run the local bills QA browser app.

Usage:
    python -m scripts.run_qa_app
    python -m scripts.run_qa_app --host 127.0.0.1 --port 5050 (pick a new one if already in use)
"""

from __future__ import annotations

import argparse
import atexit
import socket
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.qa import build_qa_browser_runtime, load_qa_config


def kill_process_on_port(port: int) -> None:
    """Kill any process already bound to *port* so this run can claim it cleanly."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        port_free = probe.connect_ex(("127.0.0.1", port)) != 0
    if port_free:
        return

    result = subprocess.run(
        ["netstat", "-ano"],
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        # Match lines like "  TCP  127.0.0.1:5050  ...  LISTENING  <PID>"
        if f":{port} " in line and "LISTENING" in line:
            pid = line.strip().split()[-1]
            subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
            print(f"Stopped stale process (PID {pid}) that was holding port {port}.")
            return

    print(f"Port {port} appears in use but no LISTENING process found; proceeding anyway.")


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
    target_port = args.port or config.app.port
    target_host = args.host or config.app.host
    kill_process_on_port(target_port)
    runtime.app.run(
        host=target_host,
        port=target_port,
        debug=False,
    )


if __name__ == "__main__":
    main()
