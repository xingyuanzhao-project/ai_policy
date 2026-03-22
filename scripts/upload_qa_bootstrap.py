"""Upload the QA corpus and cache into Render bootstrap mode.

Usage:
    python -m scripts.upload_qa_bootstrap --base-url https://ai-policy-qa.onrender.com --token <token>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024
_DEFAULT_RETRY_COUNT = 3
_DEFAULT_RETRY_DELAY_SECONDS = 2.0


def parse_args() -> argparse.Namespace:
    """Parse the upload CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Upload the QA corpus and ready cache into Render bootstrap mode."
    )
    parser.add_argument("--base-url", required=True, help="Bootstrap app base URL.")
    parser.add_argument("--token", required=True, help="Bootstrap auth token.")
    parser.add_argument(
        "--corpus-path",
        default="data/ncsl/us_ai_legislation_ncsl_text.jsonl",
        help="Local path to the QA corpus JSONL file.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/qa_cache",
        help="Local path to the ready QA cache directory.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=_DEFAULT_CHUNK_SIZE,
        help="Upload chunk size in bytes.",
    )
    return parser.parse_args()


def main() -> None:
    """Upload the QA corpus and cache files into the remote bootstrap app."""

    args = parse_args()
    corpus_path = Path(args.corpus_path).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    files_to_upload = [(corpus_path, Path("us_ai_legislation_ncsl_text.jsonl"))]
    files_to_upload.extend(
        (
            cache_file,
            Path("qa_cache") / cache_file.relative_to(cache_dir),
        )
        for cache_file in sorted(cache_dir.rglob("*"))
        if cache_file.is_file()
    )

    total_files = len(files_to_upload)
    for index, (local_path, remote_relative_path) in enumerate(files_to_upload, start=1):
        print(f"[{index}/{total_files}] Uploading {remote_relative_path.as_posix()} ...")
        upload_file(
            base_url=args.base_url,
            token=args.token,
            local_path=local_path,
            remote_relative_path=remote_relative_path,
            chunk_size=args.chunk_size,
        )

    print("Upload complete.")


def upload_file(
    *,
    base_url: str,
    token: str,
    local_path: Path,
    remote_relative_path: Path,
    chunk_size: int,
) -> None:
    """Upload one file in ordered chunks and verify its final digest."""

    expected_sha256 = sha256_file(local_path)
    total_size = local_path.stat().st_size
    offset = 0
    with open(local_path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            finalize = offset + len(chunk) == total_size
            response_payload = post_chunk(
                base_url=base_url,
                token=token,
                remote_relative_path=remote_relative_path,
                offset=offset,
                chunk=chunk,
                finalize=finalize,
                expected_sha256=expected_sha256 if finalize else "",
            )
            offset = int(response_payload["size"])

    status_payload = fetch_status(
        base_url=base_url,
        token=token,
        remote_relative_path=remote_relative_path,
    )
    if int(status_payload.get("size", -1)) != total_size:
        raise RuntimeError(
            f"Remote size mismatch for {remote_relative_path}: "
            f"{status_payload.get('size')} != {total_size}"
        )
    remote_sha256 = str(status_payload.get("sha256", ""))
    if remote_sha256 != expected_sha256:
        raise RuntimeError(
            f"Remote SHA-256 mismatch for {remote_relative_path}: "
            f"{remote_sha256} != {expected_sha256}"
        )


def post_chunk(
    *,
    base_url: str,
    token: str,
    remote_relative_path: Path,
    offset: int,
    chunk: bytes,
    finalize: bool,
    expected_sha256: str,
) -> dict[str, object]:
    """Send one chunk to the bootstrap upload endpoint."""

    query = urllib.parse.urlencode(
        {
            "path": remote_relative_path.as_posix(),
            "offset": offset,
            "finalize": int(finalize),
            "sha256": expected_sha256,
        }
    )
    upload_url = f"{base_url.rstrip('/')}/api/bootstrap/upload?{query}"
    headers = {
        "Content-Type": "application/octet-stream",
        "X-Bootstrap-Token": token,
    }
    return request_json(upload_url, headers=headers, method="POST", data=chunk)


def fetch_status(
    *,
    base_url: str,
    token: str,
    remote_relative_path: Path,
) -> dict[str, object]:
    """Fetch status for one uploaded file and include its SHA-256 digest."""

    query = urllib.parse.urlencode(
        {
            "path": remote_relative_path.as_posix(),
            "include_sha256": 1,
        }
    )
    status_url = f"{base_url.rstrip('/')}/api/bootstrap/status?{query}"
    headers = {"X-Bootstrap-Token": token}
    return request_json(status_url, headers=headers, method="GET")


def request_json(
    url: str,
    *,
    headers: dict[str, str],
    method: str,
    data: bytes | None = None,
) -> dict[str, object]:
    """Execute one JSON HTTP request with a small retry loop."""

    last_error: Exception | None = None
    for attempt in range(1, _DEFAULT_RETRY_COUNT + 1):
        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read().decode("utf-8")
                return json.loads(payload)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt == _DEFAULT_RETRY_COUNT:
                break
            time.sleep(_DEFAULT_RETRY_DELAY_SECONDS)
    assert last_error is not None
    raise last_error


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one local file."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
