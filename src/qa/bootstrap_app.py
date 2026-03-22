"""Temporary authenticated bootstrap app for Render disk uploads.

- Starts quickly without loading the heavy QA runtime.
- Accepts authenticated chunk uploads into `/var/data` so the corpus and ready
  vector cache can be staged before the real QA app boots.
- Exposes lightweight status routes used during the hosted bootstrap flow.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path

from flask import Flask, abort, jsonify, request

_DATA_ROOT = Path("/var/data")
_TOKEN_HEADER = "X-Bootstrap-Token"


def create_bootstrap_app() -> Flask:
    """Create the lightweight bootstrap app used only during Render disk staging."""

    token = os.environ.get("QA_BOOTSTRAP_TOKEN", "").strip()
    if not token:
        raise RuntimeError("QA_BOOTSTRAP_TOKEN must be set when bootstrap mode is enabled")

    app = Flask(__name__)

    def require_token() -> None:
        provided_token = request.headers.get(_TOKEN_HEADER, "")
        if not hmac.compare_digest(provided_token, token):
            abort(401)

    def resolve_target_path(raw_path: str) -> Path:
        normalized_path = raw_path.strip().replace("\\", "/")
        if not normalized_path:
            raise ValueError("path is required")

        candidate = (_DATA_ROOT / normalized_path).resolve()
        data_root = _DATA_ROOT.resolve()
        if candidate == data_root or data_root not in candidate.parents:
            raise ValueError("path must stay within /var/data")
        return candidate

    @app.get("/")
    def index():
        return jsonify(
            {
                "mode": "bootstrap",
                "data_root": str(_DATA_ROOT),
                "corpus_exists": (_DATA_ROOT / "us_ai_legislation_ncsl_text.jsonl").exists(),
                "cache_manifest_exists": (_DATA_ROOT / "qa_cache" / "manifest.json").exists(),
            }
        )

    @app.get("/api/bootstrap/status")
    def status():
        require_token()
        raw_path = str(request.args.get("path", ""))
        if not raw_path:
            return jsonify(
                {
                    "mode": "bootstrap",
                    "data_root": str(_DATA_ROOT),
                    "corpus_exists": (
                        _DATA_ROOT / "us_ai_legislation_ncsl_text.jsonl"
                    ).exists(),
                    "cache_manifest_exists": (
                        _DATA_ROOT / "qa_cache" / "manifest.json"
                    ).exists(),
                }
            )

        try:
            target_path = resolve_target_path(raw_path)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        if not target_path.exists():
            return jsonify(
                {
                    "path": raw_path,
                    "exists": False,
                }
            )

        payload: dict[str, object] = {
            "path": raw_path,
            "exists": True,
            "is_file": target_path.is_file(),
            "is_dir": target_path.is_dir(),
        }
        if target_path.is_file():
            payload["size"] = target_path.stat().st_size
            if request.args.get("include_sha256") == "1":
                payload["sha256"] = _sha256_file(target_path)
        return jsonify(payload)

    @app.post("/api/bootstrap/upload")
    def upload():
        require_token()

        raw_path = str(request.args.get("path", ""))
        offset = int(str(request.args.get("offset", "0")))
        finalize = str(request.args.get("finalize", "0")) == "1"
        expected_sha256 = str(request.args.get("sha256", "")).strip().lower()
        if offset < 0:
            return jsonify({"error": "offset must be >= 0"}), 400

        try:
            target_path = resolve_target_path(raw_path)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        target_path.parent.mkdir(parents=True, exist_ok=True)
        chunk = request.get_data(cache=False)

        if offset == 0:
            write_mode = "wb"
        else:
            current_size = target_path.stat().st_size if target_path.exists() else 0
            if current_size != offset:
                return (
                    jsonify(
                        {
                            "error": "offset mismatch",
                            "expected_offset": current_size,
                            "received_offset": offset,
                        }
                    ),
                    409,
                )
            write_mode = "ab"

        with open(target_path, write_mode) as handle:
            handle.write(chunk)

        response_payload: dict[str, object] = {
            "path": raw_path,
            "bytes_written": len(chunk),
            "size": target_path.stat().st_size,
        }
        if finalize:
            actual_sha256 = _sha256_file(target_path)
            response_payload["sha256"] = actual_sha256
            response_payload["sha256_matches"] = (
                not expected_sha256 or actual_sha256 == expected_sha256
            )
            if expected_sha256 and actual_sha256 != expected_sha256:
                return jsonify(response_payload), 422
        return jsonify(response_payload)

    return app


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest for one file."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["create_bootstrap_app"]
