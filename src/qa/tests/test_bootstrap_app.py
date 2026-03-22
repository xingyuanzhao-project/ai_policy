"""Tests for the temporary Render bootstrap upload app."""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from src.qa.bootstrap_app import create_bootstrap_app


class BootstrapAppTests(unittest.TestCase):
    """Verify the authenticated bootstrap app handles uploads safely."""

    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self._original_data_root = os.environ.get("QA_BOOTSTRAP_DATA_ROOT")
        self._original_token = os.environ.get("QA_BOOTSTRAP_TOKEN")
        os.environ["QA_BOOTSTRAP_TOKEN"] = "bootstrap-secret"
        self._data_root = Path(self._temporary_directory.name)

        import src.qa.bootstrap_app as bootstrap_app_module

        self._bootstrap_app_module = bootstrap_app_module
        self._original_module_data_root = bootstrap_app_module._DATA_ROOT
        bootstrap_app_module._DATA_ROOT = self._data_root
        self.addCleanup(self._restore_bootstrap_globals)

        app = create_bootstrap_app()
        self._client = app.test_client()

    def _restore_bootstrap_globals(self) -> None:
        self._bootstrap_app_module._DATA_ROOT = self._original_module_data_root
        if self._original_token is None:
            os.environ.pop("QA_BOOTSTRAP_TOKEN", None)
        else:
            os.environ["QA_BOOTSTRAP_TOKEN"] = self._original_token
        if self._original_data_root is None:
            os.environ.pop("QA_BOOTSTRAP_DATA_ROOT", None)
        else:
            os.environ["QA_BOOTSTRAP_DATA_ROOT"] = self._original_data_root

    def test_status_requires_token(self) -> None:
        """Verify protected routes reject missing bootstrap auth."""

        response = self._client.get("/api/bootstrap/status")
        self.assertEqual(response.status_code, 401)

    def test_upload_supports_chunked_file_writes(self) -> None:
        """Verify chunk uploads append correctly and report the final digest."""

        target_path = "qa_cache/manifest.json"
        first_chunk = b'{"status":'
        second_chunk = b'"ready"}'
        expected_sha256 = hashlib.sha256(first_chunk + second_chunk).hexdigest()

        first_response = self._client.post(
            f"/api/bootstrap/upload?path={target_path}&offset=0",
            data=first_chunk,
            headers={"X-Bootstrap-Token": "bootstrap-secret"},
        )
        self.assertEqual(first_response.status_code, 200)

        second_response = self._client.post(
            f"/api/bootstrap/upload?path={target_path}&offset={len(first_chunk)}&finalize=1&sha256={expected_sha256}",
            data=second_chunk,
            headers={"X-Bootstrap-Token": "bootstrap-secret"},
        )
        self.assertEqual(second_response.status_code, 200)
        payload = second_response.get_json()
        assert payload is not None
        self.assertTrue(payload["sha256_matches"])

        status_response = self._client.get(
            f"/api/bootstrap/status?path={target_path}&include_sha256=1",
            headers={"X-Bootstrap-Token": "bootstrap-secret"},
        )
        self.assertEqual(status_response.status_code, 200)
        status_payload = status_response.get_json()
        assert status_payload is not None
        self.assertEqual(status_payload["size"], len(first_chunk + second_chunk))
        self.assertEqual(status_payload["sha256"], expected_sha256)

    def test_upload_rejects_path_escape(self) -> None:
        """Verify uploads cannot escape the Render data directory."""

        response = self._client.post(
            "/api/bootstrap/upload?path=../outside.txt&offset=0",
            data=b"blocked",
            headers={"X-Bootstrap-Token": "bootstrap-secret"},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
