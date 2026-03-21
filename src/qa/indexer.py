"""Index building, manifest validation, and cache loading for QA retrieval.

- Reuses the existing corpus loader and chunk builder to derive retrieval units
  from the bill corpus.
- Persists a manifest, chunk payloads, and batch embedding files so large builds
  can resume safely after interruption.
- Does not own query-time answer synthesis or Flask route behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src.ner.schemas.inference_unit_builder import ChunkingConfig, InferenceUnitBuilder
from src.ner.storage.corpus_store import CorpusStore

from .artifacts import (
    INDEX_STATUS_BUILDING,
    INDEX_STATUS_READY,
    IndexManifest,
    IndexedChunk,
    validate_index_manifest,
    validate_indexed_chunk,
)
from .config import QAConfig
from .gemini_client import OpenAICompatibleClient

_INDEX_FORMAT_VERSION = 2
_MANIFEST_FILENAME = "manifest.json"
_CHUNKS_FILENAME = "chunks.json"
_EMBEDDINGS_DIRNAME = "embeddings"
_FINGERPRINT_CHUNK_BYTES = 1024 * 1024
_MAX_EMBED_RETRY_ATTEMPTS = 12
_DEFAULT_RETRY_DELAY_SECONDS = 20.0


class IndexStateError(RuntimeError):
    """Raised when the persisted QA index is missing, stale, or internally inconsistent."""


def build_indexed_chunks(
    corpus_path: Path,
    *,
    chunk_size: int,
    overlap: int,
    max_bills: int | None = None,
) -> list[IndexedChunk]:
    """Build typed retrieval chunks directly from the configured bill corpus."""

    corpus_store = CorpusStore(corpus_path)
    bills = corpus_store.load()
    if max_bills is not None:
        bills = bills[:max_bills]
    builder = InferenceUnitBuilder(
        ChunkingConfig(
            chunk_size=chunk_size,
            overlap=overlap,
        )
    )
    indexed_chunks: list[IndexedChunk] = []
    for bill in bills:
        for chunk in builder.build(bill):
            indexed_chunk = IndexedChunk(
                chunk_id=chunk.chunk_id,
                bill_id=chunk.bill_id,
                text=chunk.text,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                state=bill.state,
                title=bill.title,
                status=bill.status,
                summary=bill.summary,
                bill_url=bill.bill_url,
            )
            validate_indexed_chunk(indexed_chunk)
            indexed_chunks.append(indexed_chunk)
    return indexed_chunks


@dataclass(slots=True)
class LoadedIndex:
    """Fully loaded local QA index ready for retrieval."""

    manifest: IndexManifest
    chunks: list[IndexedChunk]
    embeddings: np.ndarray


class QAIndexer:
    """Build and load the local QA retrieval index."""

    def __init__(
        self,
        project_root: Path,
        config: QAConfig,
        provider_client: OpenAICompatibleClient | None = None,
    ) -> None:
        """Initialize the QA indexer."""

        self._project_root = project_root
        self._config = config
        self._provider_client = provider_client

    def build_or_resume(
        self,
        force_rebuild: bool = False,
        max_bills: int | None = None,
    ) -> LoadedIndex:
        """Build or resume the local QA index.

        Args:
            force_rebuild: Whether to discard an incompatible cache and rebuild it.
            max_bills: Optional cap on the number of bills used to build the index.

        Returns:
            Fully loaded QA index after the build completes.
        """

        if self._provider_client is None:
            raise IndexStateError("Provider client is required to build the QA index")

        corpus_path = self._require_corpus_path()
        corpus_fingerprint = self._fingerprint_file(corpus_path)
        existing_manifest = self._load_manifest_if_present()

        if existing_manifest is not None and not self._manifest_matches_inputs(
            existing_manifest,
            corpus_path=corpus_path,
            corpus_fingerprint=corpus_fingerprint,
            bill_limit=max_bills,
        ):
            if not force_rebuild:
                raise IndexStateError(
                    "Existing QA index is stale. Re-run the build with force_rebuild=True "
                    "or use the build script's --force-rebuild flag."
                )
            self._reset_cache_dir()
            existing_manifest = None

        if existing_manifest is not None and existing_manifest.status == INDEX_STATUS_READY:
            try:
                return self.load_ready_index(bill_limit=max_bills)
            except IndexStateError:
                # Fall through and repair the incomplete or corrupt cache by rebuilding.
                existing_manifest = None

        indexed_chunks = self._build_indexed_chunks(corpus_path, max_bills=max_bills)
        if not indexed_chunks:
            raise IndexStateError("QA indexing produced zero retrieval chunks from the corpus")

        if force_rebuild:
            self._reset_cache_dir()
            existing_manifest = None

        self._prepare_cache_dir()
        self._write_chunks(indexed_chunks)

        completed_batch_count = self._count_completed_batches(len(indexed_chunks))
        manifest = self._make_manifest(
            corpus_path=corpus_path,
            corpus_fingerprint=corpus_fingerprint,
            total_chunks=len(indexed_chunks),
            completed_batch_count=completed_batch_count,
            status=INDEX_STATUS_BUILDING,
            bill_limit=max_bills,
        )
        self._write_manifest(manifest)

        for batch_index, chunk_batch in enumerate(self._iter_chunk_batches(indexed_chunks)):
            batch_path = self._embedding_batch_path(batch_index)
            if self._is_batch_file_valid(batch_path, len(chunk_batch)):
                continue
            embeddings = self._embed_documents_with_retry(
                [chunk.text for chunk in chunk_batch]
            )
            matrix = np.vstack(embeddings).astype(np.float32, copy=False)
            np.save(batch_path, matrix)
            manifest.completed_batch_count = self._count_completed_batches(len(indexed_chunks))
            manifest.built_at_utc = self._utcnow()
            self._write_manifest(manifest)

        manifest.status = INDEX_STATUS_READY
        manifest.completed_batch_count = self._expected_batch_count(len(indexed_chunks))
        manifest.built_at_utc = self._utcnow()
        self._write_manifest(manifest)
        return self.load_ready_index(bill_limit=max_bills)

    def load_ready_index(self, bill_limit: int | None = None) -> LoadedIndex:
        """Load and validate a ready QA index from disk."""

        corpus_path = self._require_corpus_path()
        manifest = self._load_manifest()
        if manifest.status != INDEX_STATUS_READY:
            raise IndexStateError("Persisted QA index is not ready; build it before starting the app")

        current_fingerprint = self._fingerprint_file(corpus_path)
        if not self._manifest_matches_inputs(
            manifest,
            corpus_path=corpus_path,
            corpus_fingerprint=current_fingerprint,
            bill_limit=bill_limit,
        ):
            raise IndexStateError(
                "Persisted QA index is stale because the corpus, chunking settings, "
                "or embedding model no longer match the manifest"
            )

        chunks = self._load_chunks()
        embeddings = self._load_embeddings_matrix(manifest.total_chunks)
        if len(chunks) != embeddings.shape[0]:
            raise IndexStateError(
                "Persisted QA chunks did not align with the stored embedding matrix"
            )
        return LoadedIndex(
            manifest=manifest,
            chunks=chunks,
            embeddings=embeddings,
        )

    def _build_indexed_chunks(
        self,
        corpus_path: Path,
        max_bills: int | None = None,
    ) -> list[IndexedChunk]:
        """Derive indexed chunks from the configured bill corpus."""

        return build_indexed_chunks(
            corpus_path,
            chunk_size=self._config.chunking.chunk_size,
            overlap=self._config.chunking.overlap,
            max_bills=max_bills,
        )

    def _load_manifest(self) -> IndexManifest:
        """Load the persisted QA manifest from disk."""

        manifest = self._load_manifest_if_present()
        if manifest is None:
            raise IndexStateError("QA index manifest is missing; build the index first")
        return manifest

    def _load_manifest_if_present(self) -> IndexManifest | None:
        """Load the manifest when it exists."""

        manifest_path = self._manifest_path()
        if not manifest_path.exists():
            return None
        with open(manifest_path, encoding="utf-8") as handle:
            return IndexManifest.from_dict(json.load(handle))

    def _load_chunks(self) -> list[IndexedChunk]:
        """Load persisted indexed chunks from disk."""

        chunks_path = self._chunks_path()
        if not chunks_path.exists():
            raise IndexStateError("QA chunks file is missing; rebuild the index")
        with open(chunks_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise IndexStateError("QA chunks file must contain a list of chunk payloads")
        return [IndexedChunk.from_dict(chunk_payload) for chunk_payload in payload]

    def _load_embeddings_matrix(self, expected_total_chunks: int) -> np.ndarray:
        """Load the persisted embedding matrix by concatenating batch files."""

        matrices: list[np.ndarray] = []
        for batch_index, batch_row_count in enumerate(
            self._iter_batch_sizes(expected_total_chunks)
        ):
            batch_path = self._embedding_batch_path(batch_index)
            if not batch_path.exists():
                raise IndexStateError(
                    f"Embedding batch '{batch_path.name}' is missing; rebuild or resume the index"
                )
            matrix = np.load(batch_path)
            if matrix.ndim != 2 or matrix.shape[0] != batch_row_count:
                raise IndexStateError(
                    f"Embedding batch '{batch_path.name}' has an unexpected shape {matrix.shape}"
                )
            matrices.append(matrix.astype(np.float32, copy=False))

        if not matrices:
            raise IndexStateError("No embedding batches were found in the QA cache")
        embeddings = np.vstack(matrices)
        if embeddings.shape[0] != expected_total_chunks:
            raise IndexStateError(
                "Loaded embedding rows did not match the manifest total chunk count"
            )
        return embeddings

    def _prepare_cache_dir(self) -> None:
        """Create the QA cache directory structure if it does not yet exist."""

        self._embedding_batches_dir().mkdir(parents=True, exist_ok=True)

    def _reset_cache_dir(self) -> None:
        """Delete the generated QA cache directory before a forced rebuild."""

        cache_dir = self._cache_dir()
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    def _write_chunks(self, indexed_chunks: list[IndexedChunk]) -> None:
        """Persist indexed chunk metadata aligned to the embedding rows."""

        self._prepare_cache_dir()
        with open(self._chunks_path(), "w", encoding="utf-8") as handle:
            json.dump(
                [chunk.to_dict() for chunk in indexed_chunks],
                handle,
                indent=2,
                ensure_ascii=False,
            )

    def _write_manifest(self, manifest: IndexManifest) -> None:
        """Persist the QA manifest to disk."""

        validate_index_manifest(manifest)
        self._prepare_cache_dir()
        with open(self._manifest_path(), "w", encoding="utf-8") as handle:
            json.dump(manifest.to_dict(), handle, indent=2, ensure_ascii=False)

    def _make_manifest(
        self,
        corpus_path: Path,
        corpus_fingerprint: str,
        total_chunks: int,
        completed_batch_count: int,
        status: str,
        bill_limit: int | None,
    ) -> IndexManifest:
        """Construct a manifest for the current QA build inputs."""

        manifest = IndexManifest(
            index_format_version=_INDEX_FORMAT_VERSION,
            status=status,
            corpus_path=str(corpus_path.resolve()),
            corpus_fingerprint=corpus_fingerprint,
            chunk_size=self._config.chunking.chunk_size,
            overlap=self._config.chunking.overlap,
            provider_api_base_url=self._config.provider.api_base_url,
            embedding_model=self._config.models.embedding_model,
            batch_size=self._config.index.batch_size,
            total_chunks=total_chunks,
            completed_batch_count=completed_batch_count,
            built_at_utc=self._utcnow(),
            bill_limit=bill_limit,
        )
        validate_index_manifest(manifest)
        return manifest

    def _manifest_matches_inputs(
        self,
        manifest: IndexManifest,
        corpus_path: Path,
        corpus_fingerprint: str,
        bill_limit: int | None,
    ) -> bool:
        """Return whether the persisted manifest matches the current runtime inputs.

        The manifest retains the original absolute corpus path for operator
        visibility, but portability is determined by the corpus fingerprint plus
        the other retrieval-shaping inputs. This lets one ready cache move from
        a local machine to Render without being rejected solely because the
        mounted corpus path changed.
        """

        return (
            manifest.corpus_fingerprint == corpus_fingerprint
            and manifest.chunk_size == self._config.chunking.chunk_size
            and manifest.overlap == self._config.chunking.overlap
            and manifest.provider_api_base_url == self._config.provider.api_base_url
            and manifest.embedding_model == self._config.models.embedding_model
            and manifest.batch_size == self._config.index.batch_size
            and manifest.bill_limit == bill_limit
        )

    def _count_completed_batches(self, total_chunks: int) -> int:
        """Count valid embedding batch files currently present on disk."""

        completed = 0
        for batch_index, batch_row_count in enumerate(self._iter_batch_sizes(total_chunks)):
            if self._is_batch_file_valid(self._embedding_batch_path(batch_index), batch_row_count):
                completed += 1
        return completed

    def _is_batch_file_valid(self, batch_path: Path, expected_row_count: int) -> bool:
        """Return whether an embedding batch file already satisfies its contract."""

        if not batch_path.exists():
            return False
        try:
            matrix = np.load(batch_path)
        except Exception:
            return False
        return matrix.ndim == 2 and matrix.shape[0] == expected_row_count

    def _embed_documents_with_retry(self, texts: list[str]) -> list[np.ndarray]:
        """Embed one batch of chunk texts with retry handling for transient quota errors."""

        if self._provider_client is None:
            raise IndexStateError("Provider client is required to build embeddings")

        for attempt in range(_MAX_EMBED_RETRY_ATTEMPTS + 1):
            try:
                return self._provider_client.embed_documents(texts)
            except Exception as error:
                if not self._is_retryable_embed_error(error):
                    raise
                if attempt >= _MAX_EMBED_RETRY_ATTEMPTS:
                    raise
                retry_delay_seconds = self._extract_retry_delay_seconds(error)
                if retry_delay_seconds > 0.0:
                    time.sleep(retry_delay_seconds)

        raise RuntimeError("Embedding batch retry loop exited unexpectedly")

    def _iter_chunk_batches(self, indexed_chunks: list[IndexedChunk]) -> list[list[IndexedChunk]]:
        """Split indexed chunks into deterministic embedding batches."""

        batch_size = self._config.index.batch_size
        return [
            indexed_chunks[offset : offset + batch_size]
            for offset in range(0, len(indexed_chunks), batch_size)
        ]

    def _iter_batch_sizes(self, total_chunks: int) -> list[int]:
        """Return the expected row count for each persisted batch file."""

        batch_size = self._config.index.batch_size
        return [
            min(batch_size, total_chunks - offset)
            for offset in range(0, total_chunks, batch_size)
        ]

    def _expected_batch_count(self, total_chunks: int) -> int:
        """Return the expected number of embedding batches."""

        return math.ceil(total_chunks / self._config.index.batch_size)

    def _fingerprint_file(self, path: Path) -> str:
        """Return a sha256 fingerprint for one corpus file."""

        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(_FINGERPRINT_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _require_corpus_path(self) -> Path:
        """Resolve and validate the configured corpus path."""

        corpus_path = self._config.resolve_corpus_path(self._project_root)
        if not corpus_path.exists():
            raise IndexStateError(
                f"Configured QA corpus file does not exist: '{corpus_path}'"
            )
        return corpus_path

    def _cache_dir(self) -> Path:
        """Return the resolved QA cache directory."""

        return self._config.resolve_cache_dir(self._project_root)

    def _manifest_path(self) -> Path:
        """Return the manifest path inside the QA cache directory."""

        return self._cache_dir() / _MANIFEST_FILENAME

    def _chunks_path(self) -> Path:
        """Return the persisted chunk metadata path."""

        return self._cache_dir() / _CHUNKS_FILENAME

    def _embedding_batches_dir(self) -> Path:
        """Return the directory holding embedding batch files."""

        return self._cache_dir() / _EMBEDDINGS_DIRNAME

    def _embedding_batch_path(self, batch_index: int) -> Path:
        """Return the path for one embedding batch file."""

        return self._embedding_batches_dir() / f"batch_{batch_index:05d}.npy"

    def _utcnow(self) -> str:
        """Return the current UTC timestamp in ISO 8601 format."""

        return datetime.now(UTC).replace(microsecond=0).isoformat()

    def _is_retryable_embed_error(self, error: Exception) -> bool:
        """Return whether an embedding error is likely transient and worth retrying."""

        error_code = getattr(error, "code", None)
        error_status = str(getattr(error, "status", "")).upper()
        if isinstance(error, TypeError) and "NoneType" in str(error):
            return True
        return error_code in {429, 500, 503} or error_status in {
            "RESOURCE_EXHAUSTED",
            "UNAVAILABLE",
            "INTERNAL",
        }

    def _extract_retry_delay_seconds(self, error: Exception) -> float:
        """Extract a retry delay from a Gemini API error, falling back to a safe default."""

        for details_payload in (
            getattr(error, "details", None),
            getattr(error, "body", None),
            _load_error_response_json(error),
        ):
            retry_delay_seconds = _extract_retry_delay_from_payload(details_payload)
            if retry_delay_seconds is not None:
                return retry_delay_seconds
        return _DEFAULT_RETRY_DELAY_SECONDS


def _load_error_response_json(error: Exception) -> dict[str, object] | None:
    """Load a JSON error body from an OpenAI-compatible response object when present."""

    response = getattr(error, "response", None)
    if response is None:
        return None
    json_method = getattr(response, "json", None)
    if not callable(json_method):
        return None
    try:
        payload = json_method()
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_retry_delay_from_payload(
    payload: object,
) -> float | None:
    """Return a parsed retry delay from one provider error payload, if present."""

    if not isinstance(payload, dict):
        return None
    nested_details = payload.get("error", payload).get("details", [])
    if not isinstance(nested_details, list):
        return None
    for detail in nested_details:
        if not isinstance(detail, dict):
            continue
        retry_delay = detail.get("retryDelay")
        if retry_delay is None:
            continue
        try:
            return max(float(str(retry_delay).rstrip("s")), 0.0)
        except ValueError:
            continue
    return None


__all__ = ["IndexStateError", "LoadedIndex", "QAIndexer"]
