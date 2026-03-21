"""Build or resume the local QA index.

Usage:
    python -m scripts.build_qa_index
    python -m scripts.build_qa_index --force-rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.qa import OpenAICompatibleClient, QAIndexer, load_provider_api_key, load_qa_config


def parse_args() -> argparse.Namespace:
    """Parse command-line flags for the index builder."""

    parser = argparse.ArgumentParser(description="Build or resume the local bills QA index")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Discard an incompatible persisted cache before rebuilding the index",
    )
    parser.add_argument(
        "--corpus-path",
        default=None,
        help="Optional corpus path override used for indexing or demos",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory override used for indexing or demos",
    )
    parser.add_argument(
        "--max-bills",
        type=int,
        default=None,
        help="Optional limit on the number of bills used for the build",
    )
    return parser.parse_args()


def main() -> None:
    """Build or resume the local QA index and print a short summary."""

    args = parse_args()
    config = load_qa_config(_PROJECT_ROOT)
    if args.corpus_path:
        config.corpus_path = args.corpus_path
    if args.cache_dir:
        config.index.cache_dir = args.cache_dir
    config.validate()

    api_key = load_provider_api_key(config.provider)
    provider_client = OpenAICompatibleClient(
        api_key=api_key,
        api_base_url=config.provider.api_base_url,
        embedding_model=config.models.embedding_model,
        answer_model=config.models.answer_model,
    )
    try:
        loaded_index = QAIndexer(
            project_root=_PROJECT_ROOT,
            config=config,
            provider_client=provider_client,
        ).build_or_resume(
            force_rebuild=args.force_rebuild,
            max_bills=args.max_bills,
        )
    finally:
        provider_client.close()

    print(
        "Built QA index with "
        f"{loaded_index.manifest.total_chunks} chunks in "
        f"{loaded_index.manifest.completed_batch_count} batches."
    )


if __name__ == "__main__":
    main()
