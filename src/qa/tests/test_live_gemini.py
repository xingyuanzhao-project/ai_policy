"""Opt-in live provider smoke test for the local QA app."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from src.qa import OpenAICompatibleClient, RetrievedChunk, load_provider_api_key, load_qa_config

LIVE_TESTS_ENABLED = os.environ.get("GEMINI_RUN_LIVE_TESTS") == "1"


@unittest.skipUnless(
    LIVE_TESTS_ENABLED,
    "Set GEMINI_RUN_LIVE_TESTS=1 to run the live Gemini smoke test",
)
class GeminiLiveSmokeTests(unittest.TestCase):
    """Verify the configured provider client can embed and answer a grounded prompt."""

    def test_provider_client_embeds_and_answers(self) -> None:
        """Verify live provider calls work with the configured QA models."""

        project_root = Path(__file__).resolve().parents[3]
        config = load_qa_config(project_root)
        try:
            api_key = load_provider_api_key(config.provider)
        except Exception as error:  # pragma: no cover - only exercised in live mode
            self.skipTest(str(error))

        client = OpenAICompatibleClient(
            api_key=api_key,
            api_base_url=config.provider.api_base_url,
            embedding_model=config.models.embedding_model,
            answer_model=config.models.answer_model,
        )
        try:
            embeddings = client.embed_documents(
                ["Bills may require algorithmic impact assessments and disclosures."]
            )
            self.assertEqual(len(embeddings), 1)
            self.assertGreater(embeddings[0].shape[0], 0)

            answer = client.generate_answer(
                question="What can the bill require?",
                retrieved_chunks=[
                    RetrievedChunk(
                        rank=1,
                        score=1.0,
                        chunk_id=1,
                        bill_id="BILL-001",
                        text="Bills may require algorithmic impact assessments and disclosures.",
                        start_offset=0,
                        end_offset=64,
                        state="CA",
                        title="AI Accountability Act",
                        status="Introduced",
                    )
                ],
            )
            self.assertTrue(answer)
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
