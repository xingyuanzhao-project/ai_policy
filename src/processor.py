"""AI Definition Extraction Processor.

This module provides an OOP interface for extracting AI definitions
from legislation text using a local vLLM server.

Supports both synchronous and asynchronous extraction.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd
from openai import AsyncOpenAI, OpenAI
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of a single extraction.

    Attributes:
        index: Row index from the source DataFrame.
        state: State where the legislation was introduced.
        bill_id: Bill identifier.
        bill_url: URL of the bill.
        definition_found: Whether an AI definition was found.
        definition: The extracted definition text, empty string if not found.
    """

    index: int
    state: str
    bill_id: str
    bill_url: str
    definition_found: bool
    definition: str


class ExtractorMixin(ABC):
    """Base mixin for AI definition extractors.

    Provides shared configuration, data loading, and result handling
    for both sync and async implementations.

    Attributes:
        base_url: The vLLM server base URL.
        api_key: API key for the server.
        model_name: Name of the LLM model to use.
        temperature: Sampling temperature for generation.
        max_tokens: Maximum tokens for response.
        prompt_template: Prompt template dictionary for extraction.
        output_schema: JSON schema for structured output.
        df: Loaded DataFrame with legislation data.
        results: Dictionary of extraction results keyed by row index.
        error_count: Number of rows that failed extraction.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        prompt_template: dict[str, Any],
        output_schema: dict[str, Any],
    ) -> None:
        """Initialize the extractor.

        Args:
            base_url: The vLLM server base URL.
            api_key: API key for the server.
            model_name: Name of the LLM model to use.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens for response.
            prompt_template: Prompt template dictionary for extraction.
            output_schema: JSON schema for structured output.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_template = prompt_template
        self.output_schema = output_schema

        self.df: pd.DataFrame | None = None
        self.results: dict[int, ExtractionResult] = {}
        self.error_count: int = 0

        self._prompt_template_str = json.dumps(self.prompt_template)

    def load_data(self, csv_path: str, encoding: str = "utf-8") -> int:
        """Load legislation data from CSV file.

        Args:
            csv_path: Path to the CSV file containing legislation data.
            encoding: File encoding. Defaults to "utf-8".

        Returns:
            Number of rows loaded.
        """
        self.df = pd.read_csv(csv_path, encoding=encoding)
        return len(self.df)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert extraction results to pandas DataFrame.

        Returns:
            DataFrame with columns: state, bill_id, bill_url, definition_found, definition.

        Raises:
            RuntimeError: If no results available.
        """
        if not self.results:
            raise RuntimeError("No results available. Call run() first.")

        records = [
            {
                "state": r.state,
                "bill_id": r.bill_id,
                "bill_url": r.bill_url,
                "definition_found": r.definition_found,
                "definition": r.definition,
            }
            for r in self.results.values()
        ]
        return pd.DataFrame(records)

    def save_results(self, output_path: str, encoding: str = "utf-8") -> None:
        """Save extraction results to CSV file.

        Args:
            output_path: Path for the output CSV file.
            encoding: File encoding. Defaults to "utf-8".

        Raises:
            RuntimeError: If no results available.
        """
        df = self.to_dataframe()
        df.to_csv(output_path, encoding=encoding, index=False)

    def _build_prompt(self, text: str) -> str:
        """Build prompt from template with text substitution.

        Args:
            text: Legislation text to insert into template.

        Returns:
            Formatted prompt string.
        """
        return self._prompt_template_str.replace("{text}", text)

    def _parse_response(self, content: str) -> dict[str, Any]:
        """Parse JSON response from model.

        Args:
            content: Raw response content string.

        Returns:
            Parsed dictionary with definition_found and definition keys.
        """
        return json.loads(content)

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to vLLM server."""
        pass

    @abstractmethod
    def extract_single(self, text: str) -> dict[str, Any]:
        """Extract AI definition from a single legislation text.

        Args:
            text: The legislation text to analyze.

        Returns:
            Dictionary with definition_found and definition keys.
        """
        pass

    @abstractmethod
    def run(self, progress: bool = True) -> int:
        """Process all rows and extract AI definitions.

        Args:
            progress: Whether to show progress bar. Defaults to True.

        Returns:
            Number of rows processed.
        """
        pass


class AIDefinitionExtractor(ExtractorMixin):
    """Synchronous AI definition extractor.

    Extracts AI definitions from legislation text using vLLM
    with synchronous API calls.

    Attributes:
        client: Synchronous OpenAI client instance.
        max_retries: Maximum number of retries for transient errors.

    Example:
        extractor = AIDefinitionExtractor(
            base_url="http://localhost:8000/v1",
            api_key="dummy",
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            temperature=0.0,
            max_tokens=1024,
            prompt_template=prompt_config,
            output_schema=schema_config,
        )
        extractor.connect()
        extractor.load_data("data.csv")
        extractor.run()
        extractor.save_results("output.csv")
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        prompt_template: dict[str, Any],
        output_schema: dict[str, Any],
        max_retries: int = 2,
    ) -> None:
        """Initialize the synchronous extractor.

        Args:
            base_url: The vLLM server base URL.
            api_key: API key for the server.
            model_name: Name of the LLM model to use.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens for response.
            prompt_template: Prompt template dictionary for extraction.
            output_schema: JSON schema for structured output.
            max_retries: Maximum retries for transient errors. Defaults to 2.
        """
        super().__init__(
            base_url, api_key, model_name, temperature, max_tokens, prompt_template, output_schema
        )
        self.client: OpenAI | None = None
        self.max_retries = max_retries

    def connect(self) -> None:
        """Establish connection to vLLM server."""
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            max_retries=self.max_retries,
        )
        logger.info(f"Sync client connected (max_retries: {self.max_retries})")

    def extract_single(self, text: str) -> dict[str, Any]:
        """Extract AI definition from a single legislation text.

        Args:
            text: The legislation text to analyze.

        Returns:
            Dictionary with keys:
                - definition_found (bool): Whether definition was found.
                - definition (str): The extracted definition or empty string.

        Raises:
            RuntimeError: If client is not connected.
        """
        if self.client is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        prompt = self._build_prompt(text)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={"guided_json": self.output_schema},
        )

        return self._parse_response(response.choices[0].message.content)

    def run(self, progress: bool = True) -> int:
        """Process all rows and extract AI definitions synchronously.

        Args:
            progress: Whether to show progress bar. Defaults to True.

        Returns:
            Number of rows processed.

        Raises:
            RuntimeError: If data is not loaded or client not connected.
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        if self.client is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        self.results.clear()
        self.error_count = 0
        iterator = (
            tqdm(self.df.itertuples(), total=len(self.df), desc="Processing")
            if progress
            else self.df.itertuples()
        )

        for row in iterator:
            text = str(row.text) if pd.notna(row.text) else ""

            try:
                parsed = self.extract_single(text)
                self.results[row.Index] = ExtractionResult(
                    index=row.Index,
                    state=row.state,
                    bill_id=row.bill_id,
                    bill_url=row.bill_url,
                    definition_found=parsed["definition_found"],
                    definition=parsed["definition"],
                )
            except Exception as e:
                self.error_count += 1
                error_msg = str(e)[:200]
                logger.warning(f"[ERROR] Row {row.Index}: {error_msg}")
                self.results[row.Index] = ExtractionResult(
                    index=row.Index,
                    state=row.state,
                    bill_id=row.bill_id,
                    bill_url=row.bill_url,
                    definition_found=False,
                    definition="",
                )

        logger.info(f"Processing complete. Errors: {self.error_count}/{len(self.results)}")
        return len(self.results)


class AsyncAIDefinitionExtractor(ExtractorMixin):
    """Asynchronous AI definition extractor.

    Extracts AI definitions from legislation text using vLLM
    with asynchronous API calls for improved throughput.

    Attributes:
        client: Asynchronous OpenAI client instance.
        concurrency: Maximum concurrent requests.
        max_retries: Maximum number of retries for transient errors.

    Example:
        extractor = AsyncAIDefinitionExtractor(
            base_url="http://localhost:8000/v1",
            api_key="dummy",
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            temperature=0.0,
            max_tokens=1024,
            prompt_template=prompt_config,
            output_schema=schema_config,
            concurrency=10,
        )
        await extractor.connect()
        extractor.load_data("data.csv")
        await extractor.run()
        extractor.save_results("output.csv")
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        prompt_template: dict[str, Any],
        output_schema: dict[str, Any],
        concurrency: int = 10,
        max_retries: int = 2,
    ) -> None:
        """Initialize the asynchronous extractor.

        Args:
            base_url: The vLLM server base URL.
            api_key: API key for the server.
            model_name: Name of the LLM model to use.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens for response.
            prompt_template: Prompt template dictionary for extraction.
            output_schema: JSON schema for structured output.
            concurrency: Maximum concurrent requests. Defaults to 10.
            max_retries: Maximum retries for transient errors. Defaults to 2.
        """
        super().__init__(
            base_url, api_key, model_name, temperature, max_tokens, prompt_template, output_schema
        )
        self.client: AsyncOpenAI | None = None
        self.concurrency = concurrency
        self.max_retries = max_retries
        self._active_requests = 0

    async def connect(self) -> None:
        """Establish connection to vLLM server."""
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            max_retries=self.max_retries,
        )
        logger.info(
            f"Async client connected | model: {self.model_name} | "
            f"max_concurrency: {self.concurrency} | max_retries: {self.max_retries}"
        )

    async def extract_single(self, text: str) -> dict[str, Any]:
        """Extract AI definition from a single legislation text asynchronously.

        Args:
            text: The legislation text to analyze.

        Returns:
            Dictionary with keys:
                - definition_found (bool): Whether definition was found.
                - definition (str): The extracted definition or empty string.

        Raises:
            RuntimeError: If client is not connected.
        """
        if self.client is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        prompt = self._build_prompt(text)

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={"guided_json": self.output_schema},
        )

        return self._parse_response(response.choices[0].message.content)

    async def _process_row(
        self,
        row: tuple,
        semaphore: asyncio.Semaphore,
    ) -> ExtractionResult:
        """Process a single row with semaphore-controlled concurrency.

        Args:
            row: DataFrame row as namedtuple.
            semaphore: Semaphore for concurrency control.

        Returns:
            ExtractionResult for the row.
        """
        async with semaphore:
            self._active_requests += 1
            logger.info(f"[START] Row {row.Index:4d} | Active: {self._active_requests}")

            text = str(row.text) if pd.notna(row.text) else ""

            try:
                parsed = await self.extract_single(text)
                self._active_requests -= 1
                logger.info(f"[DONE]  Row {row.Index:4d} | Active: {self._active_requests}")

                return ExtractionResult(
                    index=row.Index,
                    state=row.state,
                    bill_id=row.bill_id,
                    bill_url=row.bill_url,
                    definition_found=parsed["definition_found"],
                    definition=parsed["definition"],
                )
            except Exception as e:
                self._active_requests -= 1
                self.error_count += 1
                error_msg = str(e)[:200]
                logger.warning(f"[ERROR] Row {row.Index:4d} | {error_msg}")

                return ExtractionResult(
                    index=row.Index,
                    state=row.state,
                    bill_id=row.bill_id,
                    bill_url=row.bill_url,
                    definition_found=False,
                    definition="",
                )

    async def run(self, progress: bool = True) -> int:
        """Process all rows and extract AI definitions asynchronously.

        Args:
            progress: Whether to show progress bar. Defaults to True.

        Returns:
            Number of rows processed.

        Raises:
            RuntimeError: If data is not loaded or client not connected.
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        if self.client is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        self.results.clear()
        self.error_count = 0
        semaphore = asyncio.Semaphore(self.concurrency)

        rows = list(self.df.itertuples())
        tasks = [self._process_row(row, semaphore) for row in rows]

        if progress:
            pbar = tqdm(total=len(tasks), desc="Processing")
            for coro in asyncio.as_completed(tasks):
                result = await coro
                self.results[result.index] = result
                pbar.update(1)
            pbar.close()
        else:
            results = await asyncio.gather(*tasks)
            for result in results:
                self.results[result.index] = result

        logger.info(f"Processing complete. Errors: {self.error_count}/{len(self.results)}")
        return len(self.results)

    async def close(self) -> None:
        """Close the async client connection."""
        if self.client is not None:
            await self.client.close()
            self.client = None
