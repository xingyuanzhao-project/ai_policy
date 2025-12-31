"""
Runner script for AI Definition Extraction Pipeline.

Usage:
    python scripts/run_definition_extraction.py

Requires:
    - vLLM server running on localhost:8000
    - Input CSV at data/ncsl/us_ai_legislation_ncsl_text.csv
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path for src import
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from src.processor import AIDefinitionExtractor, AsyncAIDefinitionExtractor
from src.utilities import preflight_check

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load pipeline configuration from YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_prompts(prompts_path: Path) -> dict:
    """Load prompts configuration from JSON file.

    Args:
        prompts_path: Path to the prompts JSON file.

    Returns:
        Prompts configuration dictionary.
    """
    with open(prompts_path, "r") as f:
        return json.load(f)


def log_config(config: dict, prompts: dict) -> None:
    """Log configuration settings.

    Args:
        config: Pipeline configuration dictionary.
        prompts: Prompts configuration dictionary.
    """
    llm = config["llm"]
    async_cfg = config["async"]

    logger.info("=" * 60)
    logger.info("PIPELINE CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Input CSV:        {config['input_csv']}")
    logger.info(f"Output CSV:       {config['output_csv']}")
    logger.info("-" * 60)
    logger.info("LLM Settings:")
    logger.info(f"  Base URL:       {llm['base_url']}")
    logger.info(f"  Model:          {llm['model_name']}")
    logger.info(f"  Temperature:    {llm['temperature']}")
    logger.info(f"  Max Tokens:     {llm['max_tokens']}")
    logger.info(f"  Max Retries:    {llm.get('max_retries', 2)}")
    logger.info("-" * 60)
    logger.info("Execution Settings:")
    logger.info(f"  Async Mode:     {async_cfg['async_execution']}")
    logger.info(f"  Max Concurrency:{async_cfg['max_concurrency']}")
    logger.info("-" * 60)
    logger.info(f"Prompt Task:      {prompts['prompt_template'].get('task', 'N/A')}")
    logger.info("=" * 60)


def run_sync(extractor: AIDefinitionExtractor, input_csv: Path, output_csv: Path) -> None:
    """Run synchronous extraction pipeline.

    Args:
        extractor: Synchronous extractor instance.
        input_csv: Path to input CSV file.
        output_csv: Path to output CSV file.
    """
    extractor.connect()

    logger.info(f"Loading data from: {input_csv}")
    n_rows = extractor.load_data(str(input_csv))
    logger.info(f"Loaded {n_rows} rows")

    extractor.run(progress=True)

    extractor.save_results(str(output_csv))
    logger.info(f"Results saved to: {output_csv}")

    results_df = extractor.to_dataframe()
    found_count = results_df["definition_found"].sum()
    error_count = results_df["error"].notna().sum()
    logger.info(f"Summary: {found_count} definitions found, {error_count} errors, {len(results_df)} total")


async def run_async(extractor: AsyncAIDefinitionExtractor, input_csv: Path, output_csv: Path) -> None:
    """Run asynchronous extraction pipeline.

    Args:
        extractor: Asynchronous extractor instance.
        input_csv: Path to input CSV file.
        output_csv: Path to output CSV file.
    """
    await extractor.connect()

    logger.info(f"Loading data from: {input_csv}")
    n_rows = extractor.load_data(str(input_csv))
    logger.info(f"Loaded {n_rows} rows")

    await extractor.run(progress=True)

    extractor.save_results(str(output_csv))
    logger.info(f"Results saved to: {output_csv}")

    results_df = extractor.to_dataframe()
    found_count = results_df["definition_found"].sum()
    error_count = results_df["error"].notna().sum()
    logger.info(f"Summary: {found_count} definitions found, {error_count} errors, {len(results_df)} total")

    await extractor.close()


def main():
    """Execute the AI definition extraction pipeline."""
    project_root = Path(__file__).parent.parent
    settings_dir = project_root.joinpath("settings")

    config = load_config(settings_dir.joinpath("config.yml"))
    prompts = load_prompts(settings_dir.joinpath("prompts.json"))

    # Log configuration
    log_config(config, prompts)

    llm_config = config["llm"]

    # Pre-flight checks
    logger.info("-" * 60)
    logger.info("PRE-FLIGHT CHECKS")
    logger.info("-" * 60)
    preflight = preflight_check(llm_config["base_url"], llm_config["model_name"])

    if not preflight.passed:
        logger.error("Pre-flight checks FAILED. Aborting pipeline.")
        for error in preflight.errors:
            logger.error(f"  - {error}")
        sys.exit(1)

    logger.info("Pre-flight checks PASSED")
    logger.info("-" * 60)

    input_csv = project_root.joinpath(config["input_csv"])
    output_csv = project_root.joinpath(config["output_csv"])
    async_config = config["async"]

    use_async = async_config["async_execution"]
    max_concurrency = async_config["max_concurrency"]
    max_retries = llm_config.get("max_retries", 2)

    if use_async:
        logger.info("Starting ASYNC extraction...")
        extractor = AsyncAIDefinitionExtractor(
            base_url=llm_config["base_url"],
            api_key=llm_config["api_key"],
            model_name=llm_config["model_name"],
            temperature=llm_config["temperature"],
            max_tokens=llm_config["max_tokens"],
            prompt_template=prompts["prompt_template"],
            output_schema=prompts["output_schema"],
            concurrency=max_concurrency,
            max_retries=max_retries,
        )
        asyncio.run(run_async(extractor, input_csv, output_csv))
    else:
        logger.info("Starting SYNC extraction...")
        extractor = AIDefinitionExtractor(
            base_url=llm_config["base_url"],
            api_key=llm_config["api_key"],
            model_name=llm_config["model_name"],
            temperature=llm_config["temperature"],
            max_tokens=llm_config["max_tokens"],
            prompt_template=prompts["prompt_template"],
            output_schema=prompts["output_schema"],
            max_retries=max_retries,
        )
        run_sync(extractor, input_csv, output_csv)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
