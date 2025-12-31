"""Utility functions for AI Definition Extraction Pipeline.

Provides pre-flight checks and validation utilities.
"""

import logging
import subprocess
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


@dataclass
class PreflightResult:
    """Result of pre-flight checks.

    Attributes:
        passed: Whether all checks passed.
        gpu_available: Whether GPU is available.
        gpu_name: Name of GPU if available, None otherwise.
        llm_available: Whether LLM server is reachable.
        llm_model: Model name from server if available, None otherwise.
        errors: List of error messages for failed checks.
    """

    passed: bool
    gpu_available: bool
    gpu_name: str | None
    llm_available: bool
    llm_model: str | None
    errors: list[str]


def check_gpu() -> tuple[bool, str | None]:
    """Check GPU availability using nvidia-smi.

    Returns:
        Tuple of (is_available, gpu_name or None).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split("\n")[0]
            return True, gpu_name
        return False, None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False, None


def check_llm(base_url: str, model_name: str, timeout: int = 10) -> tuple[bool, str | None]:
    """Check LLM server availability and model.

    Args:
        base_url: Base URL of the LLM server (e.g., http://localhost:8000/v1).
        model_name: Expected model name.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (is_available, model_id or None).
    """
    try:
        # Remove trailing /v1 if present for models endpoint
        url = base_url.rstrip("/")
        if url.endswith("/v1"):
            models_url = f"{url}/models"
        else:
            models_url = f"{url}/v1/models"

        response = requests.get(models_url, timeout=timeout)
        response.raise_for_status()

        data = response.json()
        models = data.get("data", [])

        if not models:
            return False, None

        # Check if expected model is available
        for model in models:
            model_id = model.get("id", "")
            if model_name in model_id or model_id in model_name:
                return True, model_id

        # Return first model if expected not found
        return True, models[0].get("id")

    except requests.RequestException:
        return False, None
    except (KeyError, ValueError):
        return False, None


def preflight_check(base_url: str, model_name: str) -> PreflightResult:
    """Run pre-flight checks for GPU and LLM availability.

    Args:
        base_url: Base URL of the LLM server.
        model_name: Expected model name.

    Returns:
        PreflightResult with check outcomes.
    """
    errors = []

    # Check GPU
    gpu_available, gpu_name = check_gpu()
    if gpu_available:
        logger.info(f"GPU Check: PASSED | {gpu_name}")
    else:
        logger.warning("GPU Check: FAILED | nvidia-smi not available or no GPU detected")
        errors.append("nvidia-smi not available or no GPU detected")

    # Check LLM
    llm_available, llm_model = check_llm(base_url, model_name)
    if llm_available:
        logger.info(f"LLM Check: PASSED | {llm_model}")
    else:
        logger.error(f"LLM Check: FAILED | Cannot reach {base_url}")
        errors.append(f"LLM server not reachable at {base_url}")

    passed = gpu_available and llm_available

    return PreflightResult(
        passed=passed,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        llm_available=llm_available,
        llm_model=llm_model,
        errors=errors,
    )
