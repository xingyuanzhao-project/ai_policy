"""NER agent implementations and shared agent helpers.

- Re-exports the shared agent contract and all concrete NER agents.
- Provides one import surface for prompt execution and structured parsing
  helpers.
- Does not own orchestration or runtime bootstrap behavior.
"""

from .base import AgentResult, BaseAgent
from .eval_assembler import EvalAssembler
from .granularity_refiner import GranularityRefiner, RefinementRequest
from .shared import AgentExecutionConfig, PromptExecutor, StructuredOutputParser
from .zero_shot_annotator import ZeroShotAnnotator

__all__ = [
    "AgentExecutionConfig",
    "AgentResult",
    "BaseAgent",
    "EvalAssembler",
    "GranularityRefiner",
    "PromptExecutor",
    "RefinementRequest",
    "StructuredOutputParser",
    "ZeroShotAnnotator",
]

