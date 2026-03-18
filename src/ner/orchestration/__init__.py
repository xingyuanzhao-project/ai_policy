"""Stage orchestration for the NER multi-agent pipeline.

- Re-exports the orchestrator and stage-failure type used by runtime code.
- Groups the explicit multi-stage control flow for the NER system.
- Does not define prompts, schemas, or storage implementations.
"""

from .orchestrator import Orchestrator, StageFailure

__all__ = ["Orchestrator", "StageFailure"]

