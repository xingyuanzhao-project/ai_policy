"""Stage modules for the nine-stage evaluation pipeline.

- Each ``stageN_*.py`` module exposes a single ``run`` entry point invoked by
  ``src.eval.evals`` orchestrator.
- Stages read and write through ``src.eval.io``, ``src.eval.cache``, and
  ``src.eval.judge``; they never touch the extractor run files directly
  except via the shared loader.
- Stages are independent: given the shared cache layout, any stage can be
  re-run without invalidating earlier work.
"""
