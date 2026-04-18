"""Evaluation harness for the bills QA app.

- Hosts the hand-curated ground-truth question set (``ground_truth.json``).
- Provides a runner that calls ``QAService`` end-to-end and scores answers.
- Writes per-question results and an aggregate report to ``results.json`` and
  ``report.md`` inside this directory.
"""
