"""Flask app factory for the local bills QA browser interface.

- Exposes a human-facing form route and a small JSON API route over the shared
  QA service.
- Renders the configured answer-model options without owning model-validation
  logic itself.
- Keeps route behavior thin so retrieval and answer logic stay testable outside
  HTTP.
- Does not build indexes or load configuration from disk by itself.
"""

from __future__ import annotations

from flask import Flask, jsonify, render_template_string, request

from .service import QAService

_PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Bill QA</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 2rem auto; max-width: 900px; line-height: 1.5; }
      textarea { width: 100%; min-height: 120px; padding: 0.75rem; }
      select { padding: 0.5rem; min-width: 320px; }
      button { padding: 0.6rem 1rem; margin-top: 0.75rem; }
      .panel { border: 1px solid #ccc; border-radius: 8px; padding: 1rem; margin-top: 1rem; }
      .error { color: #b00020; }
      .citation { margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid #eee; }
      code { background: #f3f3f3; padding: 0.1rem 0.25rem; }
    </style>
  </head>
  <body>
    <h1>AI Bills QA</h1>
    <p>Ask a question about the indexed state AI bills.</p>
    <form method="post" action="/">
      <label for="question">Question:</label>
      <textarea id="question" name="question">{{ question }}</textarea>
      <label for="answer_model">Model Selection</label>
      <select id="answer_model" name="answer_model">
        {% for answer_model_option in answer_model_options %}
          <option value="{{ answer_model_option.option_id }}" {% if answer_model_option.option_id == selected_answer_model %}selected{% endif %}>{{ answer_model_option.label }}</option>
        {% endfor %}
      </select>
      <div>
        <button type="submit">Ask</button>
      </div>
    </form>
    {% if error %}
      <div class="panel error">{{ error }}</div>
    {% endif %}
    {% if result %}
      <div class="panel">
        <h2>Answer</h2>
        <p><strong>Model:</strong> {{ result.answer_model }}</p>
        <p>{{ result.answer }}</p>
      </div>
      <div class="panel">
        <h2>Citations</h2>
        {% if result.citations %}
          {% for citation in result.citations %}
            <div class="citation">
              <div><strong>[{{ citation.rank }}]</strong> <code>{{ citation.bill_id }}</code> | {{ citation.state }} | {{ citation.title or "Untitled bill" }}</div>
              <div>Status: {{ citation.status or "N/A" }} | Offsets: {{ citation.start_offset }}:{{ citation.end_offset }} | Score: {{ "%.4f"|format(citation.score) }}</div>
              <p>{{ citation.text }}</p>
            </div>
          {% endfor %}
        {% else %}
          <p>No citations were returned.</p>
        {% endif %}
      </div>
    {% endif %}
  </body>
</html>
"""


def create_app(qa_service: QAService) -> Flask:
    """Create the Flask application for the local QA UI."""

    app = Flask(__name__)
    app.config["qa_service"] = qa_service
    app.config["available_answer_models"] = qa_service.available_answer_models
    app.config["answer_model_options"] = qa_service.answer_model_options
    app.config["default_answer_model"] = qa_service.default_answer_model

    def render_page(
        *,
        question: str,
        result,
        error: str | None,
        selected_answer_model: str,
    ) -> str:
        """Render the browser UI with the shared page context."""

        return render_template_string(
            _PAGE_TEMPLATE,
            question=question,
            result=result,
            error=error,
            answer_model_options=qa_service.answer_model_options,
            selected_answer_model=selected_answer_model,
        )

    @app.get("/")
    def index() -> str:
        return render_page(
            question="",
            result=None,
            error=None,
            selected_answer_model=qa_service.default_answer_model,
        )

    @app.post("/")
    def ask_form() -> str:
        question = request.form.get("question", "")
        answer_model = str(
            request.form.get("answer_model", qa_service.default_answer_model)
        )
        if not question.strip():
            return render_page(
                question=question,
                result=None,
                error="Please enter a question before submitting.",
                selected_answer_model=answer_model,
            )
        try:
            result = qa_service.answer_question(question, answer_model=answer_model)
            return render_page(
                question=question,
                result=result,
                error=None,
                selected_answer_model=result.answer_model,
            )
        except Exception as error:
            return render_page(
                question=question,
                result=None,
                error=str(error),
                selected_answer_model=answer_model,
            )

    @app.post("/api/ask")
    def ask_api():
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", ""))
        answer_model = str(payload.get("answer_model", ""))
        if not question.strip():
            return jsonify({"error": "question is required"}), 400
        try:
            result = qa_service.answer_question(question, answer_model=answer_model)
            return jsonify(result.to_dict())
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except Exception as error:
            return jsonify({"error": str(error)}), 502

    return app


__all__ = ["create_app"]
