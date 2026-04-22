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
      .inferred-filters { margin-top: 0.5rem; font-size: 0.9rem; color: #555; }
      details.trace-turn { margin-bottom: 0.5rem; }
      details.trace-turn > summary { cursor: pointer; font-weight: bold; padding: 0.25rem 0; }
      .trace-role { margin-top: 0.5rem; font-size: 0.85rem; color: #555; text-transform: uppercase; }
      .trace-block { background: #f8f8f8; border: 1px solid #eee; padding: 0.5rem; margin: 0.25rem 0 0.75rem 0; max-height: 300px; overflow: auto; white-space: pre-wrap; font-size: 0.85rem; }
    </style>
  </head>
  <body>
    <h1>AI Bills QA</h1>
    <p>Ask a question about the indexed state AI bills. Filters such as state, year, status, or topic are inferred from the question automatically.</p>
    <form method="post" action="/">
      <label for="question">Question:</label>
      <textarea id="question" name="question">{{ question }}</textarea>
      <label for="answer_model">Model Selection</label>
      <select id="answer_model" name="answer_model">
        {% for answer_model_option in answer_model_options %}
          <option value="{{ answer_model_option.option_id }}" {% if answer_model_option.option_id == selected_answer_model %}selected{% endif %}>{{ answer_model_option.label }}</option>
        {% endfor %}
      </select>
      <div style="margin-top: 0.5rem;">
        <label style="font-weight: normal;">
          <input type="checkbox" name="show_trace" value="1" {% if show_trace_checked %}checked{% endif %}>
          Show trace
        </label>
      </div>
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
        {% if inferred_filters_display %}
          <div class="inferred-filters"><strong>Filters inferred from question:</strong> {{ inferred_filters_display }}</div>
        {% else %}
          <div class="inferred-filters"><strong>Filters inferred from question:</strong> none</div>
        {% endif %}
        <p>{{ result.answer }}</p>
      </div>
      <div class="panel">
        <h2>Citations</h2>
        {% if result.citations %}
          {% for citation in result.citations %}
            <div class="citation">
              <div><strong>[{{ citation.rank }}]</strong> <code>{{ citation.bill_id }}</code> | {{ citation.state }}{% if citation.year %} | {{ citation.year }}{% endif %} | {{ citation.title or "Untitled bill" }}</div>
              <div>Status: {{ citation.status or "N/A" }} ({{ citation.status_bucket }}){% if citation.topics_list %} | Topics: {{ citation.topics_list|join(", ") }}{% endif %} | Offsets: {{ citation.start_offset }}:{{ citation.end_offset }} | Score: {{ "%.4f"|format(citation.score) }}</div>
              <p>{{ citation.text }}</p>
            </div>
          {% endfor %}
        {% else %}
          <p>No citations were returned.</p>
        {% endif %}
      </div>
      {% if result.trace %}
        <div class="panel">
          <h2>Planner trace</h2>
          <p style="font-size:0.85rem;color:#666;">Each turn shows what the planner LLM sent, the tool calls it issued, and the raw tool results (truncated to 4000 chars).</p>
          {% for turn in result.trace %}
            <details class="trace-turn" {% if turn.turn == 0 %}open{% endif %}>
              <summary>Turn {{ turn.turn }}{% if turn.role == 'seed' %} &mdash; seed prompts{% endif %}</summary>
              {% if turn.role == 'seed' %}
                {% for msg in turn.messages %}
                  <div class="trace-role">{{ msg.role }}</div>
                  <pre class="trace-block">{{ msg.content }}</pre>
                {% endfor %}
              {% else %}
                {% if turn.assistant_content %}
                  <div class="trace-role">assistant</div>
                  <pre class="trace-block">{{ turn.assistant_content }}</pre>
                {% endif %}
                {% for call in turn.tool_calls %}
                  <div class="trace-role">tool call: {{ call.name }}</div>
                  <pre class="trace-block">{{ call.arguments | tojson(indent=2) }}</pre>
                  <div class="trace-role">tool result</div>
                  <pre class="trace-block">{{ call.result if call.result is not none else '(none)' }}</pre>
                {% endfor %}
              {% endif %}
            </details>
          {% endfor %}
        </div>
      {% endif %}
    {% endif %}
  </body>
</html>
"""


def _format_inferred_filters(applied_filters: dict | None) -> str:
    """Render a short human-readable summary of filters inferred by the LLM.

    A scalar field prints as ``key=value``. A list-valued field prints as
    ``key in [v1, v2]`` so the user can see at a glance that several values
    matched via OR-within-field.
    """

    if not applied_filters:
        return ""
    parts: list[str] = []
    year = applied_filters.get("year")
    if year is not None and year != "" and year != 0:
        parts.append(_format_filter_pair("year", year))
    for key, label in (("state", "state"), ("status_bucket", "status")):
        value = applied_filters.get(key)
        if value:
            parts.append(_format_filter_pair(label, value))
    topics = applied_filters.get("topics") or []
    if topics:
        parts.append(f"topics in [{', '.join(str(topic) for topic in topics)}]")
    return "; ".join(parts)


def _format_filter_pair(label: str, value) -> str:
    """Render one ``label=value`` pair, switching to ``label in [...]`` for lists."""

    if isinstance(value, (list, tuple)):
        items = [str(item) for item in value]
        if len(items) == 1:
            return f"{label}={items[0]}"
        return f"{label} in [{', '.join(items)}]"
    return f"{label}={value}"


def create_app(qa_service: QAService, *, show_trace: bool = False) -> Flask:
    """Create the Flask application for the local QA UI.

    ``show_trace`` sets the default state of the per-request "Show planner
    trace" checkbox on first page load. The checkbox itself is always
    rendered so a hosted deploy can flip it per question without a code
    change. The trace panel only renders when the service actually captured
    a trace for the response.
    """

    app = Flask(__name__)
    app.config["qa_service"] = qa_service
    app.config["available_answer_models"] = qa_service.available_answer_models
    app.config["answer_model_options"] = qa_service.answer_model_options
    app.config["default_answer_model"] = qa_service.default_answer_model
    app.config["show_trace_default"] = bool(show_trace)

    def render_page(
        *,
        question: str,
        result,
        error: str | None,
        selected_answer_model: str,
        show_trace_checked: bool,
    ) -> str:
        """Render the browser UI with the shared page context."""

        applied = getattr(result, "applied_filters", None) if result is not None else None
        return render_template_string(
            _PAGE_TEMPLATE,
            question=question,
            result=result,
            error=error,
            answer_model_options=qa_service.answer_model_options,
            selected_answer_model=selected_answer_model,
            inferred_filters_display=_format_inferred_filters(applied),
            show_trace_checked=show_trace_checked,
        )

    @app.get("/")
    def index() -> str:
        return render_page(
            question="",
            result=None,
            error=None,
            selected_answer_model=qa_service.default_answer_model,
            show_trace_checked=app.config["show_trace_default"],
        )

    @app.post("/")
    def ask_form() -> str:
        question = request.form.get("question", "")
        answer_model = str(
            request.form.get("answer_model", qa_service.default_answer_model)
        )
        capture_trace = bool(request.form.get("show_trace"))
        if not question.strip():
            return render_page(
                question=question,
                result=None,
                error="Please enter a question before submitting.",
                selected_answer_model=answer_model,
                show_trace_checked=capture_trace,
            )
        try:
            result = qa_service.answer_question(
                question,
                answer_model=answer_model,
                capture_trace=capture_trace,
            )
            return render_page(
                question=question,
                result=result,
                error=None,
                selected_answer_model=result.answer_model,
                show_trace_checked=capture_trace,
            )
        except Exception as error:
            return render_page(
                question=question,
                result=None,
                error=str(error),
                selected_answer_model=answer_model,
                show_trace_checked=capture_trace,
            )

    @app.post("/api/ask")
    def ask_api():
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", ""))
        answer_model = str(payload.get("answer_model", ""))
        raw_filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else None
        raw_capture_trace = payload.get("capture_trace", payload.get("show_trace"))
        capture_trace = None if raw_capture_trace is None else bool(raw_capture_trace)
        if not question.strip():
            return jsonify({"error": "question is required"}), 400
        try:
            result = qa_service.answer_question(
                question,
                answer_model=answer_model,
                filters=raw_filters,
                capture_trace=capture_trace,
            )
            return jsonify(result.to_dict())
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except Exception as error:
            return jsonify({"error": str(error)}), 502

    return app


__all__ = ["create_app"]
