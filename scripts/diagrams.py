"""Workflow diagrams for the two NER pipelines.

Outputs two PNGs under ``plot/diagrams/``:

- ``method_a_multi_turn_pipeline.png``   -- ``src/ner``       (entry: ``scripts/run_ner.py``)
- ``method_b_skill_driven_agent.png``    -- ``src/skill_ner`` (entry: ``scripts/run_skill_ner.py``)

Layout rules (Method A):

    1 --> 2 --> 3 --> 4 --> 5            (row 1: 5 nodes, left-to-right)
                            |
                            v
                   9 <-- 8 <-- 7 <-- 6   (row 2: 4 nodes, right-to-left)

The rightmost node of row 2 (node 6) sits directly below the rightmost node
of row 1 (node 5).  Every shape of the same kind uses the same width and
height; labels do not stretch the shape.  The flow is strictly linear; there
is no branching.

Run:

    .venv_mac/bin/python scripts/diagrams.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "plot" / "diagrams"
METHOD_A_PATH = OUTPUT_DIR / "method_a_multi_turn_pipeline.png"
METHOD_B_PATH = OUTPUT_DIR / "method_b_skill_driven_agent.png"


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

BLUE_FACE = "#e6f1fb"
BLUE_EDGE = "#0f5e86"
ORANGE_FACE = "#fff5eb"
ORANGE_EDGE = "#b25a1e"
GREEN_FACE = "#e6f6ee"
GREEN_EDGE = "#20895e"
TEXT_DARK = "#1f2933"
TEXT_MUTED = "#55606b"
ARROW_COLOR = "#1f2933"


# ---------------------------------------------------------------------------
# Uniform shape sizes.  Same-kind shapes use identical width and height.
# Labels never stretch the shape; if a label is long, the font is reduced.
# ---------------------------------------------------------------------------

BOX_W, BOX_H = 22.0, 12.0
STACK_W, STACK_H = 22.0, 12.0
DOC_W, DOC_H = 14.0, 17.0
STACK_LAYERS = 3
STACK_OFFSET = 1.3


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _setup(fig_size, xlim, ylim):
    fig, ax = plt.subplots(figsize=fig_size, dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(*xlim)
    # Invert y so larger y is lower on screen; simplifies positioning reads.
    ax.set_ylim(ylim[1], ylim[0])
    ax.axis("off")
    return fig, ax


def _title(ax, x, y, title, subtitle=None):
    ax.text(
        x, y, title,
        ha="left", va="top",
        fontsize=16, fontweight="bold", color=TEXT_DARK,
    )
    if subtitle:
        ax.text(
            x, y + 3.4, subtitle,
            ha="left", va="top",
            fontsize=10, color=TEXT_MUTED,
        )


def _auto_fontsize(text, base=10.5, box_width=BOX_W):
    """Pick a font size that fits ``text`` into ``box_width`` axis units.

    Does not resize the box; only the font.  Keeps all shapes uniform.
    """
    # Rough heuristic tuned empirically for Method A/B widest labels.
    limit = box_width - 2.0
    # Approximate character-width in axis units at fontsize 10.5:
    char_w = 0.78
    width_at_base = len(text) * char_w
    if width_at_base <= limit:
        return base
    return max(8.2, base * limit / width_at_base)


def _box(ax, cx, cy, face, edge, text):
    """Draw a uniform rounded rectangle centred at ``(cx, cy)``."""
    x, y = cx - BOX_W / 2, cy - BOX_H / 2
    ax.add_patch(
        FancyBboxPatch(
            (x, y), BOX_W, BOX_H,
            boxstyle="round,pad=0,rounding_size=1.4",
            facecolor=face, edgecolor=edge, linewidth=1.7,
        )
    )
    ax.text(
        cx, cy, text,
        ha="center", va="center",
        fontsize=_auto_fontsize(text, base=10.5, box_width=BOX_W),
        fontweight="bold", color=TEXT_DARK,
    )


def _stack(ax, cx, cy, face, edge, text, layers=STACK_LAYERS, off=STACK_OFFSET):
    """Draw an artifact stack centred at ``(cx, cy)``.

    Draws back-to-front so the front layer sits on the top-left.  The visual
    centroid of the whole stack equals ``(cx, cy)``.  The header label is
    placed above the top-left face; no text is drawn inside the stack, so the
    shape size remains fixed regardless of label length.
    """
    for j in range(layers):
        dx = ((layers - 1) / 2.0 - j) * off
        dy = ((layers - 1) / 2.0 - j) * off
        x = cx - STACK_W / 2 + dx
        y = cy - STACK_H / 2 + dy
        ax.add_patch(
            FancyBboxPatch(
                (x, y), STACK_W, STACK_H,
                boxstyle="round,pad=0,rounding_size=1.1",
                facecolor=face, edgecolor=edge, linewidth=1.3,
            )
        )
    top_cx = cx - (layers - 1) / 2.0 * off
    top_y = cy - STACK_H / 2 - (layers - 1) / 2.0 * off
    ax.text(
        top_cx, top_y - 1.6, text,
        ha="center", va="bottom",
        fontsize=_auto_fontsize(text, base=10.3, box_width=STACK_W + 4),
        fontweight="bold", color=TEXT_DARK,
    )


def _doc(ax, cx, cy, face, edge, text, sub=None):
    """Draw a document icon centred at ``(cx, cy)``.  Label sits below."""
    x, y = cx - DOC_W / 2, cy - DOC_H / 2
    fold = 3.0
    ax.add_patch(
        Polygon(
            [(x, y), (x + DOC_W - fold, y), (x + DOC_W, y + fold),
             (x + DOC_W, y + DOC_H), (x, y + DOC_H)],
            closed=True, facecolor=face, edgecolor=edge, linewidth=1.7,
        )
    )
    ax.add_patch(
        Polygon(
            [(x + DOC_W - fold, y), (x + DOC_W - fold, y + fold),
             (x + DOC_W, y + fold)],
            closed=True, facecolor="white", edgecolor=edge, linewidth=1.2,
        )
    )
    for ty in (y + DOC_H * 0.32, y + DOC_H * 0.48,
               y + DOC_H * 0.64, y + DOC_H * 0.80):
        ax.plot([x + 2, x + DOC_W - 2], [ty, ty],
                color=edge, alpha=0.35, linewidth=0.9)
    ax.text(
        cx, y + DOC_H + 2.2, text,
        ha="center", va="top",
        fontsize=10.5, fontweight="bold", color=TEXT_DARK,
    )
    if sub:
        ax.text(
            cx, y + DOC_H + 6.2, sub,
            ha="center", va="top",
            fontsize=8.8, color=TEXT_MUTED,
        )


def _arrow(ax, x1, y1, x2, y2, color=ARROW_COLOR, width=2.2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>",
            mutation_scale=18,
            color=color,
            linewidth=width,
            shrinkA=0, shrinkB=0,
        )
    )


def _box_hw():
    return BOX_W / 2


def _box_vh():
    return BOX_H / 2


def _stack_hw():
    return STACK_W / 2 + (STACK_LAYERS - 1) / 2.0 * STACK_OFFSET


def _stack_vh():
    return STACK_H / 2 + (STACK_LAYERS - 1) / 2.0 * STACK_OFFSET


def _doc_hw():
    return DOC_W / 2


# ---------------------------------------------------------------------------
# Method A
# ---------------------------------------------------------------------------


def build_method_a():
    fig, ax = _setup(fig_size=(15.0, 8.2), xlim=(0, 145), ylim=(0, 85))
    _title(
        ax, x=3, y=3,
        title="Method A  --  Multi-turn three-stage pipeline  (src/ner)",
        subtitle="Entry: scripts/run_ner.py",
    )

    # Five aligned columns shared by row 1 and row 2.
    xs = [14, 42, 70, 98, 126]
    y1 = 32
    y2 = 66

    # --- Row 1 (5 nodes, left-to-right) ------------------------------------
    _doc(ax, cx=xs[0], cy=y1, face="white", edge=BLUE_EDGE,
         text="Bill text", sub="BillRecord")
    _box(ax, cx=xs[1], cy=y1, face=BLUE_FACE, edge=BLUE_EDGE,
         text="Chunking")
    _stack(ax, cx=xs[2], cy=y1, face=BLUE_FACE, edge=BLUE_EDGE,
           text="ContextChunk[]")
    _box(ax, cx=xs[3], cy=y1, face=ORANGE_FACE, edge=ORANGE_EDGE,
         text="ZeroShotAnnotator")
    _stack(ax, cx=xs[4], cy=y1, face=GREEN_FACE, edge=GREEN_EDGE,
           text="Candidate pool+evidence")

    # --- Row 2 (4 nodes, right-to-left flow) -------------------------------
    # Visual x-positions (right to left): xs[4], xs[3], xs[2], xs[1]
    # Flow order:                         6        7        8       9
    _box(ax, cx=xs[4], cy=y2, face=ORANGE_FACE, edge=ORANGE_EDGE,
         text="EvalAssembler")
    _stack(ax, cx=xs[3], cy=y2, face=GREEN_FACE, edge=GREEN_EDGE,
           text="Grouped sets")
    _box(ax, cx=xs[2], cy=y2, face=ORANGE_FACE, edge=ORANGE_EDGE,
         text="GranularityRefiner")
    _stack(ax, cx=xs[1], cy=y2, face=GREEN_FACE, edge=GREEN_EDGE,
           text="RefinedQuadruplet[]")

    # --- Arrows: row 1 (left to right) -------------------------------------
    _arrow(ax, xs[0] + _doc_hw(),   y1, xs[1] - _box_hw(),   y1)
    _arrow(ax, xs[1] + _box_hw(),   y1, xs[2] - _stack_hw(), y1)
    _arrow(ax, xs[2] + _stack_hw(), y1, xs[3] - _box_hw(),   y1)
    _arrow(ax, xs[3] + _box_hw(),   y1, xs[4] - _stack_hw(), y1)

    # --- Vertical drop: node 5 -> node 6 (same x column) -------------------
    _arrow(ax, xs[4], y1 + _stack_vh(), xs[4], y2 - _box_vh())

    # --- Arrows: row 2 (right to left; 6 -> 7 -> 8 -> 9) -------------------
    _arrow(ax, xs[4] - _box_hw(),   y2, xs[3] + _stack_hw(), y2)
    _arrow(ax, xs[3] - _stack_hw(), y2, xs[2] + _box_hw(),   y2)
    _arrow(ax, xs[2] - _box_hw(),   y2, xs[1] + _stack_hw(), y2)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Method B
# ---------------------------------------------------------------------------


def build_method_b():
    fig, ax = _setup(fig_size=(15.5, 7.8), xlim=(0, 155), ylim=(0, 74))
    _title(
        ax, x=3, y=3,
        title="Method B  --  Skill-driven agent  (src/skill_ner)",
        subtitle="Entry: scripts/run_skill_ner.py",
    )

    doc_x = 14
    bill_y = 22
    skill_y = 52

    row_y = 37
    xs = [46, 78, 112, 140]  # build_messages, run_tool_loop_async,
                             # parse_agent_response, RefinedQuadruplet[]

    tool_y = 14  # read_section sits above run_tool_loop_async

    # --- Inputs ------------------------------------------------------------
    _doc(ax, cx=doc_x, cy=bill_y, face="white", edge=BLUE_EDGE,
         text="Bill text", sub="BillRecord")
    _doc(ax, cx=doc_x, cy=skill_y, face="white", edge=BLUE_EDGE,
         text="Agentic skill", sub="ner_extraction.md")

    # --- Main row ----------------------------------------------------------
    _box(ax, cx=xs[0], cy=row_y, face=BLUE_FACE, edge=BLUE_EDGE,
         text="Build prompt")
    _box(ax, cx=xs[1], cy=row_y, face=ORANGE_FACE, edge=ORANGE_EDGE,
         text="Agent loop")
    _box(ax, cx=xs[2], cy=row_y, face=BLUE_FACE, edge=BLUE_EDGE,
         text="Parse response")
    _stack(ax, cx=xs[3], cy=row_y, face=GREEN_FACE, edge=GREEN_EDGE,
           text="RefinedQuadruplet[]")

    # --- Tool (above run_tool_loop_async) ---------------------------------
    _box(ax, cx=xs[1], cy=tool_y, face=ORANGE_FACE, edge=ORANGE_EDGE,
         text="Tool call")

    # --- Arrows: inputs -> build_messages (diagonal) -----------------------
    _arrow(ax, doc_x + _doc_hw(), bill_y,
           xs[0] - _box_hw(), row_y - 2.0)
    _arrow(ax, doc_x + _doc_hw(), skill_y,
           xs[0] - _box_hw(), row_y + 2.0)

    ax.text(
        (doc_x + _doc_hw() + xs[0] - _box_hw()) / 2,
        (bill_y + row_y - 2.0) / 2 - 1.4,
        "bill text meta",
        ha="center", va="center",
        fontsize=9, color=TEXT_MUTED, fontstyle="italic",
    )

    # --- Arrow: Tool call -> Bill text -------------------------------------
    _arrow(
        ax,
        xs[1] - _box_hw(), tool_y,
        doc_x + _doc_hw(), bill_y - DOC_H / 2,
    )

    # --- Arrows: main flow ------------------------------------------------
    _arrow(ax, xs[0] + _box_hw(), row_y, xs[1] - _box_hw(),  row_y)
    _arrow(ax, xs[1] + _box_hw(), row_y, xs[2] - _box_hw(),  row_y)
    _arrow(ax, xs[2] + _box_hw(), row_y, xs[3] - _stack_hw(), row_y)

    # --- Arrows: tool interaction (bidirectional, vertical) ---------------
    # Call: run_tool_loop_async -> read_section
    _arrow(ax, xs[1] - 2.0, row_y - _box_vh(),
           xs[1] - 2.0, tool_y + _box_vh())
    # Result: read_section -> run_tool_loop_async
    _arrow(ax, xs[1] + 2.0, tool_y + _box_vh(),
           xs[1] + 2.0, row_y - _box_vh())
    ax.text(
        xs[1] + (BOX_W / 2) + 2.0, (row_y + tool_y) / 2,
        "multi-turn\ntool loop",
        ha="left", va="center",
        fontsize=9, color=TEXT_MUTED, fontstyle="italic",
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig_a = build_method_a()
    fig_a.savefig(
        METHOD_A_PATH, dpi=220, bbox_inches="tight",
        facecolor="white", edgecolor="none",
    )
    plt.close(fig_a)
    print(f"wrote {METHOD_A_PATH}")

    fig_b = build_method_b()
    fig_b.savefig(
        METHOD_B_PATH, dpi=220, bbox_inches="tight",
        facecolor="white", edgecolor="none",
    )
    plt.close(fig_b)
    print(f"wrote {METHOD_B_PATH}")


if __name__ == "__main__":
    main()
