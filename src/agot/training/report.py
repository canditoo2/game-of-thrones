"""Render a training run log as a self-contained HTML page.

``RunLog`` appends one JSON record per iteration to a ``*.log.jsonl`` file. This
module turns that file into a single HTML page with no JavaScript, no external
assets and no plotting dependency. The charts are plain inline SVG drawn here.

The page is a *snapshot*: it reloads itself on a timer, but a static file cannot
re-read the log, so the data only changes when the file is written again. The
``agot report --watch`` subcommand re-renders it on an interval, which is what
makes an open browser tab track a live run.

The pure entry point is :func:`render`; :func:`render_file` writes the page and
is what the ``agot report`` subcommand calls. Keeping this torch-free matters:
importing ``agot.training.checkpoint`` would pull in PyTorch, so the path helper
below mirrors its ``log_path`` instead.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

#: Single-seat win rate of a coin-flip six-player game: the "do nothing" line.
PARITY = 1.0 / 6.0
#: Documented baseline: ``GreedyPolicy`` wins 38.3% against five random seats.
GREEDY_BASELINE = 0.383

_BLUE = "#2b6cb0"
_ORANGE = "#c05621"
_GREEN = "#2f855a"
_GREY = "#a0aec0"
_DARK = "#1a202c"

_WIDTH = 760
_HEIGHT = 250
_PAD_L = 58
_PAD_R = 18
_PAD_T = 12
_PAD_B = 34

_STYLE = """
body {
  margin: 0 auto; padding: 24px 20px 60px; max-width: 860px;
  font: 15px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif;
  background: #f7fafc; color: #1a202c;
}
h1 { font-size: 19px; margin: 0 0 4px; }
h2 { font-size: 14px; margin: 0 0 6px; color: #2d3748; }
.sub { color: #718096; font-size: 12px; margin: 0 0 18px; }
.stats { display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 18px; }
.stat {
  background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
  padding: 8px 12px; min-width: 92px;
}
.stat .k {
  display: block; font-size: 11px; text-transform: uppercase;
  letter-spacing: .04em; color: #718096;
}
.stat .v { font-size: 17px; font-weight: 600; }
section {
  background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
  padding: 12px 14px 8px; margin: 0 0 14px;
}
svg { display: block; width: 100%; height: auto; }
.legend { margin: 0 0 4px; font-size: 12px; color: #4a5568; }
.legend span { margin-right: 14px; }
.legend i { font-style: normal; font-weight: 700; }
.note { color: #718096; font-size: 12px; }
.empty {
  background: #fff; border: 1px dashed #cbd5e0; border-radius: 8px;
  padding: 28px; text-align: center; color: #718096;
}
"""


@dataclass
class Series:
    """One line on a chart, optionally with a shaded confidence band."""

    label: str
    points: list[tuple[float, float]]
    colour: str = _BLUE
    band: list[tuple[float, float, float]] | None = None


def log_path_for(checkpoint: str | Path) -> Path:
    """Companion JSONL log for a checkpoint.

    Mirrors :func:`agot.training.checkpoint.log_path` without importing it, so
    this module never pulls in PyTorch. ``tests/test_report.py`` pins the two
    against each other when torch is installed.
    """
    checkpoint = Path(checkpoint)
    return checkpoint.with_name(checkpoint.stem + ".log.jsonl")


def load_records(path: str | Path) -> list[dict]:
    """Read a JSONL run log, skipping blank or unparsable lines."""
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def render(
    records: Sequence[Mapping],
    *,
    title: str = "agot training",
    refresh: int = 30,
    target_iterations: int = 0,
) -> str:
    """Build the HTML page for a sequence of run-log records."""
    rows = [record for record in records if isinstance(record, Mapping)]
    losses = _points(rows, "policy_loss")
    values = _points(rows, "value_loss")
    seconds = _points(rows, "seconds")
    evals = _points(rows, "eval")
    band = _band(rows)

    if not rows:
        body = (
            '<p class="empty">No log records yet. '
            "This page reloads automatically.</p>"
        )
        return _document(title, refresh, body)

    done = int(rows[-1].get("iteration") or 0)

    stats = [
        _stat("iterations", f"{done} / {target_iterations}" if target_iterations else str(done)),
        _stat("games / iter", str(rows[-1].get("games", "?"))),
        _stat("buffer", str(rows[-1].get("buffer", "?"))),
    ]
    if evals:
        best_x, best_y = max(evals, key=lambda point: point[1])
        stats.append(_stat("best eval", f"{best_y:.3f} @ {int(best_x)}"))
        stats.append(_stat("last eval", f"{evals[-1][1]:.3f}"))
    average = _average_last(seconds, window=20)
    if average is not None:
        stats.append(_stat("sec / iter", f"{average:.0f}"))
        if target_iterations and target_iterations > done:
            stats.append(
                _stat("ETA", _duration((target_iterations - done) * average))
            )

    sections: list[str] = []
    if evals:
        markers = [max(evals, key=lambda point: point[1])]
        sections.append(
            _section(
                "Eval win rate",
                "single seat vs five opponents; higher is better",
                series=[Series("win rate", evals, _BLUE, band)],
                hlines=[
                    (PARITY, _GREY, "parity 0.167"),
                    (GREEDY_BASELINE, _ORANGE, "greedy 0.383"),
                ],
                legend=[
                    (_BLUE, "win rate"),
                    (_GREY, "parity (1/6)"),
                    (_ORANGE, "greedy baseline"),
                    (_DARK, "best (hollow)"),
                ],
                markers=markers,
                y_lo=0.0,
                y_hi=max(0.6, max(high for _, _, high in band) * 1.1) if band else 1.0,
            )
        )
    else:
        sections.append(
            _section(
                "Eval win rate",
                "single seat vs five opponents; higher is better",
                body='<p class="note">No evaluation recorded yet '
                "(the first one lands on the first eval interval).</p>",
            )
        )
    sections.append(
        _section(
            "Policy loss",
            "cross-entropy against the search policy",
            series=[Series("policy loss", losses, _GREEN)],
            y_lo=0.0,
            y_hi=_ceiling(losses),
        )
    )
    sections.append(
        _section(
            "Value loss",
            "MSE against the game outcome",
            series=[Series("value loss", values, _ORANGE)],
            y_lo=0.0,
            y_hi=_ceiling(values),
        )
    )
    sections.append(
        _section(
            "Seconds per iteration",
            "self-play plus training; wall clock",
            series=[Series("seconds", seconds, _GREY)],
            y_lo=0.0,
            y_hi=_ceiling(seconds),
            y_format="{:.0f}",
        )
    )

    body = (
        '<div class="stats">' + "".join(stats) + "</div>" + "".join(sections)
    )
    return _document(title, refresh, body)


def render_file(
    log: str | Path,
    *,
    out: str | Path | None = None,
    title: str | None = None,
    refresh: int = 30,
    target_iterations: int = 0,
) -> Path:
    """Render ``log`` to an HTML file and return its path.

    The default output keeps the log's stem: ``agot.log.jsonl`` -> ``agot.log.html``.
    """
    log = Path(log)
    records = load_records(log)
    page = render(
        records,
        title=title or f"agot training - {log.stem}",
        refresh=refresh,
        target_iterations=target_iterations,
    )
    out = Path(out) if out else log.with_suffix(".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


def _points(rows: Sequence[Mapping], key: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for record in rows:
        iteration = record.get("iteration")
        value = record.get(key)
        if iteration is None or value is None:
            continue
        points.append((float(iteration), float(value)))
    return points


def _band(rows: Sequence[Mapping]) -> list[tuple[float, float, float]]:
    band: list[tuple[float, float, float]] = []
    for record in rows:
        iteration = record.get("iteration")
        low = record.get("eval_low")
        high = record.get("eval_high")
        if iteration is None or low is None or high is None:
            continue
        band.append((float(iteration), float(low), float(high)))
    return band


def _average_last(points: Sequence[tuple[float, float]], window: int) -> float | None:
    tail = [value for _, value in points[-window:]]
    if not tail:
        return None
    return sum(tail) / len(tail)


def _ceiling(points: Sequence[tuple[float, float]]) -> float:
    peak = max((value for _, value in points), default=1.0)
    return max(peak * 1.1, 0.1)


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.0f} min"
    return f"{minutes / 60.0:.1f} h"


def _stat(label: str, value: str) -> str:
    return (
        f'<div class="stat"><span class="k">{html.escape(label)}</span>'
        f'<span class="v">{html.escape(value)}</span></div>'
    )


def _section(
    heading: str,
    subtitle: str,
    *,
    series: Sequence[Series] = (),
    hlines: Sequence[tuple[float, str, str]] = (),
    legend: Sequence[tuple[str, str]] = (),
    markers: Sequence[tuple[float, float]] = (),
    body: str = "",
    y_lo: float = 0.0,
    y_hi: float = 1.0,
    y_format: str = "{:.2f}",
) -> str:
    if body:
        return (
            f"<section><h2>{html.escape(heading)}</h2>"
            f'<p class="legend">{html.escape(subtitle)}</p>{body}</section>'
        )
    all_points = [point for item in series for point in item.points]
    x_hi = max((x for x, _ in all_points), default=1.0)
    svg = _line_chart(
        series=series,
        x_lo=0.0,
        x_hi=x_hi,
        y_lo=y_lo,
        y_hi=y_hi,
        hlines=hlines,
        markers=markers,
        y_format=y_format,
    )
    keys = (
        " ".join(
            f'<span><i style="color:{colour}">&#9679;</i> {html.escape(label)}</span>'
            for colour, label in legend
        )
        if legend
        else ""
    )
    return (
        f"<section><h2>{html.escape(heading)}</h2>"
        f'<p class="legend">{html.escape(subtitle)}'
        + (f" &middot; {keys}" if keys else "")
        + f"</p>{svg}</section>"
    )


def _line_chart(
    *,
    series: Sequence[Series],
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    hlines: Sequence[tuple[float, str, str]] = (),
    markers: Sequence[tuple[float, float]] = (),
    y_format: str = "{:.2f}",
) -> str:
    plot_w = _WIDTH - _PAD_L - _PAD_R
    plot_h = _HEIGHT - _PAD_T - _PAD_B
    if x_hi <= x_lo:
        x_hi = x_lo + 1.0
    if y_hi <= y_lo:
        y_hi = y_lo + 1.0

    def sx(value: float) -> float:
        return _PAD_L + (value - x_lo) / (x_hi - x_lo) * plot_w

    def sy(value: float) -> float:
        return _PAD_T + (y_hi - value) / (y_hi - y_lo) * plot_h

    parts = [
        f'<svg viewBox="0 0 {_WIDTH} {_HEIGHT}" role="img" '
        'xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="{_PAD_L}" y="{_PAD_T}" width="{plot_w}" height="{plot_h}" '
        'fill="none" stroke="#cbd5e0"/>',
    ]
    for index in range(4):
        value = y_lo + (y_hi - y_lo) * index / 3
        y = sy(value)
        parts.append(
            f'<line x1="{_PAD_L}" y1="{y:.1f}" x2="{_PAD_L + plot_w}" y2="{y:.1f}" '
            'stroke="#e2e8f0"/>'
        )
        parts.append(
            f'<text x="{_PAD_L - 6}" y="{y + 3:.1f}" text-anchor="end" '
            f'font-size="10" fill="#718096">{y_format.format(value)}</text>'
        )
    for index in range(4):
        value = x_lo + (x_hi - x_lo) * index / 3
        parts.append(
            f'<text x="{sx(value):.1f}" y="{_PAD_T + plot_h + 16}" '
            f'text-anchor="middle" font-size="10" fill="#718096">'
            f"{int(round(value))}</text>"
        )
    for value, colour, label in hlines:
        y = sy(value)
        parts.append(
            f'<line x1="{_PAD_L}" y1="{y:.1f}" x2="{_PAD_L + plot_w}" y2="{y:.1f}" '
            f'stroke="{colour}" stroke-dasharray="5 4"/>'
        )
        parts.append(
            f'<text x="{_PAD_L + plot_w - 4}" y="{y - 4:.1f}" text-anchor="end" '
            f'font-size="10" fill="{colour}">{html.escape(label)}</text>'
        )
    for item in series:
        if not item.band:
            continue
        up = " ".join(f"{sx(x):.1f},{sy(high):.1f}" for x, _, high in item.band)
        down = " ".join(
            f"{sx(x):.1f},{sy(low):.1f}" for x, low, _ in reversed(item.band)
        )
        parts.append(
            f'<polygon points="{up} {down}" fill="{item.colour}" opacity="0.16"/>'
        )
    for item in series:
        if not item.points:
            continue
        if len(item.points) == 1:
            x, y = item.points[0]
            parts.append(
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.5" '
                f'fill="{item.colour}"/>'
            )
            continue
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in item.points)
        parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{item.colour}" '
            'stroke-width="1.6" stroke-linejoin="round"/>'
        )
    for x, y in markers:
        parts.append(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="none" '
            'stroke="#1a202c" stroke-width="1.4"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _document(title: str, refresh: int, body: str) -> str:
    meta = (
        f'<meta http-equiv="refresh" content="{int(refresh)}">' if refresh > 0 else ""
    )
    hint = (
        f"Auto-refreshes every {int(refresh)} s."
        if refresh > 0
        else "Auto-refresh disabled."
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"{meta}\n"
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{_STYLE}</style>\n</head>\n<body>\n"
        f"<h1>{html.escape(title)}</h1>\n"
        f'<p class="sub">{html.escape(hint)} '
        "This page is a snapshot of the JSONL run log, so a reload only shows "
        "the last time it was rendered. Regenerate it with <code>agot report "
        "--watch</code> to follow a live run.</p>\n"
        f"{body}\n"
        '<p class="note">Reading: the eval chart marks the parity line (a '
        "coin-flip six-player game) and the greedy baseline. A candidate is "
        "useful once its interval clears the greedy line.</p>\n"
        "</body>\n</html>\n"
    )
