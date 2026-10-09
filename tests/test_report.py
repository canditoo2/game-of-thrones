"""Tests for the HTML training report.

The report is deliberately torch-free, so these tests run everywhere.
"""

from __future__ import annotations

import pytest

from agot.training.report import (
    GREEDY_BASELINE,
    PARITY,
    load_records,
    log_path_for,
    render,
    render_file,
)


def _records() -> list[dict]:
    return [
        {"iteration": 1, "seconds": 80.0, "policy_loss": 2.0, "value_loss": 0.40, "buffer": 100},
        {"iteration": 2, "seconds": 90.0, "policy_loss": 1.9, "value_loss": 0.30, "buffer": 200},
        {
            "iteration": 3,
            "seconds": 70.0,
            "policy_loss": 1.8,
            "value_loss": 0.20,
            "buffer": 300,
            "eval": 0.25,
            "eval_low": 0.12,
            "eval_high": 0.45,
        },
        {"iteration": 4, "seconds": 85.0, "policy_loss": 1.7, "value_loss": 0.10, "buffer": 400},
        {
            "iteration": 5,
            "seconds": 88.0,
            "policy_loss": 1.6,
            "value_loss": 0.15,
            "buffer": 500,
            "eval": 0.29,
            "eval_low": 0.15,
            "eval_high": 0.49,
        },
    ]


def test_log_path_for_mirrors_checkpoint_helper(tmp_path) -> None:
    pytest.importorskip("torch")
    from agot.training.checkpoint import log_path

    assert log_path_for("ckpt/agot.pt") == log_path("ckpt/agot.pt")
    assert log_path_for(tmp_path / "run.pt").name == "run.log.jsonl"


def test_load_records_skips_blanks_and_garbage(tmp_path) -> None:
    path = tmp_path / "run.log.jsonl"
    path.write_text(
        '{"iteration": 1}\n\n  \nnot json\n{"iteration": 2}\n',
        encoding="utf-8",
    )
    assert [record["iteration"] for record in load_records(path)] == [1, 2]


def test_load_records_missing_file_is_empty(tmp_path) -> None:
    assert load_records(tmp_path / "absent.log.jsonl") == []


def test_render_contains_data_and_charts() -> None:
    page = render(_records(), title="run2", refresh=30, target_iterations=120)

    assert page.startswith("<!DOCTYPE html>")
    assert "<title>run2</title>" in page
    assert page.count("<svg") == 4  # eval, policy, value, seconds
    assert "0.290" in page  # last/best eval from the stats row
    assert "win rate" in page and "greedy baseline" in page
    assert "best eval" in page
    assert "120" in page  # the target in the iteration stat
    assert 'content="30"' in page


def test_render_respects_refresh_zero() -> None:
    page = render(_records(), refresh=0)
    assert 'http-equiv="refresh"' not in page
    assert "Auto-refresh disabled." in page


def test_render_without_evals_still_renders() -> None:
    page = render([{"iteration": 1, "seconds": 5.0, "policy_loss": 2.0, "value_loss": 0.5}])
    assert "<svg" in page
    assert "No evaluation recorded yet" in page


def test_render_empty_log() -> None:
    page = render([])
    assert "No log records yet" in page
    assert page.startswith("<!DOCTYPE html>")


def test_render_handles_partial_records() -> None:
    page = render(
        [
            {"iteration": 1, "policy_loss": 2.0},
            {"iteration": 2, "value_loss": 0.5},
            {"iteration": 3},
        ]
    )
    assert page.startswith("<!DOCTYPE html>")


def test_render_file_defaults_to_html_sibling(tmp_path) -> None:
    log = tmp_path / "agot.log.jsonl"
    log.write_text('{"iteration": 1, "policy_loss": 2.0}\n', encoding="utf-8")

    out = render_file(log)
    assert out == tmp_path / "agot.log.html"
    assert out.exists()
    assert "<svg" in out.read_text(encoding="utf-8")


def test_parity_and_greedy_are_distinct_references() -> None:
    assert PARITY == pytest.approx(1 / 6, abs=1e-9)
    assert GREEDY_BASELINE > PARITY
