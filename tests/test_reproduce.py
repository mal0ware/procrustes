"""The reproduce script is deterministic: two runs write identical tables."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reproduce.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("reproduce", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_fast_runs_write_identical_tables(tmp_path: Path) -> None:
    reproduce = _load()
    first, second = tmp_path / "a", tmp_path / "b"
    reproduce.run(first, fast=True, make_figures=False)
    reproduce.run(second, fast=True, make_figures=False)
    names = sorted(p.name for p in first.iterdir())
    assert names == sorted(p.name for p in second.iterdir())
    assert {"headline.csv", "monte_carlo.csv", "pbo_sampling.csv", "paper_examples.csv"} <= set(
        names
    )
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_headline_table_says_noise(tmp_path: Path) -> None:
    """Smoke check on the fast run: the winner of a skill-less search looks significant
    on its own and is not significant after any correction."""
    reproduce = _load()
    reproduce.run(tmp_path, fast=True, make_figures=False)
    rows = {}
    for line in (tmp_path / "headline.csv").read_text(encoding="utf-8").splitlines()[1:]:
        metric, noise, control = line.split(",")
        rows[metric] = (float(noise), float(control))
    assert rows["p_single_test"][0] < 0.05
    assert rows["dsr"][0] < 0.95
    assert rows["p_bonferroni"][0] > 0.05
    assert rows["p_bhy"][0] > 0.05
    assert rows["selected_is_real"] == (0.0, 1.0)
