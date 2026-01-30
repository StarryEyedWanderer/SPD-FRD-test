import json

import numpy as np
import pytest

from spd_t4.cli import analyze
from spd_t4.io.artifacts import save_series
from spd_t4.io.schema import RunMeta


def _write_series(run_dir):
    steps = 200
    t = np.arange(steps) * 0.01
    mu = np.linspace(-1, 1, steps)
    x = np.zeros(steps)
    y = np.zeros(steps)
    x[120:] = 1.0
    y[120:] = 1.0
    series = {"t": t, "mu": mu, "x": x, "y": y}
    meta = RunMeta(dt=0.01, T=steps, mu0=-1, mu1=1, omega=2, sigma=0.1, seed=1)
    save_series(run_dir, series, meta.model_dump())


def test_delta_after_baseline_for_eps_and_earlywarn(tmp_path):
    run_dir = tmp_path / "run"
    _write_series(run_dir)

    analyze(
        run=run_dir,
        window=10,
        baseline_frac=0.4,
        theta_quantile_value=0.9,
        delta_metric="eps",
        delta_quantile=0.9,
        delta_persist=0,
        event_window=10,
        event_k=3.0,
    )
    with open(run_dir / "analysis.json", encoding="utf-8") as handle:
        analysis = json.load(handle)
    assert analysis["t_delta"] is None or analysis["t_delta"] >= analysis["baseline_end"]

    analyze(
        run=run_dir,
        window=10,
        baseline_frac=0.4,
        theta_quantile_value=0.9,
        delta_metric="earlywarn",
        delta_quantile=0.9,
        delta_persist=0,
        event_window=10,
        event_k=3.0,
    )
    with open(run_dir / "analysis.json", encoding="utf-8") as handle:
        analysis = json.load(handle)
    assert analysis["t_delta"] is None or analysis["t_delta"] >= analysis["baseline_end"]


def test_empty_baseline_errors(tmp_path):
    run_dir = tmp_path / "run"
    _write_series(run_dir)
    with pytest.raises(ValueError, match="Baseline window is empty"):
        analyze(
            run=run_dir,
            window=10,
            baseline_frac=0.0,
            theta_quantile_value=0.9,
            delta_metric="eps",
            delta_quantile=0.9,
            delta_persist=0,
            event_window=10,
            event_k=3.0,
        )


def test_earlywarn_outputs(tmp_path):
    run_dir = tmp_path / "run"
    _write_series(run_dir)
    analyze(
        run=run_dir,
        window=10,
        baseline_frac=0.4,
        theta_quantile_value=0.9,
        delta_metric="earlywarn",
        delta_quantile=0.9,
        delta_persist=0,
        event_window=10,
        event_k=3.0,
    )
    with open(run_dir / "analysis.json", encoding="utf-8") as handle:
        analysis = json.load(handle)
    derived = dict(np.load(run_dir / "derived.npz"))
    assert analysis["theta_sigma"] is not None
    assert "sigma" in derived
