import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from spd_t4.config import EventConfig, FitConfig, SimConfig, ThresholdConfig
from spd_t4.io.artifacts import load_series, save_analysis, save_series
from spd_t4.io.schema import RunMeta
from spd_t4.metrics.change_point import event_time_from_radius
from spd_t4.metrics.early_warning import rolling_ac1, rolling_var
from spd_t4.metrics.tension import sse_tension
from spd_t4.metrics.thresholds import baseline_indices, theta_quantile
from spd_t4.models.rolling_fit import rolling_linear_predictions
from spd_t4.plots.plot_diagnostics import plot_early_warning
from spd_t4.plots.plot_phase import plot_phase
from spd_t4.plots.plot_timeseries import plot_eps_theta
from spd_t4.simulate.hopf import simulate_hopf

app = typer.Typer(add_completion=False)


@app.command()
def simulate(
    dt: float = 0.01,
    T: int = 2000,
    mu0: float = -1.0,
    mu1: float = 1.0,
    omega: float = 2.0,
    sigma: float = 0.1,
    seed: int = 1,
    out: Path = typer.Option(..., help="Output run directory."),
):
    config = SimConfig(
        dt=dt,
        T=T,
        mu0=mu0,
        mu1=mu1,
        omega=omega,
        sigma=sigma,
        seed=seed,
    )
    series = simulate_hopf(config)
    meta = RunMeta(**config.model_dump()).model_dump()
    save_series(out, series, meta)
    typer.echo(f"Saved series to {out}")


@app.command()
def analyze(
    run: Path = typer.Option(..., help="Run directory."),
    model: str = typer.Option("linear2d", help="Model type."),
    window: int = 500,
    baseline_frac: float = 0.25,
    theta_quantile_value: float = typer.Option(0.99, "--theta-quantile"),
    event_window: int = 200,
    event_k: float = 5.0,
):
    series, meta = load_series(run)
    states = np.column_stack([series["x"], series["y"]])
    radius = np.sqrt(series["x"] ** 2 + series["y"] ** 2)

    fit_config = FitConfig(window_size=window, model_type=model)
    preds = rolling_linear_predictions(states, fit_config.window_size)
    eps = sse_tension(states, preds)

    threshold_config = ThresholdConfig(
        baseline_fraction=baseline_frac,
        quantile=theta_quantile_value,
    )
    base_idx = baseline_indices(series["mu"], threshold_config.baseline_fraction)
    base_idx = base_idx[~np.isnan(eps[base_idx])]
    if len(base_idx) == 0:
        base_idx = np.where(~np.isnan(eps))[0]
    theta = theta_quantile(eps, base_idx, threshold_config.quantile)

    event_config = EventConfig(rolling_window=event_window, k_sigma=event_k)
    baseline_radius = radius[base_idx]
    event_threshold = baseline_radius.mean() + event_config.k_sigma * baseline_radius.std()
    t_star = event_time_from_radius(radius, event_config.rolling_window, event_threshold)

    t_delta = int(np.where(eps >= theta)[0][0]) if np.any(eps >= theta) else None
    lead_time = None if t_star is None or t_delta is None else t_star - t_delta

    rolling_var_series = rolling_var(radius, event_config.rolling_window)
    rolling_ac1_series = rolling_ac1(radius, event_config.rolling_window)

    analysis = {
        "theta": theta,
        "t_star": t_star,
        "t_delta": t_delta,
        "lead_time": lead_time,
        "event_threshold": event_threshold,
        "baseline_indices": base_idx.tolist(),
    }
    derived = {
        "eps": eps,
        "preds": preds,
        "radius": radius,
        "rolling_var": rolling_var_series,
        "rolling_ac1": rolling_ac1_series,
    }
    save_analysis(run, analysis, derived)
    typer.echo(json.dumps(analysis, indent=2))


@app.command()
def plot(
    run: Path = typer.Option(..., help="Run directory."),
    out: Path = typer.Option(..., help="Output figures directory."),
):
    series, _ = load_series(run)
    derived = dict(np.load(run / "derived.npz"))
    with open(run / "analysis.json", encoding="utf-8") as handle:
        analysis = json.load(handle)

    out.mkdir(parents=True, exist_ok=True)
    time = series["t"]
    plot_eps_theta(
        time,
        derived["radius"],
        derived["eps"],
        analysis["theta"],
        analysis["t_star"],
        analysis["t_delta"],
        out / "eps_theta.png",
    )
    plot_phase(series["x"], series["y"], out / "phase.png")
    plot_early_warning(
        time,
        derived["rolling_var"],
        derived["rolling_ac1"],
        out / "early_warning.png",
    )
    typer.echo(f"Saved figures to {out}")


@app.command()
def sweep(
    seeds: list[int] = typer.Option(...),
    sigma: list[float] = typer.Option(...),
    out: Path = typer.Option(..., help="Output sweep directory."),
):
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in seeds:
        for noise in sigma:
            run_dir = out / f"run_seed{seed}_sigma{noise:.2f}"
            simulate(
                dt=0.01,
                T=2000,
                mu0=-1.0,
                mu1=1.0,
                omega=2.0,
                sigma=noise,
                seed=seed,
                out=run_dir,
            )
            analyze(
                run=run_dir,
                model="linear2d",
                window=500,
                baseline_frac=0.25,
                theta_quantile_value=0.99,
                event_window=200,
                event_k=5.0,
            )
            with open(run_dir / "analysis.json", encoding="utf-8") as handle:
                analysis = json.load(handle)
            analysis["seed"] = seed
            analysis["sigma"] = noise
            records.append(analysis)

    df = pd.DataFrame(records)
    df.to_csv(out / "summary.csv", index=False)
    typer.echo(f"Saved sweep summary to {out / 'summary.csv'}")
