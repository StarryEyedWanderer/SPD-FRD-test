# SPD/FRD Test T4 — Quantity→Quality Thresholded Transitions

This mini-project implements **Test T4**: validate the Dialectical Operator (Δ) by
simulating a Hopf bifurcation under slowly drifting stress (μ), then detecting a
qualitative regime change via a tension threshold.

## Overview

**Hypothesis:** As stress accumulates, tension ε(t) crosses a threshold θ and a
qualitative transition occurs. We model a Hopf normal form (stable fixed point →
limit cycle), measure prediction error tension, and compare its threshold crossing
with a ground-truth event time derived from radius changes.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 1) Simulate

```bash
spd-t4 simulate \
  --dt 0.01 --T 2000 \
  --mu0 -1.0 --mu1 1.0 \
  --omega 2.0 --sigma 0.10 \
  --seed 1 \
  --out runs/run_001
```

### 2) Analyze

```bash
spd-t4 analyze \
  --run runs/run_001 \
  --model linear2d \
  --window 500 \
  --baseline-frac 0.25 \
  --theta-quantile 0.99 \
  --event-window 200 \
  --event-k 5.0
```

### 3) Plot

```bash
spd-t4 plot --run runs/run_001 --out figures/run_001
```

## Outputs

Each run directory contains:

* `series.npz` + `meta.json` from simulation
* `analysis.json` + `derived.npz` from analysis
* `figures/` from plotting

## Interpretation

* **Regime shift**: radius increases past baseline; event time t* detected via rolling mean.
* **Dialectical operator**: Δ triggers when ε(t) ≥ θ.
* **Success criteria**: tΔ occurs near (or before) t* with small tolerance, and lead time L = t* − tΔ is positive.

## Falsification guidance

If transitions occur but ε(t) never approaches θ across many seeds/noise levels, the
definition of tension/threshold should be revised.
