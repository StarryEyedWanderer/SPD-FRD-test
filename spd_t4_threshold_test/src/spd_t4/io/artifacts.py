import json
from pathlib import Path

import numpy as np


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_series(out_dir, series, meta):
    out_path = Path(out_dir)
    ensure_dir(out_path)
    np.savez(out_path / "series.npz", **series)
    with open(out_path / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def load_series(run_dir):
    run_path = Path(run_dir)
    series = dict(np.load(run_path / "series.npz"))
    with open(run_path / "meta.json", encoding="utf-8") as handle:
        meta = json.load(handle)
    return series, meta


def save_analysis(run_dir, analysis, derived):
    run_path = Path(run_dir)
    np.savez(run_path / "derived.npz", **derived)
    with open(run_path / "analysis.json", "w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
