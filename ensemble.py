"""Range of outcomes for the calibrated World3-03 (GLUE method).

Method (Beven & Binley 1992, Generalized Likelihood Uncertainty Estimation):
  1. Latin-hypercube sample N parameter sets in a box of +/-SPAN around the
     calibrated optimum (clipped to the calibration bounds).
  2. Score each on TRAIN-window NRMSD only (same objective as calibrate.py).
  3. Keep the sets whose score is <= best * (1 + TOL). These all fit the data
     about equally well.
  4. Run each kept set to 2100, restarting from the observed cohorts at the
     latest data year, and report the 5/50/95th percentiles per year.

The band shows parameter uncertainty only: the spread across settings the
data can't tell apart. It does NOT include the model's structural error (see
the food-per-capita and fossil-fuel mismatches), so the true uncertainty is
wider.

Usage: python ensemble.py [--n 1200] [--span 0.25] [--tol 0.10] [--workers 4]
Outputs: results/ensemble.json, results/ensemble_bands.csv, results/fig_ensemble.png
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

import calibrate as C
from model import run_world3

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
BAND_VARS = ["pop", "le", "cbr", "cdr", "io", "so", "iopc", "f", "fpc", "ppolx", "nrur", "nrfr"]


def _score(x):
    return C.objective(np.asarray(x))


def _project(args):
    params, rs = args
    r = run_world3(params, year_max=2100, restart=rs)
    r.index = r.index.astype(int)
    return r[BAND_VARS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--span", type=float, default=0.25)
    ap.add_argument("--tol", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    R = json.loads((RES / "calibrated_params.json").read_text())
    best = R["calibrated_params"]
    rs_now = R["restart_nowcast"]
    names = C.NAMES
    lo = np.array([max(C.BOUNDS[n][0], best[n] * (1 - a.span)) for n in names])
    hi = np.array([min(C.BOUNDS[n][1], best[n] * (1 + a.span)) for n in names])
    X = qmc.scale(qmc.LatinHypercube(d=len(names), seed=a.seed).random(a.n), lo, hi)
    xbest = np.array([best[n] for n in names])
    X = np.vstack([xbest, X])  # include the optimum itself

    t0 = time.time()
    with Pool(a.workers) as pool:
        scores = np.array(pool.map(_score, list(X), chunksize=16))
    f_best = float(scores.min())
    thresh = f_best * (1 + a.tol)
    keep = np.where(scores <= thresh)[0]
    note = f"accepted {len(keep)}/{len(X)} sets with train NRMSD <= {thresh:.4f} (best {f_best:.4f} * {1+a.tol:.2f})"
    if len(keep) < 30:  # too few to estimate percentiles: fall back to the 30 best, and say so
        keep = np.argsort(scores)[:30]
        note += f" -> FEWER THAN 30; using the 30 best (worst kept {scores[keep].max():.4f})"
    print(note, f"[{time.time()-t0:.0f}s]")

    sets = [{**C.FIXED, **{n: float(v) for n, v in zip(names, X[i])}} for i in keep]
    with Pool(a.workers) as pool:
        runs = pool.map(_project, [(p, rs_now) for p in sets], chunksize=4)
    cube = {v: pd.concat([r[v] for r in runs], axis=1) for v in BAND_VARS}
    rows = []
    for v, df in cube.items():
        q = df.quantile([0.05, 0.5, 0.95], axis=1).T
        q.columns = ["p05", "p50", "p95"]
        q["var"] = v
        rows.append(q.reset_index().rename(columns={"index": "year"}))
    bands = pd.concat(rows, ignore_index=True)
    bands.to_csv(RES / "ensemble_bands.csv", index=False)

    def peak_years(v):
        return [int(df_.idxmax()) for _, df_ in cube[v].loc[1950:2100].items()]

    summary = {
        "method": "GLUE (Beven & Binley 1992)",
        "n_sampled": int(len(X)), "span": a.span, "tol": a.tol, "seed": a.seed,
        "best_train_nrmsd": f_best, "threshold": thresh, "n_kept": int(len(keep)), "note": note,
        "restart": rs_now,
        "param_ranges_kept": {n: [float(X[keep, i].min()), float(X[keep, i].max())] for i, n in enumerate(names)},
        "y2026": {v: {k: float(bands[(bands["var"] == v) & (bands["year"] == 2026)][k].iloc[0])
                      for k in ("p05", "p50", "p95")} for v in BAND_VARS},
        "peak_year_p05_p50_p95": {v: [float(np.percentile(peak_years(v), q)) for q in (5, 50, 95)]
                                  for v in ("pop", "io", "f", "ppolx")},
        "elapsed_s": round(time.time() - t0, 1),
    }
    (RES / "ensemble.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("n_kept", "y2026", "peak_year_p05_p50_p95")}, indent=1))

    # Figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from targets import indexed, load_empirical
    emp = load_empirical()
    panels = [("pop", "Population (billion)", 1e9, False), ("le", "Life expectancy (yr)", 1, False),
              ("io", "Industrial output (index 2000=1)", 1, True), ("fpc", "Food per capita (index 2000=1)", 1, True),
              ("ppolx", "Pollution (index 2000=1)", 1, True), ("nrfr", "Resources remaining (fraction)", 1, None)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (v, lab, sc, idx) in zip(axes.flat, panels):
        df = cube[v].loc[1950:2100]
        if idx:
            df = df / df.loc[2000]
        q = df.quantile([0.05, 0.5, 0.95], axis=1).T / sc
        ax.fill_between(q.index, q[0.05], q[0.95], color="#d62728", alpha=0.25, label="5–95% range")
        ax.plot(q.index, q[0.5], color="#d62728", lw=2, label="median")
        if v in emp:
            d = indexed(emp[v]) if idx else emp[v]
            ax.plot(d.index, d.values / sc, "k.", ms=3.5, label="observed")
        ax.axvline(2026.77, color="k", lw=0.8)
        ax.set_title(lab, fontsize=10); ax.set_xlim(1950, 2100)
    axes.flat[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Calibrated World3-03: {len(keep)} parameter sets that fit equally well (GLUE), "
                 f"restarted from observed {rs_now['year']} age groups", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(RES / "fig_ensemble.png", dpi=130)


if __name__ == "__main__":
    main()
