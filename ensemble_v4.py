"""Range of outcomes for pyworld4 (v4: v2 economy + re-fitted demographic tables), GLUE method.

Method (Beven & Binley 1992, Generalized Likelihood Uncertainty Estimation):
  1. Latin-hypercube sample N sets of 25 numbers around the v4 optimum:
       - v4's 13 numbers (M1..M4 / SFSN / FCE_TOCLIP warps, dcfsn, mtfn):
         +/- SPAN x (width of their calibration bounds), since warps sit near 0
       - v2's 12 other scalars (economy etc.): +/- SPAN relative, clipped to bounds
  2. Score each with v4's own objective on 1960-2014 data only:
       J = OS/OS_ref + R/R_ref + 0.5 C/C_ref (calibrate_v4.py)
  3. Keep sets with J <= J_best * (1 + TOL).
  4. Run each kept set to 2100 with a 20-year data spin-up of the age groups
     (2003..2023), and report 5/50/95th percentiles per year.

The band is parameter uncertainty only. It does NOT include the structural
bias found in held-out testing (about -0.16%/yr population drift from World3's
fixed-fraction age-group transitions), so the true uncertainty is wider.

Usage: python ensemble_v4.py [--n 1500] [--span 0.05] [--tol 0.10] [--workers 4]
Outputs: results/ensemble_v4.json, results/ensemble_v4_bands.csv, results/fig_ensemble_v4.png
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

import calibrate as C2
import calibrate_v4 as V4
from model import run_world3
from targets import assimilate_from

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
BAND_VARS = ["pop", "p1", "p2", "p3", "p4", "le", "cbr", "cdr", "io", "so", "iopc", "f", "fpc", "ppolx", "nrur", "nrfr"]
SPIN = 20

REC = json.loads((RES / "v4.json").read_text())
X4 = np.array([REC["x_v4"][n] for n in V4.NAMES])
SCALARS = [n for n in C2.NAMES if n not in V4.NAMES]  # v2 scalars not re-tuned in v4
S0 = np.array([V4.V2[n] for n in SCALARS])
NAMES = V4.NAMES + SCALARS


def unpack(x) -> tuple[dict, dict]:
    p, t = V4.build(x[:len(V4.NAMES)])
    p = {**p, **{n: float(v) for n, v in zip(SCALARS, x[len(V4.NAMES):])}}
    return p, t


def _score(x) -> float:
    try:
        p, t = unpack(np.asarray(x))
        os_, r_, _ = V4.onestep_and_rates(p, t, V4.TRAIN[1], V4.TRAIN_RATE_YEARS)
        j = os_ / V4.OS_REF + r_ / V4.R_REF + 0.5 * V4.curve(p, t, "train") / V4.C_REF
        return j if np.isfinite(j) else 1e3
    except Exception:
        return 1e3


def _project(x):
    p, t = unpack(np.asarray(x))
    r = run_world3(p, year_max=2100, restart=assimilate_from(V4.EMP, V4.LAST, start=V4.LAST - SPIN), tables=t)
    r.index = r.index.astype(int)
    return r[BAND_VARS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--span", type=float, default=0.05)
    ap.add_argument("--tol", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    b4 = np.array([V4.BOUNDS[n] for n in V4.NAMES])
    w4 = a.span * (b4[:, 1] - b4[:, 0])
    bs = np.array([C2.BOUNDS[n] for n in SCALARS])
    lo = np.concatenate([np.maximum(b4[:, 0], X4 - w4), np.maximum(bs[:, 0], S0 * (1 - a.span))])
    hi = np.concatenate([np.minimum(b4[:, 1], X4 + w4), np.minimum(bs[:, 1], S0 * (1 + a.span))])
    xbest = np.concatenate([X4, S0])
    X = np.vstack([xbest, qmc.scale(qmc.LatinHypercube(d=len(NAMES), seed=a.seed).random(a.n), lo, hi)])

    t0 = time.time()
    with Pool(a.workers) as pool:
        scores = np.array(pool.map(_score, list(X), chunksize=16))
    f_best = float(scores.min())
    thresh = f_best * (1 + a.tol)
    keep = np.where(scores <= thresh)[0]
    note = f"accepted {len(keep)}/{len(X)} sets with J <= {thresh:.4f} (best {f_best:.4f} x {1 + a.tol:.2f}; v4 optimum J={scores[0]:.4f})"
    if len(keep) < 30:
        keep = np.argsort(scores)[:30]
        note += f" -> FEWER THAN 30; using the 30 best (worst kept {scores[keep].max():.4f})"
    print(note, f"[{time.time() - t0:.0f}s]", flush=True)

    with Pool(a.workers) as pool:
        runs = pool.map(_project, [X[i] for i in keep], chunksize=4)
    cube = {v: pd.concat([r[v] for r in runs], axis=1) for v in BAND_VARS}
    members = pd.concat({v: cube[v].set_axis(range(len(runs)), axis=1) for v in ("pop", "p1", "p2", "p3", "p4", "le", "cbr", "cdr")},
                        axis=1)
    members.to_csv(RES / "ensemble_v4_members.csv")
    pd.DataFrame(X[keep], columns=NAMES).assign(J=scores[keep]).to_csv(RES / "ensemble_v4_kept_sets.csv", index=False)
    rows = []
    for v, df in cube.items():
        q = df.quantile([0.05, 0.5, 0.95], axis=1).T
        q.columns = ["p05", "p50", "p95"]
        q["var"] = v
        rows.append(q.reset_index().rename(columns={"index": "year"}))
    bands = pd.concat(rows, ignore_index=True)
    bands.to_csv(RES / "ensemble_v4_bands.csv", index=False)

    def band(v, y):
        s = bands[(bands["var"] == v) & (bands["year"] == y)]
        return {k: float(s[k].iloc[0]) for k in ("p05", "p50", "p95")}

    def peak_years(v):
        return [int(df_.idxmax()) for _, df_ in cube[v].loc[2000:2100].items()]

    unp = pd.read_csv(ROOT / "data" / "cohorts_un_projection.csv")
    unp = unp.set_index(unp.columns[0])
    un_cols = {c.lower(): c for c in unp.columns}
    un_pop = {int(y): float(unp.loc[y, [un_cols[c] for c in ("p1", "p2", "p3", "p4")]].sum()) for y in range(2024, 2031)}
    summary = {
        "method": "GLUE (Beven & Binley 1992) around pyworld4 v4",
        "n_sampled": int(len(X)), "span": a.span, "tol": a.tol, "seed": a.seed, "dims": NAMES,
        "best_J": f_best, "v4_optimum_J": float(scores[0]), "threshold": thresh, "n_kept": int(len(keep)), "note": note,
        "spinup": [V4.LAST - SPIN, V4.LAST],
        "param_ranges_kept": {n: [float(X[keep, i].min()), float(X[keep, i].max())] for i, n in enumerate(NAMES)},
        "y2026": {v: band(v, 2026) for v in BAND_VARS},
        "pop_2024_2030_vs_UN_projection": {y: {**band("pop", y), "UN_proj": un_pop[y],
                                               "p50_vs_UN_pct": 100 * (band("pop", y)["p50"] / un_pop[y] - 1),
                                               "UN_inside_band": band("pop", y)["p05"] <= un_pop[y] <= band("pop", y)["p95"]}
                                           for y in range(2024, 2031)},
        "y2050": {v: band(v, 2050) for v in ("pop", "le", "io", "fpc")},
        "y2100": {v: band(v, 2100) for v in ("pop", "le")},
        "peak_year_p05_p50_p95": {v: [float(np.percentile(peak_years(v), q)) for q in (5, 50, 95)]
                                  for v in ("pop", "io", "f", "ppolx")},
        "peak_pop_p05_p50_p95": [float(np.percentile(cube["pop"].loc[2000:2100].max(), q)) for q in (5, 50, 95)],
        "elapsed_s": round(time.time() - t0, 1),
    }
    (RES / "ensemble_v4.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("n_kept", "pop_2024_2030_vs_UN_projection", "peak_year_p05_p50_p95",
                                              "peak_pop_p05_p50_p95", "y2050", "y2100")}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from targets import indexed
    emp = V4.EMP
    panels = [("pop", "Population (billion)", 1e9, False), ("le", "Life expectancy (yr)", 1, False),
              ("cbr", "Births per 1000", 1, False), ("cdr", "Deaths per 1000", 1, False),
              ("io", "Industrial output (index 2000=1)", 1, True), ("fpc", "Food per capita (index 2000=1)", 1, True),
              ("ppolx", "Pollution (index 2000=1)", 1, True), ("nrfr", "Resources remaining (fraction)", 1, None)]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, (v, lab, sc, idx) in zip(axes.flat, panels):
        df = cube[v].loc[1950:2100]
        if idx:
            df = df / df.loc[2000]
        q = df.quantile([0.05, 0.5, 0.95], axis=1).T / sc
        ax.fill_between(q.index, q[0.05], q[0.95], color="#1f77b4", alpha=0.25, label="5–95% range")
        ax.plot(q.index, q[0.5], color="#1f77b4", lw=2, label="median")
        if v in emp:
            d = indexed(emp[v]) if idx else emp[v]
            ax.plot(d.index, d.values / sc, "k.", ms=3.5, label="observed")
        if v == "pop":
            ys = sorted(un_pop)
            ax.plot(ys, [un_pop[y] / sc for y in ys], "--", color="#ff7f0e", lw=1.5, label="UN WPP2024 projection")
        ax.axvline(2026.77, color="k", lw=0.8)
        ax.set_title(lab, fontsize=10); ax.set_xlim(1950, 2100)
    axes.flat[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"pyworld4 (v4): {len(keep)} settings that fit 1960–2014 equally well (GLUE), "
                 f"20-yr age-group spin-up to {V4.LAST}. Parameter uncertainty only.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(RES / "fig_ensemble_v4.png", dpi=130)

    # Zoom: population 2015-2035
    fig2, ax = plt.subplots(figsize=(9, 5))
    q = cube["pop"].loc[2015:2035].quantile([0.05, 0.5, 0.95], axis=1).T / 1e9
    ax.fill_between(q.index, q[0.05], q[0.95], color="#1f77b4", alpha=0.25, label="pyworld4 5–95%")
    ax.plot(q.index, q[0.5], color="#1f77b4", lw=2, label="pyworld4 median")
    e = emp["pop"].loc[2015:]
    ax.plot(e.index, e.values / 1e9, "k.", ms=6, label="observed (UN WPP2024 estimates)")
    ax.plot(sorted(un_pop), [un_pop[y] / 1e9 for y in sorted(un_pop)], "--", color="#ff7f0e", label="UN WPP2024 projection")
    ax.axvline(2026.77, color="k", lw=0.8)
    ax.set_title("Population 2015–2035: pyworld4 range vs UN"); ax.set_ylabel("billion"); ax.legend(fontsize=8)
    fig2.tight_layout(); fig2.savefig(RES / "fig_ensemble_v4_zoom.png", dpi=130)


if __name__ == "__main__":
    main()
