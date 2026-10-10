"""Widen the pyworld4 GLUE range with the model's own held-out forecast errors.

The GLUE band (ensemble_v4.py) covers parameter uncertainty only. Held-out
testing showed a structural population drift. This script adds it:

  1. Held-out residuals of the v4 optimum, same setup as the GLUE run
     (20-year age-group spin-up): for every origin o = 2014..2022 and horizon
     h = 1..(2023 - o), e[o,h] = model / observed - 1. None of these years were
     used in fitting.
  2. Error model: each origin has its own drift rate d_o (least squares
     e[o,h] ~ d_o * h). Across origins: mean d, spread s_d; leftover noise s0.
     Predicted error at horizon h for a new origin:
         e_h = d*h + t_{n-1} * sqrt(s_d^2 (1 + 1/n) h^2 + s0^2)
     (Student-t because n = 9 origins is small.)
  3. Widened band: each of the 30 GLUE members x 2000 draws of e_h,
     corrected population = member / (1 + e_h). Percentiles 5/50/95.
  4. Check: leave-one-origin-out. Refit the error model without origin o and
     test whether o's observed values fall inside the predicted 90% range.

Usage: python widen_v4.py
Outputs: results/ensemble_v4_widened.json, results/fig_ensemble_v4_widened.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from model import run_world3
from targets import assimilate_from, load_empirical

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
EMP = load_empirical()
LAST = int(EMP["pop"].index.max())
SPIN = 20
ORIGINS = list(range(2014, LAST))
DRAWS = 2000
rng = np.random.default_rng(2026)

V4 = json.loads((RES / "v4.json").read_text())
P4, T4 = V4["params_v4"], V4["tables_v4"]


def residuals() -> dict[int, dict[int, float]]:
    out = {}
    for o in ORIGINS:
        r = run_world3(P4, year_max=LAST, restart=assimilate_from(EMP, o, start=o - SPIN), tables=T4)
        r.index = r.index.astype(int)
        out[o] = {h: float(r.loc[o + h, "pop"] / EMP["pop"].loc[o + h] - 1) for h in range(1, LAST - o + 1)}
    return out


def fit(res: dict[int, dict[int, float]]) -> dict:
    d = {}
    noise = []
    for o, eh in res.items():
        h = np.array(list(eh)); e = np.array(list(eh.values()))
        d[o] = float((h * e).sum() / (h * h).sum())
        noise.extend(e - d[o] * h)
    dv = np.array(list(d.values()))
    n = len(dv)
    return {"d_per_origin": d, "d_mean": float(dv.mean()), "s_d": float(dv.std(ddof=1)),
            "s0": float(np.sqrt(np.mean(np.square(noise)))), "n": n}


def err_scale(m: dict, h: np.ndarray) -> np.ndarray:
    return np.sqrt(m["s_d"] ** 2 * (1 + 1 / m["n"]) * h ** 2 + m["s0"] ** 2)


def main() -> None:
    res = residuals()
    m = fit(res)
    print(f"held-out drift: mean {100*m['d_mean']:+.4f}%/yr, spread {100*m['s_d']:.4f}%/yr, "
          f"noise {100*m['s0']:.4f}%, n={m['n']} origins")
    print("  per origin %/yr:", {o: round(100 * v, 3) for o, v in m["d_per_origin"].items()})

    # Leave-one-origin-out coverage of the 90% range
    hits = tot = 0
    loo = {}
    for o in ORIGINS:
        mo = fit({k: v for k, v in res.items() if k != o})
        tq = stats.t.ppf([0.05, 0.95], df=mo["n"] - 1)
        for h, e in res[o].items():
            lo, hi = mo["d_mean"] * h + tq * err_scale(mo, np.array(h, float))
            inside = bool(lo <= e <= hi)
            hits += inside; tot += 1
            loo.setdefault(o, {})[h] = {"err_pct": 100 * e, "lo_pct": 100 * lo, "hi_pct": 100 * hi, "inside": inside}
    print(f"leave-one-origin-out: {hits}/{tot} held-out errors inside the predicted 90% range ({100*hits/tot:.0f}%)")

    mem = pd.read_csv(RES / "ensemble_v4_members.csv", header=[0, 1], index_col=0)
    mem.index = mem.index.astype(int)
    pop = mem["pop"]
    years = list(range(LAST, 2041))
    tq_draws = stats.t.rvs(df=m["n"] - 1, size=DRAWS, random_state=rng)
    unp = pd.read_csv(ROOT / "data" / "cohorts_un_projection.csv").set_index("Year")
    un = {int(y): float(unp.loc[y, ["p1", "p2", "p3", "p4"]].sum()) for y in unp.index if y <= 2040}
    rows = {}
    for y in years:
        h = y - LAST
        P = pop.loc[y].to_numpy()
        if h == 0:
            samp = P
        else:
            e = m["d_mean"] * h + tq_draws * err_scale(m, np.array(h, float))
            samp = (P[:, None] / (1 + e[None, :])).ravel()
        q = np.percentile(samp, [5, 50, 95])
        q_par = np.percentile(P, [5, 50, 95])
        rows[y] = {"param_only": dict(zip(("p05", "p50", "p95"), map(float, q_par))),
                   "widened": dict(zip(("p05", "p50", "p95"), map(float, q)))}
        if y in un:
            rows[y]["UN_proj"] = un[y]
            rows[y]["UN_inside_param_only"] = bool(q_par[0] <= un[y] <= q_par[2])
            rows[y]["UN_inside_widened"] = bool(q[0] <= un[y] <= q[2])
            rows[y]["widened_p50_vs_UN_pct"] = float(100 * (q[1] / un[y] - 1))

    out = {"method": __doc__.split("\n\n")[1].strip(), "error_model": m, "loo_coverage_90": {"hits": hits, "total": tot},
           "loo_detail": loo, "residuals_pct": {o: {h: 100 * e for h, e in d.items()} for o, d in res.items()},
           "bands": rows}
    (RES / "ensemble_v4_widened.json").write_text(json.dumps(out, indent=2))

    print(f"\n{'year':>4}  {'parameter-only 5/50/95 (B)':>28}  {'widened 5/50/95 (B)':>28}  {'UN proj':>7}  in-widened  median-vs-UN")
    for y in [2024, 2025, 2026, 2027, 2028, 2030, 2035, 2040]:
        r = rows[y]; a, b = r["param_only"], r["widened"]
        unv = r.get("UN_proj")
        print(f"{y:>4}  {a['p05']/1e9:8.3f} {a['p50']/1e9:8.3f} {a['p95']/1e9:8.3f}    {b['p05']/1e9:8.3f} {b['p50']/1e9:8.3f} {b['p95']/1e9:8.3f}"
              + (f"  {unv/1e9:7.3f}  {str(r['UN_inside_widened']):>10}  {r['widened_p50_vs_UN_pct']:+.2f}%" if unv else ""))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ys = np.array(years)
    g = lambda k, q: np.array([rows[y][k][q] for y in years]) / 1e9  # noqa: E731
    ax.fill_between(ys, g("widened", "p05"), g("widened", "p95"), color="#2ca02c", alpha=0.2,
                    label="widened 5–95% (settings + held-out model error)")
    ax.plot(ys, g("widened", "p50"), color="#2ca02c", lw=2, label="widened median (drift-corrected)")
    ax.fill_between(ys, g("param_only", "p05"), g("param_only", "p95"), color="#1f77b4", alpha=0.35,
                    label="settings-only 5–95% (GLUE)")
    ax.plot(ys, g("param_only", "p50"), color="#1f77b4", lw=1.5, ls=":", label="GLUE median")
    e = EMP["pop"].loc[2015:]
    ax.plot(e.index, e.values / 1e9, "k.", ms=6, label="observed (UN WPP2024 estimates)")
    uy = sorted(un)
    ax.plot(uy, [un[y] / 1e9 for y in uy], "--", color="#ff7f0e", lw=1.5, label="UN WPP2024 projection")
    ax.axvline(2026.77, color="k", lw=0.8)
    ax.set_xlim(2015, 2040); ax.set_ylabel("billion")
    ax.set_title(f"pyworld4 population: settings-only vs widened range "
                 f"(leave-one-out 90% coverage {100*hits/tot:.0f}%)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(RES / "fig_ensemble_v4_widened.png", dpi=130)


if __name__ == "__main__":
    main()
