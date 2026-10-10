"""pyworld4 step 2: re-fit World3's demographic lookup tables (no source edits).

World3-03's lookup tables live in functions_table_world3.json and are swapped
at runtime (model.run_world3(tables=...)), the same mechanism as the
documented set_world3_table_functions(modified_json). Equations untouched.

Base: v2 scalar parameters (realistic economy, no pre-2040 collapse). Tuned
(13 numbers, smooth warps only, to limit overfitting):
  M1..M4      mortality by age group vs life expectancy:
              y' = y * exp(a_i + b_i * (LE - 70) / 10)            (8)
  SFSN        social family-size norm vs delayed income per capita:
              y' = y * exp(sa + sb * (DIOPC - 400) / 400)          (2)
  FCE_TOCLIP  fertility-control effectiveness: 1 - y' = (1 - y) * exp(fa)  (1)
  dcfsn, mtfn scalar fertility constants                           (2)

Fit on 1960-2014 only, with observed age groups fed in every year (data
assimilation), so every year gives a one-step-ahead test of births/deaths:
  J = OS/OS_ref + R/R_ref + 0.5 * C/C_ref     (v2 scores 2.5)
  OS = one-step error: RMS rel. error of total pop + 0.25 * sum RMS of p1..p4
  R  = mean NRMSD of crude birth rate, crude death rate, life expectancy (1970-2014)
  C  = v2's whole-curve weighted train score (free run, guard against breaking
       the other sectors)

Held-out check (2015-2023, never used): multi-year population forecasts with a
20-year data spin-up (origins 2014..2022), birth/death/life-expectancy errors,
whole-curve test score, and the 2026 figure vs the UN WPP2024 projection.

Usage: python calibrate_v4.py [--maxiter 60] [--popsize 8] [--workers 4] [--polish 300]
Writes results/v4.json and results/tables_v4.json (drop-in for
World3.set_world3_table_functions).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

from model import TABLE_X, TABLE_Y, TABLES_JSON, run_world3
from targets import TRAIN, assimilate_from, compare, load_empirical, weighted_total

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
EMP = load_empirical()
COH = ("p1", "p2", "p3", "p4")
FIRST_COH = int(max(EMP[c].index.min() for c in COH))
LAST = int(EMP["pop"].index.max())
SPIN = 20
UN2026 = {"pop": 8300678395, "p1": 1.99926e9, "p2": 3.65500e9, "p3": 1.76622e9, "p4": 8.80196e8}

V2 = json.loads((RES / "calibrated_params.json").read_text())["calibrated_params"]

BOUNDS = {**{f"m{i}_a": (-0.7, 0.7) for i in range(1, 5)},
          **{f"m{i}_b": (-0.7, 0.7) for i in range(1, 5)},
          "sfsn_a": (-0.5, 0.5), "sfsn_b": (-0.5, 0.5), "fce_a": (-1.5, 1.5),
          "dcfsn": (2.0, 6.0), "mtfn": (6.0, 14.0)}
NAMES = list(BOUNDS)
X0 = np.array([0.0] * 11 + [V2["dcfsn"], V2["mtfn"]])


def build(x) -> tuple[dict, dict]:
    v = dict(zip(NAMES, map(float, x)))
    tabs = {}
    for i in range(1, 5):
        n = f"M{i}"
        xs, ys = np.array(TABLE_X[n]), np.array(TABLE_Y[n])
        tabs[n] = list(ys * np.exp(v[f"m{i}_a"] + v[f"m{i}_b"] * (xs - 70) / 10))
    xs, ys = np.array(TABLE_X["SFSN"]), np.array(TABLE_Y["SFSN"])
    tabs["SFSN"] = list(ys * np.exp(v["sfsn_a"] + v["sfsn_b"] * (xs - 400) / 400))
    ys = np.array(TABLE_Y["FCE_TOCLIP"])
    tabs["FCE_TOCLIP"] = list(np.clip(1 - (1 - ys) * np.exp(v["fce_a"]), 0, 1))
    params = {**V2, "dcfsn": v["dcfsn"], "mtfn": v["mtfn"]}
    return params, tabs


def _obs(key, years):
    s = EMP[key]
    return np.array([s.loc[y] if y in s.index else np.nan for y in years], dtype=float)


def _nrmsd(m: np.ndarray, d: np.ndarray) -> float:
    ok = np.isfinite(m) & np.isfinite(d)
    return float(np.sqrt(np.mean((m[ok] - d[ok]) ** 2)) / np.mean(d[ok]))


def onestep_and_rates(params, tabs, end: int, rate_years) -> tuple[float, float, dict]:
    r = run_world3(params, year_max=end, restart=assimilate_from(EMP, end), tables=tabs)
    r.index = r.index.astype(int)
    yrs = list(range(FIRST_COH + 1, end + 1))
    pred = {c: r.loc[yrs, f"{c}_pred"].to_numpy() for c in COH}
    obs = {c: _obs(c, yrs) for c in COH}
    e_pop = sum(pred.values()) / sum(obs.values()) - 1
    os_ = float(np.sqrt(np.mean(e_pop ** 2)) + 0.25 * sum(np.sqrt(np.mean((pred[c] / obs[c] - 1) ** 2)) for c in COH))
    ry = list(rate_years)
    rates = {k: _nrmsd(r.loc[ry, k].to_numpy(), _obs(k, ry)) for k in ("cbr", "cdr", "le")}
    return os_, float(np.mean(list(rates.values()))), rates


def curve(params, tabs, window: str) -> float:
    ymax = TRAIN[1] if window == "train" else 2100
    return weighted_total(compare(run_world3(params, year_max=ymax, tables=tabs), EMP), window)


TRAIN_RATE_YEARS = range(1970, TRAIN[1] + 1)
_os, _r, _ = onestep_and_rates(*build(X0), TRAIN[1], TRAIN_RATE_YEARS)
OS_REF, R_REF, C_REF = _os, _r, curve(*build(X0), "train")


def objective(x) -> float:
    try:
        p, t = build(x)
        os_, r_, _ = onestep_and_rates(p, t, TRAIN[1], TRAIN_RATE_YEARS)
        j = os_ / OS_REF + r_ / R_REF + 0.5 * curve(p, t, "train") / C_REF
        return j if np.isfinite(j) else 1e3
    except Exception:
        return 1e3


def heldout(params, tabs) -> dict:
    """Everything here uses 2015+ data that tuning never saw."""
    errs = {h: [] for h in range(1, 6)}
    for o in range(TRAIN[1], LAST):
        hmax = min(5, LAST - o)
        r = run_world3(params, year_max=o + hmax, restart=assimilate_from(EMP, o, start=o - SPIN), tables=tabs)
        r.index = r.index.astype(int)
        for h in range(1, hmax + 1):
            errs[h].append(100 * (r.loc[o + h, "pop"] / EMP["pop"].loc[o + h] - 1))
    fc = {h: {"n": len(e), "mean_abs": float(np.mean(np.abs(e))), "max_abs": float(np.max(np.abs(e))),
              "signed": float(np.mean(e))} for h, e in errs.items() if e}
    os_t, _, rates_t = onestep_and_rates(params, tabs, LAST, range(TRAIN[1] + 1, LAST + 1))
    r = run_world3(params, year_max=2035, restart=assimilate_from(EMP, LAST, start=LAST - SPIN), tables=tabs)
    r.index = r.index.astype(int)
    now = {c: {"value": float(r.loc[2026, c]), "vs_UN_pct": 100 * (r.loc[2026, c] / UN2026[c] - 1)} for c in ("pop",) + COH}
    path = {y: {k: float(r.loc[y, k]) for k in ("pop", "le", "cbr", "cdr")} for y in range(LAST, 2036)}
    return {"forecast_pop_pct": fc, "rates_nrmsd_2015_2023": rates_t,
            "curve_test_nrmsd": curve(params, tabs, "test"), "nowcast2026": now, "path_2023_2035": path}


def write_tables_json(tabs: dict, path: Path) -> None:
    full = copy.deepcopy(json.loads(TABLES_JSON.read_text()))
    for t in full:
        if t["y.name"] in tabs:
            t["y.values"] = [round(float(v), 8) for v in tabs[t["y.name"]]]
    path.write_text(json.dumps(full, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--polish", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    print(f"refs (v2 base): OS={OS_REF:.5f} R={R_REF:.4f} C={C_REF:.4f}  J(v2)={objective(X0):.4f}", flush=True)
    t0, hist = time.time(), []

    def cb(xk, convergence):
        hist.append(objective(xk))
        print(f"  gen {len(hist):3d}  J={hist[-1]:.4f}  conv={convergence:.3f}  t={time.time()-t0:.0f}s", flush=True)

    b = [BOUNDS[n] for n in NAMES]
    de = differential_evolution(objective, bounds=b, x0=X0, seed=a.seed, popsize=a.popsize, maxiter=a.maxiter,
                                tol=1e-6, mutation=(0.5, 1.0), recombination=0.7, polish=False,
                                updating="deferred", workers=a.workers, callback=cb, init="latinhypercube")
    print(f"DE done: J={de.fun:.4f} nfev={de.nfev} t={time.time()-t0:.0f}s; Powell ({a.polish})...", flush=True)
    pw = minimize(objective, de.x, method="Powell", bounds=b, options={"maxfev": a.polish, "xtol": 1e-4, "ftol": 1e-6})
    xb = pw.x if pw.fun < de.fun else de.x
    p4, t4 = build(xb)
    print(f"Powell: J={pw.fun:.4f} t={time.time()-t0:.0f}s", flush=True)

    models = {"BAU": ({}, None), "v2": (V2, None), "v4": (p4, t4)}
    rec = {"created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "objective": "J = OS/OS_ref + R/R_ref + 0.5 C/C_ref",
           "refs": {"OS": OS_REF, "R": R_REF, "C": C_REF}, "bounds": BOUNDS, "x_v4": dict(zip(NAMES, map(float, xb))),
           "optimizer": {"de_nfev": int(de.nfev), "de_fun": float(de.fun), "powell_fun": float(pw.fun),
                         "elapsed_s": round(time.time() - t0, 1), "history": hist},
           "params_v4": p4, "tables_v4": t4, "train": {}, "heldout": {}}
    for k, (p, t) in models.items():
        os_, r_, rates = onestep_and_rates(p, t, TRAIN[1], TRAIN_RATE_YEARS)
        rec["train"][k] = {"onestep": os_, "rates_mean": r_, "rates": rates, "curve": curve(p, t, "train")}
        rec["heldout"][k] = heldout(p, t)
    write_tables_json(t4, RES / "tables_v4.json")
    (RES / "v4.json").write_text(json.dumps(rec, indent=2, default=float))

    print("\nHELD-OUT (2015-2023, never used in tuning)")
    print("population forecast, mean |err| % (signed) by years ahead, 20-yr data spin-up:")
    for k in models:
        f = rec["heldout"][k]["forecast_pop_pct"]
        print(f"  {k:4s} " + "  ".join(f"h{h}={f[h]['mean_abs']:.3f}({f[h]['signed']:+.3f})" for h in f))
    for k in models:
        h = rec["heldout"][k]
        print(f"  {k:4s} rates NRMSD {({q: round(v, 4) for q, v in h['rates_nrmsd_2015_2023'].items()})}"
              f"  curve test {h['curve_test_nrmsd']:.4f}  2026 pop {h['nowcast2026']['pop']['value']/1e9:.4f}B"
              f" ({h['nowcast2026']['pop']['vs_UN_pct']:+.2f}% vs UN proj)  p4 {h['nowcast2026']['p4']['vs_UN_pct']:+.1f}%"
              f"  LE 2023->2030 {h['path_2023_2035'][2023]['le']:.1f}->{h['path_2023_2035'][2030]['le']:.1f}")
    print("\nv4 x:", {n: round(v, 4) for n, v in rec["x_v4"].items()})


if __name__ == "__main__":
    main()
