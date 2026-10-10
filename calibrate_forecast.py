"""v3: calibrate World3-03 for short-horizon forecast skill (steps 1+2).

Step 1 - finish the search: widened bounds for the parameters that ended at
         their limits in v2 (pptd, imef, dcfsn), warm start from v2,
         differential evolution, then a Powell fine-search pass.
Step 2 - tune for the right goal: restart from observed age groups at origins
         inside the TRAIN window (1995, 2002, 2009) and score population error
         over the next H=5 years. Combined objective:

             J = C / C_ref + F / F_ref

         C = whole-curve weighted train NRMSD (13 targets, as in v2)
         F = forecast error: mean |pop error| over origins x horizons,
             plus 0.25 x mean |age-group error|
         C_ref = C(v2), F_ref = F(v2), so v2 scores J = 2 and each term
         counts as a relative improvement.

Nothing after 2014 is used for tuning. Fair check (written to
results/v3_forecast.json): restart at every origin 2014..2022 and measure
the 1..5-year-ahead error against UN data that was never used in tuning.

Usage: python calibrate_forecast.py [--maxiter 70] [--popsize 8] [--workers 4] [--polish 300]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

import calibrate as C2  # v2 definitions (FIXED constants, RECAL23, defaults)
from model import run_world3
from targets import TRAIN, compare, load_empirical, restart_from, weighted_total

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
EMP = load_empirical()
COH = ("p1", "p2", "p3", "p4")

ORIGINS_TRAIN = (1995, 2002, 2009)
H = 5
assert max(ORIGINS_TRAIN) + H <= TRAIN[1], "forecast windows must stay inside TRAIN"

BOUNDS = {**C2.BOUNDS,
          "pptd": (20.0, 250.0),   # v2 ended at 149.5 with bound 150
          "imef": (0.02, 0.25),    # v2 ended at 0.058 with bound 0.05
          "dcfsn": (2.0, 6.0)}     # v2 ended at 4.78, close to bound 5
NAMES = list(BOUNDS)
FIXED = C2.FIXED


def params_from(x) -> dict:
    return {**FIXED, **{n: float(v) for n, v in zip(NAMES, x)}}


def curve_score(p: dict) -> float:
    run = run_world3(p, year_max=TRAIN[1])
    return weighted_total(compare(run, EMP), "train")


def forecast_errors(p: dict, origins, horizon: int, last_year: int | None = None) -> dict:
    """{origin: {h: {'pop': rel_err, 'p1': ..}}}; restart from observed cohorts at each origin."""
    out = {}
    for o in origins:
        hmax = horizon if last_year is None else min(horizon, last_year - o)
        if hmax < 1:
            continue
        r = run_world3(p, year_max=o + hmax, restart=restart_from(EMP, o))
        r.index = r.index.astype(int)
        out[o] = {h: {"pop": r.loc[o + h, "pop"] / EMP["pop"].loc[o + h] - 1,
                      **{c: r.loc[o + h, c] / EMP[c].loc[o + h] - 1 for c in COH}}
                  for h in range(1, hmax + 1)}
    return out


def forecast_score(p: dict) -> float:
    fe = forecast_errors(p, ORIGINS_TRAIN, H)
    pop = [abs(e["pop"]) for o in fe.values() for e in o.values()]
    coh = [abs(e[c]) for o in fe.values() for e in o.values() for c in COH]
    return float(np.mean(pop) + 0.25 * np.mean(coh))


_V2 = json.loads((RES / "calibrated_params.json").read_text())["calibrated_params"]
C_REF = curve_score(_V2)
F_REF = forecast_score(_V2)


def objective(x) -> float:
    try:
        p = params_from(x)
        j = curve_score(p) / C_REF + forecast_score(p) / F_REF
        return j if np.isfinite(j) else 1e3
    except Exception:
        return 1e3


def test_skill(p: dict, horizon: int = 5) -> dict:
    """Fair check: origins 2014..last-1, errors vs UN data never used in tuning."""
    last = int(EMP["pop"].index.max())
    fe = forecast_errors(p, range(TRAIN[1], last), horizon, last_year=last)
    by_h = {}
    for h in range(1, horizon + 1):
        pe = [o[h]["pop"] for o in fe.values() if h in o]
        if pe:
            by_h[h] = {"n_origins": len(pe), "mean_abs_pct": 100 * float(np.mean(np.abs(pe))),
                       "max_abs_pct": 100 * float(np.max(np.abs(pe))), "mean_signed_pct": 100 * float(np.mean(pe))}
    return {"by_horizon": by_h,
            "per_origin_pop_pct": {o: {h: 100 * e["pop"] for h, e in d.items()} for o, d in fe.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxiter", type=int, default=70)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--polish", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    x0 = np.array([np.clip(_V2[n], *BOUNDS[n]) for n in NAMES])
    print(f"C_ref (v2 curve) = {C_REF:.4f}   F_ref (v2 forecast) = {F_REF:.5f}   J(v2) = {objective(x0):.4f}")
    t0 = time.time()
    hist = []

    def cb(xk, convergence):
        hist.append(objective(xk))
        print(f"  gen {len(hist):3d}  J={hist[-1]:.4f}  conv={convergence:.3f}  t={time.time()-t0:.0f}s", flush=True)

    de = differential_evolution(objective, bounds=[BOUNDS[n] for n in NAMES], x0=x0, seed=a.seed,
                                popsize=a.popsize, maxiter=a.maxiter, tol=1e-6, mutation=(0.5, 1.0),
                                recombination=0.7, polish=False, updating="deferred", workers=a.workers,
                                callback=cb, init="latinhypercube")
    print(f"DE done: J={de.fun:.4f} nfev={de.nfev} t={time.time()-t0:.0f}s; Powell fine-search ({a.polish} evals)...", flush=True)
    pw = minimize(objective, de.x, method="Powell", bounds=[BOUNDS[n] for n in NAMES],
                  options={"maxfev": a.polish, "xtol": 1e-4, "ftol": 1e-6})
    xb = pw.x if pw.fun < de.fun else de.x
    best = params_from(xb)
    print(f"Powell: J={pw.fun:.4f} (kept {'Powell' if pw.fun < de.fun else 'DE'}) t={time.time()-t0:.0f}s")

    bau = {}
    c_bau, f_bau = curve_score(bau), forecast_score(bau)
    rec = {
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "objective": "J = C/C_ref + F/F_ref", "C_ref": C_REF, "F_ref": F_REF,
        "origins_train": ORIGINS_TRAIN, "horizon": H, "bounds": BOUNDS,
        "optimizer": {"de_nfev": int(de.nfev), "de_nit": int(de.nit), "de_fun": float(de.fun),
                      "powell_nfev": int(pw.nfev), "powell_fun": float(pw.fun),
                      "elapsed_s": round(time.time() - t0, 1), "history": hist},
        "params_v3": best,
        "J": {"BAU": c_bau / C_REF + f_bau / F_REF, "v2": objective(x0), "v3": float(min(pw.fun, de.fun))},
        "curve_score": {"BAU": c_bau, "v2": C_REF, "v3": curve_score(best)},
        "forecast_score_train": {"BAU": f_bau, "v2": F_REF, "v3": forecast_score(best)},
        "test_skill": {"BAU": test_skill(bau), "v2": test_skill(_V2), "v3": test_skill(best)},
    }
    # Whole-curve test-window score (no restart) for completeness
    rec["curve_test_nrmsd"] = {k: weighted_total(compare(run_world3(p, year_max=2100), EMP), "test")
                               for k, p in (("BAU", bau), ("v2", _V2), ("v3", best))}
    # 2026 figure: restart from the latest observed age groups
    last = int(EMP["p1"].index.max())
    r = run_world3(best, year_max=2100, restart=restart_from(EMP, last))
    r.index = r.index.astype(int)
    rec["nowcast"] = {"restart_year": last, "pop": {y: float(r.loc[y, "pop"]) for y in range(last, 2031)},
                      **{c: float(r.loc[2026, c]) for c in COH}}
    (RES / "v3_forecast.json").write_text(json.dumps(rec, indent=2, default=float))

    print("\nFair check: mean |population error| (%) by forecast horizon, origins 2014..2022")
    print(f"{'h':>3} " + " ".join(f"{k:>10s}" for k in ("BAU", "v2", "v3")))
    for h in range(1, H + 1):
        row = [rec["test_skill"][k]["by_horizon"].get(h, {}).get("mean_abs_pct", float('nan')) for k in ("BAU", "v2", "v3")]
        print(f"{h:>3} " + " ".join(f"{v:10.3f}" for v in row))
    print("\nwhole-curve test NRMSD (no restart):", {k: round(v, 4) for k, v in rec["curve_test_nrmsd"].items()})
    print("2026 population (restart 2023):", f"{rec['nowcast']['pop'][2026]/1e9:.4f} B")
    print("\nv3 params:", {n: round(best[n], 4) for n in NAMES})


if __name__ == "__main__":
    main()
