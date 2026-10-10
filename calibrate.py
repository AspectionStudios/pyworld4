"""Calibrate World3-03 constants against real data (train 1970-2014, test 2015-latest).

Start point: Nebel et al. (2023) "Recalibration23" (J. Ind. Ecol., Table 2).
We re-optimize 12 influential constants with scipy differential_evolution,
using only TRAIN-window data. During optimization the model is simulated
only up to TRAIN[1], so test-period data cannot influence the fit.

Usage:
    python calibrate.py [--maxiter 40] [--popsize 8] [--workers -1] [--seed 42]
Outputs:
    results/calibrated_params.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from model import VENDOR_COMMIT, run_world3
from targets import TARGETS, TEST, TRAIN, compare, load_empirical, restart_from, weighted_total

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# Nebel et al. 2023, Table 2 ("Improved value"). Where the table rounds to 2
# decimals, the value is recomputed from the table's relative-change column
# (default * (1 + change/100)).
RECAL23 = {
    "alic": 15.24,        # avg lifetime of industrial capital (alic1)
    "pptd": 116.38,       # persistent pollution transmission delay
    "hsid": 38.24,        # health services impact delay
    "lpd": 33.84,         # lifetime perception delay
    "ppgf1": 1.5344,      # persistent pollution generation factor 1
    "lfpf": 1.0221,       # labor force participation fraction  (>1 -> unphysical, see report)
    "alln": 1351.20,      # average life of land normal
    "palt": 4.22e9,       # potentially arable land total
    "nri": 1.3023e12,     # nonrenewable resources initial
    "imti": 11.064,       # industrial material toxic index
    "imef": 0.11017,      # industrial material emission factor
    "pl": 0.10405,        # processing loss
    "frpm": 0.020668,     # fraction of res. utilization on pollution generation
    "sfpc": 233.69,       # subsistence food per capita (NB: overwritten to 230 in PyWorld3-03 population init)
    "faipm": 0.000944,    # fraction of agricultural inputs as pollution
    "sd": 0.060452,       # social discount
    "mtfn": 9.4536,       # maximum total fertility normal
    "amti": 0.76870,      # agricultural material toxicity index
    "sad": 13.384,        # social adjustment delay
    "fspd": 0.61160,      # food shortage perception delay
    "uildt": 0.5330,      # urban-industrial land development time
}

# Parameters re-optimized here: name -> (lower, upper). Bounds are kept
# physically plausible; lfpf in particular is capped below 1.
BOUNDS = {
    "alic": (8.0, 25.0),
    "alsc": (10.0, 30.0),
    "icor": (2.0, 4.5),
    "pptd": (20.0, 150.0),
    "imef": (0.05, 0.25),
    "nri": (0.7e12, 3.0e12),
    "uildt": (0.5, 15.0),
    "hsid": (10.0, 50.0),
    "lpd": (10.0, 50.0),
    "mtfn": (8.0, 14.0),
    "dcfsn": (2.5, 5.0),
    "lfpf": (0.55, 0.90),
    "len": (20.0, 40.0),   # life expectancy normal: sets the LE level directly
    "sad": (5.0, 30.0),    # social adjustment delay: speed of fertility transition
}
NAMES = list(BOUNDS)
WORLD3_03_DEFAULTS = {"alic": 14.0, "alsc": 20.0, "icor": 3.0, "pptd": 20.0, "imef": 0.1,
                      "nri": 1e12, "uildt": 10.0, "hsid": 20.0, "lpd": 20.0, "mtfn": 12.0,
                      "dcfsn": 3.8, "lfpf": 0.75, "len": 28.0, "sad": 20.0}

FIXED = {k: v for k, v in RECAL23.items() if k not in BOUNDS}
EMP = load_empirical()  # module-level so spawned worker processes load it once


def params_from(x) -> dict:
    return {**FIXED, **{n: float(v) for n, v in zip(NAMES, x)}}


def objective(x) -> float:
    try:
        run = run_world3(params_from(x), year_max=TRAIN[1])
        score = weighted_total(compare(run, EMP), "train")
        return score if np.isfinite(score) else 1e3
    except Exception:
        return 1e3


def evaluate(params: dict, restart: dict | None = None) -> dict:
    run = run_world3(params, year_max=2100, restart=restart)
    sc = compare(run, EMP)
    return {"per_target": sc,
            "total_train": weighted_total(sc, "train"),
            "total_test": weighted_total(sc, "test"),
            "restart": restart}


def warm_start() -> dict:
    """Best values from the previous run (v1) where available, else Recal23, else defaults."""
    base = {**WORLD3_03_DEFAULTS, **RECAL23}
    prev = RESULTS / "v1_calibrated_params.json"
    if prev.exists():
        base.update(json.loads(prev.read_text())["calibrated_params"])
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    ws = warm_start()
    x0 = np.array([np.clip(ws[n], *BOUNDS[n]) for n in NAMES])
    print(f"Calibrating {len(NAMES)} params on TRAIN {TRAIN}; test window {TEST[0]}-latest is held out.")
    print(f"x0 objective (warm start from v1): {objective(x0):.4f}")

    t0 = time.time()
    history = []

    def cb(xk, convergence):
        f = objective(xk)
        history.append(f)
        print(f"  gen {len(history):3d}  best train NRMSD={f:.4f}  conv={convergence:.3f}  t={time.time()-t0:.0f}s", flush=True)

    res = differential_evolution(
        objective, bounds=[BOUNDS[n] for n in NAMES], x0=x0, seed=a.seed,
        popsize=a.popsize, maxiter=a.maxiter, tol=1e-5, mutation=(0.5, 1.0),
        recombination=0.7, polish=False, updating="deferred", workers=a.workers,
        callback=cb, init="latinhypercube",
    )
    elapsed = time.time() - t0

    calibrated = params_from(res.x)
    v1 = json.loads((RESULTS / "v1_calibrated_params.json").read_text())["calibrated_params"] \
        if (RESULTS / "v1_calibrated_params.json").exists() else None
    scenarios = {
        "BAU_World3-03_default": {},
        "Recalibration23_Nebel2023": dict(RECAL23),
        **({"Calibrated_v1": v1} if v1 else {}),
        "Calibrated_2026": calibrated,
    }
    evals = {k: evaluate(p) for k, p in scenarios.items()}
    # Restart validation: swap in observed cohorts at the END of training (2014),
    # then score 2015-latest. Those years were never seen, so this is a fair test.
    rs_val = restart_from(EMP, TRAIN[1])
    evals["Calibrated_2026+restart2014"] = evaluate(calibrated, rs_val)
    evals["BAU+restart2014"] = evaluate({}, rs_val)
    # Production nowcast: restart at the latest observed cohort year. Its
    # test score is NOT a fair test (2023 is in the test window), so it's
    # stored only for the 2026 figures.
    last_coh = int(EMP["p1"].index.max())
    rs_now = restart_from(EMP, last_coh)

    manifest = (ROOT / "data" / "manifest.json").read_bytes()
    out = {
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "vendor": {"repo": "https://github.com/TimSchell98/PyWorld3-03", "commit": VENDOR_COMMIT},
        "data_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "train_window": TRAIN, "test_window_start": TEST[0],
        "targets": [t.__dict__ for t in TARGETS],
        "optimizer": {"method": "scipy.optimize.differential_evolution", "seed": a.seed,
                      "popsize": a.popsize, "maxiter": a.maxiter, "nfev": int(res.nfev),
                      "nit": int(res.nit), "success": bool(res.success), "message": str(res.message),
                      "elapsed_s": round(elapsed, 1), "cpu_count": os.cpu_count(),
                      "history_train_nrmsd": history},
        "bounds": BOUNDS,
        "world3_03_defaults": WORLD3_03_DEFAULTS,
        "recalibration23": RECAL23,
        "calibrated_params": calibrated,
        "scenarios": scenarios,
        "restart_validation": rs_val,
        "restart_nowcast": rs_now,
        "evaluation": evals,
    }
    (RESULTS / "calibrated_params.json").write_text(json.dumps(out, indent=2, default=float))

    print(f"\nDone in {elapsed:.0f}s, nfev={res.nfev}.")
    print(f"{'scenario':30s} {'train':>8s} {'test':>8s}")
    for k, e in evals.items():
        print(f"{k:30s} {e['total_train']:8.4f} {e['total_test']:8.4f}")
    print("\nCalibrated params:")
    for n in NAMES:
        print(f"  {n:6s} default={WORLD3_03_DEFAULTS[n]:<10.4g} recal23={RECAL23.get(n, float('nan')):<10.4g} -> {calibrated[n]:.4g}")


if __name__ == "__main__":
    main()
