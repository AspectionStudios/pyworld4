"""pyworld4 step 1: data-driven spin-up (assimilation) before forecasting.

For each forecast origin o (2014..2022, all held out from tuning) the model's
four age cohorts are overwritten with UN WPP2024 estimates every year from
o - L .. o (L = spin-up length). L = 0 is the old single restart. Then the model
runs free for 1..5 years and is scored against observed population.

Also: 2026 figure from origin 2023, compared with the UN WPP2024 *projection*.

Usage: python option1_assimilate.py
Writes results/option1_assimilate.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model import run_world3
from targets import assimilate_from, load_empirical

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
EMP = load_empirical()
COH = ("p1", "p2", "p3", "p4")
H = 5
LAST = int(EMP["pop"].index.max())
FIRST_COH = int(max(EMP[c].index.min() for c in COH))
SPINUPS = {"single (L=0)": 0, "L=5": 5, "L=20": 20, f"full (from {FIRST_COH})": None}
UN2026 = {"pop": 8300678395, "p1": 1.99926e9, "p2": 3.65500e9, "p3": 1.76622e9, "p4": 8.80196e8}

PARAMS = {
    "BAU": {},
    "v2": json.loads((RES / "calibrated_params.json").read_text())["calibrated_params"],
    "v3": json.loads((RES / "v3_forecast.json").read_text())["params_v3"],
}


def spin(o: int, L: int | None) -> list[dict]:
    return assimilate_from(EMP, o, start=None if L is None else o - L)


def skill(p: dict, L: int | None) -> dict:
    errs = {h: [] for h in range(1, H + 1)}
    for o in range(2014, LAST):
        hmax = min(H, LAST - o)
        r = run_world3(p, year_max=o + hmax, restart=spin(o, L))
        r.index = r.index.astype(int)
        for h in range(1, hmax + 1):
            errs[h].append(100 * (r.loc[o + h, "pop"] / EMP["pop"].loc[o + h] - 1))
    return {h: {"n": len(e), "mean_abs": float(np.mean(np.abs(e))), "max_abs": float(np.max(np.abs(e))),
                "signed": float(np.mean(e))} for h, e in errs.items() if e}


def nowcast(p: dict, L: int | None) -> dict:
    r = run_world3(p, year_max=2030, restart=spin(LAST, L))
    r.index = r.index.astype(int)
    return {c: {"value": float(r.loc[2026, c]), "vs_UN_pct": 100 * (r.loc[2026, c] / UN2026[c] - 1)}
            for c in ("pop",) + COH}


def main() -> None:
    out = {}
    for pn, p in PARAMS.items():
        for sn, L in SPINUPS.items():
            out[f"{pn} | {sn}"] = {"skill": skill(p, L), "nowcast2026": nowcast(p, L)}
            s, n = out[f"{pn} | {sn}"]["skill"], out[f"{pn} | {sn}"]["nowcast2026"]
            print(f"{pn:4s} {sn:18s} " + " ".join(f"h{h}={s[h]['signed']:+.3f}" for h in s)
                  + f" | 2026 pop {n['pop']['value']/1e9:.4f}B ({n['pop']['vs_UN_pct']:+.2f}%)"
                  + f" p4 {n['p4']['vs_UN_pct']:+.1f}%", flush=True)
    (RES / "option1_assimilate.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
