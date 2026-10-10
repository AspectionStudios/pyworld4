"""Empirical targets + error metric shared by calibrate.py and report.py.

Each target pairs a World3 output variable with a real-data series. Variables
measured in the same units as the model (population, demographic rates, life
expectancy) are compared directly. Proxy variables whose units are abstract
in World3 (industrial output, services, food, pollution, resource use) are
compared as an index, with both model and data divided by their own value in
the anchor year, ANCHOR = 2000.

Metric: NRMSD = RMSE(model, data) / mean(data), the same as Nebel et al. (2023)
eq. 2. It is computed separately over the TRAIN and TEST windows. The data is
not smoothed: a forward-backward filter would pull test-period information
into the training years.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
ANCHOR = 2000
TRAIN = (1970, 2014)
TEST = (2015, 2030)  # truncated automatically to the last observed year


@dataclass(frozen=True)
class Target:
    key: str          # short id
    w3var: str        # World3 output column
    label: str
    weight: float
    indexed: bool     # True -> compare as index (value / value[ANCHOR])
    unit: str


TARGETS = [
    Target("pop", "pop", "Population", 1.0, False, "people"),
    Target("p1", "p1", "Population aged 0-14 (UN WPP)", 0.25, False, "people"),
    Target("p2", "p2", "Population aged 15-44 (UN WPP)", 0.25, False, "people"),
    Target("p3", "p3", "Population aged 45-64 (UN WPP)", 0.25, False, "people"),
    Target("p4", "p4", "Population aged 65+ (UN WPP)", 0.25, False, "people"),
    Target("le", "le", "Life expectancy", 0.5, False, "years"),
    Target("cbr", "cbr", "Crude birth rate", 0.5, False, "per 1000/yr"),
    Target("cdr", "cdr", "Crude death rate", 0.5, False, "per 1000/yr"),
    Target("io", "io", "Industrial output (industry value added, const. 2015 US$)", 0.5, True, "index 2000=1"),
    Target("so", "so", "Service output (services value added, const. 2015 US$)", 0.5, True, "index 2000=1"),
    Target("fpc", "fpc", "Food per capita (FAO kcal/cap/day)", 0.7, True, "index 2000=1"),
    Target("ppolx", "ppolx", "Persistent pollution (CO2 ppm above 280)", 0.5, True, "index 2000=1"),
    Target("nrur", "nrur", "Nonrenewable resource use (fossil fuel TWh)", 0.5, True, "index 2000=1"),
]


def restart_from(emp: dict[str, pd.Series], year: int) -> dict:
    """Restart values: the observed cohort stocks for `year`."""
    return {"year": year, **{c: float(emp[c].loc[year]) for c in ("p1", "p2", "p3", "p4")}}


def assimilate_from(emp: dict[str, pd.Series], end: int, start: int | None = None) -> list[dict]:
    """Observed cohort stocks for every year start..end (default start = first cohort year).

    Passed as `restart=` to run_world3, the model's cohorts are overwritten every
    year up to `end`, so internal delay/perception states are driven by the
    observed population history. Nothing after `end` is used.
    """
    first = int(max(emp[c].index.min() for c in ("p1", "p2", "p3", "p4")))
    s = first if start is None else max(start, first)
    return [restart_from(emp, y) for y in range(s, end + 1)]


def _series(name: str, col: str | None = None) -> pd.Series:
    df = pd.read_csv(DATA / f"{name}.csv")
    c = col or [x for x in df.columns if x != "Year"][0]
    s = pd.Series(df[c].values, index=df["Year"].astype(int).values, name=name).astype(float)
    return s.dropna()


def load_empirical() -> dict[str, pd.Series]:
    """Return {target_key: Series(year -> value)} of raw (un-indexed) observations from 1960 on."""
    pop = _series("population")
    gdp = _series("gdp_const")
    ind = _series("industry_share") / 100.0
    srv = _series("services_share") / 100.0
    fossil = pd.read_csv(DATA / "fossil_energy.csv").set_index("Year")[["Gas", "Oil", "Coal"]]
    fossil = fossil.dropna(how="any").sum(axis=1)
    fossil.index = fossil.index.astype(int)
    co2 = _series("co2_ppm") - 280.0
    coh = pd.read_csv(DATA / "cohorts.csv").set_index("Year")
    coh.index = coh.index.astype(int)
    emp = {
        "pop": pop,
        **{c: coh[c].astype(float) for c in ("p1", "p2", "p3", "p4")},
        "le": _series("life_expectancy"),
        "cbr": _series("birth_rate"),
        "cdr": _series("death_rate"),
        "io": (gdp * ind).dropna(),
        "so": (gdp * srv).dropna(),
        "fpc": _series("food_kcal"),
        "ppolx": co2,
        "nrur": fossil,
    }
    return {k: v[v.index >= 1960].sort_index() for k, v in emp.items()}


def indexed(s: pd.Series, anchor: int = ANCHOR) -> pd.Series:
    return s / float(s.loc[anchor])


def nrmsd(model: pd.Series, data: pd.Series, lo: int, hi: int) -> tuple[float, int]:
    yrs = [y for y in data.index if lo <= y <= hi and y in model.index]
    if not yrs:
        return float("nan"), 0
    m = model.loc[yrs].values
    d = data.loc[yrs].values
    if not np.all(np.isfinite(m)):
        return float("inf"), len(yrs)
    return float(np.sqrt(np.mean((m - d) ** 2)) / np.mean(d)), len(yrs)


def compare(run: pd.DataFrame, emp: dict[str, pd.Series]) -> dict[str, dict]:
    """Per-target train/test NRMSD for a model run (DataFrame indexed by year)."""
    run = run.copy()
    run.index = run.index.astype(int)
    out = {}
    for t in TARGETS:
        m = run[t.w3var]
        d = emp[t.key]
        if t.indexed:
            m, d = indexed(m), indexed(d)
        tr, ntr = nrmsd(m, d, *TRAIN)
        te, nte = nrmsd(m, d, *TEST)
        out[t.key] = {"train": tr, "test": te, "n_train": ntr, "n_test": nte,
                      "test_last_year": int(d.index.max())}
    return out


def weighted_total(scores: dict[str, dict], window: str) -> float:
    num = den = 0.0
    for t in TARGETS:
        v = scores[t.key][window]
        if scores[t.key][f"n_{window}"] == 0 or np.isnan(v):
            continue  # no data in this window for this target
        num += t.weight * v
        den += t.weight
    return num / den if den else float("nan")
