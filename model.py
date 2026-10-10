"""Thin wrapper around the vendored World3-03 model (PyWorld3-03).

The vendored package is also named `pyworld3`, the same as the pip
`pyworld3` (1974 model) already installed in site-packages. We put the vendor
path first on sys.path and then assert that the vendored copy is the one that
actually got imported.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

VENDOR = Path(__file__).resolve().parent / "vendor" / "PyWorld3-03"
VENDOR_COMMIT = "1f1b2f40637e2c77202df8ed12d7f9fb225340d7"

if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
for _m in [m for m in sys.modules if m == "pyworld3" or m.startswith("pyworld3.")]:
    del sys.modules[_m]  # make sure a previously imported pip pyworld3 doesn't shadow the vendored one
import pyworld3 as _pw3  # noqa: E402
from pyworld3 import World3  # noqa: E402

assert str(VENDOR) in _pw3.__file__, f"wrong pyworld3 imported: {_pw3.__file__}"

# Policy-switch parameters exist as <name>1 (before pyear) and <name>2 (after
# pyear). We always run with pyear=4000 (no policy switch), so a calibrated
# value gets written to both variants to keep them consistent.
PAIRED = {"alic", "alsc", "icor", "fioac", "scor", "alai"}

OUTPUT_VARS = ["pop", "p1", "p2", "p3", "p4", "cbr", "cdr", "le", "fpc", "ppolx", "nrur", "nrfr",
               "iopc", "sopc", "io", "f", "so", "ciopc", "ef", "hwi"]

import json as _json  # noqa: E402

from scipy.interpolate import interp1d  # noqa: E402

TABLES_JSON = VENDOR / "pyworld3" / "functions_table_world3.json"
_TABLES = _json.loads(TABLES_JSON.read_text())
TABLE_X = {t["y.name"]: list(map(float, t["x.values"])) for t in _TABLES}  # default x grids (read-only)
TABLE_Y = {t["y.name"]: list(map(float, t["y.values"])) for t in _TABLES}  # default y values (read-only)


def expand(params: dict) -> dict:
    """Map calibration names (e.g. 'alic') onto World3 kwargs ('alic1', 'alic2')."""
    kw = {}
    for k, v in params.items():
        if k in PAIRED:
            kw[f"{k}1"] = v
            kw[f"{k}2"] = v
        else:
            kw[k] = v
    return kw


def run_world3(params: dict | None = None, year_max: int = 2100, dt: float = 1.0,
               fast: bool = True, restart: dict | list | None = None,
               tables: dict[str, list[float]] | None = None) -> pd.DataFrame:
    """Run World3-03 (2004 reference structure, no policy switch).

    restart: optional {"year": Y, "p1": ..., "p2": ..., "p3": ..., "p4": ...},
        or a list of such dicts (one per year) for data assimilation.
        Restart from observed data: at year Y, the four population cohort stocks
        are replaced with the observed values. The replacement happens inside
        the state update, so total population, births, deaths and every other
        variable computed from the cohorts in that step use the observed values.
        All other stocks (capital, land, pollution, resources) evolve as normal.
        The model's own value just before each overwrite is returned as
        p1_pred..p4_pred (NaN in years without an overwrite).
    tables: optional {TABLE_NAME: [y values]} replacing the y-values of World3
        lookup tables (same x grid as functions_table_world3.json). This is the
        runtime equivalent of set_world3_table_functions(modified_json).

    Returns a DataFrame indexed by year with OUTPUT_VARS as columns.
    """
    preds: dict[str, np.ndarray] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = World3(year_min=1900, year_max=year_max, dt=dt, pyear=4000)
        w.init_world3_constants(**expand(params or {}))
        w.init_world3_variables()
        w.set_world3_table_functions()
        if tables:
            for name, yv in tables.items():
                xv = TABLE_X[name]
                if len(yv) != len(xv):
                    raise ValueError(f"table {name}: {len(yv)} y-values for {len(xv)} x-values")
                setattr(w, name.lower() + "_f", interp1d(xv, yv, bounds_error=False, fill_value=(yv[0], yv[-1])))
        w.set_world3_delay_functions()
        if restart is not None:
            if not fast:
                raise ValueError("restart requires fast=True (patched state updates)")
            entries = [restart] if isinstance(restart, dict) else list(restart)
            by_k: dict[int, dict] = {}
            for e in entries:
                ks = np.where(np.isclose(w.time, e["year"]))[0]
                if len(ks) != 1:
                    raise ValueError(f"restart year {e['year']} not on the time grid")
                by_k[int(ks[0])] = e
            for c in ("p1", "p2", "p3", "p4"):
                orig = getattr(w, f"_update_state_{c}")
                arr = getattr(w, c)
                vals = {k: float(e[c]) for k, e in by_k.items()}
                pred = preds.setdefault(c, np.full(len(w.time), np.nan))

                def patched(k, j, jk, _orig=orig, _arr=arr, _vals=vals, _pred=pred):
                    _orig(k, j, jk)
                    if k in _vals:
                        _pred[k] = _arr[k]
                        _arr[k] = _vals[k]
                setattr(w, f"_update_state_{c}", patched)  # instance attr shadows the class method
        w.run_world3(fast=fast)
    data = {v: np.asarray(getattr(w, v), dtype=float) for v in OUTPUT_VARS if hasattr(w, v)}
    data.update({f"{c}_pred": a for c, a in preds.items()})
    return pd.DataFrame(data, index=pd.Index(np.round(w.time, 6), name="year"))


if __name__ == "__main__":
    base_safe = run_world3(fast=False)
    base_fast = run_world3(fast=True)
    diff = (base_safe - base_fast).abs().max()
    print("max |safe - fast| per variable:\n", diff.to_string())
    print(base_fast.loc[[1970, 2000, 2020, 2026], ["pop", "le", "iopc", "ppolx", "nrfr"]])
