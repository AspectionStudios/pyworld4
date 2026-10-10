"""Download real empirical world series used to calibrate World3-03.

All data comes from Our World in Data (OWID) grapher CSV endpoints, which
redistribute UN WPP, FAO, NOAA, World Bank and Energy Institute data.
Nothing is synthesized here. Every file is saved verbatim under data/raw/,
and the World-level series are extracted to data/<name>.csv. Provenance
(URL, timestamp, year range, row count) goes to data/manifest.json.

Usage:  python fetch_data.py
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"
UA = {"User-Agent": "pyworld3-calibration/1.0 (research; contact: local user)"}

# name -> (OWID grapher slug, preferred entity, World3 variable, description)
SERIES: dict[str, tuple[str, str, str, str]] = {
    "population": ("population", "World", "pop", "Total population (UN WPP)"),
    "birth_rate": ("crude-birth-rate", "World", "cbr", "Crude birth rate per 1000 (UN WPP)"),
    "death_rate": ("crude-death-rate", "World", "cdr", "Crude death rate per 1000 (UN WPP)"),
    "life_expectancy": ("life-expectancy", "World", "le", "Period life expectancy at birth (UN WPP)"),
    "food_kcal": ("daily-per-capita-caloric-supply", "World", "fpc", "Food supply kcal/capita/day (FAO)"),
    "fossil_energy": ("global-fossil-fuel-consumption", "World", "nrur", "Global fossil fuel consumption TWh (Energy Institute)"),
}

# Primary-source series fetched outside OWID (the OWID slugs 404'd).
NOAA_CO2 = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.csv"
WB = "https://api.worldbank.org/v2/country/WLD/indicator/{ind}?format=json&per_page=500"
WB_SERIES: dict[str, tuple[str, str, str]] = {
    "gdp_const": ("NY.GDP.MKTP.KD", "iopc/sopc", "World GDP, constant 2015 US$ (World Bank WDI)"),
    "industry_share": ("NV.IND.TOTL.ZS", "iopc", "Industry incl. construction value added, % of GDP (World Bank WDI)"),
    "services_share": ("NV.SRV.TOTL.ZS", "sopc", "Services value added, % of GDP (World Bank WDI)"),
}

# UN World Population Prospects 2024, mid-year population by 5-year age group (thousands).
WPP_AGE = ("https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/CSV_FILES/"
           "WPP2024_PopulationByAge5GroupSex_Medium.csv.gz")
WPP_LAST_ESTIMATE_YEAR = 2023  # WPP2024: estimates through 2023, Medium-variant projections from 2024
COHORTS = {  # World3 population cohorts
    "p1": ["0-4", "5-9", "10-14"],
    "p2": ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44"],
    "p3": ["45-49", "50-54", "55-59", "60-64"],
    "p4": ["65-69", "70-74", "75-79", "80-84", "85-89", "90-94", "95-99", "100+"],
}

BASE = "https://ourworldindata.org/grapher/{slug}.csv?v=1&csvType=full&useColumnShortNames=false"


def fetch_url(url: str) -> bytes:
    last = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # network hiccup -> retry with backoff
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to download {url}: {last}")


def fetch(slug: str) -> bytes:
    return fetch_url(BASE.format(slug=slug))


def extract_world(raw: bytes, entity: str) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    ent_col = "Entity" if "Entity" in df.columns else df.columns[0]
    sub = df[df[ent_col] == entity]
    if sub.empty:
        raise ValueError(f"entity {entity!r} not found; entities sample: {df[ent_col].unique()[:10]}")
    value_cols = [c for c in sub.columns if c not in (ent_col, "Code", "Year", "Day")]
    if "Year" not in sub.columns:
        raise ValueError(f"no Year column; columns={list(sub.columns)}")
    # The first numeric value column is the indicator. Projection columns
    # (e.g. UN 'medium variant') are kept separately so they never get mixed
    # in with the observed values.
    out = sub[["Year"] + value_cols].copy()
    return out.sort_values("Year").reset_index(drop=True)


def _record(manifest, name, url, w3var, desc, df, **extra):
    manifest[name] = {
        "source_url": url,
        "world3_variable": w3var,
        "description": desc,
        "columns": list(df.columns),
        "year_min": int(df["Year"].min()),
        "year_max": int(df["Year"].max()),
        "rows": int(len(df)),
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    }
    print(f"[ok] {name:20s} {df['Year'].min()}-{df['Year'].max()} rows={len(df)} cols={list(df.columns)[1:]}")


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    failures = []

    # 1) OWID grapher series
    for name, (slug, entity, w3var, desc) in SERIES.items():
        try:
            raw = fetch(slug)
            (RAW / f"{slug}.csv").write_bytes(raw)
            df = extract_world(raw, entity)
            df.to_csv(OUT / f"{name}.csv", index=False)
            _record(manifest, name, BASE.format(slug=slug), w3var, desc, df, owid_slug=slug, entity=entity)
        except Exception as e:
            failures.append(name)
            print(f"[FAIL] {name}: {e}", file=sys.stderr)

    # 2) NOAA Mauna Loa annual-mean CO2 (ppm)
    try:
        raw = fetch_url(NOAA_CO2)
        (RAW / "co2_annmean_mlo.csv").write_bytes(raw)
        df = pd.read_csv(io.BytesIO(raw), comment="#")
        df = df.rename(columns={"year": "Year", "mean": "CO2 ppm"})[["Year", "CO2 ppm"]]
        df.to_csv(OUT / "co2_ppm.csv", index=False)
        _record(manifest, "co2_ppm", NOAA_CO2, "ppolx", "Mauna Loa annual mean CO2, ppm (NOAA GML)", df)
    except Exception as e:
        failures.append("co2_ppm")
        print(f"[FAIL] co2_ppm: {e}", file=sys.stderr)

    # 3) World Bank WDI, World aggregate (WLD)
    for name, (ind, w3var, desc) in WB_SERIES.items():
        url = WB.format(ind=ind)
        try:
            raw = fetch_url(url)
            (RAW / f"wb_{ind}.json").write_bytes(raw)
            payload = json.loads(raw)
            rows = [(int(r["date"]), float(r["value"])) for r in payload[1] if r["value"] is not None]
            df = pd.DataFrame(sorted(rows), columns=["Year", ind])
            df.to_csv(OUT / f"{name}.csv", index=False)
            _record(manifest, name, url, w3var, desc, df, wb_indicator=ind)
        except Exception as e:
            failures.append(name)
            print(f"[FAIL] {name}: {e}", file=sys.stderr)

    # 4) UN WPP 2024 age structure -> World3 cohorts p1..p4
    try:
        gz = RAW / "WPP2024_PopulationByAge5GroupSex_Medium.csv.gz"
        if not gz.exists():
            gz.write_bytes(fetch_url(WPP_AGE))
        parts = []
        for chunk in pd.read_csv(gz, compression="gzip", chunksize=200_000, low_memory=False,
                                 usecols=["LocID", "Location", "Time", "AgeGrp", "PopTotal", "Variant"]):
            parts.append(chunk[chunk["LocID"] == 900])
        w = pd.concat(parts)
        if w.empty:
            raise ValueError("World (LocID 900) not found in WPP file")
        piv = w.pivot_table(index="Time", columns="AgeGrp", values="PopTotal", aggfunc="sum")
        missing = [g for gs in COHORTS.values() for g in gs if g not in piv.columns]
        if missing:
            raise ValueError(f"missing WPP age groups: {missing}")
        coh = pd.DataFrame({c: piv[gs].sum(axis=1) * 1000.0 for c, gs in COHORTS.items()})  # thousands -> people
        coh.index = coh.index.astype(int)
        coh.index.name = "Year"
        coh["total"] = coh[list(COHORTS)].sum(axis=1)
        est = coh[coh.index <= WPP_LAST_ESTIMATE_YEAR].reset_index()
        proj = coh[coh.index > WPP_LAST_ESTIMATE_YEAR].reset_index()
        est.to_csv(OUT / "cohorts.csv", index=False)
        proj.to_csv(OUT / "cohorts_un_projection.csv", index=False)
        _record(manifest, "cohorts", WPP_AGE, "p1,p2,p3,p4",
                "UN WPP2024 mid-year population by World3 cohort (0-14, 15-44, 45-64, 65+), estimates only", est)
        _record(manifest, "cohorts_un_projection", WPP_AGE, "(comparison only)",
                "UN WPP2024 Medium-variant PROJECTION, never used as a calibration target", proj)
    except Exception as e:
        failures.append("cohorts")
        print(f"[FAIL] cohorts: {e}", file=sys.stderr)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if failures:
        print(f"\n{len(failures)} series failed: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
