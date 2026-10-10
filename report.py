"""Build charts + markdown report from results/calibrated_params.json (+ results/ensemble.json if present).

Usage: python report.py
Outputs: results/fig_targets.png, results/fig_world3_standard.png, results/fig_restart_validation.png,
         results/report.md
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from model import run_world3  # noqa: E402
from targets import TARGETS, TEST, TRAIN, indexed, load_empirical  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
DATA = ROOT / "data"
NOW = 2026.77  # 2026-10-09
STYLE = {
    "BAU_World3-03_default": dict(color="#888888", ls="--", lw=1.4, label="BAU (World3-03 default)"),
    "Recalibration23_Nebel2023": dict(color="#1f77b4", ls="-.", lw=1.4, label="Recalibration23 (Nebel 2023)"),
    "Calibrated_v1": dict(color="#ff9896", ls="-", lw=1.4, label="Calibrated v1 (previous run)"),
    "Calibrated_2026": dict(color="#d62728", ls="-", lw=2.2, label="Calibrated v2 (this run)"),
}
LABEL = {**{k: v["label"] for k, v in STYLE.items()},
         "Calibrated_2026+restart2014": "Calibrated v2 + restart at 2014 (observed age groups)",
         "BAU+restart2014": "BAU + restart at 2014 (observed age groups)"}


def _j(name: str) -> dict | None:
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def pyworld4_section(A, emp) -> None:
    """Everything below is read from results/*.json written by the pyworld4 scripts."""
    v3, o1, v4 = _j("v3_forecast.json"), _j("option1_assimilate.json"), _j("v4.json")
    ens, wid = _j("ensemble_v4.json"), _j("ensemble_v4_widened.json")
    if v4 is None:
        A("_pyworld4 results not found (run calibrate_v4.py)._\n")
        return
    A("**pyworld4** = the original World3-03 engine (PyWorld3-03, source untouched) + v2 settings + re-fitted "
      "demographic lookup tables ([tables_v4.json](tables_v4.json), a drop-in for "
      "`World3.set_world3_table_functions()`) + a 20-year data spin-up of the four age groups before each forecast. "
      "Changes that would need new equations are out of scope (see *Lessons for zigworld*).\n")
    A(f"_v4 fit: {v4['optimizer']['de_nfev']} DE + Powell evaluations, {v4['optimizer']['elapsed_s']:.0f}s, "
      f"objective {v4['objective']}, J(v2) = 2.5 → J(v4) = {v4['optimizer']['powell_fun']:.4f}. "
      "Tuning uses 1960–2014 only; every number marked held-out uses 2015–2023 data never seen in tuning._\n")

    # Held-out forecast skill
    A("## Held-out population forecasts (origins 2014–2022, mean |error| %, signed in brackets)\n")
    rows = []
    if v3:
        for k in ("BAU", "v2", "v3"):
            bh = v3["test_skill"][k]["by_horizon"]
            rows.append((f"{k}, single restart", {int(h): (d["mean_abs_pct"], d["mean_signed_pct"]) for h, d in bh.items()}))
    if o1:
        for k in ("BAU | L=20", "v2 | L=20"):
            if k in o1:
                rows.append((k.replace(" | L=20", ", 20-yr spin-up"),
                             {int(h): (d["mean_abs"], d["signed"]) for h, d in o1[k]["skill"].items()}))
    rows.append(("**v4 (pyworld4), 20-yr spin-up**",
                 {int(h): (d["mean_abs"], d["signed"]) for h, d in v4["heldout"]["v4"]["forecast_pop_pct"].items()}))
    A("| Model | 1 yr | 2 yr | 3 yr | 4 yr | 5 yr |\n|---|---|---|---|---|---|")
    for lab, d in rows:
        A(f"| {lab} | " + " | ".join(f"{d[h][0]:.3f} ({d[h][1]:+.3f})" if h in d else "—" for h in range(1, 6)) + " |")
    A("\n> **BAU looks best on population only, for the wrong reasons.** Its simulated economy collapses after ~2020: "
      "life expectancy " + (f"{v4['heldout']['BAU']['path_2023_2035']['2023']['le']:.0f} → "
                            f"{v4['heldout']['BAU']['path_2023_2035']['2030']['le']:.0f}" if 'BAU' in v4['heldout'] else "falls")
      + " by 2030 (observed ≈ 73), and its rising death rate coincided with COVID. A population-only score can be "
      "fooled, so birth rate, death rate and life expectancy are scored too.\n")
    A("![held-out skill](fig_v4_skill.png)\n")

    A("## Held-out demographic rates and overall fit (2015–2023)\n")
    A("| | BAU | v2 | **v4** |\n|---|---|---|---|")
    H = v4["heldout"]
    for key, lab in (("cbr", "birth rate NRMSD"), ("cdr", "death rate NRMSD"), ("le", "life expectancy NRMSD")):
        A(f"| {lab} | " + " | ".join(f"{H[m]['rates_nrmsd_2015_2023'][key]:.4f}" for m in ("BAU", "v2", "v4")) + " |")
    A("| all 13 curves, test NRMSD (no restart) | " + " | ".join(f"{H[m]['curve_test_nrmsd']:.4f}" for m in ("BAU", "v2", "v4")) + " |")
    A("| life expectancy 2023 → 2030 | " + " | ".join(
        f"{H[m]['path_2023_2035']['2023']['le']:.1f} → {H[m]['path_2023_2035']['2030']['le']:.1f}" for m in ("BAU", "v2", "v4")) + " |")
    for c, lab in (("pop", "2026 population vs UN projection"), ("p4", "2026 age 65+ vs UN projection")):
        A(f"| {lab} | " + " | ".join(f"{H[m]['nowcast2026'][c]['vs_UN_pct']:+.2f}%" for m in ("BAU", "v2", "v4")) + " |")

    A("\n### v4 table adjustments\n")
    A("Smooth warps only (scale and tilt), to limit overfitting: `M1..M4`: y·exp(a + b·(LE−70)/10); "
      "`SFSN`: y·exp(a + b·(DIOPC−400)/400); `FCE_TOCLIP`: 1−y' = (1−y)·exp(a).\n")
    A("| number | value | bounds |\n|---|---|---|")
    for n, v in v4["x_v4"].items():
        lo, hi = v4["bounds"][n]
        A(f"| `{n}` | {v:.4f} | [{lo:g}, {hi:g}] |")

    A("\n### What still limits accuracy: built into the equations\n")
    A("World3 moves people between its 4 age groups at fixed yearly fractions (1/15, 1/30, 1/20), as if each group "
      "were spread evenly by age. Baby boomers break this, so the 65+ group grows ≈1%/yr too fast in one-step checks. "
      "The fit compensates by raising old-age mortality (`m3_a`, `m4_a` ≈ +0.38, i.e. ≈ +46%), which pushes the death "
      "rate up as the population ages and causes the remaining negative population drift. Fixing it needs new "
      "equations → zigworld.\n")

    if ens:
        A(f"## Range of outcomes ({ens['method']})\n")
        A(f"_{ens['note']}; 25 numbers sampled (v4's 13 + v2's 12 economy settings), span {ens['span']}._\n")
        A("| 2026 | 5% | median | 95% |\n|---|---|---|---|")
        for v, lab, sc, fmt in (("pop", "Population (B)", 1e9, "{:.4f}"), ("p1", "0–14 (B)", 1e9, "{:.4f}"),
                                ("p2", "15–44 (B)", 1e9, "{:.4f}"), ("p3", "45–64 (B)", 1e9, "{:.4f}"),
                                ("p4", "65+ (B)", 1e9, "{:.4f}"), ("le", "Life expectancy (yr)", 1, "{:.1f}"),
                                ("cbr", "Birth rate", 1, "{:.2f}"), ("cdr", "Death rate", 1, "{:.2f}")):
            q = ens["y2026"][v]
            A(f"| {lab} | {fmt.format(q['p05']/sc)} | {fmt.format(q['p50']/sc)} | {fmt.format(q['p95']/sc)} |")
        A("\n| Peak year | 5% | median | 95% |\n|---|---|---|---|")
        for v, q in ens["peak_year_p05_p50_p95"].items():
            A(f"| `{v}` | {q[0]:.0f} | {q[1]:.0f} | {q[2]:.0f} |")
        pk = ens["peak_pop_p05_p50_p95"]
        A(f"\nPeak population {pk[0]/1e9:.2f} / {pk[1]/1e9:.2f} / {pk[2]/1e9:.2f} B (5/50/95%).\n")
        A("> **After ~2040 these are not forecasts.** v2's economy contains a food-system collapse (modelled food per "
          "capita falls while observed keeps rising), and the warped tables are pushed far outside the life-expectancy "
          "range seen in 1960–2014. Treat the long-run tail as World3's overshoot-and-collapse scenario.\n")
        A("![ensemble v4](fig_ensemble_v4.png)\n")

    if wid:
        m = wid["error_model"]
        c = wid["loo_coverage_90"]
        A("## Widened range: settings + the model's own held-out error\n")
        A(f"Held-out drift of v4 (origins 2014–2022, 20-yr spin-up): mean **{100*m['d_mean']:+.3f}%/yr**, spread "
          f"{100*m['s_d']:.3f}%/yr between origins, noise {100*m['s0']:.3f}% (n = {m['n']}, Student-t). "
          f"Leave-one-origin-out check: **{c['hits']}/{c['total']}** held-out errors inside the predicted 90% range "
          f"({100*c['hits']/c['total']:.0f}%).\n")
        A("| Year | settings only 5/50/95 (B) | **widened 5/50/95 (B)** | UN WPP2024 projection (B) | UN inside widened? |\n"
          "|---|---|---|---|---|")
        for y in ("2024", "2025", "2026", "2027", "2028", "2030", "2035", "2040"):
            r = wid["bands"][y]
            a, b = r["param_only"], r["widened"]
            un = r.get("UN_proj")
            A(f"| {y} | {a['p05']/1e9:.3f} / {a['p50']/1e9:.3f} / {a['p95']/1e9:.3f} | "
              f"**{b['p05']/1e9:.3f} / {b['p50']/1e9:.3f} / {b['p95']/1e9:.3f}** | "
              + (f"{un/1e9:.3f} | {'yes' if r['UN_inside_widened'] else '**no**'} |" if un else "— | — |"))
        A("\n> Close agreement with the UN projection is **two models agreeing**, not verified accuracy: both extend "
          "recent trends, and UN numbers after 2023 are projections. The real check is the next UN estimates "
          "(WPP 2026). The longest tested horizon is 9 years, so years after ~2032 are extrapolation.\n")
        A("![widened](fig_ensemble_v4_widened.png)\n")

    A("## v3 (negative result, kept for the record)\n")
    if v3:
        A(f"Tuning the 14 scalar settings for 5-year forecasts from 1995/2002/2009 improved the training score "
          f"(J {v3['J']['v2']:.3f} → {v3['J']['v3']:.3f}) but made held-out forecasts worse (table above) and the "
          f"whole-curve test score worse ({v3['curve_test_nrmsd']['v2']:.4f} → {v3['curve_test_nrmsd']['v3']:.4f}). "
          "Errors keep the same sign every year: a growth-rate bias that scalar settings only flip. "
          "This motivated the table re-fit.\n")

    A("## Lessons for zigworld\n")
    A("1. **Single-year age groups** (0–100+, both sexes) instead of 4 buckets: removes the ageing error and the "
      "compensating over-mortality.\n2. **Fit to age-specific rates** (UN WPP fertility and mortality by age), not only "
      "totals.\n3. **Data spin-up / assimilation as a built-in feature**, not a runtime patch.\n4. **Multi-measure scoring** "
      "(population + birth/death rates + life expectancy + economy): a single number was fooled by BAU.\n5. **Speed**: "
      "≈0.3 s per Python run limits searches (up to 17k runs here) and ensembles.\n")

    # Figure: held-out skill by horizon
    fig, ax = plt.subplots(figsize=(9, 5))
    for lab, d in rows:
        hs = [h for h in range(1, 6) if h in d]
        ax.plot(hs, [d[h][1] for h in hs], "o-", lw=2.4 if "v4" in lab else 1.3, label=lab.replace("**", ""))
    ax.axhspan(-0.1, 0.1, color="green", alpha=0.12, label="±0.1%")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("years ahead"); ax.set_ylabel("mean signed population error (%)")
    ax.set_title("Held-out population forecasts, origins 2014–2022")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RES / "fig_v4_skill.png", dpi=130); plt.close(fig)


def main() -> None:
    R = json.loads((RES / "calibrated_params.json").read_text())
    E = R["evaluation"]
    emp = load_empirical()
    runs = {k: run_world3(p, year_max=2100) for k, p in R["scenarios"].items()}
    rs_val, rs_now = R["restart_validation"], R["restart_nowcast"]
    cal_p = R["calibrated_params"]
    runs_rs = {"Calibrated_2026+restart2014": run_world3(cal_p, year_max=2100, restart=rs_val),
               "BAU+restart2014": run_world3({}, year_max=2100, restart=rs_val)}
    nowcast = run_world3(cal_p, year_max=2100, restart=rs_now)
    for r in [*runs.values(), *runs_rs.values(), nowcast]:
        r.index = r.index.astype(int)
    unproj = pd.read_csv(DATA / "cohorts_un_projection.csv").set_index("Year")

    # --- Fig 1: every target
    n = len(TARGETS)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4.2 * rows))
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, t in zip(axes.flat, TARGETS):
        d = emp[t.key]
        sc = 1e9 if t.unit == "people" else 1
        if t.indexed:
            d = indexed(d)
        for k, r in runs.items():
            m = indexed(r[t.w3var]) if t.indexed else r[t.w3var]
            m = m.loc[1950:2100]
            ax.plot(m.index, m.values / sc, **STYLE[k])
        ax.plot(d.index, d.values / sc, "k.", ms=3.5, label="Observed data")
        ax.axvspan(TRAIN[0], TRAIN[1], color="#2ca02c", alpha=0.06)
        ax.axvspan(TEST[0], d.index.max(), color="#ff7f0e", alpha=0.15)
        ax.axvline(NOW, color="k", lw=0.8)
        ax.set_title(t.label, fontsize=10)
        ax.set_ylabel("billion people" if t.unit == "people" else t.unit, fontsize=8)
        ax.set_xlim(1950, 2100)
        s = E["Calibrated_2026"]["per_target"][t.key]
        ax.text(0.02, 0.97, f"v2 NRMSD\ntrain {s['train']:.4f} | test {s['test']:.4f}",
                transform=ax.transAxes, va="top", fontsize=7.5, bbox=dict(fc="white", ec="0.8", alpha=0.85))
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=10, frameon=False)
    fig.suptitle("World3-03 vs observed data. Green: training 1970–2014 · Orange: held-out test 2015–latest · "
                 "Black line: Oct 2026 · No restart in these runs", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(RES / "fig_targets.png", dpi=120)
    plt.close(fig)

    # --- Fig 2: restart validation, population % error by year
    yrs = [y for y in emp["pop"].index if y >= TEST[0]]
    fig, ax = plt.subplots(figsize=(10, 5))
    curves = {"Calibrated v2, no restart": runs["Calibrated_2026"],
              "Calibrated v2 + restart 2014": runs_rs["Calibrated_2026+restart2014"],
              "BAU + restart 2014": runs_rs["BAU+restart2014"],
              "BAU, no restart": runs["BAU_World3-03_default"]}
    pct = {}
    for lab, r in curves.items():
        e = [100 * (r.loc[y, "pop"] / emp["pop"].loc[y] - 1) for y in yrs]
        pct[lab] = e
        ax.plot(yrs, e, "o-", label=lab, lw=2 if "v2 + restart" in lab else 1.2)
    ax.axhspan(-0.1, 0.1, color="green", alpha=0.12, label="±0.1% band")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("population error vs UN (%)")
    ax.set_title("Held-out test: world population error in 2015–2023 (fit and restart use data through 2014 only)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RES / "fig_restart_validation.png", dpi=130); plt.close(fig)

    # --- Fig 3: standard run, calibrated (2023 restart) vs BAU
    cal, bau = nowcast, runs["BAU_World3-03_default"]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    series = [("nrfr", "Resources (fraction left)", "#8c564b"), ("io", "Industrial output", "#ff7f0e"),
              ("f", "Food", "#2ca02c"), ("pop", "Population", "#1f77b4"), ("ppolx", "Pollution", "#9467bd")]
    for var, lab, c in series:
        peakv = np.nanmax(cal[var].loc[1900:2100])
        for r, ls, a in ((bau, ":", 0.55), (cal, "-", 1.0)):
            v = r[var].loc[1900:2100]
            ax.plot(v.index, np.clip(v / peakv, 0, 1.15), ls, color=c, alpha=a,
                    lw=2 if ls == "-" else 1.4, label=lab if ls == "-" else None)
    ax.axvline(NOW, color="k", lw=1); ax.text(NOW + 1, 1.02, "Oct 2026", fontsize=9)
    ax.set_xlim(1900, 2100); ax.set_ylim(0, 1.15)
    ax.set_ylabel("normalized to calibrated-run peak (clipped at 1.15)")
    ax.set_title(f"World3-03 'standard run': calibrated v2 restarted from {rs_now['year']} data (solid) vs original BAU (dotted)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(RES / "fig_world3_standard.png", dpi=130); plt.close(fig)

    # --- Markdown
    L = []
    A = L.append
    A("# pyworld4: World3-03 calibrated to observed data (through 2023–2025)\n")
    pyworld4_section(A, emp)
    A("\n---\n\n# Earlier version (v2): + UN age structure, restart from data, range of outcomes\n")
    A(f"_Generated {R['created_local']} · model: PyWorld3-03 @ `{R['vendor']['commit'][:8]}` · "
      f"data manifest sha256 `{R['data_manifest_sha256'][:12]}` · optimizer nfev={R['optimizer']['nfev']}, "
      f"{R['optimizer']['nit']} generations, {R['optimizer']['elapsed_s']:.0f}s_\n")
    A("## Fit quality (weighted NRMSD over all targets, lower is better)\n")
    A("| Scenario | Train 1970–2014 | **Test 2015–latest (held out)** |\n|---|---|---|")
    for k, e in E.items():
        A(f"| {LABEL.get(k, k)} | {e['total_train']:.4f} | **{e['total_test']:.4f}** |")
    A("\nThe *restart at 2014* rows swap in the observed age groups at the end of training, so 2015–latest is still unseen.\n")

    A("## Population: held-out error by year (%)\n")
    A("| Year | UN observed | " + " | ".join(pct) + " |\n|---|---|" + "---|" * len(pct))
    for i, y in enumerate(yrs):
        A(f"| {y} | {emp['pop'].loc[y]/1e9:.4f} B | " + " | ".join(f"{pct[k][i]:+.3f}%" for k in pct) + " |")
    A("\n| Summary | " + " | ".join(pct) + " |\n|---|" + "---|" * len(pct))
    A("| max abs. error | " + " | ".join(f"{max(abs(v) for v in pct[k]):.3f}%" for k in pct) + " |")
    A("| mean abs. error | " + " | ".join(f"{np.mean(np.abs(pct[k])):.3f}%" for k in pct) + " |")

    A("\n### Per target (test-window NRMSD)\n")
    keys = [k for k in ("BAU_World3-03_default", "Recalibration23_Nebel2023", "Calibrated_v1", "Calibrated_2026",
                        "Calibrated_2026+restart2014") if k in E]
    A("| Target | weight | last obs. | " + " | ".join(LABEL[k] for k in keys) + " |\n|---|---|---|" + "---|" * len(keys))
    for t in TARGETS:
        c = E["Calibrated_2026"]["per_target"][t.key]
        A(f"| {t.label} | {t.weight} | {c['test_last_year']} | " +
          " | ".join(f"{E[k]['per_target'][t.key]['test']:.4f}" for k in keys) + " |")

    A("\n## Parameters\n")
    v1p = R["scenarios"].get("Calibrated_v1", {})
    A("| Param | World3-03 default | Recalibration23 | v1 | **v2** | bounds |\n|---|---|---|---|---|---|")
    for p, (lo, hi) in R["bounds"].items():
        rc = R["recalibration23"].get(p)
        A(f"| `{p}` | {R['world3_03_defaults'][p]:.4g} | {('%.4g' % rc) if rc is not None else '—'} | "
          f"{('%.4g' % v1p[p]) if p in v1p else '—'} | **{cal_p[p]:.4g}** | [{lo:.3g}, {hi:.3g}] |")

    A(f"\n## 2026 figures (calibrated v2, restarted from observed {rs_now['year']} age groups)\n")
    A("> 2026 hasn't been fully observed yet, so the UN column is the **UN WPP2024 Medium-variant projection**, "
      "not an observation.\n")
    A("| Variable | Model 2026 | UN WPP2024 projection 2026 | difference |\n|---|---|---|---|")
    for c, lab in (("pop", "Total population"), ("p1", "0–14"), ("p2", "15–44"), ("p3", "45–64"), ("p4", "65+")):
        un = unproj.loc[2026, "total" if c == "pop" else c]
        m = nowcast.loc[2026, c]
        A(f"| {lab} | {m/1e9:.4f} B | {un/1e9:.4f} B | {100*(m/un-1):+.2f}% |")

    ens = RES / "ensemble.json"
    if ens.exists():
        S = json.loads(ens.read_text())
        A(f"\n## Range of outcomes ({S['method']}): {S['n_kept']} equally good parameter sets\n")
        A(f"_{S['note']}_\n")
        A("| Variable (2026) | 5% | median | 95% |\n|---|---|---|---|")
        for v, lab, sc, fmt in (("pop", "Population (B)", 1e9, "{:.4f}"), ("le", "Life expectancy (yr)", 1, "{:.1f}"),
                                ("cbr", "Birth rate", 1, "{:.2f}"), ("cdr", "Death rate", 1, "{:.2f}")):
            q = S["y2026"][v]
            A(f"| {lab} | {fmt.format(q['p05']/sc)} | {fmt.format(q['p50']/sc)} | {fmt.format(q['p95']/sc)} |")
        A("\n| Peak year | 5% | median | 95% |\n|---|---|---|---|")
        for v, q in S["peak_year_p05_p50_p95"].items():
            A(f"| `{v}` | {q[0]:.0f} | {q[1]:.0f} | {q[2]:.0f} |")
        A("\n> The range covers **parameter** uncertainty only. Structural error in the model (food per capita, "
          "fossil fuel use) isn't included, so the real uncertainty is wider.\n")
        A("![ensemble](fig_ensemble.png)\n")

    A("\n![restart validation](fig_restart_validation.png)\n\n![targets](fig_targets.png)\n\n"
      "![standard run](fig_world3_standard.png)\n")
    (RES / "report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
