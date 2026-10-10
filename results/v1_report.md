# World3-03 calibrated to observed data (through 2023–2025)

_Generated 2026-10-09T22:12:06-0700 · model: PyWorld3-03 @ `1f1b2f40` · data manifest sha256 `b2cf80084945`_

## Fit quality (NRMSD, lower is better; weighted mean over targets)

| Scenario | Train 1970–2014 | **Test 2015–latest (held out)** |
|---|---|---|
| BAU (World3-03 default) | 0.0841 | **0.1510** |
| Recalibration23 (Nebel 2023) | 0.0745 | **0.1876** |
| Calibrated to 2026 (this work) | 0.0509 | **0.0838** |

### Per target

| Target | weight | last obs. | BAU test | Recal23 test | Calibrated train | **Calibrated test** |
|---|---|---|---|---|---|---|
| Population | 1.0 | 2023 | 0.0568 | 0.0616 | 0.0061 | **0.0025** |
| Life expectancy | 0.5 | 2023 | 0.0531 | 0.1604 | 0.0713 | **0.0782** |
| Crude birth rate | 0.5 | 2023 | 0.1081 | 0.2865 | 0.0381 | **0.0354** |
| Crude death rate | 0.5 | 2023 | 0.1363 | 0.0919 | 0.0486 | **0.0759** |
| Industrial output (industry value added, const. 2015 US$) | 0.5 | 2025 | 0.2008 | 0.3241 | 0.0578 | **0.0795** |
| Service output (services value added, const. 2015 US$) | 0.5 | 2024 | 0.1124 | 0.1559 | 0.0257 | **0.0734** |
| Food per capita (FAO kcal/cap/day) | 0.7 | 2023 | 0.1669 | 0.1238 | 0.0407 | **0.1226** |
| Persistent pollution (CO2 ppm above 280) | 0.5 | 2025 | 0.5286 | 0.4111 | 0.0453 | **0.1413** |
| Nonrenewable resource use (fossil fuel TWh) | 0.5 | 2025 | 0.0843 | 0.2247 | 0.1737 | **0.2115** |

## Parameters

| Param | World3-03 default | Recalibration23 | **Calibrated** | bounds |
|---|---|---|---|---|
| `alic` | 14 | 15.24 | **11.43** | [8, 25] |
| `alsc` | 20 | — | **27.73** | [10, 30] |
| `icor` | 3 | — | **2.707** | [2, 4.5] |
| `pptd` | 20 | 116.4 | **142.9** | [20, 150] |
| `imef` | 0.1 | 0.1102 | **0.07281** | [0.05, 0.25] |
| `nri` | 1e+12 | 1.302e+12 | **2.582e+12** | [7e+11, 3e+12] |
| `uildt` | 10 | 0.533 | **11.77** | [0.5, 15] |
| `hsid` | 20 | 38.24 | **11.35** | [10, 50] |
| `lpd` | 20 | 33.84 | **37.7** | [10, 50] |
| `mtfn` | 12 | 9.454 | **8.577** | [8, 14] |
| `dcfsn` | 3.8 | — | **4.08** | [2.5, 5] |
| `lfpf` | 0.75 | 1.022 | **0.7416** | [0.55, 0.9] |

All other constants are fixed at the Recalibration23 values (Nebel et al. 2023, Table 2).

## Snapshot: model vs latest observation

| Variable | Latest observed | Calibrated model, same year | Calibrated model, **2026** | BAU, 2026 |
|---|---|---|---|---|
| pop | 8.092 B (2023) | 8.079 B | **8.294 B** | 7.633 B |
| le | 73.2 yr (2023) | 78.7 yr | **79.1 yr** | 71.0 yr |
| cbr | 16.3 (2023) | 17.3 | **16.9** | 13.0 |
| cdr | 7.6 (2023) | 8.3 | **8.3** | 10.4 |

## Projected peaks (calibrated vs BAU)

| Variable | BAU peak year | Calibrated peak year |
|---|---|---|
| `pop` | 2028 | 2053 |
| `io` | 2018 | 2049 |
| `f` | 2017 | 2034 |
| `iopc` | 2017 | 2048 |
| `fpc` | 1996 | 1992 |
| `ppolx` | 2039 | 2100 |

![targets](fig_targets.png)

![standard run](fig_world3_standard.png)
