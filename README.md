# PyWorld4

PyWorld4 is an advanced calibration, tuning, and diagnostic fork of the vendored [PyWorld3-03](https://github.com/cvanwynsberghe/pyworld3) system dynamics model. 

It was built specifically to push the classical World3 4-bucket demographic equations to their absolute mathematical limits. By tuning structural lookup curves against modern UN demographic data (1970–2023), PyWorld4 exposes exactly where the classical model equations structurally fail, laying the definitive ground-truth benchmark for a from-scratch rewrite: **Zigworld**.

## Key Features

* **Data Assimilation (Spin-Up):** Built-in mechanics to "warm up" the model using 20 years of real age-group data, completely eliminating the standard 1970 initialization errors.
* **Non-Linear Table Warping:** Dynamically overrides and re-calibrates the hardcoded PyWorld3 lookup tables (M1-M4, SFSN, FCE_TOCLIP) without touching the core vendored source code.
* **GLUE Ensembles:** Generates uncertainty bands using the Generalized Likelihood Uncertainty Estimation (GLUE) methodology.
* **Structural Drift Compensation:** Statistically widens the projection bands using a Student-t distribution based on the model's historical held-out drift to provide honest, real-world uncertainty ranges.

## The Structural Limit (Why Zigworld?)

Through extensive calibration, PyWorld4 proved that the classic World3 demographic model hits a hard mathematical ceiling. Because it uses fixed fractional maturation across only 4 age buckets (e.g., 0-14, 15-44), it assumes populations are evenly distributed within those buckets. Real-world anomalies—like the aging baby boomer generation—break this assumption, causing the 65+ bucket to grow ~1%/yr too fast in the model. 

To compensate, the model artificially forces mortality up, dragging life expectancy down. **PyWorld4 definitively proved this is unfixable via parameter tuning.** 

The next generation of the model (**Zigworld**) will solve this by implementing single-year cohort tracking (0–100+).

## Running the Model

PyWorld4 requires Python 3.8+ and standard scientific libraries (`numpy`, `scipy`, `matplotlib`). 

To generate the full ensemble and the final Markdown report:

```bash
# Generate the full evaluation report
python3 report.py
```

## Results

All generated graphs, logs, JSON arrays, and the final compiled report are available in the `results/` directory.

Read the final conclusive report here: **[results/report.md](results/report.md)**

## License

The vendored PyWorld3-03 code retains its original **CeCILL 2.1** Free Software License. The overarching PyWorld4 scripts and calibration engines are licensed under the **GNU General Public License v3.0 (GPL-3.0)**.
