# EP2 – Exercise 4: spherical collapse, formulations A, B, C

## Files
| path (from the repository root) | content |
|---|---|
| `scripts/exercise_4.py` | abstract class `Background` (background, integration, shooting `tune_delta_i`, turnaround/virial `turnaround_virial`), formulations `A`, `B`, `C`, `virial_y` |
| `scripts/exercise_1.py` | `Cosmology` class (background, growth rate `f_i`, growth factor) used by `exercise_4.py` |
| `notebooks/exercise_4.ipynb` | checks, tables and figures; writes `report_ex4/figures/*.pdf` and `report_ex4/data/*.csv` |
| `report_ex4/ex4_report.tex`, `ex4_report.pdf` | LaTeX report |
| `report_ex4/data/` | saved results: `required_outputs.csv`, `eds_benchmarks.csv`, `lcdm_grid.csv`, `convergence_ai.csv`, `convergence_tol.csv`, `convergence_threshold.csv` |
| `report_ex4/figures/` | `growing_mode.pdf`, `collapse_epoch.pdf`, `turnaround.pdf`, `lcdm_grid.pdf`, `failure_modes.pdf` |

## Dependencies
Python 3.13 with numpy 2.4, scipy 1.17, matplotlib 3.10, pandas 3.0, jupyter/nbconvert (the `cosmo` conda env).
LaTeX: `pdflatex`/`latexmk` with amsmath, booktabs, graphicx, hyperref, xcolor, placeins, microtype, lmodern.

## Run
```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace exercise_4.ipynb    #  regenerates figures and data

cd ../report_ex4
latexmk -pdf ex4_report.tex                                            # builds ex4_report.pdf
```

Single run from the command line (inside `scripts`), printing all required outputs:
```bash
python exercise_4.py C 0.3 0                                  # formulation, Omega_m0, z_c
python exercise_4.py A 1.0 1 --threshold 1e6 --a_i 1e-4       # optional: --threshold --a_i --rtol --atol
```

Single run from Python (inside `scripts`):
```python
from exercise_4 import A, B, C
C(0.3).turnaround_virial(z_c=0.0)          # optional keywords: threshold, a_i, rtol, atol
```

## Settings used 
- Initial epoch `a_i = 1e-5`, growing-mode initial conditions with `f_i` from the Exercise 1 ODE.
- Integrator `scipy.integrate.solve_ivp`, DOP853, `rtol = 1e-10`, `atol = 1e-13`.
- Collapse event (terminal): `delta = 1e8` (A, B), `g = 1e-8` (C). Turnaround event: `theta = -3` (recorded only).
- Shooting: `brentq` on `ln(a_coll/a_c)`, bracket ±3 % around `1.686 D(a_i)/D(a_c)` (widened if needed), `xtol = 1e-14*guess`, `rtol = 1e-14`.
- Virial: root of `4 λ y³ − 2(1+λ) y + 1 = 0` in (0, 1/2], formal-collapse epoch, `Δ_vir,c = Ω_m(a_c) Δ_vir,m`.
