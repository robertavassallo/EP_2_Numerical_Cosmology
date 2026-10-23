# Numerical Cosmology — EP2

For each exercise (linear growth, perturbation-theory mode coupling, Lagrangian initial conditions, spherical collapse and Boltzmann-solver validation) there is a script with the main functions (`scripts/`) and a notebook with intermediate results, checks and figures (`notebooks/`).

## Exercises
| # | Topic | Script | Notebook | Extra |
|---|---|---|---|---|
| 1 | Linear growth: growing-mode ODE in $N=\ln a$, $D(a)$, $f(a)$, quadrature check | `scripts/exercise_1.py` | `notebooks/exercise_1.ipynb` | |
| 2 | Mode coupling: second-order SPT density, power spectrum and bispectrum | `scripts/exercise_2.py` | `notebooks/exercise_2.ipynb` | |
| 3 | Lagrangian initial conditions: Gaussian field, Zel'dovich displacements, CIC power spectrum | `scripts/exercise_3.py` | `notebooks/exercise_3.ipynb` | |
| 4 | Spherical collapse in three formulations (A, B, C), turnaround and virial | `scripts/exercise_4.py` | `notebooks/exercise_4.ipynb` | LaTeX report in `report_ex4/` |
| 5 | Boltzmann solvers: $P_m$ reconstruction from the CLASS transfer, convergence, CLASS vs CAMB | `scripts/exercise_5.py` | `notebooks/exercise_5.ipynb` | CLASS input and output tables in `ex5_class/` |

Exercise 4 uses the `Cosmology` class of Exercise 1; Exercise 2 uses the Gaussian-field generator of Exercise 3.

## Dependencies
Python 3.13 with
- numpy 2.4, scipy 1.17, matplotlib 3.10, pandas 3.0, jupyter
- [CAMB](https://camb.readthedocs.io) 1.6.5 (Exercises 3 and 5)
- [CLASS](https://github.com/lesgourg/class_public) v3.4.1 through its Python wrapper `classy` (Exercise 5)

```bash
pip install numpy scipy matplotlib pandas jupyter classy camb
```

## Run
Run the notebooks from inside `notebooks/` (they import the scripts from `../scripts`):
```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace exercise_1.ipynb
```
Exercise 5 caches its convergence runs (about 10 min) in `ex5_class/output/precision_ratios.npz`; delete that file to recompute them.
See `report_ex4/README.md` for building the Exercise 4 report.
