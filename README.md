# SANR

Polarised neutron reflectometry with arbitrary polarisation: build a layer
stack, simulate its reflectivity for any incident / analysed polarisation, and
fit it to measured channels.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python setup.py build_ext --inplace   # compiled transfer matrix, 1.5-2x faster fits
```

The last step needs a C compiler (on macOS: `xcode-select --install`). Without
it everything works with the numpy version of the same product, only slower.
Build again after editing `model/_transfer.pyx`.

## Run

```bash
python main.py                 # the GUI
python make_app.py             # macOS: build SANR.app (Dock name and icon), then open it
python -m pytest tests         # the tests (pip install pytest first)
```

## Windows installer

`SANR-<version>-setup.exe` installs SANR as a Windows program (Start menu,
optional desktop icon, uninstaller, no administrator rights needed).  It is
built on Windows, so GitHub builds it: **Actions → Windows installer → Run
workflow** (or push a tag `v1.0`, which also makes a draft release); the
installer is the run's artifact `SANR-windows-installer`.  On a Windows PC
with Python 3.12, Inno Setup 6 and the Microsoft C++ Build Tools:

```bat
pip install -r requirements.txt pyinstaller
python make_windows_installer.py
```

The installed program keeps its sessions in `Documents\SANR`.

## Layout

| Path | What it holds |
|------|---------------|
| `main.py` | Main window, menus and session files (File → Save / Open session: model, settings and the imported data files; File → Import Licorne session: a Licorne session folder) |
| `style.py`, `icons/` | The dark theme and its icons; `icons/app.png` is the app icon |
| **`model/`** | **The physics, no GUI** |
| `model/stack.py` | `Layer`, `Stack`: roughness profile (additive, ordered or independent interfaces, `Stack.interface_crossing`), slicing, transfer matrix, `Stack.reflectivities(Q, pairs)`; fittable `scale` and `background`. The module docstring explains the physics and its conventions. |
| `model/_transfer.pyx`, `setup.py` | The transfer-matrix product compiled with Cython (a loop per Q, two 2×2 products when M is collinear); `Stack.build_transfer_matrix` uses it when built, numpy otherwise |
| `model/roughness.py` | Licorne's interface kernel (manual App. 10.1): window, closed-form J, renormalisation, sigma conversions |
| `model/polarisation.py` | Named channels (`++`, `+-`, `-+`, `--`, `+`, `-`) as (P0, P) pairs, and the Pi / Pa form the GUI uses |
| `model/licorne_io.py` | `load_licorne_model`: a Licorne export (`parameters.m`, `profile.dat`) as a Licorne-exact Stack (`Stack.set_licorne_exact`), its axes rotated so that M and P lie in the film plane (`licorne_axis_map`); `load_licorne_session`: a whole saved Licorne session folder (model, fit bounds, background, `Norm_factor`, TOF resolution, `rexp*.dat` channels, `rtheory*.dat` curves) and notes on what is not reproduced |
| `model/licorne_resolution.py` | Licorne's resolution convolution on the data grid (`resolut`, modes 1–3) as a sparse weight matrix |
| **`fit/`** | **Fitting** |
| `fit/problem.py` | `FitProblem`: free parameters, data, model and cost, shared by both methods |
| `fit/joint.py` | `JointProblem`: several datasets fitted together with one model, each parameter shared, own to a dataset or not fitted in it (the `FitProblem` interface, for DE, multistart and DREAM) |
| `fit/datasets_panel.py` | "Datasets to fit" panel (right side of the Fit tab): the datasets of a joint fit and each parameter's mode in each |
| `fit/panel.py` | The Fit tool box of the Experimental tab (general / DE / local L-BFGS-B / multistart / DREAM pages and their actions) |
| `fit/workers.py` | Process pools for DE (and its L-BFGS-B) and DREAM that receive the fit problem once, as each worker starts, instead of with every task |
| `fit/de/de_fit.py` | `run_de`: differential-evolution fit (scipy), optionally over worker processes; `run_lbfgs`: L-BFGS-B alone from the current values (the local page) |
| `fit/de/de_window.py` | DE window (also for a local fit): best cost per generation (or iteration), profile and reflectivity of the selected one (one tab per dataset), send to the Simulation tab |
| `fit/dream/dream.py` | DREAM / DREAM(ZS) sampler (Vrugt 2016), NumPy only: resumable state, `ess` |
| `fit/dream/dream_fit.py` | `PosteriorProblem` (the `FitProblem` as a Gaussian log posterior), `sample` (also continuing a run), `save_result` / `load_run` (a saved run with its data, rebuilt without the GUI), `summary_table`, `derived`, predictive and profile bands |
| `fit/dream/dream_plots.py` | Matplotlib figures of a DREAM run for scripts and reports (traces, corner, predictive and SLD-profile bands) |
| `fit/dream/dream_window.py` | DREAM window (pyqtgraph): progress, time left, stop / continue, summary and warnings, live trace and corner, predictive and SLD-profile bands (one tab per dataset); save chains, load MAP into the model, posterior bounds |
| `fit/dream/NOTES_dream.md` | Implementation notes and status of the DREAM plan |
| **`tabs/`** | **The tabs** |
| `tabs/simulation.py` | Simulation tab: layer editor, general parameters, polarisation table, profile and reflectivity plots |
| `tabs/experimental.py` | Experimental tab: data import, datasets, per-channel polarisation, plots; hosts the Fit tool box (`fit.panel`) |
| `tabs/geometry.py` | Geometry tab: 3-D view (OpenGL) of the beam, Pi / Pa, the polarisation axis and a layer's M split into what the channels measure (in plane, ∥ P, ⊥ P) |
| **`tests/`** | Physics and fit tests; `supermatrix.py` is an independent reference implementation (Ruehm, Toperverg & Dosch, PRB 60, 16073); `licorne_reference.py` ports Licorne 1.4.2's MATLAB engine (roughness slabs, supermatrix, spin average, resolution) as the oracle the Licorne tests compare against; `data/licorne/` holds the Licorne exports; `data/rms_additive_reference.npz` freezes the additive rms profile (from `scripts/make_rms_reference.py`) |
| **`scripts/plot_stack.py`** | Stand-alone matplotlib plot of a stack's profile and reflectivity |
| `scripts/make_rms_reference.py` | Writes `tests/data/rms_additive_reference.npz` |
| `make_app.py` | Builds `SANR.app` (macOS launcher with the app's name and icon; machine-specific, not in git) |
| `make_windows_installer.py`, `installer/sanr.iss`, `.github/workflows/windows-installer.yml` | The Windows installer: PyInstaller, then Inno Setup, on GitHub's Windows machines |
| `selftest.py` | `SANR --self-test <file>`: checks a built program (kernel, a reflectivity, the worker pool) |
| `sessions/` | Sessions and saved DREAM chains (File → Open session); `session_Cr2Te3*.json` are the examples in git, the rest stay local |
