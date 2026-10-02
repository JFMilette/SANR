# SANR

Polarised neutron reflectometry with arbitrary polarisation: build a layer
stack, simulate its reflectivity for any incident / analysed polarisation, and
fit it to measured channels.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py                 # the GUI
python make_app.py             # macOS: build SANR.app (Dock name and icon), then open it
python -m pytest tests         # the tests (pip install pytest first)
```

## Layout

| Path | What it holds |
|------|---------------|
| `main.py` | Main window, menus and session files (File → Save / Open session: model, settings and the imported data files; File → Import Licorne session: a Licorne session folder) |
| `style.py`, `icons/` | The dark theme and its icons; `icons/app.png` is the app icon |
| **`model/`** | **The physics, no GUI** |
| `model/stack.py` | `Layer`, `Stack`: roughness profile, slicing, transfer matrix, `Stack.reflectivities(Q, pairs)`; fittable `scale` and `background`. The module docstring explains the physics and its conventions. |
| `model/roughness.py` | Licorne's interface kernel (manual App. 10.1): window, closed-form J, renormalisation, sigma conversions |
| `model/polarisation.py` | Named channels (`++`, `+-`, `-+`, `--`, `+`, `-`) as (P0, P) pairs, and the Pi / Pa form the GUI uses |
| `model/licorne_io.py` | `load_licorne_model`: a Licorne export (`parameters.m`, `profile.dat`) as a Licorne-exact Stack (`Stack.set_licorne_exact`), its axes rotated so that M and P lie in the film plane (`licorne_axis_map`); `load_licorne_session`: a whole saved Licorne session folder (model, fit bounds, background, `Norm_factor`, TOF / MONO resolution, `rexp*.dat` channels, `rtheory*.dat` curves) and notes on what is not reproduced |
| `model/licorne_resolution.py` | Licorne's resolution convolution on the data grid (`resolut`, modes 1–3) as a sparse weight matrix |
| **`fit/`** | **Fitting** |
| `fit/problem.py` | `FitProblem`: free parameters, data, model and cost, shared by both methods |
| `fit/panel.py` | The Fit tool box of the Experimental tab (general / DE / DREAM pages and their actions) |
| `fit/de/de_fit.py` | `run_de`: differential-evolution fit (scipy), optionally over worker processes |
| `fit/de/de_window.py` | DE window: best cost per generation, profile and reflectivity of the selected one, send to the Simulation tab |
| `fit/dream/dream.py` | DREAM / DREAM(ZS) sampler (Vrugt 2016), NumPy only: resumable state, `ess` |
| `fit/dream/dream_fit.py` | `PosteriorProblem` (the `FitProblem` as a Gaussian log posterior), `sample` (also continuing a run), `save_result` / `load_run` (a saved run with its data, rebuilt without the GUI), `summary_table`, `derived`, predictive and profile bands |
| `fit/dream/dream_plots.py` | Matplotlib figures of a DREAM run for scripts and reports (traces, corner, predictive and SLD-profile bands) |
| `fit/dream/dream_window.py` | DREAM window (pyqtgraph): progress, time left, stop / continue, summary and warnings, live trace and corner, predictive and SLD-profile bands; save chains, load MAP into the model, posterior bounds |
| `fit/dream/NOTES_dream.md` | Implementation notes and status of the DREAM plan |
| **`tabs/`** | **The tabs** |
| `tabs/simulation.py` | Simulation tab: layer editor, general parameters, polarisation table, profile and reflectivity plots |
| `tabs/experimental.py` | Experimental tab: data import, datasets, per-channel polarisation, plots; hosts the Fit tool box (`fit.panel`) |
| `tabs/geometry.py` | Geometry tab: 3-D view (OpenGL) of the beam, Pi / Pa, the polarisation axis and a layer's M split into what the channels measure (in plane, ∥ P, ⊥ P) |
| **`tests/`** | Physics and fit tests; `supermatrix.py` is an independent reference implementation (Ruehm, Toperverg & Dosch, PRB 60, 16073); `licorne_reference.py` ports Licorne 1.4.2's MATLAB engine (roughness slabs, supermatrix, spin average, resolution) as the oracle the Licorne tests compare against; `data/licorne/` holds the Licorne exports |
| **`scripts/plot_stack.py`** | Stand-alone matplotlib plot of a stack's profile and reflectivity |
| `make_app.py` | Builds `SANR.app` (macOS launcher with the app's name and icon; machine-specific, not in git) |
| `session_Cr2Te3*.json` | Example sessions (File → Open session) |
