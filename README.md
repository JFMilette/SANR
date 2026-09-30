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
python -m pytest tests         # the tests (pip install pytest first)
```

## Layout

| Path | What it holds |
|------|---------------|
| `model/stack.py` | `Layer`, `Stack`: roughness profile, slicing, transfer matrix, `Stack.reflectivities(Q, pairs)`. The module docstring explains the physics and its conventions. |
| `model/polarisation.py` | Named channels (`++`, `+-`, `-+`, `--`, `+`, `-`) as (P0, P) pairs, and the Pi / Pa form the GUI uses |
| `model/fit.py` | `FitProblem` and `run_de`: differential-evolution fit of polarised channels |
| `tabs/simulation.py` | Simulation tab: layer editor, polarisation table, profile and reflectivity plots |
| `tabs/experimental.py` | Experimental tab: data import, per-channel polarisation, fitting |
| `tabs/fitwindow.py` | Fit window: best cost per generation, profile and reflectivity of the selected one, send to the Simulation tab |
| `main.py` | Main window, menus and session files (File → Save / Open session: model, settings and the imported data files) |
| `tests/` | Physics and fit tests; `supermatrix.py` is an independent reference implementation (Ruehm, Toperverg & Dosch, PRB 60, 16073) |
| `Cr2Te3.json`, `CrSb.json` | Example models (File → Open session) |
| `test.py` | Stand-alone matplotlib plot of a stack's profile and reflectivity |
