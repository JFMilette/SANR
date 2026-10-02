# DREAM sampling — Phase 0 notes

> **Paths since the clean-up:** `model/fit.py` is now `fit/problem.py`
> (`FitProblem`) + `fit/de/de_fit.py` (`run_de`); `model/dream.py`,
> `model/bayes.py`, `model/bayes_plots.py` are `fit/dream/dream.py`,
> `dream_fit.py`, `dream_plots.py`; `tabs/fitwindow.py` and
> `tabs/bayeswindow.py` are `fit/de/de_window.py` and
> `fit/dream/dream_window.py`; the Fit tool box is `fit/panel.py`;
> `tests/test_fit.py` and `tests/test_bayes_pnr.py` are `tests/test_de.py`
> and `tests/test_dream_fit.py`.  The notes below keep the names of the
> time they were written.

Answers to §0 of `dream_implementation_plan.md`, from the code at commit
2c576ca (branch `licorne-roughness`) plus the uncommitted Licorne-session
import.  Decisions and implementation status at the end.

## Layout

The plan's file names map onto this repo as follows. There is no `pnr/`
package and no `calc_reflectance`.

| Plan | Here |
|------|------|
| `fit.py` | `model/fit.py` |
| `stack.py` | `model/stack.py` |
| `polarisation.py` | `model/polarisation.py` |
| `simulation.py` (resolution) | resolution is in `model/stack.py` (`Stack._resolve`); `tabs/simulation.py` is only the GUI tab |
| `experimental.py` | `tabs/experimental.py` (GUI), `tabs/fitwindow.py` (progress window) |
| per-channel `calc_reflectance` | `Stack.reflectivity(Q, P0, P)`, a one-pair call of `reflectivities` |

`dream.py` and `test_dream.py` were supplied afterwards and are now
`model/dream.py` and `tests/test_dream.py`.

## 1. `model/fit.py`

- **Free parameters.** `Stack.free_parameters()` returns
  `[(layer index, attr, value, min, max)]` for every `Layer.fit[attr]` with
  `vary=True`. `attr` is one of `FIT_PARAMS`: `thickness`, `NSLD_real`,
  `NSLD_img`, `MSLD_rho`, `MSLD_theta`, `MSLD_phi`, `roughness_sigma`, in
  layer order, then attribute order. `Stack.fit_allowed` excludes the
  thickness of the semi-infinite media and the fronting's roughness. There
  are no names beyond `(layer.name, attr)`, and **no links or
  constraints**: every parameter is independent, with box bounds.
- **Vector → Stack.** `FitProblem.apply(x)` sets each attribute with
  `setattr`, and `FitProblem.model(x)` then calls `build_sublayers()` and
  one `reflectivities(self.Q, self.pairs)` on the union of all channels' Q.
  DE works in normalised `u ∈ [0,1]^n` (`to_x` / `to_u`).
- **Objective.** `'chi2'`: the mean over all points of `((R_model − R)/dR)²`
  in **linear R**. Points with `dR ≤ 0` use `dR = |R|`. `'log'`: the mean
  of `(log10 R_model − log10 R)²`. Non-finite residuals are replaced by 1e3
  (a penalty), not raised.
- **Background double-count.** There is none in the model.
  `Stack.reflectivities` adds `self.background` once per pair, so R+ is
  R++ + R+- + bg. The data side is separate: `tabs/experimental.py
  half_polarized` builds R± from measured R++ + R+- for asymmetry display
  only, never for fitting.
- **Already GUI-free and picklable.** `FitProblem` deep-copies the stack and
  holds only numpy arrays. `run_de` sends it to `spawn` worker processes
  (`_PoolMap`), and `tests/test_fit.py::test_workers_fit_like_one_process`
  covers this. **No GUI entanglement**, so the §0 stop condition is not
  triggered.

## 2. `model/stack.py`

- `reflectivities(Q, pairs)` **exists**. All pairs come from one
  transfer-matrix pass (`_trace` for a non-magnetic fronting,
  `_lab_channels` + weights for a magnetic one), with resolution, then
  background once per pair.
- Existing tests (`tests/test_stack.py`):
  - `test_batched_equals_single`: ≤ 1e-14, all four NSF/SF channels plus
    R+/R−, on a **noncollinear** Fe/Cr/Co stack (different θ and φ), with a
    non-magnetic or magnetic fronting, `q_in_fronting` on/off, and
    resolution on/off.
  - `test_no_analyser_counts_background_once`
  - `test_matches_supermatrix_eq5`: an independent reference implementation.
  - `test_total_reflection_is_unitary`: magnetic backing.
- **Gap for P2.** No batched-vs-single comparison with a **magnetic
  backing**. Also, "batched = single" is partly tautological here, because
  `reflectivity` is `reflectivities` with one pair. The independent check is
  the supermatrix test.

## 3. `model/polarisation.py`

- Efficiencies are the **lengths of the P0 / P vectors** of each channel,
  applied in `_density` (ρ = (1 + P·σ)/2) inside `_trace`. A channel gets its
  vectors in the GUI (own pair, or the Simulation tab's Polarisation box),
  and `FitProblem` takes them as fixed `P0`, `P` per data channel.
- **They are not fittable.** The same is true of background
  (`Stack.background`, not in `FIT_PARAMS`), and **there is no scale /
  normalisation factor at all**. The plan's nuisance parameters (§5.2 tests
  4 and 5, §8 Q2) therefore need new fit parameters first, which also
  changes the DE fit.

## 4. Resolution

- In `Stack._resolve`, applied **inside the model to every pair**, before
  the background. It is a fixed Gaussian quadrature (101 nodes over ±RES_SPAN
  σ) of an ideal R computed on an adaptive Q grid and interpolated.
- Modes: `'mono'` (fixed λ, dλ/λ, dθ) and `'tof'` (per-angle bands, Licorne's
  `resolution.m` template; `test_tof_resolution_matches_licorne_script`).
- **No Licorne-style discrete convolution / MaxwellWidth** exists.
- The DE cost and any likelihood built on `FitProblem.model` see only
  convolved curves.
- Open discrepancy: on IPTS 27232 S1 `best_103` with magnetism removed, the
  imported model differs from Licorne's `rtheory1.dat` by a median of 6.5 %
  (up to 50 % in the minima, where Licorne's curve is more smeared). No
  constant σ factor or norm factor removes it. This bears directly on §8 Q1.

## 5. Determinism (P4) and failures (P5)

- `FitProblem.model(x)` **mutates its own stack copy** (`apply` +
  `build_sublayers`). It is still a pure function of x, because every free
  attribute is overwritten and the sublayers are rebuilt on each call. There
  is no global state and no identity-keyed cache. It is safe per process,
  but **not thread-safe** (one `FitProblem` must not be shared between
  threads).
- Failures: NaN or inf comes back as data, and `residuals` maps it to 1e3.
  `build_sublayers` / `_check_modes` raise `ValueError` for bad modes. A
  log-likelihood wrapper must catch exceptions and non-finite values and
  return `-inf`, which is not what the DE penalty does.

## 6. GUI fit launch (`tabs/experimental.py`)

- `ExperimentalTab.start_fit` builds the `data` list (Q, R, dR, P0, P per
  channel of the chosen dataset, within the Simulation tab's Q range), makes
  a `FitProblem`, opens `FitWindow`, and starts a `FitThread` (a `QThread`
  running `run_de`).
- Progress goes out through the `progress(x, cost, generation)` signal.
  Cancel works through `cancelled=lambda: self._stop`, polled per evaluation
  and every 0.05 s while a worker generation is out.
- Results are sent to the Simulation tab from the fit window
  (`_send_fit`), and the Simulation tab is disabled during a fit.
- A DREAM run fits the same pattern: a `QThread` running `sample(...)`, with
  `callback` emitting progress and returning the stop flag.

## 7. Timing (§5.3)

One `FitProblem.__call__` on the `best_103` model, 4 channels × 204 Q,
d = 16, Apple-silicon laptop, one process:

| Resolution | Time per evaluation |
|------------|--------------------:|
| On (TOF) | 33 ms |
| Off | 2.7 ms |

The resolution quadrature is about 92 % of the cost, and 33 ms is above the
plan's 20 ms threshold, so `_resolve` should be profiled before tuning the
sampler. For scale: 30 000 generations × 3 chains ≈ 50 min serial, or about
17 min on 3 workers.

## Decisions (JF, 2026-10-01)

1. **Resolution:** keep the current scheme (`Stack._resolve`); no
   Licorne-style discrete convolution.
2. **Polarisation:** the channels' pairs as set in the GUI (Simulation tab
   or the channel's own), fixed; efficiencies are not fitted.
3. **Joint fits** of several fields or temperatures: later, not now.
4. **Thin-layer J_eff:** nothing to do.
5. **Scale and background** are fittable for both DE and DREAM.

## Status

| Plan item | State |
|-----------|-------|
| P1 background once | Already true; the tests now also cover scale |
| P2 batched channels | Already true; magnetic-backing case added (`test_batched_equals_single_magnetic_backing`) |
| P3 resolution | Inside the model (`Stack._resolve`), unchanged |
| P4 determinism | Yes per process (see §5); a fixed seed gives identical chains with 1 or 3 workers |
| P5 failures → −inf | `PosteriorProblem` catches every exception and non-finite R |
| Nuisance parameters | `Stack.scale` (R = scale·R + background), and `Stack.fit` entries for `scale` and `background`; `free_parameters()` lists them with layer index `None`; Simulation tab rows with Min / Max / Fit |
| `dream.py` / `test_dream.py` | Copied unchanged.  Additions only: cancel (callback returns True), `ess()`.  `tests/conftest.py` puts `fit/dream/` on the path for `from dream import ...` |
| `bayes.py` §3.1, §3.3 | Done.  Samples in normalised u ∈ [0,1]^d and returns physical units.  `x0='de'` uses the best members of `run_de`'s final population (`run_de(full_output=True)`) |
| §3.2 `reparam='inplane'` | **Not done** (kept simple); the DE fit's Halperin warning still shows when rho and phi are both varied |
| §3.4 results | `save_result` / `load_result` (.npz + JSON meta: names, bounds, data hash, sources, git hash, mode flags, stack), `summary_table` / `format_summary` (median, ±1σ, 95 %, R̂, ESS, MAP χ²/N), `derived`, `posterior_warnings` |
| §3.4 plots | `model/bayes_plots.py`: trace, corner, predictive band (+ spin asymmetry for R+/R− or R++/R−− on one Q grid), SLD band from the slabs |
| §4.1 checkpoint / resume | Resume done: `DreamResult.state` + `dream(resume=...)` / `bayes.sample(resume=...)` continue a stopped or finished run bit for bit (test: stop + continue = one run); the state is saved in the .npz, so a loaded run can be continued (GUI: Continue in the DREAM window).  No periodic checkpoint files during a run |
| §4.4 snooker | Not done (optional) |
| §5.2 tests | 1 (noise-free), 3 (DE agreement), 4 (background), 5 (scale) in `tests/test_bayes_pnr.py`, plus sigma_free, priors, failures, pickling, cancel, save/load, DE start.  **Not done:** 2 (coverage, slow) and 6 (needs reparam).  Test 4 has no old double-count path to compare against: the code never had one |
| §6 GUI | "Bayesian sampling (DREAM)" box in the Experimental tab and `tabs/bayeswindow.py` |

Deviations from the plan's wording:

- **§5.2 test 3.** The plan asks for DE χ² ≥ MAP χ² − 1e-6. With a uniform
  prior the MAP is the χ² minimum, so the polished DE optimum should be at
  least as good as any visited state. The test asserts DE χ² ≤ MAP χ² +
  1e-6 instead.
- **§5.2 test 5.** "Posterior contains the truth" is checked as within ±3σ
  of the median, not as inside the 95 % interval. On one noisy dataset with
  5 parameters, some 95 % interval misses the truth about 1 time in 4 by
  design. In the seeded test, the Fe thickness sits 2.3σ off the truth, at
  the same place as the DE optimum.
