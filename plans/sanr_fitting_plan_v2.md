# SANR — fitting robustness and Licorne-exact fixes — implementation plan v2

Target: `github.com/JFMilette/SANR`, branch `licorne-exact`, HEAD `628d173`.
v2 replaces `sanr_fitting_plan_v1.md` and the open items of `licorne_compat_plan_v1.md`.

## 0. Before starting

### 0.1 Files

| File, as delivered | Put it at | Status |
|---|---|---|
| `sanr_fitting_plan_v2.md` | `plans/sanr_fitting_plan_v2.md` | this plan |
| `repeated_q_fix.diff` | `plans/repeated_q_fix.diff` | **apply** (item B1); verified on a fresh checkout of `628d173` |
| `dream_snooker_tempering_prototype.diff` | `plans/dream_snooker_tempering_prototype.diff` | **reference** for D1/D2: applies cleanly and is validated, but it is a prototype; implement D1/D2 to the interface below (applying it first as a starting point is fine) |
| `bench_fit.py` | `benchmarks/bench_fit.py` | benchmark script, tested on both sessions; not part of pytest |

### 0.2 Test data

JF copies two Licorne 1.4.2 session folders into the repo:

```
tests/data/private/session_A/   from HC-230104A_chi5_6_8Afinal.zip
tests/data/private/session_B/   from Fit_thickness.zip
```

Copy only these files from each: `parameters.m`, `profile.dat`, `profile_sublayers.dat`, `q.dat`, `rexp1.dat`, `rexp2.dat`, `rtheory1.dat`, `rtheory2.dat`, `resolution.m`, `version.txt`. Leave out the `IPTS2233/` and `EB_040710/` subfolders of A (raw data, `.opj`). Keep B exactly as uploaded: its Licorne fit is unfinished, which is fine for its role as a fixed benchmark problem.

Add both to `tests/data/licorne/README.md`:

- **A:** 12 free parameters (the session's own fit flags). Its `q.dat` repeats Q = 0.0226139 and Q = 0.0617789 where the TOF angles are stitched. Its Licorne fit stopped at the 50-iteration LM limit (`LMNiter=50`), not at a minimum.
- **B:** 7 free parameters: thicknesses of layers 1–3 and roughness of layers 1–3 and the substrate. Its NSLDs and ρ are held at unconverged values. It is a benchmark problem, not a physics result.

Tests that need these folders skip if they are absent, like `test_licorne_end_to_end.py` does now.

### 0.3 Running the tests

```
QT_QPA_PLATFORM=offscreen python -m pytest -q tests          # ~3.5 min on one core
QT_QPA_PLATFORM=offscreen python -m pytest -q -m slow tests  # the slow sampler tests
```

Add a `slow` marker in `tests/conftest.py`, with `pytest_configure` registering it and `-m "not slow"` as the default through `pytest.ini`. Mark any test that runs longer than about 20 s as slow.

### 0.4 Conventions

These are the repo's existing conventions:

- Module docstrings are the documentation. Every rule reproduced from Licorne cites the file and line it comes from.
- Commit once per item, with that item's tests.
- Never loosen a Licorne-exact reference tolerance to make a test pass.

### 0.5 Order

1. Phase 1, B1 first.
2. Phase 3: D1, D2, D4, D5, D6.
3. Phases 4–7.
4. Phase 2 and D3, in any order after that.

### 0.6 Existing tests these changes affect

| Test | Affected by | Change |
|---|---|---|
| `test_licorne_resolution.py::test_licorne_scheme_unsorted_Q` | B1 | already updated in `repeated_q_fix.diff` |
| `test_licorne_resolution.py::test_kernel_row_sums` | B1 | already updated in `repeated_q_fix.diff` |
| `test_dream_fit.py::test_cancel_returns_partial` (asserts `seen[0].shape == (3, 3)`) | D4 | pass `n_chains=3` explicitly |
| `test_dream_fit.py`, `test_dream.py`: tests not about the defaults | D4 | pass the old settings explicitly (`n_chains=3, p_snooker=0, beta_start=1`), so their meaning and runtime do not change |
| `test_licorne_session.py::test_axis_map_svd` | B6 | its rank-3 case (a 90° tilt) must still raise |

## 1. Evidence (measured on HEAD plus B1)

**Both sessions reproduce Licorne.** SANR Licorne-exact against `rtheory1/2`: A 4.8e-6, B 9.2e-6 (the files' 6 significant figures). χ²/N at the session values: A 9.0249, B 13.4354.

**Landscape** (`bench_fit.py landscape`, L-BFGS-B):

| | A (24 random + current) | B (30 random + current) |
|---|---|---|
| Best χ²/N | 4.927 (only from the current values) | 13.318 |
| Random starts within +1 of the best | 0 % | 94 % |
| Fits with ≥ 1 parameter on a bound | 100 % | 100 % |

**DREAM at HEAD** (archive ZS, Latin-hypercube start):

- **A, n_gen 5000:**
  - 3 chains: χ²/N 12.07, 10.36, 6.63 (R̂ 3.4, 1.5, 194);
  - 8 chains: 10.85, 7.56;
  - 8 chains with the error scale: 12.64, 11.52.
- **B, n_gen 8000, 7 runs:** all χ²/N 13.29, R̂ ≤ 1.09.
- **A, warm-started at the 4.927 minimum** (8 chains, 3000 generations): stays at 4.93. Layer 5 thickness piles on its 100 Å bound (49 % of samples) and the substrate roughness on 50 Å (26 %).

**Synthetic sampler tests at HEAD:**

- **Exact:** flat 5-D box; half-normal at a bound.
- **Two-mode target `TWO_MODES`:** d = 10, modes at 0.25 and 0.75 in every coordinate, sd 0.03, weights 0.3 / 0.7, bounds [0, 1]. With 3 chains, seeds 1–5 give mode-2 weights 0.64, 0.67, **0.00**, 0.54, **0.00**, and R̂ ≈ 1.0 in both zero cases (false convergence).

**Prototype (`dream_snooker_tempering_prototype.diff`):**

- Snooker is exact. At p_snooker 0.1 / 0.5 the flat-box KS is 0.018 / 0.024, half-normal means are within 3 %, and the 10-D correlated Gaussian mean and covariance are within 0.094 sd.
- `TWO_MODES` with 8 chains, snooker 0.1 and tempered burn-in (β 1e-3 → 1 over 3000 generations, n_gen 30000), seeds 1–5: weights 0.86, 0.66, 0.61, 0.65, 0.68 (mean 0.69), R̂ ≤ 1.26.
- **Session A, same recipe, n_gen 5000, seeds 1/2:** all 8 chains in ONE basin, but χ²/N 6.95 and 12.37, not 4.93. Tempering prevents split chains but does not solve A's global search, hence Phases 4–5.

## 2. Phase 1 — bugs

### B1. Repeated Q in the Licorne-exact convolution — apply `repeated_q_fix.diff`

**Problem.** Licorne's `resolut` multiplies by spacings and never divides by them, so a repeated Q inside the grid (common where TOF angles are stitched) is valid. SANR refuses it, so session A cannot be loaded.

**The diff:**

- `model/licorne_resolution.kernel` accepts non-decreasing Q.
- It raises `ValueError('…first or last Q is repeated…')` when Q[0] == Q[1] or Q[-1] == Q[-2]. The constant tail steps by the end spacing, so the loop would never end; Licorne hangs there too.
- `Stack._resolve_licorne` drops its distinct-Q check.
- `FitProblem._groups` keeps repeated points (a stable argsort plus an inverse index, instead of `np.unique`).
- The two tests in §0.6 are updated.

**Add** `tests/test_licorne_sessions.py::test_session_A_reproduces_licorne` (skips if absent):

- the kernel equals `licorne_reference.resolut` to ≤ 1e-13 on A's grid with its sigma;
- SANR equals `rtheory1/2` to ≤ 1e-5;
- `FitProblem` χ²/N at the session values is 9.0249 ± 1e-4.

Do the same for B (≤ 1e-5; χ²/N 13.4354 ± 1e-4).

### B2. The fit window and the filters shrink the Licorne convolution grid

**Problem.** Licorne computes the model on ALL data points and only then selects `QPOINTS` (`fit_reflection.m`: `reflection_m(expandrough(M),Q_dQ{n},…)` on the full grid, then `Rmod(QPOINTS{n},k)`). `FitProblem.__init__` builds each channel's grid from `Q[ok]`, where `ok` also includes the Q window, `R > 0` for the log cost, and `dR > 0`.

**Change** `fit/problem.py`, for `resolution['scheme'] == 'licorne'` only:

- the convolution grid of channel k is `Q[np.isfinite(Q)]` (all its points, in the stable sort order of B1);
- the fitted points stay `ok`, and become indices into that grid;
- `_groups` groups channels by their full grid.

`npoints`, residuals and costs are unchanged: they use the fitted points only. The quadrature scheme is unchanged.

**Test** `test_licorne_resolution.py::test_fit_window_keeps_full_grid`, on the `fixture1_stack()` of that file (Q1 = `linspace(0.0106348, 0.24841, 204)`, `BEST_103` resolution):

- `FitProblem(..., qmax=0.22).model(x)` equals `st.reflectivities(Q1, pairs)[Q1 <= 0.22]` to ≤ 1e-12 (HEAD: 33 % at Q = 0.219);
- the same for the log cost with R ≤ 0 at every third point above Q = 0.2 (HEAD: 1.5 %).

### B3. Model curves on the wrong grid in Licorne-exact mode

**Problem.** In Licorne-exact mode the curve depends on the grid it is computed on. At HEAD the Simulation tab's grid shifts it by up to 11 % near Q_c.

| Where | What happens at HEAD |
|---|---|
| `tabs/experimental.py` `add_simulation` and `_update_live` | evaluate on `sim.last_Q`, i.e. `linspace(qmin, qmax, nq)` |
| `fit/panel.py` `start_fit` | passes `np.linspace(qmin, qmax, sim.nq.value())` to `DEWindow.start` |

**Change.** Add `model_grid(stack, dataset_channels)` in `tabs/experimental.py`:

- with `resolution['enabled']` and `scheme == 'licorne'`, it returns each channel's full data grid (the B2 grid);
- otherwise it returns the current linspace.

The live simulation dataset uses the grid of the dataset selected for fitting (`experimental` session `fit.dataset`). If none is selected it keeps `sim.last_Q`, and the Simulation tab shows: "Licorne convolution: this curve depends on its Q grid; compare with data on the data's Q". `DEWindow` gets per-channel grids the same way, and `simulation_channels` is called once per distinct grid.

**Test** (`tests/test_simulation_tab.py` style, offscreen): the DE window's model for fixture1's dataset equals `FitProblem.model` at the fitted points to ≤ 1e-12.

### B4. Live curves ignore the per-channel norm

`add_simulation` and `_update_live` pass the norms of the dataset selected for fitting (`{ch: d['norm']}`) to `simulation_channels`.

Test: with a dataset whose channel has norm 0.5, the live channel equals 0.5 × the unnormalised model + background.

### B5. sinh(S d)/S at S = 0 (`model/stack.py`, `_slab_matrices`)

Replace `s1S, s3S = s1/S1, s3/S3` with:

```python
with np.errstate(invalid='ignore', divide='ignore'):
    s1S = np.where(S1 == 0, d, s1/S1)
    s3S = np.where(S3 == 0, d, s3/S3)
```

Remove the `xfail(strict=True)` marker from `test_licorne_roughness.py::test_slab_at_its_own_critical_Q`; it then passes (verified).

### B6. The import refuses negligible out-of-plane tilts (`model/licorne_io.py`, `licorne_axis_map`)

**Problem.** The rank test `sv > 1e-9·sv[0]` refuses a 0.01° tilt, as left behind by a Licorne angle fit.

**Change.** New constant `AXIS_TILT_MAX = 0.5` (degrees). The signature becomes `licorne_axis_map(par, top, profile=None, notes=None)`.

1. Take the SVD of the stacked unit vectors V (as now). The normal is n = Vt[-1] for rank ≥ 2; for rank 1, use the existing rule.
2. Compute `tilt = max_v degrees(arcsin(|v·n|))`.
3. If `tilt > AXIS_TILT_MAX`, raise the existing "not coplanar" ValueError, with the tilt added to the message.
4. Otherwise build R as now from e1 and n. If `tilt > 1e-9`, append to `notes`: "out-of-plane parts up to {tilt:.3g}° dropped (Licorne used them; specular PNR cannot see them)".

`load_licorne_model` and `load_licorne_session` pass `notes` through. `sample_angles` already zeroes components below 1e-12. Here, set the elevation to 0 for every vector when `tilt ≤ AXIS_TILT_MAX`.

Tests:

- v127_chi3_137 with `Layers(3).msld=[2.60371e-06,9.83039,90.01]` imports, with a note containing "0.01", and every `MSLD_phi == 0`.
- The existing 90° case still raises.

### B7. DREAM docstring (`fit/dream/dream.py`, module docstring)

Replace "cannot permanently lose a mode" with: "keeps every mode some chain has visited; a mode never visited is not found (see the start options and the tempered burn-in)".

## 3. Phase 2 — Licorne import completeness (`model/licorne_io.py`)

Licorne source excerpts are quoted so that no access to Licorne is needed.

### I1. Q_mult

`q.dat` is raw. `run_refl.m`:

```matlab
Q=UD{1}; ... Q=Q*Par.Model.Q_mult; Q_dQ=[Q,resolution(Par,Q)];
```

So Q·Q_mult is used everywhere, including in the resolution formula. In `load_licorne_session`:

- the data Q (the `q_file` channels and the theory curves) become Q·Q_mult;
- `qrange` uses the scaled Q.

The Experimental tab reads Q from `q.dat` through `file_ref`, so add a `q_mult` field to the dataset channel dict, applied when the file is read. Save it in the session.

### I2. Rexp_mult

The rexp files are raw. `licorne_cb.m` (≈ lines 8516 and 8762):

```matlab
for k=1:Pol_num, Rexp(:,k)=Rexp(:,k)*Rexp_mult(k); Err(:,k)=Err(:,k)*Rexp_mult(k); end
```

Add `r_mult` to the channel dict and apply it to R and dR when reading. Remove I1/I2's "not applied" note from `import_notes`.

Tests for I1/I2: copy fixture1-like data into a `tmp_path` session folder with `Q_mult=1.01;` and `Rexp_mult=[2,1,1,1,1,1];` edited into `parameters.m`. Check that the imported Q, R and dR are scaled exactly.

### I3. QLims becomes the fit range

`main.py`, `_licorne_state`: set the simulation `qmin`/`qmax` to `QLims ∩ [min Q, max Q]`, rounded outwards as now. This is safe only after B2.

### I4. ExpFit becomes per-channel fitting

A channel with `ExpFit(k)=0` is imported with `'fit': False`. `FitPanel._fit_inputs` skips it, and the tree draws it greyed out with the tooltip "not fitted (Licorne ExpFit = 0)".

### I5. Coupled parameters

A nonzero `<field>_fit` value is a parameter NUMBER, and equal numbers are fitted as one parameter. Session A ties layer 2 and layer 3 NSLD (both `re_nsld_fit=6`). Licorne's mapping (`licorne_cb.m` ≈ 8936–8995), for members j with start values v_j and bounds [min_j, max_j]:

- shared parameter S, start S0 = Σ v_j, bounds [Σ min_j, Σ max_j];
- for S ≥ S0, each member is v_j + (S − S0)(max_j − v_j)/(Σ max − S0);
- for S < S0, each member is v_j + (S − S0)(min_j − v_j)/(Σ min − S0).

Import equal numbers as a T1 tie group with mode `'licorne'`.

### I6. Missing ResolutionFun

Keep mode 2, which matches saved rtheory files before 1.2.3. Add the note: "Licorne 1.4.2 would recompute this session with ResolutionFun 3 (read_param.m)".

## 4. Phase 3 — DREAM (`fit/dream/dream.py`, `fit/dream/dream_fit.py`, `fit/panel.py`)

### D1. Snooker updates (DREAM(ZS) only)

**Interface.** `dream(..., p_snooker: float = 0.0)`; `sample(..., p_snooker=0.1)` (see D4); also stored in the resume state and in `save_result` metadata. Old states load with 0.

**Algorithm** (ter Braak & Vrugt 2008), per chain and generation, with probability `p_snooker` when `archive=True`:

```
draw three distinct archive indices -> z, z1, z2
e = x - z;  if e.e == 0: make a regular DREAM move instead
gamma_s ~ U(1.2, 2.2)
x' = x + gamma_s * e * (e.(z1 - z2)) / (e.e)          # all d coordinates, no crossover
log_jac = (d - 1) * (log ||x' - z|| - log ||x - z||)
```

- **Bounds.** A snooker proposal is NEVER folded or reflected. If any coordinate leaves [lo, hi], it is rejected without evaluating `logpdf` (log p = −∞). This is what keeps the move exact.
- **Metropolis.** log α = log p(x') − log p(x) + log_jac.
- **Adaptation.** Snooker moves do not enter the pCR statistics J and n_id.
- **Evaluations.** Count only proposals actually evaluated.

The prototype diff implements exactly this.

**Tests** (new `tests/test_dream_sampler.py`, using `dream()` vectorised):

| Test | Setup | Pass |
|---|---|---|
| `test_snooker_flat_box` | logpdf = 0 on [0, 1]^5, 3 chains, n_gen 20000, seed 1, p_snooker 0.1 and 0.5 | max KS (every 10th sample, each coordinate, against uniform) < 0.04; mass in the outer 5 % per side / 0.1 in [0.95, 1.05] |
| `test_snooker_half_normal` | −½Σ(x/sd)², sd = (0.05, 0.1, 0.2, 0.3), bounds [0, 1]^4, seed 2 | means / (sd·√(2/π)) within [0.96, 1.04] |
| `test_snooker_gaussian` | as `test_dream.test_gaussian` with p_snooker 0.1 and 0.5 | max\|mean\|/sd < 0.15, max\|ΔCov\|/(sd_i sd_j) < 0.15 |
| `test_snooker_resume` | n_gen 400 in one run vs 200 + resume to 400, p_snooker 0.3 | bit-identical chains |

Measured on the prototype: KS 0.018 / 0.024; edge ratio 1.004 / 1.032; half-normal ratios within 0.981–1.027; Gaussian 0.072 / 0.080–0.094.

### D2. Tempered burn-in

**Interface.**

- `dream(..., logpdf_split: Optional[Callable] = None, temper_until: Optional[int] = None, beta_start: float = 1.0)`.
- `logpdf_split(X) -> (logprior (N,), loglike (N,))` (vectorised, or through `map_func`).
- `temper_until` defaults to `adapt_until`. Tempering is active iff `beta_start < 1`, and then `logpdf_split` is required.
- `PosteriorProblem.split(x) -> (logprior, loglike)`, with loglike −∞ where the prior is −∞. `_Unit` gets a matching `split(u)`.
- `sample(..., beta_start=1e-3)` (see D4) wires `logpdf_split`.

**Algorithm.**

- **Bookkeeping.** Keep `lp_X`, `ll_X` per chain; `logp_X = lp_X + ll_X` (stored and used as now). Evaluate each proposal once, as `(lpp, llp)`; `logp_p = lpp + llp`.
- **Schedule.** For t ≤ temper_until, β(t) = beta_start ** (1 − t/temper_until), so β = 1 at t = temper_until.
- **Acceptance.** While β < 1, log α = (lpp + β·llp) − (lp_X + β·ll_X), plus log_jac for snooker moves; when β = 1, as before.
- **State.** `lp_X`, `ll_X`, `temper_until` and `beta_start` go into the resume state. Old states load with `beta_start = 1`.

The tempered phase is inside the adaptation phase, which `samples()` never returns, so the stored posterior stays exact.

**Tests.**

- `test_tempering_off_is_identical`: `beta_start=1` gives chains bit-identical to not passing it.
- `test_two_modes_found` (slow): `TWO_MODES` with 8 chains, p_snooker 0.1, beta_start 1e-3, adapt_until = temper_until = 3000, n_gen 30000, seeds 1–5.
  - Every seed has each mode's weight ≥ 0.05 in `samples(0.5)`.
  - The mean mode-2 weight over seeds is within 0.10 of 0.70.
  - Measured: 0.86, 0.66, 0.61, 0.65, 0.68.

### D3. Parallel tempering (optional; after D1–D2)

**Interface.** `dream(..., betas=None)`; `betas` a sorted array ending at 1, e.g. `np.geomspace(1e-2, 1, 4)`.

**Algorithm.**

- Each β has its own N chains and its own archive. The target is lp + β·ll throughout (no schedule).
- After each generation, for each adjacent pair (i, i+1), pick one chain index c uniformly and swap states and (lp, ll) with probability min(1, exp((β_i − β_{i+1})(ll_{i+1,c} − ll_{i,c}))).
- Only β = 1 chains are stored in `chains` and `logp`. Record the swap acceptance per pair in the result.

**Cost.** K·N evaluations per generation. GUI: an "advanced" checkbox with K.

**Test** (slow): `TWO_MODES`, 3 chains per β, `betas=geomspace(1e-2, 1, 4)`, n_gen 30000, seeds 1–5: each seed's mode-2 weight within 0.05 of 0.70.

### D4. New defaults

| Setting | Old | New |
|---|---|---|
| chains with archive (`sample`, GUI "auto") | 3 | 8 |
| `p_snooker` | — | 0.1 |
| `beta_start` (tempered burn-in) | — | 1e-3 |
| R̂ target (`sample`, GUI) | 1.2 | 1.1 |

Classic DREAM keeps `max(7, d + 1)`.

GUI (`fit/panel.py` `_build_dream_page`):

- a "snooker" QDoubleSpinBox (0–0.5, step 0.05);
- a "tempered burn-in" checkbox (on), with β start as a QDoubleSpinBox in log10 (−6 … 0, default −3);
- tooltips explaining both.

Saved runs from before still load (missing keys take the old behaviour).

**Regression test** (slow): session B, 7 parameters (§0.2), new defaults, n_gen 8000, seed 1 → χ²/N of the MAP ≤ 13.30 and R̂ max ≤ 1.1. HEAD gives 13.29 / ≤ 1.09.

### D5. Detect chains in different minima

**`dream_fit.posterior_warnings(result, bounds=None, ..., split_nats=10)`.** After adaptation, take each chain's mean log posterior over `samples(discard)`. If max − min > `split_nats`, warn: "chains sit in different minima (χ²/N per chain: a, b, …)", and report not-converged even when R̂ passes. χ²/N per chain is computed by `chi2_per_point` on each chain's best kept state; pass the problem in.

**The DREAM window** shows each chain's current χ²/N in the trace page.

**Tests:**

- With x0 putting 4 chains in each mode of `TWO_MODES`, 3 chains, no snooker or tempering, n_gen 3000: the warning appears.
- With all chains in one mode it does not. Document in the docstring that one-mode false convergence is NOT detectable this way, which is why D2 exists.

### D6. Start options

`sample(x0=...)` gains:

- `'current'`: N jittered copies of the problem's current values, jitter N(0, 0.002·(hi − lo)), clipped to the bounds;
- an `(N, d)` array, as now; O1 rows arrive this way.

GUI DREAM page: a "start" QComboBox: "Latin hypercube" | "DE population" (`'de'`) | "current values" (`'current'`) | "multistart minimum" (enabled after O1; uses the selected row of the multistart table).

Test: `'current'` gives N distinct starting states within 1 % of the bound width of the current values.

## 5. Phase 4 — optimisation

### O1. Multistart (new `fit/multistart.py`)

```python
run_multistart(problem, n_starts=24, method='lbfgs', seed=None, workers=1,
               cancelled=None, de_options=None) -> list[dict]
# each: {'chi2', 'x' (physical), 'u', 'hits', 'at_bound' (bool per param), 'from_current'}
```

- **Starts.** Start 0 is the current values `problem.x0()`. Starts 1..n are the first n points of `scipy.stats.qmc.Sobol(d, seed=seed).random_base2(ceil(log2 n))`.
- **Local fit.** `'lbfgs'` is `scipy.optimize.minimize(problem, u0, method='L-BFGS-B', bounds=[(0,1)]*d, options={'maxiter': 3000})`. `'de'` is `run_de(problem, maxiter=100, polish=True, seed=seed+k)`, with x0 forced to u0.
- **Distinct minima.** Merge minima with ‖u_a − u_b‖∞ < 0.02 and |χ²_a − χ²_b| < 1e-3; keep the best member and count hits. Sort by χ².
- **Workers.** `workers > 1` uses a spawn Pool, like `run_de`.

GUI: a "Multistart" page in the fit tool box with n_starts, method and a "Run" button, and a table (χ²/N, hits, parameters on a bound, values). Row actions: "Apply to model" and "Start DREAM here" (D6: N jittered copies of that row).

**Tests** (slow; skip without the sessions):

- B, 31 starts, lbfgs: best χ²/N ≤ 13.33, and ≥ 70 % of starts within +1 of the best (measured 94 %).
- A, 25 starts: best χ²/N ≤ 4.93 and `from_current` is True for it. Write the minima table as a pytest artefact; it is a benchmark, not an assertion.

### O2. DE convergence settings and restarts (`fit/de/de_fit.py`, `fit/panel.py`)

- `run_de` passes `atol` and `init` through to scipy. GUI defaults: `tol=1e-6`, `atol=0`, `init='sobol'`.
- A "restarts" spin box (1–20, default 1): run n DE fits with seeds seed+k, keep the best as the fit result, and list all their minima in the multistart table (O1 format).

At HEAD, seed 3 on session A stopped "successfully" at χ²/N 10.28 with `tol=1e-3`.

Test: on session A with restarts 3 and `maxiter=50` (fast), three minima are listed and the reported result is the best of them.

### O3. Priors in the DE cost (`fit/problem.py`)

`FitProblem` builds the prior terms from the stack (Phase 5). `__call__(u)` returns:

```
mean(residuals**2) + (-2 * sum(log prior_j(x_j))) / npoints
```

So the DE optimum is DREAM's MAP. The DE window labels the cost "χ²/N + prior" when any prior is set. `residuals()` is unchanged.

Test: a one-parameter linear toy model with a Gaussian prior (μ, τ) and Gaussian data; the DE optimum equals the analytic precision-weighted mean to 1e-6.

## 6. Phase 5 — priors from other measurements

JF often knows the value a thickness should have, but not its distribution.

**Data model.** `Layer.fit[attr]['prior']` (and `Stack.fit[attr]['prior']`) is `None` or one of:

```python
{'kind': 'gauss',    'mu': value, 'tau': tolerance}
{'kind': 'soft_box', 'a': lower,  'b': upper, 's': softness}
```

**Log densities** (normalised over ℝ; the hard bounds truncate, which is harmless for MCMC and DE):

```
gauss:    -0.5*((x-mu)/tau)**2 - log(tau*sqrt(2*pi))
soft_box: Z = (b - a) + s*sqrt(2*pi)
          inside [a, b]: -log Z;  outside: -0.5*(dist/s)**2 - log Z   (dist to the nearer edge)
```

Put them in a new `fit/priors.py`, with `log_prior(spec, x)` and `prior_terms(stack, params)` returning a list of callables or `None` aligned with `free_parameters()`.

**Defaults.** If only "expected" μ is given, `tau = 0.1*|mu|`. If μ = 0, ask for τ. For `soft_box`, `s = 0.1*(b - a)` when blank. Hard bounds stay the physical support: they are no longer the way to say "should be".

**Use.**

- `PosteriorProblem` merges the stack's priors with any `priors={name: callable}` given through the API (the API callables win on a clash).
- `save_result` stores the data priors in its metadata (callables still are not saved), and `load_run` restores them.

**GUI** (`tabs/simulation.py`, the layer editor rows of `EDITOR_PARAMS`, and the stack fit rows):

- a small "prior…" button per row opens a dialog: kind, expected, ± (shown greyed as "default 10 %" when blank), or a / b / s for soft box;
- a row with a prior shows a marker;
- stored in the session JSON; old sessions load with none.

**Diagnostics.** In both the DE report and `posterior_warnings`, flag "prior–data tension: X = … is 3.4 τ from μ" when the MAP value lies more than 3 τ from μ (gauss), or more than 3 s outside [a, b] (soft box).

**Tests.**

- Each kind integrates to 1 over a wide range (`scipy.integrate.quad`, 1e-8).
- A flat likelihood with a gauss prior: DREAM samples have mean and sd within 5 % of μ and τ.
- Session JSON and `save_result` round trips.
- O3's analytic test.

## 7. Phase 6 — error model

### E1. Suggest the error scale

After a DE, multistart or polish result with χ²/N > 2, the DREAM page ticks "fit error scale" and shows: "χ²/N = {c:.3g} at the optimum: the error bars underestimate the misfit by ×{sqrt(c):.2g}; the error scale is fitted". The user can untick it.

Test: the panel logic, offscreen.

### E2. Relative error floor

**Likelihood.** The effective errors are dR_eff² = dR² + (f·R_data)².

- In DREAM, `sample(..., error_floor='fit')` appends `log10_error_floor` in [−4, −0.5] (next to `log10_sigma_scale` if both are on).
- `make_gaussian_loglike` then computes `log_norm` per call: −Σ log dR_eff − ½N log 2π.
- In DE and `FitProblem`, `error_floor` is a fixed value f (GUI field, default 0 = off); residuals use dR_eff.

Test: a synthetic stack with 5 % multiplicative noise added to R, with dR set to the pure counting part → DREAM recovers f = 0.05 ± 0.01.

## 8. Phase 7 — ties and reparametrisation

### T1. Parameter ties

**Data model.** `Stack.ties = [{'members': [[i, attr], ...], 'mode': 'equal'|'licorne', 'v0': [...], 'lo': [...], 'hi': [...]}]`, with `v0` and the bounds frozen when the tie is made.

**`free_parameters()`.** Each group is ONE entry `(('tie', k), 'S', S, lo, hi)`:

- `'equal'`: S = the first member's value, bounds = the intersection of the members' bounds.
- `'licorne'`: as in I5.

**Everything else.**

- `FitProblem.apply` and `Stack.owner` handle `('tie', k)`: they set every member through the mode's mapping.
- `parameter_names` shows `tie(Layer 2.NSLD_real, Layer 3.NSLD_real)`.
- Members of a group are excluded from their own vary flags.
- The fit settings get a "tie with…" action.
- Ties are saved in the session.

**Tests.**

- With `'equal'`, members stay equal through a DE fit.
- With `'licorne'`, the mapping is reproduced at S0, at Σ min and at Σ max.
- Session A imports with the layer 2/3 NSLD tie. An L-BFGS-B polish from the session values reaches χ²/N 5.63 ± 0.05 (measured with the coupling emulated).

### T2. Sheet-moment mode

**Data model.** A layer may vary `MSLD_moment` = ρ·t instead of `MSLD_rho`; it is an entry in `Layer.fit` and the two are mutually exclusive in the GUI.

**Use.** `FitProblem.apply` sets the thickness first, then ρ = moment / t. A t at 0 is clipped at 1e-6 Å. The default bounds of the moment are [ρ_min·t_min, ρ_max·t_max] from the current entries.

Session A's data constrain ρ·t: 1.47–1.54 × 10⁻⁶ Å⁻¹ across every minimum found.

Test: on session A, warm-started at the 4.927 minimum, DREAM's relative 95 % width of the moment is smaller than that of ρ.

## 9. Phase 8 — benchmarks

Add `benchmarks/bench_fit.py` (delivered; tested on both sessions) and `benchmarks/README.md` with:

- the §1 tables as the reference numbers;
- the commands:

```
python benchmarks/bench_fit.py landscape tests/data/private/session_A --starts 24
python benchmarks/bench_fit.py landscape tests/data/private/session_B --starts 30 \
  --vary 1:thickness,2:thickness,3:thickness,1:roughness_sigma,2:roughness_sigma,3:roughness_sigma,4:roughness_sigma
python benchmarks/bench_fit.py dream tests/data/private/session_A --seeds 1 2 3 --n-gen 5000 --chains 3
python benchmarks/bench_fit.py dream tests/data/private/session_A --seeds 1 2 --n-gen 5000 --chains 8 \
  --p-snooker 0.1 --temper                                   # after D1/D2
```

`--p-snooker` and `--temper` pass `p_snooker` and `beta_start=1e-3` to `sample()`, so they work once D1/D2 exist. After O1, add a `multistart` mode, and after D3 a `--betas` option.

Results append to `benchmarks/results.jsonl`, which is git-ignored.

## 10. Open question for JF

May `session_A` and `session_B` go into the public repo as fixtures? If not, keep them in a local `tests/data/private/` that is git-ignored, and let the tests that need them skip.
