# SANR — per-angle stitch factors as fitted parameters — implementation plan v1

Target: `github.com/JFMilette/SANR`, branch `thickness-spread-and-fit-tools` (checked against HEAD 8c45dd7).
**Prerequisite:** none. The path with no angle factors must stay exactly as it is.

## 0. Before starting

### 0.1 Files

| File, as delivered | Put it at | Status |
|---|---|---|
| `angle_scale_plan_v1.md` | `plans/angle_scale_plan_v1.md` | this plan |
| the six S1 QuickNXS file pairs (`REF_M_47573…` to `REF_M_47589…`, `Off_Off` and `On_Off`) | `tests/data/private/angle_scale/` | private data; tests needing them skip if absent |
| `good_starting_point_1.json` (session with the 40 000-generation DREAM tab) | `tests/data/private/angle_scale/` | private; used for the regression in T8 |

### 0.2 Conventions

As in the earlier plans: module docstrings are the documentation, one commit per item with its tests, and Licorne-exact tolerances are never loosened. **Every existing test must pass unchanged**, and with every factor at 1 (or a dataset without angle information) the model must be bitwise identical to today's.

### 0.3 Order

1. D1–D2 (angle information reaches the fit).
2. M1–M2 (the factors in the model).
3. J1 (joint fits).
4. P1 (priors from the QuickNXS header).
5. F1 (fit and DREAM outputs).
6. U1–U3 (GUI).

### 0.4 Decision recorded

The factors are **ordinary fitted parameters** (like `stack.scale`): they appear in DE, multistart and DREAM, with bounds and Gaussian priors, and their posteriors come straight from the chains. Analytic marginalisation was considered and is **not** part of this plan (§9).

## 1. Evidence (analysis outside SANR, on the S1 data)

**How QuickNXS stitches** (`neutrons/quicknxs`, `models/data_manipulation.py`, `stitch_reflectivity`):

- Angle 1 is normalised so that the weighted mean of R over Q < `total_reflectivity_q_cutoff` is 1 (these files: 0.016 Å⁻¹).
- Each later angle is scaled to the previous one with Mantid `Stitch1D` over their overlap, using the selected cross-section only (`global_stitching = False` here).
- The factors multiply along the chain, and QuickNXS propagates their errors the same way.
- The same factor is applied to every cross-section of a run (`set_parameter` loops over them): **R⁺ and R⁻ of one angle always share a factor.**
- The `.dat` export keeps every run's points, scaled and sorted by Q (no merging), with θ per point in the 5th column. The header `[Data Runs]` lists each run's cumulative `scale` and `scale_err`.

**The θ column identifies the angles exactly.** In all six datasets the 5th column has exactly three values (one per run), equal in `Off_Off` and `On_Off`. The labels agree point for point with an independent reconstruction from the wavelength-bin Q template.

**Stitch uncertainties from the headers.** "Step" is the relative error added at that stitch, √(err_k² − err_{k−1}²) of the cumulative relative errors:

| Dataset | Runs | Angle 1 (plateau) | Step 1→2 | Step 2→3 | θ₁, θ₂, θ₃ (rad) |
|---|---|---|---|---|---|
| S1 4K 4.8T | 47573–75 | 0.38 % | 2.26 % | 4.55 % | 0.00770, 0.01906, 0.04564 |
| S1 4K 0.1T | 47577–79 | 0.40 % | 2.06 % | 5.28 % | 0.00800, 0.01917, 0.04583 |
| S1 4K 2T | 47580–82 | 0.39 % | 2.08 % | 4.86 % | 0.00793, 0.01914, 0.04578 |
| S1 10K 4.8T | 47583–85 | 0.30 % | 2.33 % | 4.40 % | 0.00692, 0.01821, 0.04467 |
| S1 20K 4.8T | 47586–88 | 0.30 % | 2.23 % | 4.91 % | 0.00695, 0.01813, 0.04477 |
| S1 250K 4.8T | 47589–91 | 0.28 % | 2.52 % | 3.89 % | 0.00653, 0.01759, 0.04422 |

**The plateau normalisation is sound** (weighted mean of the exported angle-1 points below 0.016 Å⁻¹: 0.9957–1.0026 in all six; the resolution-smeared model gives 0.9973 for 0.1T).

**The effect is large.** At the DREAM best state of `good_starting_point_1.json`, with the structure unchanged:

- Fixed round factors (T8 below) take the data χ² from 4157.132 to 3828.445 (N = 2678).
- The largest gain is 0.1T, the dataset that disagreed most with the others in a model-free comparison (812.997 → 639.448).
- Between datasets, the model-free per-angle differences (2–6 %) are consistent with two independent stitch errors of the sizes above.

## 2. Model — new docstring section `ANGLE (STITCH) FACTORS` in `model/stack.py`

A dataset measured at K angles (K ≥ 2, θ₁ < θ₂ < … < θ_K) has, at every point of angle k:

```
R_model = background + a_k · (R − background)        a_1 = 1,  a_k = s_2 · s_3 · … · s_k
```

where R is what the model gives today (scale, norms, resolution, Q shift, spread all included) and s_k = `Stack.stitch_k` is the factor joining angle k to angle k−1.

- **Chained, as QuickNXS does.** s_k's prior is its own step error, and the correlation between angles 2 and 3 that the reduction introduced is kept.
- **Angle 1 is the reference.** Its normalisation is `stack.scale`; there is no s_1.
- **The factor multiplies the signal, not the background.** Open question Q1 in §8.
- **Shared by all channels of the dataset.** A factor applies to every point with that θ in every channel.
- **s_2, s_3, s_4** exist (`STITCH_MAX = 4`); default 1.0, fixed, bounds (0.8, 1.25).

## 3. Data — D items

### D1. θ reaches the fit and the saved problems

- `fit/panel.py` (`_fit_inputs`, around line 1162): add `theta=d.get('theta')` to each data entry.
  The Experimental tab already parses the 5th column (`tabs/experimental.py`, `parse_data`); today it is dropped here.
- `fit/saved.py`: `problem_state` writes `theta`; `problem_from_state` reads it (`_array`, absent → None).
  Saved problems and DREAM files written before this load with θ = None (no angles).

### D2. Angle labels in `FitProblem`

In `FitProblem.__init__` (`fit/problem.py`), after the `ok` mask:

- Angles are the distinct θ values over all channels of the dataset, sorted ascending.
  Values are matched with a relative tolerance of 1e-6; a θ that matches nothing in another channel is still an angle.
- Store `self.angle_theta` (K,) and, per channel, `self.angle_of[k]`: the angle index (0-based) of each fitted point, aligned with `channels[k][2]`.
- With any channel without θ, or K = 1, set `self.angle_theta = None`.
- Overlap and repeated Q: labels go with the point, not with `self.Q`. Two points at the same Q from different angles keep different labels.

## 4. Model — M items

### M1. Stack parameters

In `model/stack.py`:

- `STACK_FIT_PARAMS` gains `'stitch_2', 'stitch_3', 'stitch_4'`; `DEFAULT_BOUNDS` gains them at (0.8, 1.25).
- `Stack.__init__`: `self.stitch_2 = self.stitch_3 = self.stitch_4 = 1.0`.
- `to_dict` and `from_dict`: written always; absent → 1.0.
- `LICORNE_EXACT` unchanged: Licorne has no such parameter, and at 1.0 nothing changes.
- `prior_terms` already finds `Stack.fit[attr]['prior']` through `stack.owner(None)`, so priors need no new code.

### M2. Applying the factors

In `FitProblem.model` (`fit/problem.py`), after `out` is filled and before it is returned:

```python
a = self.angle_factors()            # None, or (K,) with a[0] = 1
if a is not None and np.any(a != 1.0):
    bg = self.stack.background
    for k in range(len(out)):
        out[k] = bg + a[self.angle_of[k]] * (out[k] - bg)
```

- `angle_factors()` is the cumulative product of [1, stitch_2, …, stitch_K].
- **The `np.any(a != 1.0)` guard keeps the all-ones path bitwise identical**, since bg + 1·(m − bg) is not always m in floating point.
- A varied `stitch_k` with k > K is an error at construction: "stitch_3 is fitted but S1 … has 2 angles".
  So is a varied `stitch_k` on a dataset with no θ.
- Combined targets (`sum`, `asym`) need nothing more: they combine the channels' model after this step.
- Simulation (no data) is untouched: the factors only exist where there are data points with θ.

## 5. Joint fits — J1

In `fit/joint.py`:

- `stack.stitch_k` defaults to `'own'` (a factor of one dataset's stitching has no meaning in another).
  Put this in a module constant `OWN_BY_DEFAULT = ('stack.stitch_2', 'stack.stitch_3', 'stack.stitch_4')`, applied when `modes` does not name the parameter.
- `'shared'` stays allowed but emits a warning.
- A dataset with fewer angles than a varied `stitch_k` gets it as `'fixed'` at 1.0 for that dataset, with a note in the fit report. (The model check in M2 then does not fire for that part.)
- Names follow the existing rule: `stack.stitch_2 [S1 4K 0.1T]`.

## 6. Priors from the QuickNXS header — P1

New function `quicknxs_stitch_priors(path)` in `tabs/experimental.py` (next to `parse_data`):

- Reads the `[Data Runs]` table: columns by header name, values tokenised so `[a, b]` lists stay one token.
- Returns, sorted by the run's θ (from the data's 5th column):
  - `plateau`: relative error of angle 1 (`scale_err/scale`);
  - `steps`: [√(e₂² − e₁²), √(e₃² − e₂²), …] of the cumulative relative errors (0 if rounding makes the difference negative).
- When a dataset is imported with a QuickNXS header and θ, set for each k ≤ K:
  - `fit['stitch_k'] = {'vary': False, 'min': 0.8, 'max': 1.25, 'prior': {'kind': 'gauss', 'mu': 1.0, 'tau': steps[k-2]}}`;
  - `fit['scale']['prior'] = {'kind': 'gauss', 'mu': 1.0, 'tau': plateau}`, unless a prior is already set.

  Ticking Fit is left to the user.
- The `[Global Options]` values `normalize_to_unity`, `total_reflectivity_q_cutoff`, `global_stitching` and `polynomial_stitching` are shown in the dataset's info tooltip.

## 7. Outputs and GUI — F and U items

### F1. Fit and DREAM outputs

- The factors appear in the fit report and DREAM summary like any parameter. The summary adds, per dataset, the cumulative a_2, a_3 (median and 95 % interval, from the chains).
- `posterior_warnings` already reports prior tension and bound hugging; nothing new is needed.
- `fit/saved.py` and `dream_fit.save_result` carry θ (D1), so a saved run rebuilds with its angles.

### U1. Where the parameters are edited

The stack-level fit rows (where scale, background and q_shift are ticked) gain `stitch 2`, `stitch 3`, `stitch 4`. Only the rows up to the current dataset's K are shown; all are hidden when it has no θ.

### U2. Datasets panel

The joint datasets panel shows the `own` default of J1 for the stitch rows.

### U3. Residuals per angle

The fit result plot gets an option "colour residuals by angle". The experimental plot gets an option "show data corrected by the fitted factors", dividing angle-k points' (R − bg) by a_k. This is a display aid only; the fit always uses the raw data.

## 8. Open questions for JF

- **Q1. Signal or signal + background?** QuickNXS's factor multiplies everything left in the reduced R of a run, residual background included. This plan multiplies only (R − bg), because the fitted background is one number per dataset and matters only in angle 3. If you prefer QuickNXS's convention exactly, the line becomes `out[k] = a[...] * out[k]` (and the background then reads as angle 1's).
- **Q2. Bounds** (0.8, 1.25) are far outside every prior (largest step error 5.3 %), so they never bind; tighter?
- **Q3. `STITCH_MAX = 4`** enough? REF_M sets here have 3 angles.
- **Q4. Automatic priors on import (P1)**, or only through a button "Priors from the QuickNXS header"?

## 9. Not in this plan

- **Analytic marginalisation** of the factors in the likelihood: decided against for now (§0.4). It would keep DREAM's dimension down (12 extra parameters for the six S1 datasets), and could come later as an option behind the same model, M2 unchanged.
- **Per-angle Q offsets** (an angle calibration error, ΔQ/Q = Δθ/θ per angle): the S1 test at fixed structure found Δθ ≈ −1×10⁻⁴ rad in all six datasets but no χ² gain over a constant `q_shift`. Revisit after the structure is refitted with these factors.

## 10. Tests

| ID | Test | Expect |
|---|---|---|
| T1 | Whole suite unchanged; and every existing fixture with θ added but no factor varied | identical results, bitwise for the model |
| T2 | Synthetic 3-angle dataset, s_2 = 1.03 only | model points of angle 2 and 3 equal bg + 1.03·(R₀ − bg), angle 1 identical to R₀, in every channel |
| T3 | Chain: s_2 = 1.02, s_3 = 0.97 | angle 3 factor 0.9894 (product), angle 2 1.02 |
| T4 | `sum` and `asym` targets with factors | equal to combining the factored channels by hand |
| T5 | Errors | varied `stitch_3` on a 2-angle dataset, and on a dataset without θ, raise with the messages of M2 |
| T6 | Joint | names `stack.stitch_2 [name]`; default mode `own`; `shared` warns; a 2-angle part gets `stitch_3` fixed at 1 |
| T7 | Saved state | `problem_state` → `problem_from_state` round trip keeps θ and labels; an old state without θ loads with no angles |
| T8 | **Private regression** (`good_starting_point_1.json`, DREAM best state, structure as stored) with the factors below | data χ² per dataset as listed (±0.01), total 3828.445 (from 4157.132 at all ones) |
| T9 | **Private header parse**, all six files | `scale`, `scale_err`, step errors and θ as in the §1 table (step errors to 0.01 %) |
| T10 | **Synthetic recovery**: S1-like stack, the six real Q, dQ, θ grids and error bars, true factors (1.03, 0.95), noise drawn from dR, priors as P1 | DE recovers each factor within 2σ of its DREAM posterior; report the factor–roughness, factor–background and factor–spread correlations |
| T11 | Bounds and priors | a factor's prior enters `prior_penalty` and DREAM's log prior exactly as for `stack.scale` |

T8 factors and expected data χ² (structure at the stored best state, factors applied as M2):

| Dataset | s_2 | s_3 | χ² at ones | χ² with factors |
|---|---|---|---|---|
| S1 4K 0.1T | 0.95 | 1.01 | 812.997 | 639.448 |
| S1 4K 2T | 1.00 | 1.01 | 523.081 | 522.235 |
| S1 4K 4.8T | 0.99 | 0.99 | 585.036 | 572.005 |
| S1 10K 4.8T | 1.01 | 1.01 | 722.558 | 703.638 |
| S1 20K 4.8T | 1.02 | 0.94 | 593.245 | 555.353 |
| S1 250K 4.8T | 1.02 | 1.05 | 920.215 | 835.767 |

These were computed outside SANR from the same model curves (angle labels from the 5th column), so they check M2 end to end.

## 11. As built (branch `angle-stitch-factors`, 2026-11-03)

Answers: Q1 signal only (as planned); Q2, Q3 as planned (0.8–1.25, `STITCH_MAX = 4`); Q4 automatic.

Deviations:

- **D2 tolerance.** θ is *not* always equal in `Off_Off` and `On_Off` of the delivered files (e.g. S1 4K 2T angle 2: 0.01913799 and 0.01914061, 1.4e-4 relative). Angles are grouped within `SAME_ANGLE` = 5 % (stitched angles are a factor apart), not 1e-6, so one angle's spin states stay one angle. More than `MAX_ANGLES` = 12 distinct values is an angle scan: no angles. `angle_theta` is each angle's largest θ.
- **D2 counting.** The angles are counted from every channel's points (a Q range that leaves out angle 3 still has 3 angles); labels are given to the fitted points.
- **P1 timing.** The header priors are applied when a dataset's fit is built (`FitPanel._header_priors`, on that dataset's copy of the stack), not written into the model at import: datasets usually share one model, so import-time priors would be overwritten by the last dataset imported, and a joint fit's `own` factors would all get the same prior. A prior set (or removed) by hand wins; the fit report says which priors came from the header.
- **P1 run order.** The `[Data Runs]` rows are taken in their order (the stitching order, angle 1 first), not sorted by θ: the header has no per-run θ.
- **P1 tooltip.** Each channel's tool tip in the Experimental tab gives the header's plateau and step errors and the four `[Global Options]`.
- **J1 `fixed`.** A joint dataset with fewer angles gets the factor unticked at 1 in its own copy of the stack (`FitProblem(stitch_extra='fix')`), with a note; it then simply does not have the parameter.
- **U1.** The stitch rows are shown up to the largest K of the imported measured datasets (or when the model fits one or has it away from 1); the Fit tab lists a stitch factor once the model has an entry for it.
- **Not done:** U3 (residuals coloured by angle, data corrected by the factors) and T10 (synthetic DE/DREAM recovery).
