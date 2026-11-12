# SANR — show χ²/N (data only) next to the cost in every fit — implementation plan v1

Target: `github.com/JFMilette/SANR`, branch `main` (read at commit `040ea83`, SANR 0.6). Check against `session-zip` (`30eb631`) before starting.
**Purpose:** every optimizer (DE, L-BFGS-B, least squares, basin hopping, dual annealing, SHGO, DIRECT, multistart) shows the **data-only χ²/N** in its own column next to the cost it minimizes, so fits can be compared with each other and with DREAM. **Nothing changes in what is optimized:** the priors stay in the cost.

## 0. Before starting

| File, as delivered | Put it at | Status |
|---|---|---|
| `chi2_column_plan_v1.md` | `plans/chi2_column_plan_v1.md` | this plan |

Conventions as in earlier plans: module docstrings are the documentation, one commit per item with its tests, every existing test passes unchanged, and old sessions still open.

## 1. The problem

The optimizers minimise and display the **cost** of `fit.problem` COST: **χ²/N + prior penalty/N** (labelled "χ² + prior" when there are priors). DREAM shows **χ²/N without priors** (`PosteriorProblem.chi2_per_point`). The two cannot be compared directly, and nothing in the fit windows shows the data-only value.

| Where | Shows today | Data-only χ²/N? |
|---|---|---|
| Fit window history and status (all optimizers) | cost, labelled "χ² + prior" (it is per point) | no |
| Multistart: fit tab minima table | cost | no |
| Multistart: report line on the page (`_ms_done`) | labelled **"χ²/N"**, but the value includes the priors | no (mislabelled) |
| Fit window, residuals view | χ²/N of each channel | per channel only |
| DREAM | χ²/N without priors | yes |

**Measured** on JF's S1 joint fit (L-BFGS-B, 62 parameters, 2686 points, 18 Gaussian priors on the scale and stitch factors): cost 1.350679 = **χ²/N 1.342765** + priors 0.007914. The priors move the third decimal, which is the size of the differences used for model comparison (about 0.004 per parameter with BIC).

## 2. Key idea: no extra model evaluation

For the `chi2` cost, `FitProblem.__call__` returns `mean(residuals²) + prior_penalty(x) / npoints` (`JointProblem` is the same). So:

> **χ²/N = cost − prior_penalty(x) / npoints**

`prior_penalty` only evaluates the priors: **≈ 3×10⁻⁵ s**, against **0.18 s** for one model evaluation of the S1 joint fit. Checked on the S1 result: the formula and a direct `mean(residuals(x)**2)` agree to all printed digits (1.342765).

Add to `fit/problem.py` (also used by `JointProblem` through its shared methods):

```python
def chi2_per_point(self, x, cost=None):
    """chi^2/N of physical x without the priors' penalty: from `cost`
    (this problem's cost at x) when given, without evaluating the model;
    None for the 'log' cost."""
    if self.cost != 'chi2':
        return None
    if cost is None:
        return float(np.mean(self.residuals(np.asarray(x, float))**2))
    return float(cost) - self.prior_penalty(np.asarray(x, float)) / self.npoints
```

`PosteriorProblem.chi2_per_point` (DREAM) can then call it, so both report the same number by construction.

## 3. Items

| Item | Where | Change |
|---|---|---|
| **C1** | `fit/problem.py` | `chi2_per_point(x, cost=None)` as above, plus a test that both paths agree. |
| **C2** | `FitWindow` (`fit/de/de_window.py`): `start`, `_append`, `show_saved`, `_fill_minima` | **Third column "χ²/N"** in the history: `chi2_per_point(x, cost)` for each entry. The cost column header becomes **"cost (χ²/N + prior/N)"** when there are priors, else just "χ²/N"; then the third column is hidden, since it would be identical. For the `log` cost: the column shows "—", with a tooltip saying χ²/N is not defined for the log R cost. Right-aligned, 6 significant digits, the best entry in bold as now. Entries stay `(label, cost, x)`: the χ²/N is derived, not stored. |
| **C3** | `FitWindow` status lines (progress, end of fit, polish note, `show_minima`) | Show both, e.g. *"best cost 1.35068 (χ²/N 1.34276 + priors 0.00791)"*; at the end also **χ²/ν** = χ²·N/(N − k). |
| **C4** | Multistart (`fit/panel.py` `_ms_done`, minima rows) | Fix the label: report *"best χ²/N 1.3428 (cost 1.3507)"* using `chi2_per_point(m['x'], m['chi2'])`. Rename the row key `'chi2'` to `'cost'` in `fit/multistart.py`, keeping `'chi2'` as a read alias for old saved tabs. Merging minima (`MERGE_COST`) stays on the cost. |
| **C5** | Residuals view (`_draw_residuals`) | Besides each channel's χ²/N in the legend: a line under the plot with the **dataset's** χ²/N and, in a joint fit, the **total** χ²/N; note "(error floor f % included)" when a floor is set. |
| **C6** | Old sessions | Saved fit tabs restore through `show_saved`, which now computes the column from the restored problem: nothing to migrate. Check with a session saved before this change. |

**Not changed:** the optimizers' cost function, the priors, the stop criteria, DREAM's likelihood.

## 4. Tests

- **T1** (`tests/test_fit_problem*.py`): `chi2_per_point(x, cost)` equals `mean(residuals(x)**2)` to 1e-12, with and without priors, for a `FitProblem` and a `JointProblem`; `None` for the `log` cost.
- **T2** (`tests/test_fit_panel.py`, offscreen): after an L-BFGS-B fit with priors, the history has three columns; for every row, the third equals the cost minus the prior term; the headers read as in C2; without priors the third column is hidden.
- **T3:** the multistart report and minima use the corrected labels; an old saved tab with `'chi2'` rows still opens.
- **T4:** `show_saved` on a session saved before the change shows the χ²/N column.
- **T5:** DREAM's χ²/N (MAP, median) equals the optimizer's χ²/N at the same parameters.

## 5. Open questions for JF

1. Also show **χ²/ν** as a column, or only in the status line? It is χ²/N × N/(N − k), about 1.024 × for the S1 joint fit.
2. Sort the multistart minima by **cost** (what was minimized, as now) or by χ²/N? Keeping the cost is consistent with the optimizer; the χ²/N column makes the difference visible either way.
