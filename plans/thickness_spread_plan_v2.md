# SANR — thickness spread v2: whole stack or one layer, log-normal, node convergence — implementation plan

Target: `github.com/JFMilette/SANR`, branch `licorne-exact`.
**Replaces** the earlier v2 draft (independent layer groups). Groups are deferred; §9 records why.
**Prerequisite:** `thickness_spread_plan_v1.md` implemented and its tests passing. This plan builds on v1's slab-sequence plumbing (T2), node-sequence cache (T3), averaged `reflectivities` (T4) and frozen node counts in fits (F2).

## 0. Before starting

### 0.1 Files

| File, as delivered | Put it at | Status |
|---|---|---|
| `thickness_spread_plan_v2.md` | `plans/thickness_spread_plan_v2.md` | this plan; delete the earlier v2 draft if it was copied |

The v1 private test data (`tests/data/private/thickness_spread/`) is reused. Tests needing it skip if it is absent.

### 0.2 Conventions

As in v1: module docstrings are the documentation, one commit per item with its tests, and Licorne-exact tolerances are never loosened. With the target on the whole stack and the Gaussian distribution, every v1 test must pass unchanged.

### 0.3 Order

1. S1–S4 (target).
2. N1–N3 (distributions).
3. A1 (node convergence; amends v1 T5).
4. F1–F2 (fitting).
5. U1–U3 (GUI).

There is still **one spread parameter and one averaging dimension**, so the cost is the same as v1: the node count times one evaluation.

## 1. Evidence (prototype, outside SANR)

Data, stack and settings as in v1 §1 (S1 4 K 4.8 T, σ_Q = the data dQ, full structure and magnetisation refitted in every row).

| Spread applied to | Distribution | Fitted σ | χ² | ΔBIC vs whole stack | Grid points | Notes |
|---|---|---|---|---|---|---|
| Whole stack (v1) | Gaussian | 7.2 % | 498.0 | 0 | 7 | |
| BAT only (≈ 99 Å) | Gaussian | 13.1 % | 496.8 | −1 | 7 | Structure and moments as for the whole stack |
| CrSb only (≈ 41–50 Å) | Gaussian | 28 % | (485.1) | (−13) | 7 | **Invalid:** the lowest node has a negative thickness (f = −0.05) |
| CrSb only | Log-normal | 28 % | 489.0 | −9 | 11 | CrSb thickness hits its 50 Å bound; 1 QL BAT MSLD drops from 0.31 to 0.10 |
| Two independent groups (cap + CrSb, BAT + buffer) | Gaussian | 9.5 %, 10.0 % | 494.9 | +3 | 35 | Deferred, §9 |

What this shows for the design:

- **A single-layer spread is a real alternative, at no extra cost.** The data are about as well described by a spread in the BAT alone as in the whole stack. A spread in the CrSb alone is somewhat better by BIC, but only with the CrSb thickness on its bound and a changed magnetic profile. The location of the spread is weakly determined by these data. The feature is still worth having, because it lets this question be tested, but results must be compared by BIC and checked for bounds and with DREAM.
- **Gaussian spreads on one thin layer can go negative.** A single layer can need a large relative spread (28 % here); a Gaussian then puts nodes at negative thicknesses. N3 forbids that and points to the log-normal, which is always positive.
- **Log-normal and Gaussian are indistinguishable at 7 %** (from the earlier test: at most 0.24 data σ apart at σ = 7.2 %, 1.6 data σ at 15 %).

Log-normal node convergence on the whole stack (vs a 241-point brute force): σ = 0.072 needs 11 nodes for 4e-5 relative; σ = 0.15 needs more than 19 (1.7e-2 at 19). v1's φ rule gives too few for the log-normal, hence A1.

## 2. Model — extend the THICKNESS SPREAD docstring section

`Stack.thickness_spread_target` selects which thicknesses vary:

- **'stack'** (the default, v1): every interior layer is scaled by the same factor f.
- **a layer index i** (an interior layer): only layer i is scaled by f. Every other layer keeps its thickness, so the layers below the target shift in depth by the change in its thickness.

```
R(Q) = Σ_k w_k · R(Q; d_j → f_k d_j for the target layers)
```

f has mean 1 and standard deviation σ_t = `thickness_spread`, and its distribution is set by `thickness_spread_dist` (§4). Everything else is as in v1 §2: only thicknesses scale; the average is linear and taken on the ideal reflectivity, so it commutes with the resolution; `critical_edges()` and `profile()` are unchanged.

A single-layer spread is a lateral variation of that layer's thickness, for example island growth or a flux gradient of one source. Add to the docstring that it can trade off against the roughness of that layer's interfaces and against the thicknesses of its neighbours, so fits should be checked with DREAM.

## 3. Phase 1 — target (`model/stack.py`)

### S1. Attribute and serialisation

Add `Stack.thickness_spread_target = 'stack'`. Store `'stack'` or an int in `to_dict`; `from_dict` defaults to `'stack'`, so v1 sessions load unchanged.

### S2. Validation

A layer target must be an int with 1 ≤ i ≤ len(layers) − 2 (not the fronting or the backing). Raise `ValueError` otherwise, in a setter `set_spread_target(target)` that every caller uses.

### S3. Keep the target when layers change

Add `reindex_spread_target(old_layers)`, with the same identity logic as `reindex_ties` (line ~900): follow the target layer by `id()` to its new index. Call it right after every `reindex_ties` call (at HEAD `tabs/simulation.py` ~1765, ~1775, ~1795, ~1808).

If the target layer was deleted, set the target to `'stack'`, set `thickness_spread = 0` and its `fit['vary'] = False`, and show a status-bar message. This avoids silently moving a fitted spread onto the whole stack.

### S4. Which layers v1's code scales

In v1 T3, replace "every layer in `layers[1:-1]`" by `self._spread_layers()`: all interior indices for `'stack'`, or `[i]` for a layer target. Use the same list for D in the node count (A1 and v1 T5) and for the cache key.

## 4. Phase 2 — distributions (`model/stack.py`)

### N1. Attribute

Add `Stack.thickness_spread_dist` ∈ {'gaussian', 'lognormal'}, default 'gaussian', serialised with that default for older files.

### N2. One node function

Add `spread_factors(sigma, dist, n)` → `(f, w)`. x_k and h_k are `numpy.polynomial.hermite.hermgauss(n)`, w_k = h_k / √π.

- 'gaussian': f_k = 1 + √2 σ x_k (v1).
- 'lognormal': s² = ln(1 + σ²), μ = −s²/2, f_k = exp(μ + √2 s x_k). Then E[f] = 1 and SD[f] = σ, so σ means the same for both distributions and the fitted thickness stays the mean thickness.

Write the parameterisation into the docstring. Papers often use the mode instead (Morelhão et al., arXiv:2109.01914, N₀ and b = N₀ e^{σ²}); U3 reports the median and mode.

### N3. Positivity of the Gaussian

Thickness can't be negative. For 'gaussian':

- The probability below f = 0.05 is p = Φ(−0.95/σ).
- If p > 1e-6 (σ ≳ 0.19), raise `ValueError`: "Gaussian spread too large for positive thicknesses; use the log-normal".
- Otherwise drop any node with f_k ≤ 0.05 (high node counts can put a negligible-weight node there) and renormalise the weights.

Set the default Gaussian fit bound to 0.15, and the log-normal bound to 0.5. When the distribution is switched in the GUI, clip the bounds accordingly.

## 5. A1 — amend v1 T5: node count by a convergence check

Replace v1's 'auto' rule:

1. Start from n₀ = the smallest odd integer ≥ 1.5 φ + 2, with φ = Q_max · D · σ, where D is the thickness of the spread layers (S4). Take at least 5.
2. Compute R with n and with n + 4 nodes.
3. If max |R_n / R_{n+4} − 1| < `SPREAD_TOL` (default 2e-3), keep n. Otherwise n ← n + 4 and repeat, up to `SPREAD_NMAX` (default 41); at the limit, warn and keep it.

Cache the result per (σ, dist, target, layer thicknesses rounded to 0.1 Å). In a fit it runs once at the start (F2). An int setting skips the check.

## 6. Phase 3 — fitting

### F1. Parameter

Still one parameter, `stack.thickness_spread`, whatever the target. Joint modes, priors, DE, multistart and DREAM need no change. The target and distribution are model choices, not fit parameters: compare them by running separate fits.

### F2. Frozen nodes

v1 F2 unchanged, except that the node count now comes from A1 at the **upper bound** of σ for the chosen distribution and target, on the largest data Q plus 3 σ_Q.

## 7. Phase 4 — GUI (`tabs/simulation.py`)

### U1. Target

A combo box on v1's thickness-spread row: "Whole stack", then each interior layer by name (with `#i` when names repeat, as in `parameter_names`). Rebuild its items whenever layers are added, removed, renamed or moved, keeping the selection through S3. Tooltip: "Which thicknesses vary laterally: every layer together, or this layer alone".

### U2. Distribution

A combo box: Gaussian, log-normal. Show N3's error as a red note on the row instead of a dialog, so typing a value doesn't interrupt the user.

### U3. Reporting

In fit results, report σ_t with its target and distribution. For a single layer, also report the RMS thickness variation in Å, σ_t · d_i. For the log-normal, also report the median (e^μ) and mode (e^{μ − s²}) of f and the corresponding thicknesses.

## 8. Tests — extend `tests/test_thickness_spread.py`

1. **Defaults:** target 'stack' with the Gaussian is bit-identical to v1 in every v1 test configuration.
2. **Layer target matches an explicit average:** with target i, σ = 0.1 and n = 9, the built-in result equals Σ w_k `Stack.from_dict(layer i scaled by f_k).reflectivities(...)` to rtol 1e-12, for both roughness schemes.
3. **Only the target changes:** with target i, every node's slab list differs from the nominal one only in the slabs that the scaled layer i and its interface windows produce; slabs above that region are identical.
4. **One-layer stack:** for a stack with a single interior layer, target 'stack' and target 1 give identical results.
5. **Log-normal moments:** Σ w_k f_k = 1 and Σ w_k (f_k − 1)² = σ², both to 1e-10, for σ in {0.02, 0.07, 0.3} and n = 41.
6. **Log-normal → Gaussian:** at σ = 0.005 they agree to 1e-6 relative.
7. **Gaussian positivity:** σ = 0.25 raises `ValueError`; σ = 0.15 with n = 41 drops only nodes of negligible weight and still gives Σ w_k = 1.
8. **Convergence check** (`slow`): A1's count is within 2e-3 relative of a 241-point brute force, for the whole stack and for a 40 Å single layer, for σ in {0.05, 0.15} Gaussian and {0.05, 0.3} log-normal.
9. **Reindexing:** inserting a layer above the target follows it; moving it follows it; deleting it resets to 'stack' with σ = 0 and not varied. Round trip through `to_dict`/`from_dict`.
10. **Validation:** target 0, target = last index, and a non-int target raise `ValueError`.
11. **Frozen nodes:** a fit with σ free and a layer target stores an int node count on its copy only, and the cost on a fine σ grid has no jumps.
12. **Recovery:** simulate a film and substrate plus a 40 Å top layer with a log-normal spread of 0.25 on the top layer only. A fit with that target recovers σ = 0.25 ± 0.04. The same data fitted with target 'stack' give a clearly worse χ².

## 9. Deferred: independent groups

Several independent spreads (one per group of layers) were tested on the same data: two groups (cap + CrSb, BAT + buffer) gave σ = 9.5 % and 10.0 %, χ² 494.9, **ΔBIC +3** against the single whole-stack spread, at 5× the evaluations (35 vs 7 nodes) and about 3× the fit time. Each independent group multiplies the cost by its node count, and the spreads are strongly correlated with each other and with roughness. Revisit only if a sample has independent evidence (thickness maps of one material, layers grown separately) or residuals a single spread can't remove. The design (tensor-product grid, owner objects like ties, transfer-matrix reuse) is in the earlier v2 draft.
