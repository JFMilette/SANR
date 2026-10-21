# SANR — lateral thickness spread (incoherent average) — implementation plan v1

Target: `github.com/JFMilette/SANR`, branch `licorne-exact`, written against HEAD `1e31137` (13 Oct 2026).
If HEAD has moved, check the function names and line numbers quoted in §3–§5 against the current code before starting.

## 0. Before starting

### 0.1 Files

| File, as delivered | Put it at | Status |
|---|---|---|
| `thickness_spread_plan_v1.md` | `plans/thickness_spread_plan_v1.md` | this plan |
| `S1_4K_4p8T_thickness_spread.json` | `tests/data/private/thickness_spread/` | SANR session of the prototype fit (§1). Carries the new keys `thickness_spread` and `thickness_spread_nodes`, which HEAD ignores on load |
| `thickness_spread_reference_4p8T.json` | `tests/data/private/thickness_spread/` | reference curves for the acceptance test (test 10): Q, dQ, R⁺, R⁻ of the prototype and its χ² |
| `REF_M_47573_47574_47575_peak1_Specular_Off_Off.dat`, `..._On_Off.dat` | `tests/data/private/thickness_spread/` | JF copies them; S1 4 K 4.8 T |

Tests that need the private folder skip if it is absent, like `test_licorne_end_to_end.py`.

### 0.2 Running the tests

```
QT_QPA_PLATFORM=offscreen python -m pytest -q tests
QT_QPA_PLATFORM=offscreen python -m pytest -q -m slow tests
```

### 0.3 Conventions

The repo's existing conventions apply:

- Module docstrings are the documentation. The new section THICKNESS SPREAD (§2) goes in `model/stack.py`'s docstring, with the references.
- Commit once per item, with that item's tests.
- Never loosen a Licorne-exact reference tolerance to make a test pass. With `thickness_spread = 0` every existing test must pass unchanged.

### 0.4 Order

1. Phase 1 (model): T1 → T6.
2. Phase 2 (fit): F1 → F3.
3. Phase 3 (GUI): G1 → G3.
4. Phase 4 is optional and not part of v1.

## 1. Evidence (prototype, outside SANR)

The prototype computed the incoherent average with `Stack.from_dict` copies at scaled thicknesses. Dataset: S1 4 K 4.8 T (CrSb/(Bi,Sb)₂Te₃, 225 points per channel, half-polarised, P = ±0.98), with resolution σ_Q = the data file's dQ column. The model has an AlOx cap of two slabs, a CrSb top slab of 4 Å with its own MSLD, CrSb, L4, 1 QL BAT, BAT, a buffer and the substrate. Full structure and magnetisation were refitted in every row.

| Model | χ² (450 points) | ΔBIC | Spin-averaged ⟨r²⟩ | Difference ⟨r²⟩, Q < 0.07 |
|---|---|---|---|---|
| Uniform film | 646.4 | 0 | 1.68 | 1.14 |
| Uniform film, roughness bounds × 2.5 | 602.4 | −44 | 1.47 | 1.13 |
| **Thickness spread, σ_t = 7.23 ± 0.24 %** | **498.0** | **−142** | **1.23** | **0.90** |
| Thickness spread + roughness bounds × 2.5 | 492.2 | −148 | 1.24 | 0.87 |
| Thickness spread, top-slab MSLD fixed at 0 | 540.5 | −99 | 1.35 | 0.99 |

**Why this is not resolution.** Fitting per-angle resolution widths instead wanted about 7 % extra broadening at angles 2 and 3 and none at angle 1. A thickness spread smears the fringes, which depend on thickness, but not the critical edge, which depends only on the substrate. Resolution broadening can't separate the two. The uniform-film fits also pushed three roughnesses to their bounds and needed a 120 Å graded buffer; with the spread, the buffer contrast vanishes (5.81 vs substrate 5.94, σ ≈ 3 Å).

**What changes in the sample.** BAT NSLD becomes 1.80 ± 0.06, consistent with bulk Sb-rich BST (it was 1.69 ± 0.03). AlOx main becomes 3.65 ± 0.25 over 23 Å. The 1 QL BAT MSLD drops from 0.92 to 0.31 ± 0.08. The CrSb top slab keeps 1.42 ± 0.19.

**Node convergence** (Gauss–Hermite against a 161- or 241-point brute-force average over ±6σ_t; this stack, Q ≤ 0.20 Å⁻¹, D = 338 Å of interior layers):

| σ_t | φ = Q_max·D·σ_t | 3 nodes | 5 nodes | 7 nodes | 9 nodes | 11 nodes | 15 nodes |
|---|---|---|---|---|---|---|---|
| 0.072 | 4.9 rad | 1.8e-1 | 2.0e-2 | **7.9e-4** | 2.8e-4 | | |
| 0.10 | 6.8 rad | | | 3.1e-2 | | **1.5e-3** | 2.5e-4 |
| 0.15 | 10.1 rad | | | 3.8e-1 | | 1.0e-1 | **1.6e-3** |

Values are the maximum relative difference in R. In units of the data's error bars, 7 nodes at 7.2 % give 2.5e-3 σ; 5 nodes give 6e-2 σ. A fixed 5 nodes is not enough; the node count must scale with φ (T5).

**Cost.** One evaluation of this stack on 225 Q is about 0.02 s; the spread multiplies it by the node count.

**Joint check (pending).** σ_t is a property of the sample, so a joint fit over 0.1, 2 and 4.8 T must give one σ_t. Record the result here when it is run: shared σ_t, per-field σ_t, and the uniform-film reference, with the same settings.

## 2. Model — the THICKNESS SPREAD section of `model/stack.py`'s docstring

Write this section (adapting the wording) into the module docstring.

THICKNESS SPREAD — `Stack.thickness_spread` = σ_t ≥ 0, a fraction (0.07 = 7 %). The film's thickness varies across the illuminated area over lateral distances much larger than the neutron's transverse coherence length. Each region then reflects independently, and the measured specular reflectivity is the area-weighted incoherent sum of the regions' reflectivities, not the reflectivity of the averaged profile (Majkrzak, O'Donovan & Berk 2006, §1.4.4; Majkrzak et al. 2014, 2022).

The model: every interior layer (all layers except fronting and backing) is scaled by the same factor f, with f Gaussian of mean 1 and standard deviation σ_t. This is a growth-rate variation correlated across the layers. Roughness, SLDs, magnetisation and angles are unchanged; only thicknesses scale.

```
R_spread(Q) = Σ_k w_k R(Q; d_j → f_k d_j),   f_k = 1 + √2 σ_t x_k,   w_k = h_k / √π
```

where x_k, h_k are the n-point Gauss–Hermite nodes and weights (`numpy.polynomial.hermite.hermgauss(n)`; Abramowitz & Stegun §25.4.46). The w_k sum to 1.

Properties the code relies on:

- The average is linear, so it commutes with the resolution average (both schemes) and with norm, scale and background. It is taken on the ideal reflectivity, inside the callable passed to `_resolve`, so the resolution is applied once.
- `critical_edges()` depends only on the fronting and backing, so the quadrature grid is the same for every node.
- σ_t = 0 is the current code path, untouched.
- `profile()` and the plotted slabs stay the nominal (f = 1) ones.
- Every f_k must be positive: raise `ValueError` if 1 − √2 σ_t x_max ≤ 0.05.

References for the docstring:

- C. F. Majkrzak, K. V. O'Donovan, N. F. Berk, "Polarized neutron reflectometry", in *Neutron Scattering from Magnetic Materials* (Elsevier, 2006), §1.4.4.
- C. F. Majkrzak et al., "Determination of the effective transverse coherence of the neutron wave packet as employed in reflectivity investigations of condensed-matter structures. I. Measurements", *Phys. Rev. A* 89 (2014). Check the article number before citing.
- C. F. Majkrzak et al., "The effect of transverse wavefront width on specular neutron reflection", *J. Appl. Cryst.* 55 (2022).
- S. L. Morelhão, C. I. Fornari, P. H. O. Rappl, E. Abramof, *J. Appl. Cryst.* 50, 399–410 (2017), doi:10.1107/S1600576717000760: thickness distributions in Bi₂Te₃ films.
- A. R. J. Nelson, S. W. Prescott, *J. Appl. Cryst.* 52, 193–200 (2019), doi:10.1107/S1600576718017296: refnx `MixedReflectModel`, incoherent sums.

## 3. Phase 1 — `model/stack.py`

### T1. Attributes, serialisation, fit registration

- In `Stack.__init__` (next to `self.scale`, line ~664): `self.thickness_spread = 0.0` and `self.thickness_spread_nodes = 'auto'` ('auto' or an odd int ≥ 3).
- `to_dict`: add both keys. `from_dict`: `float(d.get('thickness_spread', 0.0))` and `d.get('thickness_spread_nodes', 'auto')`, so older files load with no spread.
- `STACK_FIT_PARAMS = ('scale', 'background', 'thickness_spread')` (line 251).
- `DEFAULT_BOUNDS['thickness_spread'] = (0.0, 0.2)`.

### T2. Plumb an optional slab sequence through the transfer matrix

Add a keyword `seq=None` to `build_transfer_matrix(Q)` (line ~1445), `_reflect(Q2)` (~1511), `_trace(Q, pairs)` (~1647) and `_lab_channels(Q)` (~1665). `seq=None` means `self.sublayers`, exactly as now. Pass `seq` down unchanged. No behaviour change in this item; the existing tests must pass bit for bit.

### T3. Node sequences

New method `_spread_sequences(self, Qmax)` returns `[(w_k, seq_k)]`:

1. Get the nodes from `spread_nodes(Qmax)` (T5).
2. Save the interior thicknesses as a list of floats.
3. For each f_k: set `layer.thickness = d0 * f_k` for `layers[1:-1]`, call `build_sublayers()`, keep the returned list.
4. In a `try/finally`, restore every thickness from the saved floats (do not divide by f) and restore `self.sublayers` to the object it was before.

`build_sublayers` then handles both roughness schemes, including the Licorne windows and renormalisation, the same way the prototype's `from_dict` copies did.

**Cache.** `FitProblem.model` (`fit/problem.py` ~277) calls `build_sublayers()` once, then `reflectivities` once per Q group. Add a generation counter `self._sub_gen`, incremented wherever `self.sublayers` is assigned (`_set_sublayers`, `_build_licorne`). Cache the node sequences under the key `(self._sub_gen, thickness_spread, n)`. This rebuilds once per parameter change, never per Q group.

### T4. `reflectivities`

At the top of `reflectivities` (line ~1557): if `self.thickness_spread == 0`, run the current code unchanged. Otherwise take `seqs = self._spread_sequences(Qmax)` and use averaged ideal functions:

- non-magnetic fronting: `ideal(q) = Σ w_k self._trace(q, pairs, seq=s_k)`
- magnetic fronting: `ideal(q) = Σ w_k self._lab_channels(q, seq=s_k)`, then `_weigh` as now

`Qmax` for T5 is `max(Q) + 3·max(sigma)` (sigma as resolved in `_resolve`). Norms, scale and background are applied after, as now.

### T5. Node count

`spread_nodes(self, Qmax)` returns `[(w_k, f_k)]`. For an int setting, use n directly. For 'auto':

```
φ = Qmax · D · σ_t        (D = Σ thickness of layers[1:-1])
n = clamp(odd_ceil(1.5 φ + 2), 5, 31)
```

`odd_ceil(x)` is the smallest odd integer ≥ x. The rule gives 11 nodes at the §1 point (φ = 4.9), 13 at φ = 6.8 and 19 at φ = 10.1. Each is at least as many as the convergence table shows are needed for a 2e-3 relative error (7, 11 and 15), so the rule is conservative. Then apply the f_k > 0 check from §2.

### T6. Licorne-exact

`thickness_spread` is not part of `LICORNE_EXACT`. `set_licorne_exact` does not touch it, and `is_licorne_exact()` is unaffected. Licorne has no such parameter; G2 shows a note in the GUI. The Licorne import keeps its existing note for a second incoherent fraction (`model/licorne_io.py` ~463) and sets `thickness_spread = 0`.

## 4. Phase 2 — fitting

### F1. Free parameter

`free_parameters()` picks it up through `STACK_FIT_PARAMS` (layer index None, name `stack.thickness_spread`). Joint modes 'shared', 'own' and 'fixed' then work through `fit/joint.py` with no change. Priors work through `fit.priors` like scale and background. Covered by test 7.

### F2. Freeze the node count during a fit

With 'auto', n depends on σ_t, so the cost would jump where n changes. In `FitProblem.__init__`, on the fit's own copy of the stack: if `thickness_spread_nodes == 'auto'`, compute n once from the **upper bound** of σ_t (or its value, if fixed) and the largest data Q + 3σ_Q, then store the int on the copy. Do the same in `JointProblem` with the largest Q over all parts. The user's stack keeps 'auto'.

DREAM workers rebuild the stack from `to_dict` (`fit/dream/dream_fit.py` ~473), so they inherit the frozen int.

### F3. Messages

When σ_t is free or non-zero at the start of a fit, log once: "thickness spread: n nodes, each evaluation about n× slower".

## 5. Phase 3 — GUI (`tabs/simulation.py`)

### G1. Thickness-spread row

Add a row after Background, built in the same `stack_rows` loop (line ~1216) and laid out in the same loop (~1389): value, min, max, Fit, prior button.

- Display in %, with a `dspin(0, 30, 2, 0.1)`, and store the fraction. Convert in both directions in `_load_stack_fit` (~1843) and wherever the boxes are written to the stack.
- Find every place that handles scale and background together (`grep -n "background" tabs fit`; at HEAD: `tabs/simulation.py` ~1920–1921, ~2351–2352) and add `thickness_spread` there, so fit results and recomputes carry it.
- Tooltip: "Lateral spread of the film thickness (σ/d). Incoherent average over regions larger than the neutron coherence length; all interior layers scale together."

### G2. Node setting and Licorne note

A combo box with auto, 5, 7, 9, 11, 15, 21, bound to `thickness_spread_nodes`. When Licorne-exact is on and σ_t > 0, show a note next to it: "Licorne has no thickness spread".

### G3. Profile

The profile plot stays nominal (f = 1). Add to its tooltip: "nominal thicknesses; the thickness spread is not drawn". A ±σ_t band is Phase 4.

## 6. Phase 4 — later, not in v1

- A per-layer opt-out (`Layer.spread`, default True), for example to keep the substrate-side buffer fixed. The prototype scaled all interior layers, including a 122 Å buffer.
- A log-normal distribution in thickness, as used for MnBi₂Te₄ films.
- Independent spreads per layer.
- Import of Licorne's second incoherent fraction (Fraction < 100) as a general two-region incoherent mixture.
- A ±σ_t band on the profile plot.

## 7. Tests — new `tests/test_thickness_spread.py`

1. **Zero spread is the current code.** σ_t = 0 gives `np.array_equal` results with HEAD, for both roughness schemes, a magnetic-fronting stack, resolution on and off, and an explicit `sigma`.
2. **Matches an explicit average.** At σ_t = 0.07 and n = 7, the built-in result equals Σ w_k `Stack.from_dict(scaled).reflectivities(...)` to rtol 1e-12, for both roughness schemes and both resolution schemes.
3. **Commutes with resolution.** The average of the resolved node curves equals the resolved average, to rtol 1e-12 (quadrature and Licorne schemes).
4. **Node convergence** (`slow`). On a film and substrate with D ≈ 300 Å, Q up to 0.2: 'auto' is within 5e-3 relative of a 241-point brute force for σ_t in {0.03, 0.07, 0.10, 0.15}.
5. **Critical edge preserved.** With resolution off, for Q < 0.9 Q_c of the substrate, |R(σ_t = 0.1) − R(0)| < 1e-6.
6. **Serialisation.** `to_dict`/`from_dict` round trip; a session without the keys loads with 0.0 and 'auto'.
7. **Fit registration.** `free_parameters()` includes `(None, 'thickness_spread', …)`, `parameter_names` gives `stack.thickness_spread`, and a `JointProblem` with mode 'own' names it `stack.thickness_spread [dataset]`.
8. **Recovery.** Simulate a two-layer film with σ_t = 0.05 and 3 % noise (fixed seed). A fit with only σ_t free recovers it within ±0.005.
9. **Frozen nodes.** A `FitProblem` with 'auto' and σ_t free stores an int on its copy only; the user's stack still has 'auto'. The cost on a fine σ_t grid has no step larger than its smooth neighbours.
10. **Acceptance** (private data; skip if absent). Load `S1_4K_4p8T_thickness_spread.json` with the 4.8 T data, n = 7, σ_Q = the data dQ. R⁺ and R⁻ match `thickness_spread_reference_4p8T.json` to rtol 1e-9, and χ² = 497.97 ± 0.01 (χ²/N = 1.10659).
11. **Existing suite.** Every existing test passes unchanged, including the Licorne-exact references.

## 8. Done when

- Tests 1–11 pass.
- The §1 prototype result reproduces in the GUI: load the session, set nodes to 7, compute on S1 4 K 4.8 T → χ² 497.97.
- The joint result in §1 is filled in.
