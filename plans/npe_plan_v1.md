# SANR — neural posterior estimation (NPE) front end for DREAM(ZS) — implementation plan v1

Target: `github.com/JFMilette/SANR`, branch `thickness-spread-and-fit-tools` (read at commit `8c45dd7`).
**Purpose:** find every region of parameter space compatible with the data (multimodal, strongly correlated posteriors) with a neural posterior estimator, then let DREAM(ZS) with SANR's exact likelihood produce the posterior. The network only *proposes*; DREAM *decides*. No result is ever reported from raw NPE samples.

**Generality is a hard requirement.** Nothing in this plan may depend on a particular sample, stack, channel set or instrument. The simulator *is* `FitProblem.model` / `JointProblem.model`, so every model feature SANR has (any number of layers, ties, mixtures, thickness spread, both roughness schemes, every resolution mode, magnetic fronting/backing, full or half polarisation analysis, joint datasets with shared/own parameters, priors from other measurements) is supported by construction. No physics is re-implemented here.

## 0. Before starting

### 0.1 Files

| File, as delivered | Put it at | Status |
|---|---|---|
| `npe_plan_v1.md` | `plans/npe_plan_v1.md` | this plan |

New code goes in a new package `fit/npe/`. New tests: `tests/test_npe_*.py`. New benchmark: `benchmarks/bench_npe_sim.py`.

### 0.2 Conventions

- As in earlier plans: module docstrings are the documentation, one commit per item with its tests, Licorne-exact tolerances are never loosened, and every existing test must pass unchanged.
- **Optional dependency.** `torch` (CPU) and `sbi` go in a new `requirements-npe.txt`, not in `requirements.txt`. `fit/npe` imports them lazily; the GUI and every non-NPE test must run without them. NPE tests skip when they are absent. Training tests are marked `slow` (pytest already deselects `slow`).
- **Check the installed `sbi` API before coding.** Class and function names changed between releases (e.g. `SNPE` vs `NPE`, restricted-prior / truncation helpers, SBC utilities). Pin the version that was checked in `requirements-npe.txt` and write against that version's documentation, not from memory.
- **Precision.** The simulator stays SANR's float64 numpy/Cython. The networks may train in float32 on the CPU. Do not use the Apple MPS backend for anything that touches the simulator (MPS has no float64).
- **Units and conventions are SANR's.** Parameters live in the FitProblem's normalised coordinates `u ∈ [0, 1]^n` (`to_x` / `to_u`). Angles are in turns; `MSLD_theta` is the in-plane angle from sample x and `MSLD_phi` the elevation, as in the `Layer` docstring; only ρ cos(φ) enters (HALPERIN). This plan adds no reparameterisation of its own.

### 0.3 Prerequisites (P0 — check, do not assume)

The earlier Bayesian plan gated new work on two bugs. In this branch both look fixed; **confirm with a test before starting** and report back if either is not:

1. *Half-polarised background double-counting* — the POLARISATION docstring says the background is added once per pair after the scale (`R+ = scale (R++ + R+-) + background`). Add a regression test if none exists: a half-polarised pair with a known background, checked against `scale * (R++ + R+-) + background` computed from the four channels.
2. *Batched channel calculation* — `Stack._trace` computes every pair from one transfer-matrix evaluation. Confirm `FitProblem.model` makes one `reflectivities` call per Q grid.

### 0.4 Order

1. P0 (prerequisites), B0 (benchmark).
2. S1–S3 (simulator).
3. T1–T2 (training).
4. D1–D2 (DREAM seeding).
5. V1–V3 (validation; V3 is the acceptance test).
6. Y1 (symmetry probe), A1 (scenario studies).

Everything in §4 is deferred.

## 1. Evidence (measured on this branch, outside SANR's test suite)

Stack: air / AlOx 15 Å / CrSb-like 45 Å / magnetic interface 10 Å / BST-like 90 Å / substrate, noncollinear in-plane magnetisation, `rms` scheme. Channels R+ and R− (no analyser) on 150 Q points, 0.008–0.15 Å⁻¹. Single core of an x86 cloud CPU, **not the M1**: B0 re-measures on JF's machine.

| Sublayers per interface | Slabs | Compiled product | Resolution (tof, quadrature) | Time per draw (`build_sublayers` + `reflectivities`) |
|---|---|---|---|---|
| 5 | 27 | yes | off | 2.4 ms |
| 5 | 27 | yes | on | 9.7 ms |
| 5 | 27 | no | on | 19.4 ms |
| 16 | 82 | yes | off | 4.6 ms |
| 16 | 82 | yes | on | 19.1 ms |
| 16 | 82 | no | on | 53.0 ms |

What this shows for the design:

- **The existing per-draw path is fast enough.** At about 10 ms per draw, 10⁵ simulations take roughly 17 min on one core; a joint problem with K datasets costs K times that. Sequential NPE needs 10⁴–10⁵ simulations, so no new vectorised simulator is needed for v1. A batched-over-draws simulator is deferred (§4.1) until B0 or T1 shows the simulator is the bottleneck.
- **The resolution convolution is ~75 % of the cost** with the quadrature scheme. Fidelity reductions for training (S2) should look at resolution settings as well as sublayer counts.
- **Fixed sublayer counts do not give fixed slab counts.** Over 300 random draws (thicknesses 5–120 Å, roughness 0–15 Å), the slab count varied from 14 to 29 with 5 sublayers per interface, in both the `rms` and `licorne` schemes; still 25–29 with every roughness > 0. `build_sublayers` merges coincident edges (`np.unique` and the 1e-4 Å filter), e.g. two windows clipped at the same mid-layer depth, and a zero-width window collapses to one edge. This is harmless for the per-draw path used here; it matters only for the deferred batched simulator (§4.1).
- **Motivation from the existing benchmarks** (`benchmarks/README.md`): on session A, DREAM from a Latin-hypercube start does not reach the χ²/N = 4.927 minimum (chains end at 6.6–12.1, R̂ up to 194); only a warm start at that minimum does. Finding such basins from scratch is the job of this plan. V3 uses it as the acceptance test.

## 2. Design

```
FitProblem / JointProblem  ──►  Simulator (S1)  ──►  sbi (T1)  ──►  posterior q(u | data)
      (bounds, ties, priors,          prior draws,         sequential             │
       grids, resolution,             noise, features      rounds                 ▼
       scale, background)                                                 seed_chains (D1)
                                                                                  │
                                         PosteriorProblem + dream_fit.sample(x0=...)  ──►  result
```

- **One source of truth.** Parameters, bounds, ties and priors come from the problem's `params`, `lo`, `hi`, `names`, `priors`. The forward model is `problem.model(x)`. The noise model mirrors the likelihood DREAM uses (`PosteriorProblem`: Gaussian with the fitted channels' σ, including any fixed `error_floor`).
- **Sequential NPE per measured problem.** The network is trained for one observation, which can be a joint problem over several datasets (shared structure, own magnetisation). This keeps the simulation count at laptop scale and needs no composition machinery: the observation is simply all channels of all datasets.
- **The first round is amortised over the prior.** Keep the round-1 network; it is a valid (less sharp) amortised estimator over the whole prior and serves the scenario studies (A1).
- **Refinement is unchanged DREAM(ZS).** NPE samples only replace the starting states (and optionally the initial archive, D2).

## 3. Items

### B0 — Simulator benchmark on the target machine

`benchmarks/bench_npe_sim.py`: for a FitProblem loaded from a session file (argument), time `problem.model(x)` over random prior draws, with the compiled product on/off, resolution on/off and a few sublayer counts. Print draws per second per core and the slab-count range. Not part of pytest. JF runs it on the M1; the numbers go in §1 of v2.

### S1 — `fit/npe/simulator.py`: the simulator

`Simulator(problem, *, noise=True, seed=None)` where `problem` is a FitProblem or JointProblem with the `'chi2'` cost.

- `sample_prior(n, rng)` → `u (n, d)`: uniform in `[0, 1]^d` (the bounds), times the `fit.priors` terms where present. Sample those by rejection from the uniform, or from the truncated Gaussian / soft box directly; either way `log_prob` must equal the uniform plus `fit.priors.log_prior` on the bounded support, so that the NPE prior and DREAM's prior are the same distribution.
- `simulate(u)` → raw channels: `problem.model(problem.to_x(u))` per row. Failures (exceptions, non-finite output) are recorded and the row is dropped, never silently replaced; the failure rate is reported.
- `add_noise(R, rng)`: Gaussian with each channel's fitted σ as `PosteriorProblem` builds it (the FitProblem's σ including any fixed `error_floor`). Free noise parameters (`sigma_free`, `floor_free`) are not supported in v1: raise a clear error (§5, Q4).
- `features(R)`: one flat vector of all channels in a fixed order. Use a sign-safe transform, `asinh(R / c_k)` with `c_k` a fixed per-channel scale (e.g. the median σ of channel k), because noisy R can be ≤ 0 near the background. Then standardise with means and spreads computed once from round-1 simulations and stored with the network.
- Parallel evaluation through the existing spawn-process pattern of `fit.workers` (the problem pickled once per worker), with explicit per-chunk seeds so a run is reproducible regardless of the worker count.

### S2 — Training fidelity: coarse simulator with a measured gap

`coarse_problem(problem, n_sub=None, resolution=None)` returns a deep copy whose stack(s) use `n_sub` sublayers per interface (every layer's `roughness_sublayer`, both schemes) and optionally reduced resolution settings (e.g. fewer quadrature nodes). Thickness-spread node counts stay frozen as `FitProblem` freezes them.

`fidelity(problem, coarse, n_draws, rng)` evaluates both on the same prior draws, plus draws pinned to the corners of the roughness bounds (where slicing error is largest), and returns the distribution of `max_Q |R_coarse − R_full| / σ` per channel.

Rule: use the cheapest coarse settings whose 99th percentile of that statistic is below a threshold (default 0.3; configurable). The full-fidelity problem is always the one handed to DREAM.

### S3 — Simulator tests (`tests/test_npe_simulator.py`, fast, no torch)

1. With noise off and full fidelity, `simulate(u)` equals `problem.model(problem.to_x(u))` **bitwise**, for a FitProblem and for a JointProblem with shared and own parameters.
2. Ties: tied parameters appear once in `u`, and every tied member takes the value.
3. Prior: `sample_prior` stays within `[0, 1]^d`; with a Gaussian prior, the sample mean and spread match the truncated distribution to Monte-Carlo accuracy; `log_prob` agrees with `fit.priors.log_prior` up to a constant.
4. Reproducibility: same seed gives the same simulations with 1 and with 3 workers.
5. Noise: the empirical spread of `add_noise` matches the likelihood's σ per channel.
6. `fidelity` is zero when `coarse` is the full problem.
7. Generality smoke tests: a magnetic-fronting stack with full polarisation analysis, a Licorne-scheme stack, a stack with thickness spread, and a mixture all run through `simulate` and `fidelity` without special cases.

### T1 — `fit/npe/train.py`: sequential training

`train(simulator, *, rounds, sims_per_round, embedding='mlp', truncation=True, seed)` → `NPEResult`.

- Neural posterior estimation in `u`-space with a flow suited to bounded supports (map `u` to logit space, or use sbi's transform support, so samples never leave `[0, 1]^d`).
- Embedding: an MLP on the standardised feature vector (works for any number of channels and grids). Keep the architecture configurable.
- Sequential rounds with **truncated proposals** (TSNPE-style: later rounds sample the prior restricted to the high-density region of the current posterior) rather than atomic-loss SNPE, which can leak mass outside the prior. Use whatever the pinned sbi version provides for this.
- Defaults to start from, and to revise after V1: 5 rounds, 2·10⁴ simulations per round, early stopping on validation loss.
- `NPEResult` saves: the networks of every round (round 1 kept separately, §2), feature standardisation, the problem's parameter names and bounds, the coarse settings and their fidelity report, the git hash and data hash (reuse `fit.dream.dream_fit._git_hash` and `data_hash`), the sbi/torch versions and the seeds. `load_npe(path)` refuses a problem whose names, bounds or data hash differ.
- Memory: simulations are stored as float32 features (10⁵ draws × 1200 features ≈ 0.5 GB). Stream rounds to disk (`.npz` per round) so 8 GB machines are not a constraint.

### T2 — Training tests (`tests/test_npe_train.py`, `slow`)

1. **Analytic toy.** A one-layer problem (thickness and NSLD free, resolution off) with tight synthetic data: NPE posterior mean and spread match a brute-force grid posterior of the exact likelihood within a stated tolerance.
2. **Bounds.** No posterior sample leaves `[0, 1]^d`.
3. **Save/load round trip.** Identical samples for an identical seed; a changed data hash is refused.

### D1 — Seeding DREAM from NPE

`fit/npe/seed.py`: `seed_chains(npe_result, posterior_problem, n_chains, rng)` → `x0 (n_chains, d)` in physical units for `dream_fit.sample(problem, x0=...)`.

- Draw many NPE samples (e.g. 10⁴), evaluate the exact log posterior of each with `PosteriorProblem`, and keep the starting states spread across modes: cluster in `u`-space (k-means or a Gaussian mixture), allocate chains to clusters in proportion to the clusters' summed importance weights `p(x | data) / q(x)` with at least one chain per cluster above a weight threshold, and inside each cluster pick the highest-posterior samples.
- Report per cluster: weight, best χ²/N, and parameters on a bound. This is useful on its own (it lists the candidate solutions).
- No change to `dream_fit.sample` is needed for D1: it already accepts an `(n_chains, d)` `x0`.

### D2 — Optional: initial archive from NPE samples (needs JF's decision, §5 Q1)

`dream_fit.sample` currently seeds the DREAM(ZS) archive from the prior even when `x0` is given. Add an optional `z0` argument (array of initial archive states, physical units) used instead of the prior draw. Default unchanged; existing DREAM tests pass unchanged. Test: with `z0` given, the initial archive equals it; with `z0=None`, results are bitwise identical to before for a fixed seed.

### V1 — Synthetic recovery (`tests/test_npe_recovery.py`, `slow`)

For a noncollinear multilayer, a half-polarised (R+, R−) problem and a full-analysis problem, each from known parameters with realistic noise:

1. NPE-seeded DREAM recovers the true parameters within their posterior credible intervals.
2. **Known multimodality is found.** Use a case with a mirror ambiguity identified by Y1 (e.g. in-plane angle sign without an analyser): with bounds that allow both modes, D1's clusters must contain both, and DREAM must sample both with the weight the exact likelihood implies (compare with a long DREAM(ZS) run started from both modes).
3. **Joint problem.** Three synthetic datasets with shared structure and own magnetisation: recovery of shared and own parameters.

### V2 — Calibration of the amortised (round-1) network

Simulation-based calibration (rank statistics or expected coverage) of the round-1 network on a few hundred prior draws, with the utilities of the pinned sbi version. Report it in `NPEResult`. A badly calibrated round-1 network may still seed DREAM well, but must not be used for A1.

### V3 — Acceptance test on real data (benchmark, not pytest)

Extend `benchmarks/bench_fit.py` with `--seed-npe`:

1. **Session A** (private data): NPE-seeded DREAM from scratch, without the current values, must reach χ²/N ≤ 4.95 (the known 4.927 minimum) in all of 3 seeds, compared with the Latin-hypercube and `'de'` starts on the same budget. Record wall time including training.
2. **Session B:** must match the existing result (χ²/N 13.29, R̂ ≤ 1.09).
3. **One of JF's joint temperature/field datasets** (private), as a JointProblem with shared structure and own magnetisation: report the clusters D1 finds and the final DREAM result next to the current best fit.

### Y1 — Symmetry probe (`fit/npe/symmetry.py`)

`probe(problem, n_draws)`: for candidate transformations of the free parameters, test numerically whether `problem.model` is invariant on the problem's own channels. Candidates:

- in-plane angle θ → −θ, θ → θ + ½ turn, and ρ → −ρ together with θ → θ + ½, per magnetic layer and for all layers together;
- φ → −φ (elevation).

Report each transformation as exact (to 1e-12 relative), approximate (gap / σ below 0.1) or broken. It decides bounds for DREAM and for NPE; nothing is hard-coded. Tests:

- a collinear stack with full analysis, where the known symmetries hold;
- a magnetic fronting, where some break;
- a half-polarised noncollinear stack.

### A1 — Scenario studies with the amortised network (script)

`scripts/npe_scenarios.py`: given a set of scenario parameter vectors (e.g. moment at an interface vs distributed through a layer), simulate synthetic data for each with the problem's noise, sample the round-1 posterior, and report whether the scenarios' posteriors separate (posterior mass of each scenario's region). Checked against DREAM on a few of them. Use only if V2 passed.

## 4. Deferred

### 4.1 Batched-over-draws simulator

Only if B0/T1 show simulation dominates. It needs the same slab count for every draw in a batch. §1 shows the current slicing does not guarantee that even with fixed sublayer counts. Pad each draw to the batch maximum with zero-thickness slabs (their transfer matrix is exactly the identity), or add a slicing option that never merges edges. Either must reproduce `build_sublayers` profiles exactly (tests against the per-draw path).

### 4.2 Amortised training at scale

10⁶ simulations or more on a GPU cluster, for reuse across many samples.

### 4.3 Compositional or hierarchical estimators across datasets

Separately trained single-dataset estimators, combined at inference with a shared/local parameter split (Munteanu et al., arXiv:2610.29362). Not needed while sequential NPE can train on the joint problem directly.

### 4.4 GUI

A "Seed from NPE" start option in the DREAM window, after V3 passes.

### 4.5 Free noise parameters

Free noise parameters (`sigma_free`, `floor_free`) as NPE parameters.

## 5. Open questions for JF

1. **D2:** should the DREAM(ZS) archive be seeded from NPE samples, or stay prior-seeded with NPE only setting the chains' starting states? Seeding it speeds burn-in but makes the early proposal differences depend on the network.
2. **Fidelity threshold (S2):** is 0.3 σ (99th percentile, worst Q) the right bar for the coarse training simulator, given that DREAM always uses full fidelity?
3. **V3 data:** which joint dataset (sample, temperatures, fields) should be the acceptance case, and may its parameter set include the thickness spread?
4. **Noise:** do your current DREAM runs use `sigma_free` or `floor_free`? If so, 4.5 moves up.
5. **Where training runs:** M1 CPU only for v1, or should T1 also support running the simulation rounds on an Alliance cluster node and training locally?
