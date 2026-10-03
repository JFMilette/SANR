# Fitting benchmarks

`bench_fit.py` runs the fitting methods on the two private Licorne sessions
(`tests/data/private/session_A`, `session_B`, see
`tests/data/licorne/README.md`). It is not part of pytest. Results are added
as JSON lines to `benchmarks/results.jsonl`, which git ignores.

| Session | Free parameters | χ²/N at the session values |
|---|---|---|
| A (`HC-230104A_chi5_6 8Afinal`) | 12 untied (`--untie`); 11 as imported (layer 2 / 3 NSLD coupled) | 9.0249 |
| B (`Fit_thickness`) | 7 with the `--vary` below | 13.4354 |

All numbers use every Q point. Session A's own `QLims` is [0, 0.22], so
Licorne fitted A only up to Q = 0.22. The GUI import now does the same; these
benchmarks do not.

## Reference numbers

Measured with plan v2, before the fitting changes (`plans/sanr_fitting_plan_v2.md` §1).

**Landscape** (`landscape`, L-BFGS-B):

| | A (24 random + current, untied) | B (30 random + current) |
|---|---|---|
| Best χ²/N | 4.927 (only from the current values) | 13.318 |
| Random starts within +1 of the best | 0 % | 94 % |
| Fits with ≥ 1 parameter on a bound | 100 % | 100 % |

**DREAM, plain sampler** (archive ZS, Latin-hypercube start, no snooker or
tempering):

- **A, n_gen 5000:** 3 chains: χ²/N 12.07, 10.36, 6.63 (R̂ 3.4, 1.5, 194);
  8 chains: 10.85, 7.56.
- **B, n_gen 8000, 7 runs:** all χ²/N 13.29, R̂ ≤ 1.09.
- **A, warm-started at the 4.927 minimum** (8 chains, 3000 generations):
  stays at 4.93. Layer 5's thickness sits on its 100 Å bound (49 % of
  samples), the substrate roughness on 50 Å (26 %).

**DREAM with snooker 0.1 and the tempered burn-in** (8 chains, `--p-snooker 0.1
--temper`), A, n_gen 5000, seeds 1 and 2: all 8 chains in one basin, at
χ²/N 6.95 and 12.37, not 4.93. Tempering stops the chains from splitting but
does not solve A's global search. Use the multistart, then start DREAM at its
best minimum.

## Measured by the tests (`python -m pytest -m slow tests`)

- Multistart, B, 31 starts: best χ²/N ≤ 13.33, with at least 70 % of the starts within +1 of it.
- Multistart, A untied, 25 starts: best 4.927 only from the current values;
  each of the other 24 starts ends in its own minimum (5.41 to 2973). The
  table is written to `benchmarks/multistart_session_A.txt`.
- DE restarts, A untied (3 runs, maxiter 50, popsize 5, tol 1e-6): every
  run reaches 4.927.
- A as imported (11 parameters, coupled NSLD): an L-BFGS-B polish from the
  session values reaches 5.63.
- DREAM, new defaults, B, n_gen 8000, seed 1: χ²/N ≤ 13.30, R̂ ≤ 1.1.

## Commands

```
python benchmarks/bench_fit.py landscape tests/data/private/session_A --starts 24 --untie
python benchmarks/bench_fit.py landscape tests/data/private/session_B --starts 30 \
  --vary 1:thickness,2:thickness,3:thickness,1:roughness_sigma,2:roughness_sigma,3:roughness_sigma,4:roughness_sigma
python benchmarks/bench_fit.py dream tests/data/private/session_A --untie --seeds 1 2 3 --n-gen 5000 --chains 3
python benchmarks/bench_fit.py dream tests/data/private/session_A --untie --seeds 1 2 --n-gen 5000 --chains 8 \
  --p-snooker 0.1 --temper
python benchmarks/bench_fit.py multistart tests/data/private/session_A --untie --starts 24 --workers 4
```

Without `--p-snooker` or `--temper`, `dream` runs the plain sampler (snooker 0,
no tempering), so its numbers compare with the reference ones. `sample()`'s
defaults, and the GUI's, are snooker 0.1 and the tempered burn-in.
