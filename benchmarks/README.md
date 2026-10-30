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

## NPE-seeded DREAM (`plans/npe_plan_v1.md` B0, V3)

`bench_npe_sim.py` (B0) times the training simulator; `bench_fit.py dream
--seed-npe` (V3.1, V3.2) trains a network on the session and starts DREAM
at its seeds, without the current values; `bench_npe_joint.py` (V3.3) does
the same for a joint fit of a saved SANR session. Measured on JF's M1 (8
cores), 2026-10-30.

**B0, session A untied:** about 0.7–1.3 ms per draw (3 to 9 slabs per
interface, compiled product on, resolution on or off), 20–59 slabs; 10⁵
simulations take well under a minute on 8 cores, so training, not
simulation, takes the time.

**V3.1, session A untied**, 5 rounds × 2·10⁴ simulations (full fidelity: no
coarse setting was within 0.3 σ), 10⁴ seeding samples, then DREAM 8 chains,
5000 generations, snooker 0.1 and the tempered burn-in:

| Seed | MAP χ²/N | Chains at the end | Max R̂ | Wall time (training included) |
|---|---|---|---|---|
| 1 | 4.9494 | 4.97–5.02 | 3.1 | 33 min |
| 2 | 4.9895 | 4.99–5.42 | 4.1 | 33 min |
| 3 | 4.9533 | 4.98–5.06 | 4.7 | 49 min |

Every chain of every seed starts in the 4.927 basin, which the Latin
hypercube start with the same settings never reaches (6.95 and 12.37
above); the criterion χ²/N ≤ 4.95 in all 3 seeds is met by seed 1 only:
5000 generations do not converge inside the basin (R̂ 3–5). Layer 5's
thickness and the substrate roughness sit on their bounds, as in the warm
start. Rounds 3–5 drew from the envelope proposal (`fit.npe.train` ROUNDS).

**V3.2, session B** (7 parameters, `--vary` as above), the same NPE
settings, DREAM 8 chains, 8000 generations, snooker 0.1 and the tempered
burn-in: MAP χ²/N 13.2893, 13.2897, 13.2890 for seeds 1–3, every chain at
13.29–13.33, R̂ 1.08, 1.04, 1.05: it matches the reference (13.29, R̂ ≤
1.09).
