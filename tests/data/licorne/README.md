# Licorne exports used by tests/test_licorne_roughness.py

| Folder | Licorne | Source | Tests |
|---|---|---|---|
| `fixture1/` | 1.4.2 | IPTS 27232, S1_3K_4.8T_FC/2/best_103 | geometry and values (§8.4), incl. the clipped, renormalised window |
| `v127_chi3_137/`, `v127_r2_6_508/` | 1.2.7 | `tests/data/` of Licorne-Py (github.com/neutrons/Licorne-Py) | the angle rule inside sublayers (MSLD_phi, MSLD_theta per sublayer), unclipped windows |
| `fixture2/antiparallel/`, `fixture2/perpendicular/` | (to export) | 50 Å / 50 Å bilayer, rho = 5e-6, theta = 90, phi = 0/180 resp. 0/90, sigma_L = 10, tanh, N = 6 | angle rule (§8.5); skipped while absent |

| `session1/` | (to export) | a full saved Licorne session folder: `parameters.m`, `profile.dat`, `q.dat`, `rexp*.dat`, `rtheory*.dat`, `resolution.m`; ideally `Norm_factor` other than 1 and 2, one analysed channel and `ResolutionFun=3` | `tests/test_licorne_end_to_end.py::test_session_rtheory`; skipped while absent |

Private sessions (`tests/data/private/`, git-ignored; copy them in by hand,
`tests/test_licorne_sessions.py` and the benchmarks skip without them), each
`parameters.m`, `profile.dat`, `profile_sublayers.dat`, `q.dat`,
`rexp1/2.dat`, `rtheory1/2.dat`, `resolution.m`, `version.txt` of a Licorne
1.4.2 session folder:

| Folder | Source | Role |
|---|---|---|
| `session_A/` | `HC-230104A_chi5_6 8Afinal` (without `IPTS2233/`, `EB_040710/`) | 12 free parameters (the session's own fit flags). Its `q.dat` repeats Q = 0.0226139 and 0.0617789 where the TOF angles are stitched. Its Licorne fit stopped at the 50-iteration LM limit (`LMNiter=50`), not at a minimum. chi^2/N 9.0249 at the session values |
| `session_B/` | `Fit_thickness`, as saved | 7 free parameters: thicknesses of layers 1-3, roughness of layers 1-3 and the substrate; its NSLDs and rho are held at unconverged values. A benchmark problem, not a physics result. chi^2/N 13.4354 |

The model folders hold `parameters.m`, `profile.dat` and `profile_sublayers.dat`.
`tests/licorne_reference.py` (a port of Licorne's MATLAB engine) reproduces
the three `profile_sublayers.dat` to their 6 figures.
