# Licorne exports used by tests/test_licorne_roughness.py

| Folder | Licorne | Source | Tests |
|---|---|---|---|
| `fixture1/` | 1.4.2 | IPTS 27232, S1_3K_4.8T_FC/2/best_103 | geometry and values (§8.4), incl. the clipped, renormalised window |
| `v127_chi3_137/`, `v127_r2_6_508/` | 1.2.7 | `tests/data/` of Licorne-Py (github.com/neutrons/Licorne-Py) | the angle rule inside sublayers (MSLD_phi, MSLD_theta per sublayer), unclipped windows |
| `fixture2/antiparallel/`, `fixture2/perpendicular/` | (to export) | 50 Å / 50 Å bilayer, rho = 5e-6, theta = 90, phi = 0/180 resp. 0/90, sigma_L = 10, tanh, N = 6 | angle rule (§8.5); skipped while absent |

| `session1/` | (to export) | a full saved Licorne session folder: `parameters.m`, `profile.dat`, `q.dat`, `rexp*.dat`, `rtheory*.dat`, `resolution.m`; ideally `Norm_factor` other than 1 and 2, one analysed channel and `ResolutionFun=3` | `tests/test_licorne_end_to_end.py::test_session_rtheory`; skipped while absent |

The model folders hold `parameters.m`, `profile.dat` and `profile_sublayers.dat`.
`tests/licorne_reference.py` (a port of Licorne's MATLAB engine) reproduces
the three `profile_sublayers.dat` to their 6 figures.
