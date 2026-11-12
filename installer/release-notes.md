## What's new in 0.7

### DREAM: more effective samples, stricter convergence
- **About twice the effective samples for the same run time.** At the end of the adaptation phase, DREAM now drops the early burn-in states it kept for its jumps (they made most later jumps too long to be accepted). It always keeps at least 10 states per parameter.
- **Adaptation** is a setting on the DREAM page: **auto** (the default) is 10 % of the generations, at least 5000 and at most half the run; or type a number of generations. The tempered burn-in now ends halfway through the adaptation phase.
- **Samples:** choose **last half** (the default, safe) or **after adaptation** (more samples, for chains started at a fitted optimum). R̂, ESS, the Summary, the bands and the plots all follow the run's choice.
- **Stricter R̂:** convergence uses the rank-normalised split R̂ (Vehtari et al. 2021), which also sees chains that are still drifting. The default **R̂ target is 1.05** (it was 1.1).
- **ESS target** (new, 400 by default): **stop when converged** now also waits until every parameter has that many effective samples, and R̂ must stay below its target (a rise starts the count again). A run that ends with fewer gets a warning.
- **Burn-in warning:** if log p was still rising after the adaptation phase, the Summary says so and **Prune & continue** counts those generations as burn-in, drops them, and continues the run.
- **Continue** works on any ended run (stopped, converged or done; not one reopened from a session), for the number of generations in the **+ gen.** box beside it.
- Runs saved by 0.6 or earlier open as before and show the R̂ they were computed with (the older Gelman–Rubin R̂).

### Fits: χ²/N beside the cost
- Every fit window (DE, L-BFGS-B, least squares, basin hopping, dual annealing, SHGO, DIRECT, multistart) shows the **data-only χ²/N** in its own column beside the cost it minimises. With priors the cost is "cost (χ²/N + prior/N)"; without priors both are the same and one column "χ²/N" is shown. It is the same χ²/N DREAM reports, so fits and DREAM compare directly.
- The status lines give both, e.g. "best cost 1.35068 (χ²/N 1.34276 + priors 0.00791)", and at the end of a fit **χ²/ν**, χ² per degree of freedom.
- The **multistart** report gave the cost as "χ²/N": it now reads "best χ²/N … (cost …)".
- The **Residuals** plot gives the dataset's χ²/N under the plot and, in a joint fit, every dataset's together, noting the error floor when one is set.

### Sessions
- **Opening a session (or importing a Licorne one) no longer asks about unsaved changes**: only quitting SANR does.

Everything in 0.6 (sessions in one `.zip`, the R̂ tab), 0.5.1, 0.5, 0.4.1 and 0.4 is included.

## Downloads

| Your computer | File |
|---|---|
| Windows on Intel or AMD (most PCs) | `SANR-<version>-x64-setup.exe` |
| Windows on ARM (Snapdragon, e.g. Surface Pro X / Copilot+ PCs) | `SANR-<version>-arm64-setup.exe` |
| Mac with Apple silicon (M1 or later) | `SANR-<version>-macos-arm64.dmg` |

Not sure which Windows you have? **Settings → System → About → System type**: "x64-based processor" or "ARM-based processor".

**Windows:** run the setup file. No administrator rights are needed. You get a Start-menu entry, an optional desktop icon and an uninstaller; a newer version installs over an older one. Windows SmartScreen may warn that the publisher is unknown: **More info → Run anyway**.

**macOS:** open the `.dmg` and drag **SANR** onto **Applications**. The app is not signed with an Apple developer identity, so the first time macOS refuses to open it: open **System Settings → Privacy & Security**, scroll down and press **Open Anyway** next to SANR (on macOS 14 and older, right-click SANR → **Open** also works).

Sessions are saved in **Documents/SANR** by default.
