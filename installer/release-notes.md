## What's new in 0.5

- **A Q range per model:** each model now keeps its own Q min, Q max and Q points. The range sets where that model's curves are computed, which of its datasets' points are shown in the Experimental tab, and which points are fitted. In a joint fit each dataset is fitted within its own model's range, so an X-ray scan and a neutron measurement can be fitted together over different Q windows. **Copy to…** with *general parameters* ticked also copies the Q range. Sessions saved with an older version open with every model on the session's single range, so they look and fit exactly as before.
- **Monochromator resolution:** a new resolution mode for a fixed wavelength and an angle scan, such as a lab X-ray reflectometer. It takes the wavelength λ, the relative wavelength spread Δλ/λ and the angular spread Δθ (both standard deviations, not FWHM), with σ_Q = Q·√((Δθ/θ)² + (Δλ/λ)²). The resolution dialog can fit Δθ and Δλ/λ to a dataset's measured dQ.
- **Faster fits with the Licorne resolution:** when the Q range keeps only part of the data, the model is now computed only where the fitted points need it, not on the whole data grid. The results are identical. An X-ray scan measured to 4 Å⁻¹ and fitted up to 0.25 Å⁻¹ runs about 10 times faster per iteration.
- **Simulation curves in the Experimental tab:** each model's curve is computed on its own Q range. It no longer changes with the datasets chosen in the Fit tab, which could leave a curve stuck on another dataset's Q points. Ticking and unticking datasets to fit is also faster.

Everything in 0.4.1 (larger NSLD range, fit-window curves within the fitted Q range, session folder remembered) and 0.4 (a model per dataset, Copy to…, stitch factors, the three installers) is included.

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
