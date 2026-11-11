## What's new in 0.5.1

- **Export datasets as CSV:** a new **Export…** button in the Experimental tab saves one or more datasets (measured, simulated or frozen copies) as CSV files in a folder you choose. For each dataset you can save:
  - the **SLD profile** of its model: depth z, NSLD (real and imaginary), MSLD ρ, φ and θ, and the layer at each depth, followed by a table of the interfaces with their roughness σ and magnetic roughness σ_M;
  - any of the **plotted quantities**: polarized, half-polarized and non-polarized reflectivity, SA NSF, SF fraction, SA SF and SA half-polarized;
  - the reflectivities as **R, log₁₀ R and/or R·Q⁴**, with the errors scaled to match.

  Each curve gets its own Q, value, error and dQ columns. The model's curve can be written beside the data, and the export can keep only the points within the model's Q range, as plotted. Files open in Excel with Å and ⁻¹ shown correctly. When you export several datasets, each one goes in its own sub-folder.

Everything in 0.5 (a Q range per model, monochromator resolution, faster fits with the Licorne resolution), 0.4.1 (larger NSLD range, fit-window curves within the fitted Q range, session folder remembered) and 0.4 (a model per dataset, Copy to…, stitch factors, the three installers) is included.

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
