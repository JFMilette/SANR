## What's new in 0.4

- **A model per dataset.** Every imported dataset gets its own model, renamed and removed with it; each dataset is fitted with its own model.
- **Copy to…** writes the shown model's values, fit bounds, Fit ticks, priors, general parameters, resolution or whole layer stack into other models.
- **Stitch factors** for up to 12 angles, fitted per dataset and edited in the Experimental tab; the resolution can be fitted per stitched angle.
- **Faster Licorne resolution:** opening a session, switching models and changing the resolution are up to 10× faster on large datasets (same results).
- **Fit tab:** a splitter between the datasets and the parameters; parameter names tinted with their SLD-profile colours.
- **NPE tab:** Constraints and Network check.
- **Installers** for Windows x64, Windows ARM64 and macOS (Apple silicon).

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
