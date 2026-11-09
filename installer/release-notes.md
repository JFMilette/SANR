## What's new in 0.4.1

- **Larger NSLD values:** the layer editor accepts NSLD Re and Im from −1000 to 1000 ×10⁻⁶ Å⁻² (was ±100), e.g. X-ray SLDs of heavy elements (Pt, Ir, Os) for fitting XRR data.
- **Fit window:** the model curves stop at the fitted Q min and Q max, so the fit is easier to see on data that go to high Q (with the Licorne resolution they used to run over all the data's Q).
- **Session folder remembered:** once a session is opened or saved, Open session and Save session start in its folder for the rest of the run.

Everything in 0.4 (a model per dataset, Copy to…, stitch factors, faster Licorne resolution, the three installers) is included.

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
