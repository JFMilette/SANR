## What's new in 0.6

### Sessions in one file
- **Sessions are saved as a `.zip`** that holds everything: the session itself, a copy of every imported data file, and a file for each open fit or DREAM tab, named after its algorithm (e.g. `tabs/1_L-BFGS-B.json`, `tabs/2_DREAM.npz`). A session now opens on any computer, without the folders the data came from. A `.zip` session is an ordinary zip file: any unzip tool opens it.
- `.json` sessions still open, and **Save session** still offers `.json` (data files by path) if you prefer it.
- **Opening a session shows an "Opening session…" window** until everything is ready. A session with a large DREAM run no longer freezes the program ("Not responding"): the run's summary is computed in the background.
- Sessions now also keep the settings of the **DREAM**, **NPE** and **multistart** pages.

### Fitting
- **DREAM:** a new **R̂** tab plots the largest R̂ over the parameters against generation, live while the run goes and for a finished or reopened run, with the run's R̂ target as a dashed line and the generation where it was first reached. The **Summary** gives each parameter's unit, with the values in that unit (as in the Simulation tab), and the 68 % interval (16 % and 84 %) beside ±1σ and the 95 % interval. The **median** can be sent to the model: it is in the list beside **Load into model** (with its χ²/N, to compare with the MAP), and a **Load median** button sends it directly. The Chains box (and NPE's Seeds) starts at 8 and can be typed into freely; "auto" is gone. **Apply ±σ to bounds** and **Load into model** apply to every dataset of a joint run, each in its own model: the Dataset box is gone. The error scale is no longer ticked automatically after a fit.
- **Multistart:** the local fit "short DE" is now **DE** and uses the DE page's settings (Max iter, Pop, Tol, mutation, strategy, polish) instead of fixed ones. The minima are listed in the multistart's fit tab, where you select one and send it to the Simulation tab. The table on the multistart page, its buttons and the DREAM start "multistart minimum" are gone: to start DREAM there, send the minimum and use **Start = current values**. DE's **Restarts** box is gone too.
- **Licorne resolution:** when the channels are not on the same Q points, the Q-grid dialog now suggests a channel (the one covering the others' Q range, unless another is much denser).
- Fit and DREAM tabs are numbered by their place among the tabs open (1, 2, 3…), renumbered when one is closed.
- The **independent** interface order computes profiles about 25 % faster.

Everything in 0.5.1 (export datasets as CSV), 0.5 (a Q range per model, monochromator resolution, faster fits with the Licorne resolution), 0.4.1 and 0.4 is included.

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
