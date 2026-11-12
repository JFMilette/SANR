"""
Entry point: main window holding one tab per tool.

Run with   python main.py

A session file is the model JSON (Stack.to_dict) with the Simulation tab's
settings under 'simulation' and the Experimental tab's under 'experimental'
(datasets by file path, display and fit settings), and the ended fit and
DREAM tabs under 'fit_tabs' (fit.panel SESSION), reopened after the tabs
already open.  Opening a plain model file only replaces the model.
Sessions are saved as .zip by default: that JSON and a copy of every
data file it reads again (session_file), so it opens without the
data's folders; a .json session (data files by path) is still read and
written.

Settings -> Dark mode switches between the dark and light themes (style.py);
the choice is remembered for the next launch.  Settings -> Line colours (or
"Colours…" beside a reflectivity plot) sets the channels' colours in every
reflectivity plot (line_colours.py), remembered the same way.

The mouse wheel never changes a spin box or a combo box (NoWheelOnInputs): it
scrolls the panel under the pointer, so a value cannot be changed by accident.

The tabs: Geometry, Simulation (shown first, starting with no layer),
Experimental data, Fit (the fit box, fit.panel).  Each fit
(DE, L-BFGS-B, multistart) and DREAM run opens a tab of its own after the
four tools, named short (fit.panel.FitPanel._tab_name); older ones
stay until their × closes them (FitPanel.close_page), not while they run.

Built as a program (make_windows_installer.py, PyInstaller), the session
dialogs open in Documents/SANR, created when needed, as the program's own
folder is not writable; `SANR --self-test <file>` checks the build
(selftest.py).  Once a session is opened or saved, both dialogs open in
its folder for the rest of the run (MainWindow.session_dir).

File -> Import Licorne session reads a saved Licorne session folder
(model.licorne_io.load_licorne_session) into the same form: its rexp<k>.dat
channels as one dataset, Licorne's rtheory<k>.dat as a second one drawn as
lines.  File -> Import profile reads a layer table (model.profile_io) as the
model's layers, the other settings kept.
"""

import json
import math
import multiprocessing
import os
import sys

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

import line_colours
import session_file
import style
from model.licorne_io import load_licorne_session
from model.polarisation import default_vectors
from model.profile_io import load_profile
from tabs import ExperimentalTab, SimulationTab
from tabs.experimental import file_ref
from tabs.geometry import GeometryTab
from tabs.simulation import empty_stack

APP_ICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icons',
                        'app.png')
# where the session dialogs open (session_file.sessions_dir)
SESSIONS_DIR = session_file.sessions_dir()
# the folder a session's data paths are taken relative to when comparing
# states (MainWindow._state_key): any fixed one
_KEY_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILTER = 'Sessions (*.zip *.json);;All files (*)'
SAVE_FILTER = ('Session with its data (*.zip);;'
               'Session, data files by path (*.json)')
# The native macOS dialog greys out *.json files with this filter; use Qt's own.
DIALOG_OPTIONS = QtWidgets.QFileDialog.Option.DontUseNativeDialog


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('SANR')
        # where the session dialogs open: SESSIONS_DIR until a session is
        # opened or saved, then that session's folder (this run only)
        self.session_dir = SESSIONS_DIR
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)

        # no layer to start with (tabs.simulation.empty_stack)
        self.simulation = SimulationTab(empty_stack())
        self.experimental = ExperimentalTab(simulation=self.simulation)
        self.geometry = GeometryTab(self.simulation)
        self.tabs.addTab(self.geometry, 'Geometry')
        self.tabs.addTab(self.simulation, 'Simulation')
        self.tabs.addTab(self.experimental, 'Experimental data')
        self.tabs.addTab(self.experimental.fit_page, 'Fit')
        self.tabs.setCurrentWidget(self.simulation)     # the one shown first
        # fits and DREAM runs open as tabs after these, with a close button
        self.tabs.setTabsClosable(True)
        for i in range(self.tabs.count()):
            for side in (QtWidgets.QTabBar.ButtonPosition.LeftSide,
                         QtWidgets.QTabBar.ButtonPosition.RightSide):
                self.tabs.tabBar().setTabButton(i, side, None)
        self.experimental.page_tabs = self.tabs
        self.tabs.tabCloseRequested.connect(
            lambda i: self.experimental.close_page(self.tabs.widget(i)))

        menu = self.menuBar().addMenu('&File')
        menu.addAction('&Open session…', QtGui.QKeySequence.StandardKey.Open,
                       self.open_session)
        menu.addAction('&Save session…', QtGui.QKeySequence.StandardKey.Save,
                       self.save_session)
        menu.addSeparator()
        menu.addAction('&Import experimental data…', self.import_data)
        menu.addAction('Import &Licorne session…', self.import_licorne)
        menu.addAction('Import &profile…', self.import_profile)

        menu = self.menuBar().addMenu('&Settings')
        self.dark_mode = menu.addAction('&Dark mode')
        self.dark_mode.setCheckable(True)
        self.dark_mode.setChecked(style.theme == 'dark')
        # keep it in this menu (macOS moves "settings"-like actions)
        self.dark_mode.setMenuRole(QtGui.QAction.MenuRole.NoRole)
        self.dark_mode.toggled.connect(self.set_dark_mode)
        act = menu.addAction('&Line colours…',
                             lambda: line_colours.show_dialog(self))
        act.setMenuRole(QtGui.QAction.MenuRole.NoRole)

        self.resize(1600, 950)
        # the empty session counts as saved: taken again once the window
        # has settled (models made for datasets, first curves)
        self._mark_saved()
        QtCore.QTimer.singleShot(0, self._mark_saved)

    def set_dark_mode(self, on):
        theme = 'dark' if on else 'light'
        style.apply(QtWidgets.QApplication.instance(), theme)
        settings().setValue('theme', theme)

    # -- unsaved changes ------------------------------------------------------
    def _state_key(self):
        """What a session would save, to compare with the one saved or
        opened last: the model and data settings as JSON, and the fit and
        DREAM tabs open; None if it cannot be had."""
        try:
            d = self.simulation.session_state()
            d['experimental'] = self.experimental.session_state(_KEY_BASE)
            tabs = tuple(id(self.tabs.widget(i))
                         for i in range(self.tabs.count()))
            return json.dumps(d, sort_keys=True, default=_plain), tabs
        except Exception:
            return None

    def _mark_saved(self):
        """What the window holds now is saved (or was just opened)."""
        self._saved_key = self._state_key()

    def _keep_or_lose(self, title):
        """Before the session is closed or replaced: True to go on
        (nothing unsaved, or saved, or the user chose to discard it), False
        to stay.  A window not shown has no one to ask (tests)."""
        if not self.isVisible():
            return True
        key = self._state_key()
        if key is not None and key == self._saved_key:
            return True
        B = QtWidgets.QMessageBox.StandardButton
        b = QtWidgets.QMessageBox.question(
            self, title, 'The session has changes that are not saved.\n'
            'Save them first?', B.Save | B.Discard | B.Cancel, B.Save)
        if b == B.Save:
            return self.save_session()
        return b == B.Discard

    def closeEvent(self, ev):
        if not self._keep_or_lose('Quit SANR'):
            ev.ignore()
            return
        self.experimental.shutdown()
        super().closeEvent(ev)

    def _snapshot(self):
        """The session as it is (the model and data), for _restore to go
        back to if loading another one fails half way."""
        d = self.simulation.session_state()
        d['experimental'] = self.experimental.session_state(_KEY_BASE)
        return d

    def _replace(self, d, base, snapshot):
        """_restore(d, base); if it fails, the session it replaced is put
        back (snapshot, _snapshot) and the error raised."""
        try:
            return self._restore(d, base)
        except Exception:
            try:
                self._restore(snapshot, _KEY_BASE)
            except Exception:
                pass                    # the error that matters is the first
            raise

    def open_session(self):
        if self.experimental.busy():
            QtWidgets.QMessageBox.information(
                self, 'Open session',
                'Stop the running fit or sampling first.')
            return
        if not self._keep_or_lose('Open session'):
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open session', self.session_dir, MODEL_FILTER,
            options=DIALOG_OPTIONS)
        if not path:
            return
        self.session_dir = os.path.dirname(os.path.abspath(path))
        popup = self._opening_popup(path)
        try:
            try:
                if session_file.is_zip_session(path):
                    d, base = session_file.read_session(path)
                else:
                    with open(path, encoding='utf-8') as f:
                        d = json.load(f)
                    base = os.path.dirname(path)
                missing = self._replace(d, base, self._snapshot())
            except Exception as exc:
                popup.close()
                QtWidgets.QMessageBox.critical(
                    self, 'Open session',
                    'Could not load %s:\n%s' % (path, exc))
                return
            self.setWindowTitle('SANR — %s' % os.path.basename(path))
            QtWidgets.QApplication.processEvents()
            lost = self.experimental.restore_fit_tabs(d.get('fit_tabs', []))
            self._wait_for_tabs()
            self._mark_saved()
        finally:
            popup.close()
        if missing:
            QtWidgets.QMessageBox.warning(
                self, 'Open session', 'These channels could not be read '
                'again and were left out:\n\n' + '\n'.join(missing))
        if lost:
            QtWidgets.QMessageBox.warning(
                self, 'Open session', 'These fit tabs could not be opened '
                'again:\n\n' + '\n'.join(lost))

    def _opening_popup(self, path):
        """'Opening session…', shown (no button) until the session is
        ready, the DREAM tabs' summaries included."""
        popup = QtWidgets.QProgressDialog(
            'Opening session %s…' % os.path.basename(path), None, 0, 0, self)
        popup.setWindowTitle('Open session')
        popup.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        popup.setMinimumDuration(0)
        popup.setCancelButton(None)
        popup.setAutoClose(False)
        popup.setAutoReset(False)
        popup.show()
        QtWidgets.QApplication.processEvents()
        return popup

    def _wait_for_tabs(self):
        """Let the window run (the popup turning) until the reopened DREAM
        tabs have computed their summaries in the background."""
        pending = self.experimental.computing_tabs()
        if not pending:
            return
        loop = QtCore.QEventLoop()

        def check():
            if not any(w.computing for w in pending):
                loop.quit()
        for w in pending:
            w.ready.connect(check)
            w.destroyed.connect(loop.quit)
        if any(w.computing for w in pending):
            loop.exec()

    def _restore(self, d, base):
        """Load session dict d (files relative to folder `base`); returns the
        channels that could not be read."""
        self.simulation.restore_state(d)
        if 'experimental' not in d:
            return []
        return self.experimental.restore_state(d['experimental'], base)

    def import_licorne(self):
        """Replace the session with a saved Licorne session folder: model,
        fit bounds, background, resolution, the measured channels and
        Licorne's fitted curves (model.licorne_io.load_licorne_session)."""
        if self.experimental.busy():
            QtWidgets.QMessageBox.information(
                self, 'Import Licorne session',
                'Stop the running fit or sampling first.')
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Import Licorne session folder', '',
            options=DIALOG_OPTIONS
            | QtWidgets.QFileDialog.Option.ShowDirsOnly)
        if not folder or not self._keep_or_lose('Import Licorne session'):
            return
        try:
            lic = load_licorne_session(folder)
            if not lic['channels']:
                raise ValueError('no measured channel (rexp<k>.dat):\n'
                                 + '\n'.join(lic['notes']))
            d = self._licorne_state(lic, folder)
            missing = self._replace(d, folder, self._snapshot())
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, 'Import Licorne session',
                'Could not import %s:\n%s' % (folder, exc))
            return
        self.tabs.setCurrentWidget(self.experimental)
        self.setWindowTitle('SANR — %s (Licorne)' % lic['name'])
        self._mark_saved()               # as a session just opened
        notes = lic['notes'] + missing
        if notes:
            QtWidgets.QMessageBox.warning(
                self, 'Import Licorne session',
                'Imported, with these notes:\n\n' + '\n\n'.join(notes))

    def import_profile(self):
        """Replace the model's layers with a profile table
        (model.profile_io.load_profile); every other setting of the
        Simulation tab stays."""
        if self.experimental.busy():
            QtWidgets.QMessageBox.information(
                self, 'Import profile',
                'Stop the running fit or sampling first.')
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Import profile', '',
            'Profiles (*.dat *.txt);;All files (*)', options=DIALOG_OPTIONS)
        if not path:
            return
        try:
            d = self.simulation.session_state()
            d['layers'] = [dict(vars(l)) for l in load_profile(path)]
            d['ties'] = []
            d['simulation']['selected_layer'] = 1
            self.simulation.restore_state(d)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, 'Import profile',
                'Could not import %s:\n%s' % (path, exc))
            return
        self.tabs.setCurrentWidget(self.simulation)

    def _licorne_state(self, lic, folder):
        """Session dict of a load_licorne_session result; the tabs' display
        and fit settings stay as they are."""
        d = lic['stack'].to_dict()
        lo, hi = lic['qrange']
        sim = self.simulation.session_state()['simulation']
        # round outwards to the boxes' decimals, keeping every point
        f_lo = 10 ** self.simulation.qmin.decimals()
        f_hi = 10 ** self.simulation.qmax.decimals()
        sim.update(qmin=math.floor(lo * f_lo) / f_lo,
                   qmax=math.ceil(hi * f_hi) / f_hi,
                   polarisation=default_vectors(), selected_layer=1)
        d['simulation'] = sim
        defaults = default_vectors()

        def own(c):                     # None: follow the Polarisation box
            return None if c['pol'] == defaults[c['channel']] else c['pol']

        data = {c['channel']: dict(file=file_ref(c['path'], folder),
                                   q_file=file_ref(lic['q_path'], folder),
                                   start=1, pol=own(c), norm=c['norm'],
                                   q_mult=c['q_mult'], r_mult=c['r_mult'],
                                   fit=c['fit'])
                for c in lic['channels']}
        theory = {c['channel']: dict(
            values=dict(Q=c['Q'], R=c['R'], dR=np.zeros_like(c['R']),
                        dQ=np.zeros_like(c['R'])),
            path=c['path'], pol=own(c)) for c in lic['theory']}
        sets = [dict(name=lic['name'], visible=True, live=False, line=False,
                     channels=data)]
        if theory:
            sets.append(dict(name=lic['name'] + ' (Licorne fit)',
                             visible=True, live=False, line=True,
                             channels=theory))
        exp = self.experimental.session_state(folder)
        exp['sets'] = sets
        exp['fit'].update(dataset=lic['name'], joint=None)
        d['experimental'] = exp
        return d

    def import_data(self):
        self.tabs.setCurrentWidget(self.experimental)
        self.experimental.import_data()

    def save_session(self):
        """Save the session to a file the user picks; True once saved."""
        path, chosen = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save session',
            os.path.join(self.session_dir, 'session.zip'), SAVE_FILTER,
            options=DIALOG_OPTIONS)
        if not path:
            return False
        self.session_dir = os.path.dirname(os.path.abspath(path))
        # a name typed without .zip / .json gets the chosen filter's (the
        # dialog only asked about overwriting the name as typed)
        ext = '.json' if '*.json' in chosen else '.zip'
        if not path.lower().endswith(('.zip', '.json')):
            path += ext
            if os.path.exists(path) and QtWidgets.QMessageBox.question(
                    self, 'Save session', '%s already exists.\nReplace it?'
                    % os.path.basename(path)) != \
                    QtWidgets.QMessageBox.StandardButton.Yes:
                return False
        try:
            d = self.simulation.session_state()
            d['experimental'] = self.experimental.session_state(
                os.path.dirname(os.path.abspath(path)))
            d['fit_tabs'], running = self.experimental.fit_tabs_state()
            if path.lower().endswith('.json'):
                lost = session_file.only_in_session(d)
                if lost and QtWidgets.QMessageBox.question(
                        self, 'Save session', 'These data files exist only '
                        'inside the .zip session they were opened from, and '
                        'a .json session would not find them again:\n\n'
                        + '\n'.join(lost) + '\n\nSave as .json anyway?') != \
                        QtWidgets.QMessageBox.StandardButton.Yes:
                    return False
                text = json.dumps(d, indent=2, default=_plain)
                # written next to it, then put in its place: a failed write
                # leaves the session that was there whole
                part = path + '.part'
                try:
                    with open(part, 'w', encoding='utf-8') as f:
                        f.write(text)
                    os.replace(part, path)
                finally:
                    if os.path.exists(part):
                        os.remove(part)
            else:
                session_file.write_session(path, d, default=_plain)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, 'Save session', 'Could not save %s:\n%s' % (path, exc))
            return False
        self.setWindowTitle('SANR — %s' % os.path.basename(path))
        self._mark_saved()
        if running:
            QtWidgets.QMessageBox.information(
                self, 'Save session', 'Saved without these tabs, whose fit '
                'or run is still going on:\n\n' + '\n'.join(running))
        return True


def settings():
    """Preferences kept between launches (the theme, the line colours)."""
    return QtCore.QSettings('SANR', 'SANR')


class NoWheelOnInputs(QtCore.QObject):
    """Application event filter: the wheel never changes a spin box value
    or a combo box choice.  The event is ignored instead, so Qt hands it to
    the parent and the panel under the pointer scrolls as if the box were
    not there.  An open combo box list still scrolls (it is not a
    QComboBox)."""

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Type.Wheel and \
                isinstance(obj, (QtWidgets.QAbstractSpinBox,
                                 QtWidgets.QComboBox)):
            ev.ignore()
            return True
        return False


def _plain(obj):
    """json.dump fallback: numpy arrays and scalars as plain numbers."""
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    raise TypeError('%s is not JSON serialisable' % type(obj).__name__)


def set_macos_app_name(name):
    """Show `name` instead of the interpreter's (python3.12) in the macOS
    menu bar and Dock: macOS takes it from CFBundleName of the main bundle,
    which a plain python executable lacks.  Set through the Objective-C
    runtime (no pyobjc needed) before the QApplication exists; does nothing
    elsewhere and never stops the app from starting."""
    if sys.platform != 'darwin':
        return
    try:
        import ctypes
        import ctypes.util
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library('objc'))
        ctypes.cdll.LoadLibrary(ctypes.util.find_library('Foundation'))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        P = ctypes.c_void_p

        def send(*argtypes, restype=P):        # typed objc_msgSend
            return ctypes.CFUNCTYPE(restype, P, P, *argtypes)(
                ('objc_msgSend', objc))

        def sel(s):
            return objc.sel_registerName(s.encode())

        def nsstring(s):
            return send(ctypes.c_char_p)(objc.objc_getClass(b'NSString'),
                                         sel('stringWithUTF8String:'),
                                         s.encode())

        bundle = send()(objc.objc_getClass(b'NSBundle'), sel('mainBundle'))
        info = send()(bundle, sel('infoDictionary'))
        if info and send(P, restype=ctypes.c_bool)(
                info, sel('respondsToSelector:'), sel('setObject:forKey:')):
            for key in ('CFBundleName', 'CFBundleDisplayName'):
                send(P, P, restype=None)(info, sel('setObject:forKey:'),
                                         nsstring(name), nsstring(key))
        info = send()(objc.objc_getClass(b'NSProcessInfo'), sel('processInfo'))
        send(P, restype=None)(info, sel('setProcessName:'), nsstring(name))
    except Exception:                   # cosmetic only
        pass


def set_windows_app_id(app_id):
    """Group SANR's windows under its own task-bar entry, with its icon:
    Windows takes the task-bar icon from the process's AppUserModelID,
    which for `python main.py` is python.exe's (the Python icon, whatever
    setWindowIcon says).  Set before any window exists; does nothing
    elsewhere and never stops the app from starting."""
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            ctypes.c_wchar_p(app_id))
    except Exception:                   # cosmetic only
        pass


def main():
    set_macos_app_name('SANR')          # before the QApplication
    set_windows_app_id('uOttawa.SANR')
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('SANR')
    # every window's icon, and the Dock / task-bar icon
    app.setWindowIcon(QtGui.QIcon(APP_ICON))
    no_wheel = NoWheelOnInputs(app)
    app.installEventFilter(no_wheel)
    style.apply(app, settings().value('theme', style.DEFAULT))
    try:
        line_colours.set_colours(json.loads(
            settings().value('line_colours', '{}')))
    except (TypeError, ValueError, AttributeError):
        pass                            # unreadable: the defaults
    line_colours.notifier().changed.connect(
        lambda: settings().setValue('line_colours',
                                    json.dumps(line_colours.colours())))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    # a built program's worker processes (fit.workers) run this file again:
    # this hands them to multiprocessing instead of starting another SANR
    multiprocessing.freeze_support()
    if len(sys.argv) == 3 and sys.argv[1] == '--self-test':
        import selftest
        sys.exit(selftest.run(sys.argv[2]))
    main()
