"""
Entry point: main window holding one tab per tool.

Run with   python main.py

A session file is the model JSON (Stack.to_dict) with the Simulation tab's
settings under 'simulation' and the Experimental tab's under 'experimental'
(datasets by file path, display and fit settings).  Opening a plain model
file only replaces the model.

File -> Import Licorne session reads a saved Licorne session folder
(model.licorne_io.load_licorne_session) into the same form: its rexp<k>.dat
channels as one dataset, Licorne's rtheory<k>.dat as a second one drawn as
lines.
"""

import json
import math
import os
import sys

import numpy as np
from PyQt6 import QtGui, QtWidgets

import style
from model.licorne_io import load_licorne_session
from model.polarisation import default_vectors
from tabs import ExperimentalTab, SimulationTab
from tabs.experimental import file_ref
from tabs.geometry import GeometryTab

APP_ICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icons',
                        'app.png')
MODEL_FILTER = 'Session / model files (*.json);;All files (*)'
# The native macOS dialog greys out *.json files with this filter; use Qt's own.
DIALOG_OPTIONS = QtWidgets.QFileDialog.Option.DontUseNativeDialog


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('SANR')
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)

        self.simulation = SimulationTab()
        self.tabs.addTab(self.simulation, 'Simulation')
        self.experimental = ExperimentalTab(simulation=self.simulation)
        self.tabs.addTab(self.experimental, 'Experimental data')
        self.geometry = GeometryTab(self.simulation)
        self.tabs.addTab(self.geometry, 'Geometry')

        menu = self.menuBar().addMenu('&File')
        menu.addAction('&Open session…', QtGui.QKeySequence.StandardKey.Open,
                       self.open_session)
        menu.addAction('&Save session…', QtGui.QKeySequence.StandardKey.Save,
                       self.save_session)
        menu.addSeparator()
        menu.addAction('&Import experimental data…', self.import_data)
        menu.addAction('Import &Licorne session…', self.import_licorne)

        self.resize(1600, 950)

    def closeEvent(self, ev):
        self.experimental.shutdown()
        super().closeEvent(ev)

    def open_session(self):
        if self.experimental.busy():
            QtWidgets.QMessageBox.information(
                self, 'Open session',
                'Stop the running fit or sampling first.')
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open session', '', MODEL_FILTER,
            options=DIALOG_OPTIONS)
        if not path:
            return
        try:
            with open(path) as f:
                d = json.load(f)
            missing = self._restore(d, os.path.dirname(path))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, 'Open session', 'Could not load %s:\n%s' % (path, exc))
            return
        self.setWindowTitle('SANR — %s' % os.path.basename(path))
        if missing:
            QtWidgets.QMessageBox.warning(
                self, 'Open session', 'These channels could not be read '
                'again and were left out:\n\n' + '\n'.join(missing))

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
        if not folder:
            return
        try:
            lic = load_licorne_session(folder)
            if not lic['channels']:
                raise ValueError('no measured channel (rexp<k>.dat):\n'
                                 + '\n'.join(lic['notes']))
            d = self._licorne_state(lic, folder)
            missing = self._restore(d, folder)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, 'Import Licorne session',
                'Could not import %s:\n%s' % (folder, exc))
            return
        self.tabs.setCurrentWidget(self.experimental)
        self.setWindowTitle('SANR — %s (Licorne)' % lic['name'])
        notes = lic['notes'] + missing
        if notes:
            QtWidgets.QMessageBox.warning(
                self, 'Import Licorne session',
                'Imported, with these notes:\n\n' + '\n\n'.join(notes))

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
                                   start=1, pol=own(c))
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
        exp['fit']['dataset'] = lic['name']
        d['experimental'] = exp
        return d

    def import_data(self):
        self.tabs.setCurrentWidget(self.experimental)
        self.experimental.import_data()

    def save_session(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save session', 'session.json', MODEL_FILTER,
            options=DIALOG_OPTIONS)
        if not path:
            return
        # a name typed without .json gets it (the dialog only asked about
        # overwriting the name as typed)
        if not path.lower().endswith('.json'):
            path += '.json'
            if os.path.exists(path) and QtWidgets.QMessageBox.question(
                    self, 'Save session', '%s already exists.\nReplace it?'
                    % os.path.basename(path)) != \
                    QtWidgets.QMessageBox.StandardButton.Yes:
                return
        try:
            d = self.simulation.session_state()
            d['experimental'] = self.experimental.session_state(
                os.path.dirname(os.path.abspath(path)))
            text = json.dumps(d, indent=2, default=_plain)
            with open(path, 'w') as f:
                f.write(text)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, 'Save session', 'Could not save %s:\n%s' % (path, exc))
            return
        self.setWindowTitle('SANR — %s' % os.path.basename(path))


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


def main():
    set_macos_app_name('SANR')          # before the QApplication
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('SANR')
    # every window's icon, and the Dock / task-bar icon
    app.setWindowIcon(QtGui.QIcon(APP_ICON))
    style.apply(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
