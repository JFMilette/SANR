"""
Entry point: main window holding one tab per tool.

Run with   python main.py

A session file is the model JSON (Stack.to_dict) with the Simulation tab's
settings under 'simulation' and the Experimental tab's under 'experimental'
(datasets by file path, display and fit settings).  Opening a plain model
file only replaces the model.
"""

import json
import os
import sys

import numpy as np
from PyQt6 import QtGui, QtWidgets

import style
from tabs import ExperimentalTab, SimulationTab

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

        menu = self.menuBar().addMenu('&File')
        menu.addAction('&Open session…', QtGui.QKeySequence.StandardKey.Open,
                       self.open_session)
        menu.addAction('&Save session…', QtGui.QKeySequence.StandardKey.Save,
                       self.save_session)
        menu.addSeparator()
        menu.addAction('&Import experimental data…', self.import_data)

        self.resize(1600, 950)

    def closeEvent(self, ev):
        self.experimental.shutdown()
        super().closeEvent(ev)

    def open_session(self):
        if self.experimental._fit is not None:
            QtWidgets.QMessageBox.information(
                self, 'Open session', 'Stop the running fit first.')
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open session', '', MODEL_FILTER,
            options=DIALOG_OPTIONS)
        if not path:
            return
        try:
            with open(path) as f:
                d = json.load(f)
            self.simulation.restore_state(d)
            missing = self.experimental.restore_state(
                d['experimental'], os.path.dirname(path)) \
                if 'experimental' in d else []
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, 'Open session', 'Could not load %s:\n%s' % (path, exc))
            return
        self.setWindowTitle('SANR — %s' % os.path.basename(path))
        if missing:
            QtWidgets.QMessageBox.warning(
                self, 'Open session', 'These channels could not be read '
                'again and were left out:\n\n' + '\n'.join(missing))

    def import_data(self):
        self.tabs.setCurrentWidget(self.experimental)
        self.experimental.import_data()

    def save_session(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save session', 'session.json', MODEL_FILTER,
            options=DIALOG_OPTIONS)
        if not path:
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


def main():
    app = QtWidgets.QApplication(sys.argv)
    style.apply(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
