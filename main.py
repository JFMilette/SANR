"""
Entry point: main window holding one tab per tool.

Run with   python main.py
"""

import sys

import pyqtgraph as pg
from PyQt6 import QtGui, QtWidgets

from tabs import ExperimentalTab, SimulationTab

MODEL_FILTER = 'Model files (*.json);;All files (*)'
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
        self.experimental = ExperimentalTab()
        self.tabs.addTab(self.experimental, 'Experimental data')

        menu = self.menuBar().addMenu('&File')
        menu.addAction('&Open model…', QtGui.QKeySequence.StandardKey.Open,
                       self.open_model)
        menu.addAction('&Save model…', QtGui.QKeySequence.StandardKey.Save,
                       self.save_model)
        menu.addSeparator()
        menu.addAction('&Import experimental data…', self.import_data)

        self.resize(1600, 950)

    def open_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open model', '', MODEL_FILTER,
            options=DIALOG_OPTIONS)
        if path:
            try:
                self.simulation.load_model(path)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self, 'Open model', 'Could not load %s:\n%s' % (path, exc))

    def import_data(self):
        self.tabs.setCurrentWidget(self.experimental)
        self.experimental.import_data()

    def save_model(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save model', 'model.json', MODEL_FILTER,
            options=DIALOG_OPTIONS)
        if path:
            try:
                self.simulation.save_model(path)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self, 'Save model', 'Could not save %s:\n%s' % (path, exc))


def main():
    pg.setConfigOptions(antialias=True, background='w', foreground='k')
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
