"""
The "Datasets to fit" dialog of the Fit tool box (Select… on the general
page): the datasets fitted together, and how each parameter ticked Fit in
the Simulation tab is fitted (fit.joint MODES): shared by the datasets (the
default), its own value in each, or not fitted (kept at its value in the
Simulation tab).  The table has a row per parameter: its "All datasets"
column sets the mode in every ticked dataset at once ("Own value" there:
a value of its own in each), and a column per ticked dataset changes one
of them ("All datasets" then reads Mixed).  Modes only matter for two
datasets or more; one dataset ticked is fitted alone.
"""

from PyQt6 import QtCore, QtWidgets

from tabs.simulation import shown_names

# fit.joint MODES, as the combos show them
MODE_LABELS = [('shared', 'Shared'),
               ('own', 'Own value'),
               ('fixed', 'Not fitted')]
MIXED = 'mixed'             # "All datasets" when the datasets differ


class DatasetsDialog(QtWidgets.QDialog):
    """Pick datasets and the parameters' modes (see the module
    docstring)."""

    def __init__(self, names, chosen, modes, params, parent=None):
        """names: the datasets that can be fitted; chosen: those ticked;
        modes: {dataset: {parameter name: mode}} (fit.joint); params: the
        parameter names (fit.problem.parameter_names)."""
        super().__init__(parent)
        self.setWindowTitle('Datasets to fit')
        self.params = list(params)
        self.modes = {n: dict((modes or {}).get(n) or {}) for n in names}

        self.list = QtWidgets.QListWidget()
        self.list.setToolTip('Tick the datasets fitted together')
        for n in names:
            it = QtWidgets.QListWidgetItem(n)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.CheckState.Checked if n in chosen
                             else QtCore.Qt.CheckState.Unchecked)
            self.list.addItem(it)
        self.list.itemChanged.connect(lambda *_: self._build_table())

        self.table = QtWidgets.QTableWidget(len(self.params), 0)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.all_combos = []           # "All datasets" combo of each row
        self.cells = {}                # (dataset, parameter) -> its combo
        self.note = QtWidgets.QLabel(wordWrap=True)
        self.note.setProperty('role', 'muted')

        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel('<b>Datasets</b>'))
        left.addWidget(self.list, 1)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel('<b>Parameters</b>'))
        right.addWidget(self.table, 1)
        right.addWidget(self.note)
        row = QtWidgets.QHBoxLayout()
        row.addLayout(left, 1)
        row.addLayout(right, 3)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        v = QtWidgets.QVBoxLayout(self)
        v.addLayout(row, 1)
        v.addWidget(QtWidgets.QLabel(
            '<i>Shared: one value fitted to every dataset sharing it.  Own '
            'value: fitted to each dataset alone.  Not fitted: kept at its '
            'value in the Simulation tab.  "All datasets" sets every '
            'dataset; a dataset\'s column changes that one only.</i>',
            wordWrap=True))
        v.addWidget(buttons)
        self.resize(760, 420)
        self._build_table()

    def chosen(self):
        """The ticked datasets, in the list's order."""
        return [self.list.item(r).text() for r in range(self.list.count())
                if self.list.item(r).checkState()
                == QtCore.Qt.CheckState.Checked]

    def selection(self):
        """{'sets': the ticked datasets, 'modes': {dataset: {parameter:
        mode}} of theirs that are not 'shared'}."""
        sets = self.chosen()
        return {'sets': sets,
                'modes': {n: {a: m for a, m in self.modes[n].items()
                              if m != 'shared'} for n in sets
                          if any(m != 'shared'
                                 for m in self.modes[n].values())}}

    def mode(self, dataset, param):
        return self.modes[dataset].get(param, 'shared')

    @staticmethod
    def _combo(mixed=False):
        c = QtWidgets.QComboBox()
        for key, label in MODE_LABELS:
            c.addItem(label, key)
        if mixed:                      # shown, never picked
            c.addItem('Mixed', MIXED)
            c.model().item(c.count() - 1).setEnabled(False)
        return c

    def _build_table(self):
        """A row per parameter: "All datasets", then a column per ticked
        dataset; usable with two ticked datasets or more."""
        sets = self.chosen()
        t = self.table
        t.clearContents()
        t.setColumnCount(2 + len(sets))
        t.setHorizontalHeaderLabels(['Parameter', 'All datasets'] + sets)
        self.all_combos, self.cells = [], {}
        for r, p in enumerate(self.params):
            t.setItem(r, 0, QtWidgets.QTableWidgetItem(shown_names(p)))
            c = self._combo(mixed=True)
            c.setToolTip('The mode in every ticked dataset')
            c.currentIndexChanged.connect(
                lambda _, p=p, c=c: self._set_all(p, c.currentData()))
            t.setCellWidget(r, 1, c)
            self.all_combos.append(c)
            for k, n in enumerate(sets):
                c = self._combo()
                c.setCurrentIndex(c.findData(self.mode(n, p)))
                c.setToolTip('The mode in %s only' % n)
                c.currentIndexChanged.connect(
                    lambda _, n=n, p=p, c=c: self._set_one(n, p,
                                                           c.currentData()))
                t.setCellWidget(r, 2 + k, c)
                self.cells[n, p] = c
            self._show_all(r)
        # names in full; the table scrolls sideways if the window is narrow
        for k in range(t.columnCount()):
            t.resizeColumnToContents(k)
        # wide enough for every column (the dialog grows), up to 1000 px
        need = sum(t.columnWidth(k) for k in range(t.columnCount())) + \
            2 * t.frameWidth() + t.verticalScrollBar().sizeHint().width()
        t.setMinimumWidth(min(need, 1000))
        on = len(sets) > 1
        t.setEnabled(on)
        if not self.params:
            self.note.setText('No parameter is ticked Fit in the Simulation '
                              'tab.')
        elif not on:
            self.note.setText('Tick two datasets or more to fit them '
                              'together; one dataset is fitted alone, with '
                              'every parameter ticked Fit.')
        else:
            self.note.setText('')

    def _show_all(self, r):
        """Row r's "All datasets": the ticked datasets' common mode, or
        Mixed."""
        p, c = self.params[r], self.all_combos[r]
        found = {self.mode(n, p) for n in self.chosen()}
        c.blockSignals(True)
        c.setCurrentIndex(c.findData(found.pop() if len(found) == 1
                                     else 'shared' if not found else MIXED))
        c.blockSignals(False)

    def _set_all(self, p, mode):
        if mode == MIXED:
            return
        for n in self.chosen():
            self.modes[n][p] = mode
            c = self.cells[n, p]
            c.blockSignals(True)
            c.setCurrentIndex(c.findData(mode))
            c.blockSignals(False)

    def _set_one(self, n, p, mode):
        self.modes[n][p] = mode
        self._show_all(self.params.index(p))

    def accept(self):
        if not self.chosen():
            QtWidgets.QMessageBox.information(self, 'Datasets to fit',
                                              'Tick at least one dataset.')
            return
        super().accept()
