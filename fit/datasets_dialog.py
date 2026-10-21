"""
The "Datasets to fit" dialog of the Fit tool box (Select… on the general
page): the datasets fitted together, and how each parameter ticked Fit in
the Simulation tab is fitted (fit.joint MODES): shared by the datasets (the
default), its own value in each, or not fitted.  The table has a row per
parameter: its "All datasets" column sets the mode in every ticked dataset
at once ("Own value" there: a value of its own in each), and a column per
ticked dataset changes one of them ("All datasets" then reads Mixed).
Modes only matter for two datasets or more; one dataset ticked is fitted
alone.

NOT FITTED -- a box beside "Not fitted" holds the value the parameter keeps
in that dataset (fit.joint `values`), in the Simulation tab's units; left
empty it keeps the Simulation tab's current value, which the box shows
greyed.  The "All datasets" box writes its value into every ticked
dataset (empty: back to the Simulation tab's value); it shows their value
when they all have the same one.  A value that is not a number is not
taken (the box turns red).
"""

from PyQt6 import QtCore, QtWidgets

import style
from tabs.simulation import shown_names

# fit.joint MODES, as the combos show them
MODE_LABELS = [('shared', 'Shared'),
               ('own', 'Own value'),
               ('fixed', 'Not fitted')]
MIXED = 'mixed'             # "All datasets" when the datasets differ


class DatasetsDialog(QtWidgets.QDialog):
    """Pick datasets and the parameters' modes (see the module
    docstring)."""

    def __init__(self, names, chosen, modes, params, parent=None,
                 values=None, info=None):
        """names: the datasets that can be fitted; chosen: those ticked;
        modes: {dataset: {parameter name: mode}} (fit.joint); params: the
        parameter names (fit.problem.parameter_names); values: {dataset:
        {parameter name: value}} of the 'fixed' ones given a value (stored
        units); info: {parameter name: (scale, unit, current value)} for
        the boxes (scale: stored -> shown; default 1, '', no value)."""
        super().__init__(parent)
        self.setWindowTitle('Datasets to fit')
        self.params = list(params)
        self.modes = {n: dict((modes or {}).get(n) or {}) for n in names}
        self.values = {n: dict((values or {}).get(n) or {}) for n in names}
        self.info = dict(info or {})

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
        self.all_edits = []            # and its value box
        self.cells = {}                # (dataset, parameter) -> its combo
        self.edits = {}                # (dataset, parameter) -> value box
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
            'value: fitted to each dataset alone.  Not fitted: kept at the '
            'value typed beside it, or at its value in the Simulation tab '
            'when the box is empty (shown greyed).  "All datasets" sets '
            'every dataset; a dataset\'s column changes that one only.</i>',
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
        mode}} of theirs that are not 'shared', 'values': {dataset:
        {parameter: value}} of their 'fixed' ones given a value}."""
        sets = self.chosen()
        values = {n: {a: v for a, v in self.values[n].items()
                      if self.mode(n, a) == 'fixed' and a in self.params}
                  for n in sets}
        return {'sets': sets,
                'modes': {n: {a: m for a, m in self.modes[n].items()
                              if m != 'shared'} for n in sets
                          if any(m != 'shared'
                                 for m in self.modes[n].values())},
                'values': {n: v for n, v in values.items() if v}}

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

    # -- the value boxes of 'fixed' -----------------------------------------
    def _shown(self, p, v):
        """Stored value v of parameter p as its box shows it."""
        return '%.6g' % (v * self.info.get(p, (1.0,))[0])

    def _edit(self, p, tip):
        """A value box for parameter p: empty shows (greyed) the Simulation
        tab's value."""
        e = QtWidgets.QLineEdit()
        e.setMaximumWidth(110)
        scale, unit, current = self.info.get(p, (1.0, '', None))
        e.setPlaceholderText(self._shown(p, current) if current is not None
                             else 'current')
        e.setToolTip(tip + ' (%s; empty: the Simulation tab\'s value%s)'
                     % (unit or 'no unit', '' if current is None else
                        ', ' + self._shown(p, current)))
        return e

    def _parse(self, p, e):
        """The stored value typed in box e, None when it is empty; False
        (box turned red) when it is not a number."""
        text = e.text().strip()
        if not text:
            e.setStyleSheet('')
            return None
        try:
            v = float(text) / self.info.get(p, (1.0,))[0]
        except ValueError:
            e.setStyleSheet('border: 1px solid %s' % style.ERROR)
            return False
        e.setStyleSheet('')
        return v

    def _cell(self, combo, edit):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        h.addWidget(combo, 1)
        h.addWidget(edit)
        return w

    def _build_table(self):
        """A row per parameter: "All datasets", then a column per ticked
        dataset; usable with two ticked datasets or more."""
        sets = self.chosen()
        t = self.table
        t.clearContents()
        t.setColumnCount(2 + len(sets))
        t.setHorizontalHeaderLabels(['Parameter', 'All datasets'] + sets)
        self.all_combos, self.all_edits = [], []
        self.cells, self.edits = {}, {}
        for r, p in enumerate(self.params):
            t.setItem(r, 0, QtWidgets.QTableWidgetItem(shown_names(p)))
            c = self._combo(mixed=True)
            c.setToolTip('The mode in every ticked dataset')
            c.currentIndexChanged.connect(
                lambda _, p=p, c=c: self._set_all(p, c.currentData()))
            e = self._edit(p, 'The value it keeps in every ticked dataset')
            e.editingFinished.connect(
                lambda p=p, e=e: self._set_all_value(p, e))
            t.setCellWidget(r, 1, self._cell(c, e))
            self.all_combos.append(c)
            self.all_edits.append(e)
            for k, n in enumerate(sets):
                c = self._combo()
                c.setCurrentIndex(c.findData(self.mode(n, p)))
                c.setToolTip('The mode in %s only' % n)
                c.currentIndexChanged.connect(
                    lambda _, n=n, p=p, c=c: self._set_one(n, p,
                                                           c.currentData()))
                e = self._edit(p, 'The value it keeps in %s' % n)
                if p in self.values[n]:
                    e.setText(self._shown(p, self.values[n][p]))
                e.editingFinished.connect(
                    lambda n=n, p=p, e=e: self._set_value(n, p, e))
                e.setVisible(self.mode(n, p) == 'fixed')
                t.setCellWidget(r, 2 + k, self._cell(c, e))
                self.cells[n, p] = c
                self.edits[n, p] = e
            self._show_all(r)
        # names in full; the table scrolls sideways if the window is narrow
        for k in range(t.columnCount()):
            t.resizeColumnToContents(k)
        t.resizeRowsToContents()
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
        Mixed; its value box (shown when every one is not fitted) their
        common value, or empty."""
        p, c = self.params[r], self.all_combos[r]
        sets = self.chosen()
        found = {self.mode(n, p) for n in sets}
        mode = found.pop() if len(found) == 1 else \
            'shared' if not found else MIXED
        c.blockSignals(True)
        c.setCurrentIndex(c.findData(mode))
        c.blockSignals(False)
        e = self.all_edits[r]
        vals = {self.values[n].get(p) for n in sets}
        e.setText(self._shown(p, vals.pop()) if len(vals) == 1 and
                  None not in vals else '')
        e.setStyleSheet('')
        e.setVisible(mode == 'fixed')

    def _set_all(self, p, mode):
        if mode == MIXED:
            return
        for n in self.chosen():
            self.modes[n][p] = mode
            c = self.cells[n, p]
            c.blockSignals(True)
            c.setCurrentIndex(c.findData(mode))
            c.blockSignals(False)
            self.edits[n, p].setVisible(mode == 'fixed')
        self._show_all(self.params.index(p))

    def _set_one(self, n, p, mode):
        self.modes[n][p] = mode
        self.edits[n, p].setVisible(mode == 'fixed')
        self._show_all(self.params.index(p))

    def _set_value(self, n, p, e):
        v = self._parse(p, e)
        if v is False:
            return
        if v is None:
            self.values[n].pop(p, None)
        else:
            self.values[n][p] = v
        self._show_all(self.params.index(p))

    def _set_all_value(self, p, e):
        """The "All datasets" box: its value into every ticked dataset."""
        v = self._parse(p, e)
        if v is False:
            return
        for n in self.chosen():
            if v is None:
                self.values[n].pop(p, None)
            else:
                self.values[n][p] = v
            self.edits[n, p].setText('' if v is None else self._shown(p, v))
            self.edits[n, p].setStyleSheet('')

    def accept(self):
        # a box still being typed in counts (editingFinished may not have
        # fired before OK); a value that is not a number stops here
        bad = []
        for (n, p), e in self.edits.items():
            if e.isVisible() or self.mode(n, p) == 'fixed':
                v = self._parse(p, e)
                if v is False:
                    bad.append('%s in %s' % (shown_names(p), n))
                elif v is None:
                    self.values[n].pop(p, None)
                else:
                    self.values[n][p] = v
        if bad:
            QtWidgets.QMessageBox.information(
                self, 'Datasets to fit', 'Not a number: %s.' % ', '.join(bad))
            return
        if not self.chosen():
            QtWidgets.QMessageBox.information(self, 'Datasets to fit',
                                              'Tick at least one dataset.')
            return
        super().accept()
