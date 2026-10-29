"""
The "Datasets to fit" panel, on the right of the main window's Fit tab:
which datasets are fitted, the model each one is fitted with, what is
fitted and how.

DATASETS -- a row per dataset that can be fitted: ticked, it is fitted
(several ticked: together, fit.joint); its Model box links it to a model of
the Simulation tab's Model box (fit.panel MODELS; "(shown model)": whichever
the Simulation tab shows).  fit.panel turns two ticked datasets or more into
the Dataset box's joint entry, one into that dataset.  The last ticked
dataset cannot be unticked.

PARAMETERS -- a row per parameter any ticked dataset's model can fit
(Stack.free_parameters(everything=True), matched by name across the
models).  Its Fit tick ticks it in every one of those models that has it
(each model keeps its Fit ticks, its values, bounds and priors, edited in
the Simulation tab); partly ticked: ticked in some of them.  The buttons
below tick a quantity everywhere, or untick everything.  For two ticked
datasets or more, a ticked row says how each dataset fits it (fit.joint
MODES): shared by the datasets (the default), its own value in each, or not
fitted (it keeps its model's value).  "All datasets" sets every ticked
dataset at once ("Own value": a value of its own in each), a dataset's
column that one only ("All datasets" then reads Mixed); "—" where the
dataset's model has no such parameter.  A shared parameter takes its
bounds and starting value from one model, chosen in its "From" box among
the models of the datasets sharing it (default: the first one's).
"""

from PyQt6 import QtCore, QtGui, QtWidgets

from tabs.simulation import shown_names

# fit.joint MODES, as the combos show them
MODE_LABELS = [('shared', 'Shared'),
               ('own', 'Own value'),
               ('fixed', 'Not fitted')]
MIXED = 'mixed'             # "All datasets" when the datasets differ
SHOWN = '(shown model)'     # a dataset's Model box: no link
# the tick buttons: key, label, tool tip
TICKS = [('untick', 'Untick all',
          'Untick Fit on every parameter of the ticked datasets\' models; '
          'values and bounds are kept'),
         ('thickness', 'Tick thickness',
          'Tick Fit on the thickness of every film layer'),
         ('NSLD_real', 'Tick NSLD',
          'Tick Fit on the real NSLD of every layer but the fronting'),
         ('roughness_sigma', 'Tick roughness',
          'Tick Fit on the roughness σ of every interface with a roughness '
          'model')]
# the parameter table's fixed columns
FIT_COL, NAME_COL, FROM_COL, ALL_COL = range(4)


class DatasetsPanel(QtWidgets.QWidget):
    """Pick datasets, their models, the parameters fitted and their modes
    (see the module docstring)."""

    changed = QtCore.pyqtSignal()       # a dataset, link, mode or From
    # (parameter names, ticked): a Fit tick; fit.panel writes it into the
    # models and reloads the panel
    fitToggled = QtCore.pyqtSignal(object, bool)
    tickRequested = QtCore.pyqtSignal(str)     # a TICKS key

    def __init__(self, parent=None):
        super().__init__(parent)
        self.names, self.models, self.links = [], [], {}
        self.params, self.present, self.varied = [], {}, {}
        self.modes, self.sources = {}, {}
        self._chosen = []

        self.list = QtWidgets.QTableWidget(0, 2)
        self.list.setHorizontalHeaderLabels(['Dataset', 'Model'])
        self.list.verticalHeader().setVisible(False)
        self.list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.list.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.list.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.list.setToolTip('Tick the datasets fitted together, and pick '
                             'the model each one is fitted with')
        self.list.itemChanged.connect(self._ticked)

        self.table = QtWidgets.QTableWidget(0, 0)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table.itemChanged.connect(self._fit_ticked)
        self.all_combos = {}           # parameter -> "All datasets" combo
        self.from_combos = {}          # parameter -> its From combo
        self.cells = {}                # (dataset, parameter) -> its combo
        self.note = QtWidgets.QLabel(wordWrap=True)
        self.note.setProperty('role', 'muted')
        ticks = QtWidgets.QHBoxLayout()
        self.tick_buttons = {}
        for key, label, tip in TICKS:
            b = self.tick_buttons[key] = QtWidgets.QPushButton(label)
            b.setToolTip(tip + ' (in every ticked dataset\'s model)')
            b.clicked.connect(lambda _, k=key: self.tickRequested.emit(k))
            ticks.addWidget(b)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel('<b>Datasets</b>'))
        left.addWidget(self.list, 1)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel('<b>Parameters</b>'))
        right.addWidget(self.table, 1)
        right.addLayout(ticks)
        right.addWidget(self.note)
        row = QtWidgets.QHBoxLayout()
        row.addLayout(left, 1)
        row.addLayout(right, 3)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addLayout(row, 1)
        v.addWidget(QtWidgets.QLabel(
            '<i>Fit: varied in the fit (bounds, values and priors are each '
            'model\'s, in the Simulation tab).  Shared: one value fitted to '
            'every dataset sharing it, from the model chosen in From.  Own '
            'value: fitted to each dataset alone.  Not fitted: kept at its '
            'model\'s value.  "All datasets" sets every dataset; a dataset\'s '
            'column changes that one only.</i>', wordWrap=True))
        self.load()

    def load(self, names=(), chosen=(), models=(), links=None, params=(),
             present=None, varied=None, modes=None, sources=None):
        """names: the datasets that can be fitted, chosen: those ticked;
        models: the Simulation tab's models, links: {dataset: model} (none:
        the shown model); params: the parameter names of the ticked
        datasets' models (fit.problem.parameter_names of
        free_parameters(everything=True)), present: {dataset: the names its
        model has}, varied: {name: True, False or None (ticked in some of
        the models)}; modes: {dataset: {parameter name: mode}} (fit.joint),
        sources: {parameter name: model} of the shared ones, None for both:
        the panel's own, for the datasets still there."""
        self.names, self.models = list(names), list(models)
        self.links = {n: m for n, m in (links or {}).items()
                      if n in self.names and m in self.models}
        self.params = list(params)
        self.present = {n: set(v) for n, v in (present or {}).items()}
        self.varied = dict(varied or {})
        self.modes = {n: dict((self.modes if modes is None else modes)
                              .get(n) or {}) for n in self.names}
        self.sources = dict(self.sources if sources is None else sources)
        self._chosen = [n for n in self.names if n in chosen]
        t = self.list
        t.blockSignals(True)
        t.setRowCount(len(self.names))
        for r, n in enumerate(self.names):
            it = QtWidgets.QTableWidgetItem(n)
            it.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled
                        | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.CheckState.Checked if n in chosen
                             else QtCore.Qt.CheckState.Unchecked)
            t.setItem(r, 0, it)
            c = QtWidgets.QComboBox()
            c.addItems([SHOWN] + self.models)
            c.setCurrentText(self.links.get(n, SHOWN))
            c.setToolTip('The model %s is fitted with' % n)
            c.currentTextChanged.connect(
                lambda m, n=n: self._linked(n, m))
            t.setCellWidget(r, 1, c)
        t.resizeColumnToContents(1)
        t.blockSignals(False)
        self._build_table()

    # -- datasets -------------------------------------------------------------
    def _ticked(self, item):
        """A dataset ticked or unticked; the last one stays ticked."""
        if not self.chosen():
            self.list.blockSignals(True)
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            self.list.blockSignals(False)
            return
        self._chosen = self.chosen()
        self._build_table()
        self.changed.emit()

    def _linked(self, n, model):
        if model == SHOWN:
            self.links.pop(n, None)
        else:
            self.links[n] = model
        self.changed.emit()

    def chosen(self):
        """The ticked datasets, in the list's order."""
        return [self.list.item(r, 0).text()
                for r in range(self.list.rowCount())
                if self.list.item(r, 0).checkState()
                == QtCore.Qt.CheckState.Checked]

    def link(self, n):
        """The model dataset n is linked to, None for the shown one."""
        return self.links.get(n)

    def selection(self):
        """{'sets': the ticked datasets, 'modes': {dataset: {parameter:
        mode}} of theirs that are not 'shared', 'sources': {parameter:
        model} of the shared ones whose From was chosen}."""
        sets = self.chosen()
        return {'sets': sets,
                'modes': {n: {a: m for a, m in self.modes[n].items()
                              if m != 'shared'} for n in sets
                          if any(m != 'shared'
                                 for m in self.modes[n].values())},
                'sources': dict(self.sources)}

    def mode(self, dataset, param):
        return self.modes[dataset].get(param, 'shared')

    def _has(self, n, p):
        return p in self.present.get(n, ())

    # -- parameters -----------------------------------------------------------
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
        """A row per parameter: its Fit tick and name; for two ticked
        datasets or more, on a ticked row, its From model, "All datasets"
        and a column per ticked dataset."""
        sets = self.chosen()
        joint = len(sets) > 1
        t = self.table
        t.blockSignals(True)
        t.clearContents()
        t.setRowCount(len(self.params))
        t.setColumnCount(4 + len(sets) if joint else 2)
        t.setHorizontalHeaderLabels(
            ['Fit', 'Parameter'] + (['From', 'All datasets'] + sets
                                    if joint else []))
        self.all_combos, self.from_combos, self.cells = {}, {}, {}
        for r, p in enumerate(self.params):
            on = self.varied.get(p, False)
            it = QtWidgets.QTableWidgetItem()
            it.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled
                        | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.CheckState.PartiallyChecked
                             if on is None else QtCore.Qt.CheckState.Checked
                             if on else QtCore.Qt.CheckState.Unchecked)
            it.setData(QtCore.Qt.ItemDataRole.UserRole, p)
            if on is None:
                it.setToolTip('Ticked in some of the models only')
            t.setItem(r, FIT_COL, it)
            name = QtWidgets.QTableWidgetItem(shown_names(p))
            name.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            if on is not False:
                font = name.font()
                font.setBold(True)
                name.setFont(font)
            t.setItem(r, NAME_COL, name)
            if not joint or on is False:
                continue
            c = self._combo(mixed=True)
            c.setToolTip('The mode in every ticked dataset')
            c.currentIndexChanged.connect(
                lambda _, p=p, c=c: self._set_all(p, c.currentData()))
            t.setCellWidget(r, ALL_COL, c)
            self.all_combos[p] = c
            f = QtWidgets.QComboBox()
            f.setToolTip('The model whose bounds and value this shared '
                         'parameter takes')
            f.currentTextChanged.connect(
                lambda m, p=p: self._set_source(p, m))
            t.setCellWidget(r, FROM_COL, f)
            self.from_combos[p] = f
            for k, n in enumerate(sets):
                if not self._has(n, p):
                    dash = QtWidgets.QTableWidgetItem('—')
                    dash.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
                    dash.setToolTip('The model of %s has no such parameter'
                                    % n)
                    dash.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignCenter)
                    t.setItem(r, 4 + k, dash)
                    continue
                c = self._combo()
                c.setCurrentIndex(c.findData(self.mode(n, p)))
                c.setToolTip('The mode in %s only' % n)
                c.currentIndexChanged.connect(
                    lambda _, n=n, p=p, c=c: self._set_one(n, p,
                                                           c.currentData()))
                t.setCellWidget(r, 4 + k, c)
                self.cells[n, p] = c
            self._show_row(p)
        t.blockSignals(False)
        # names in full; the table scrolls sideways if the tab is narrow
        for k in range(t.columnCount()):
            t.resizeColumnToContents(k)
        t.resizeRowsToContents()
        if not self.names:
            self.note.setText('Import data to fit first.')
        elif not any(v is not False for v in self.varied.values()):
            self.note.setText('Tick Fit on the parameters to fit.')
        elif not joint:
            self.note.setText('Tick two datasets or more to fit them '
                              'together; one dataset is fitted alone, with '
                              'every parameter ticked Fit.')
        else:
            self.note.setText('')

    def _model_of(self, n):
        return self.links.get(n, SHOWN)

    def _show_row(self, p):
        """Row p's "All datasets": the ticked datasets' common mode, or
        Mixed; its From box: the models of the datasets sharing it (none:
        disabled)."""
        sets = [n for n in self.chosen() if self._has(n, p)]
        found = {self.mode(n, p) for n in sets}
        mode = found.pop() if len(found) == 1 else \
            'shared' if not found else MIXED
        c = self.all_combos[p]
        c.blockSignals(True)
        c.setCurrentIndex(c.findData(mode))
        c.blockSignals(False)
        f = self.from_combos[p]
        models = []
        for n in sets:
            if self.mode(n, p) == 'shared' and self._model_of(n) not in models:
                models.append(self._model_of(n))
        f.blockSignals(True)
        f.clear()
        f.addItems(models)
        src = self.sources.get(p)
        f.setCurrentIndex(models.index(src) if src in models else 0)
        f.setEnabled(len(models) > 1)
        f.blockSignals(False)

    def _set_all(self, p, mode):
        if mode == MIXED:
            return
        for n in self.chosen():
            if (n, p) not in self.cells:
                continue
            self.modes[n][p] = mode
            c = self.cells[n, p]
            c.blockSignals(True)
            c.setCurrentIndex(c.findData(mode))
            c.blockSignals(False)
        self._show_row(p)
        self.changed.emit()

    def _set_one(self, n, p, mode):
        self.modes[n][p] = mode
        self._show_row(p)
        self.changed.emit()

    def _set_source(self, p, model):
        if model:
            self.sources[p] = model
            self.changed.emit()

    def _fit_ticked(self, item):
        """A Fit tick: fit.panel writes it into the models (a partly ticked
        row is ticked everywhere)."""
        if item.column() != FIT_COL:
            return
        p = item.data(QtCore.Qt.ItemDataRole.UserRole)
        on = self.varied.get(p, False) is not True
        self.fitToggled.emit([p], on)
