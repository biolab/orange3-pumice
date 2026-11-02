import os.path
from typing import Optional

from AnyQt import QtWidgets, QtCore

from orangewidget.settings import Setting
from orangewidget.utils.itemmodels import PyListModel, signal_blocking
from orangewidget.widget import Msg

from Orange.data import Table, Domain
from Orange.misc import DistMatrix
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState

from orangecontrib.pumice.distance.distance import LevenshteinDistance


class InterruptException(Exception):
    pass


class DistanceRunner:
    @staticmethod
    def run(data: Table, delimiter: Optional[str], normalize: bool,
            state: TaskState) -> Optional[DistMatrix]:
        if data is None:
            return None

        def callback(i: float) -> bool:
            state.set_progress_value(i)
            if state.is_interruption_requested():
                raise InterruptException

        state.set_status("Calculating...")
        dist = LevenshteinDistance(
            callback=callback,
            delimiter=delimiter,
            normalize=normalize)
        model = dist.fit(data)
        return model(data)


class OWLevenshtein(OWWidget, ConcurrentWidgetMixin):
    name = "Levenshtein Distance"
    description = "Levenshtein distance between string attributes of two data tables."
    icon = "icons/levenshtein.svg"

    class Inputs:
        data = Input("Data", Table, default=True)

    class Outputs:
        distances = Output("Distances", DistMatrix, default=True)

    class Error(OWWidget.Error):
        memory_error = Msg("Not enough memory")
        no_string_attributes = Msg(
            "Data has no string attributes.\n"
            "Levenshtein distance can only be computed for string attributes.")

    want_main_area = False

    unselected_hint: list[str] = Setting([], schema_only=True)
    normalize: bool = Setting(True)
    split_words: bool = Setting(True)
    delimiter: str = Setting(",")
    auto_apply: bool = Setting(True)

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.data: Optional[Table] = None

        box = gui.vBox(self.controlArea, "Select attributes")

        self.attr_model = PyListModel()
        al = self.attr_list = QtWidgets.QListView(box)
        al.setSelectionMode(QtWidgets.QListView.MultiSelection)
        al.setModel(self.attr_model)
        al.selectionModel().selectionChanged.connect(self._on_selection_changed)
        al.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        al.setFocus()
        box.layout().addWidget(al)

        bbox = gui.hBox(box)
        gui.button(
            bbox, self, "Select All",
            callback=self.select_all,
            focusPolicy=QtCore.Qt.NoFocus)
        gui.button(
            bbox, self, "Select None",
            callback=self.select_none,
            focusPolicy=QtCore.Qt.NoFocus)

        box = gui.vBox(self.controlArea, "Options")
        gui.checkBox(
            box, self, "normalize", "Normalize by length of strings",
            callback=self.commit.deferred)
        hbox = gui.hBox(box)
        gui.checkBox(
            hbox, self, "split_words", "Split words by",
            callback=self._on_split_words_changed)
        gui.lineEdit(
            hbox, self, "delimiter", placeholderText=",", controlWidth=25,
            callback=self._on_delimiter_changed,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        gui.rubber(hbox)
        help = QtWidgets.QLabel("?")
        help.setFixedSize(16, 16)
        help.setAlignment(QtCore.Qt.AlignCenter)
        help.setStyleSheet("""
            QLabel {
                border: 1px solid palette(mid);
                border-radius: 8px;
                color: blue;
                font-weight: bold;
                font-size: 11px;
            }
        """)
        help.setToolTip(
            "If values consist of multiple words, they can be split.\n"
            "The distance is then computed between all pairs of words\n"
            "in two data instances, and the minimum is taken.")
        hbox.layout().addWidget(help)

        b = gui.auto_apply(self.controlArea, self, "auto_apply")
        b.setFocusPolicy(QtCore.Qt.NoFocus)

        # Debounce apply to avoid multiple calls when selecting attributes
        self.selection_timer = QtCore.QTimer(self)
        self.selection_timer.setSingleShot(True)
        self.selection_timer.setInterval(1000)
        self.selection_timer.timeout.connect(self.commit.deferred)

    @Inputs.data
    def set_data(self, data: Optional[Table]):
        def clear():
            self.data = None
            self.attrs = []
            self.commit.now()

        self.Error.clear()
        if data is None:
            clear()
            return

        str_vars = [var for var in data.domain.metas if var.is_string]
        if not str_vars:
            self.Error.no_string_attributes()
            clear()
            return

        self.data = data

        # Prevent triggering a commit: we have an unconditional commit below
        with signal_blocking(self.attr_list.selectionModel()):
            saved_hint = self.unselected_hint.copy()
            self.attr_model[:] = [var.name for var in str_vars]

            unselection = QtCore.QItemSelection()
            used = []
            for name in saved_hint:
                try:
                    idx = self.attr_model.index(self.attr_model.indexOf(name))
                    used.append(name)
                except ValueError:
                    continue
                unselection.select(idx, idx, )
            self.unselected_hint = used
            selection = QtCore.QItemSelection()
            selection.select(
                self.attr_model.index(0, 0),
                self.attr_model.index(len(str_vars) - 1, 0))
            selection.merge(unselection, QtCore.QItemSelectionModel.Deselect)
            self.attr_list.selectionModel().select(
                selection, QtCore.QItemSelectionModel.ClearAndSelect
            )

        self.commit.now()

    def _on_selection_changed(self):
        selected = {
            self.attr_model[idx.row()]
            for idx in self.attr_list.selectionModel().selectedIndexes()}
        self.unselected_hint = list(set(self.attr_model) - selected)
        self.selection_timer.start()

    def _on_split_words_changed(self):
        self.commit.deferred()

    def _on_delimiter_changed(self):
        self.split_words = bool(self.delimiter)
        self.commit.deferred()

    def select_all(self):
        self.attr_list.selectionModel().select(
            QtCore.QItemSelection(
                self.attr_model.index(0, 0),
                self.attr_model.index(len(self.attr_model) - 1, 0)
            ),
            QtCore.QItemSelectionModel.ClearAndSelect
        )

    def select_none(self):
        self.attr_list.selectionModel().clearSelection()

    @gui.deferred
    def commit(self):
        self.Error.memory_error.clear()
        if self.data is None:
            self.Outputs.distances.send(None)
            return

        domain = self.data.domain
        selected = [
            self.attr_model[idx.row()]
            for idx in self.attr_list.selectionModel().selectedIndexes()]
        if not selected:
            self.Outputs.distances.send(None)
            return

        reduced_domain = Domain(
            domain.attributes,
            domain.class_vars,
            [domain[i] for i in selected])
        reduced_data = self.data.transform(reduced_domain)

        delimiter = self.delimiter if self.split_words else None
        self.start(DistanceRunner.run, reduced_data, delimiter, self.normalize)

    def on_partial_result(self, _):
        pass

    def on_done(self, result: DistMatrix):
        assert isinstance(result, DistMatrix) or result is None
        if result is not None:
            result.row_items = result.col_items = self.data
        self.Outputs.distances.send(result)

    def on_exception(self, ex):
        if isinstance(ex, MemoryError):
            self.Error.memory_error()
        elif isinstance(ex, InterruptException):
            self.Outputs.distances.send(None)
        else:
            self.Outputs.distances.send(None)
            raise ex

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    dirname = os.path.join(os.path.dirname(__file__), "..", 'datasets')
    WidgetPreview(OWLevenshtein).run(
        set_data=Table(os.path.join(dirname, 'languages-words.xlsx'))
    )
