import os.path
from typing import Optional

import numpy as np

from AnyQt.QtCore import Qt, QSize, QAbstractTableModel
from AnyQt.QtGui import QPixmap, QFont
from AnyQt.QtWidgets import QTableView, QSizePolicy, QItemDelegate

from orangewidget.utils.itemmodels import PyListModel

from Orange.data import Table
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input
from Orange.widgets.utils.itemmodels import VariableListModel

from orangecontrib.network import Network
from orangewidget.widget import Msg

# TODO: sort person names in the combo, or eliminate combo and have a scrollable list
# TODO: text "Your friends also liked" is clipeed on projector, but not on machine?!


class PosterDelegate(QItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

    def paint(self, painter, option, index):
        painter.save()
        painter.setRenderHint(painter.Antialiasing)
        image = index.data(Qt.ItemDataRole.DecorationRole)
        rect = option.rect.adjusted(5, 5, -5, -5)
        if image is not None:
            painter.drawPixmap(
                rect.x() + (rect.width() - image.width()) // 2, rect.y(),
                image)
            rect.adjust(0, 210, 0, 0)
        font = index.data(Qt.ItemDataRole.FontRole)
        if font:
            painter.setFont(font)
        text = index.data(Qt.ItemDataRole.DisplayRole)
        align = index.data(Qt.ItemDataRole.TextAlignmentRole)
        painter.drawText(rect, align, text)
        painter.restore()


class CartoonTableModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self.chosen_items = []
        self.chosen_images = []
        self.recommended_items = []
        self.recommended_images = []
        self.recommenders = []

    def set_data(self, chosen_items, chosen_images, recommended_items,
                 recommended_images, recommenders, image_origin):
        self.beginResetModel()
        self.chosen_items = chosen_items
        self.chosen_images = chosen_images
        self.recommended_items = recommended_items
        self.recommended_images = recommended_images
        self.recommenders = recommenders
        self.image_origin = image_origin
        self.endResetModel()

    def rowCount(self, parent=None):
        return 5

    def columnCount(self, parent=None):
        return max(len(self.chosen_items), len(self.recommended_items))

    def data(self, index, role):
        row = index.row()
        column = index.column()
        if role == Qt.ItemDataRole.FontRole and row < 4:
            font = QFont()
            font.setPixelSize(18 + 2 * (row in [0, 2]))
            font.setLetterSpacing(QFont.PercentageSpacing, 110)
            if row in [0, 2]:
                font.setBold(True)
            return font
        if role == Qt.ItemDataRole.DisplayRole:
            if row == 0 and column == 0:
                return "Všeč so ti:"
            if row == 1 and column < len(self.chosen_items):
                return self.chosen_items[column]
            if row == 2 and column == 0:
                return "Tvojim prijateljem so všeč tudi:"
            if row == 3 and column < len(self.recommended_items):
                return self.recommended_items[column]
            if row == 4 and column < len(self.recommenders):
                return self.recommenders[column]
        if role == Qt.ItemDataRole.DecorationRole:
            if (row == 1
                    and self.chosen_images is not None
                    and column < len(self.chosen_images)):
                img_name = self.chosen_images[column]
            elif (row == 3
                    and self.recommended_images is not None
                    and column < len(self.recommended_images)):
                img_name = self.recommended_images[column]
            else:
                return None
            img_name = os.path.join(self.image_origin, img_name)
            if os.path.exists(img_name):
                image = QPixmap(img_name)
                return QPixmap(image).scaled(
                    150, 200, Qt.AspectRatioMode.KeepAspectRatio)
            else:
                return None
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if row in (0, 2):
                return (Qt.AlignmentFlag.AlignLeft
                        | Qt.AlignmentFlag.AlignVCenter
                        | Qt.TextDontClip)
            else:
                return (Qt.AlignmentFlag.AlignHCenter
                        | Qt.AlignmentFlag.AlignTop
                        | Qt.TextWordWrap)
        return None


class OWRecommendation(OWWidget):
    name = "Recommendation"
    description = "Demo for simple network-based recommendation algorithm"
    icon = "icons/recommendation.png"

    class Inputs:
        network = Input("Network", Network, default=True)
        item_data = Input("Items", Table)

    class Error(OWWidget.Error):
        no_choices = Msg(
            "Network does not contain user choices. Provide separate data.")
        no_item_names = Msg(
            "Network does not contain item names.")
        no_user_names_in_net = Msg(
            "Data included in the network does not contain user names.")
        user_names_mismatch = Msg(
            "Names of network nodes must match names of data columns"
        )
        invalid_node_data = Msg("Network data format is not recognized")

    selected_person_hint: Optional[str] = settings.Setting(None)
    item_column_hint: Optional[str] = settings.Setting(None)
    person_column_hint: Optional[str] = settings.Setting(None)

    want_control_area = False

    def __init__(self):
        """
        This widget is complicated because it can receive data from network
        and/or a separate signal. This is further complicated because the two
        sources are likely transposed because of the way they are constructed
        in Orange.

        Sources of data and their arrangement
        -------------------------------------

        `network.nodes` can be an instance of `Table` or a list of person names.
        If it is a `Table`, its rows correspond to persons and columns to items.

        `data`, if present, is a `Table`, whose rows correspond to items and
        columns to persons.

        `network.nodes` and `data` are transposed w.r.t. each other due to the
        way in which they are created. In the network, rows always represent
        nodes, which are persons. (TODO: what about a network of items?).
        In `data`, rows are items so that we can attach a column with images,
        and columns are persons because this is how other activities in Pumice
        arrange data.

        One or two sources (or none)
        ----------------------------

        Either `network.nodes` must be a `Table`, or there must be an input
        table (`data`). Otherwise, the widget shows an error and `person_names`
        and `items_names` are `None`. This can be used to test whether the
        widget is operational.

        If both sources are present, there must be exactly one column in
        `network.nodes` whose values equal the names of attributes in `data`.
        This column is taken to represent items. Otherwise, `item_column`
        is `None` and widget is non-operational.

        Names of nodes (persons) and items
        ----------------------------------

        `person_names` and `item_names` (both np.array with strings) contain
        names of persons and of items. If either is `None`, the data is not
        valid; the widget shows an error and is empty and non-operational.

        Depending upon the situation, the user may be able to choose columns
        whose values are used to fill those arrays.

        `person_column` and `item_column` (taken from `person_column_model` and
        `item_column_model`) contain variables with this data IF they are chosen
        by user. In situations where they can't be chosen, models are empty
        and attributes are `None`, combos are hidden. If there is only a
        single choice, the model contains it, the attribute (person_column,
        item_column) is set, but combo is hidden.

        - If `data` is not present, it is read from `network.nodes`.
          Its rows correspond to persons and user chooses how to name them:
          `person_column_model` contains string attributes from `network.nodes`,
          `person_column` is set and combo is shown if there is more than one
          choice.

          `item_names` are set to names of `network.nodes.domain.attributes`;
          the corresponding model is empty and combo is hidden.

        - If data is present, its rows correspond to items, so the user can
          choose how to name items (`item_names`, with `item_column` and
          `item_column_model`) -- if there are multiple options.
          Options include all string variables who do not have type=image;
          if all variables are type=image, they are all candidates.
          If there are no candidates, this is an error.

          `person_names` are set from `network.nodes`.


        Attributes:

            network (Network): input network (mandatory)
                - If `network.nodes` is an instance of Table, rows correspond
                to persons and columns to items.
                - Otherwise, `network.nodes` is a list of labels, representing
                persons.
                - TODO: Distances widget across columns outputs a table with
                labels and positions. This should also be treated as valid,
                but interpreted more like the latter case.

            data (Table): input table. Its *rows* represent items,
                and columns are persons. **This is the opposite from
                `network.nodes`.

            choices (np.ndarray of dtype bool):
                if `data` is given, choices equal data.X.T ordered so that
                rows' names (persons, this is X.T!) correspond to person_names
                (nodes in the network).

            person_column (StringVariable): variable with names of persons,
                or None if person can't be chosen (because it's taken from
                network)
            person_names (np.ndarray of strings): names of persons
            selected_person (string): name of selected person

            item_column (StringVariable): variable with names of items,
                or None if it can't be chosen.
            item_names (np.ndarray of strings): names of items

            image_column (StringVariable): variable with image names

          - otherwise, if
           and is equal to `data` if it exists, otherwise to `network.nodes`
        """
        super().__init__()
        self.network: Network = None
        self.data: Table = None
        self.choices = None

        self.person_column_model = VariableListModel()
        self.person_column = None
        self.person_names = None

        self.item_column_model = VariableListModel()
        self.item_names = None
        self.item_column = None

        self.selected_person = None

        self.image_column = None

        self.column_box = gui.hBox(self.mainArea)
        gui.comboBox(
            self.column_box, self, "person_column",
            label="Person name column: ", box=True,
            model=self.person_column_model,
            callback=self.on_person_column_changed,
            orientation=Qt.Horizontal)

        gui.comboBox(
            self.column_box, self, "item_column",
            label="Item column: ", box=True,
            model=self.item_column_model,
            callback=self.on_item_column_changed,
            orientation=Qt.Horizontal)

        self.persons_model = PyListModel()
        box = gui.hBox(self.mainArea)
        gui.comboBox(
            box, self, "selected_person",
            model=self.persons_model,
            callback=self.on_person_changed)

        self.friends_list = gui.widgetLabel(box, "<b>Prijatelji</b>: none")

        box3 = gui.vBox(self.mainArea, True)
        self.rec_model = CartoonTableModel()
        self.rec_table = rec = QTableView()
        self.rec_table.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.rec_table.setItemDelegate(PosterDelegate())
        self.rec_table.verticalHeader().setDefaultSectionSize(50)
        self.rec_table.setSelectionMode(QTableView.SelectionMode.NoSelection)
        rec.setModel(self.rec_model)
        rec.verticalHeader().hide()
        rec.horizontalHeader().hide()
        rec.setShowGrid(False)
        rec.horizontalHeader().setDefaultSectionSize(160)
        rec.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignHCenter)
        box3.layout().addWidget(rec)

    def sizeHint(self):
        return QSize(900, 600)

    def clear(self):
        self.Error.clear()
        self.column_box.setHidden(True)

        self.controls.person_column.box.setHidden(True)
        self.person_column_model.clear()
        self.person_column = None
        self.person_names = None

        self.persons_model.clear()
        self.selected_person = None

        self.controls.item_column.box.setHidden(True)
        self.item_column_model.clear()
        self.item_column = None
        self.item_names = None

        self.choices = None
        self.image_column = None

        self.update_page()

    @Inputs.network
    def set_network(self, network):
        self.network = network

    @Inputs.item_data
    def set_item_data(self, item_data):
        self.data = item_data

    def handleNewSignals(self):
        self.clear()
        if self.network is None:
            return
        if self.data is None and not isinstance(self.network.nodes, Table):
            self.Error.no_choices()
            return

        self.init_person_column()
        self.init_item_column()

        if self.is_valid:
            self.set_ordered_choices()
            self.init_value_list()
            if self.selected_person_hint in self.persons_model:
                self.selected_person = self.selected_person_hint

        self.update_page()

    @property
    def is_valid(self):
        return self.person_names is not None and self.item_names is not None

    def init_person_column(self):
        if self.network is None:
            return
        if (isinstance(self.network.nodes, Table)
                and self.network.nodes.domain.attributes):
            self._init_person_column_from_net()
        elif self.data is not None:
            self._init_person_column_from_data()
        else:
            self.Error.no_choices()

    def on_person_column_changed(self):
        assert self.network is not None
        assert isinstance(self.network.nodes, Table)
        self.person_column_hint = self.person_column.name
        self.person_names = self.network.nodes.get_column(self.person_column)
        self.set_ordered_choices()
        self.init_value_list()

    def init_item_column(self):
        if self.data is None:
            self._init_item_column_from_net()
        else:
            self._init_item_column_from_data()

    def on_item_column_changed(self):
        self.item_column_hint = self.item_column.name
        self.item_names = self.data.get_column(self.item_column)
        self.update_page()

    def init_value_list(self):
        self.persons_model[:] = self.person_names
        self.selected_person = self.persons_model[0]

    def set_ordered_choices(self):
        """Order of rows is determined by self.person_names.
           self.data may have a different order and needs to be reordered
           """
        if self.network is None:
            return

        if self.data is None:
            choices = self.network.nodes.X
        else:
            assert self.person_names is not None
            domain = self.data.domain
            assert len(self.person_names) == len(domain.attributes)
            order = np.array([domain.index(name) for name in self.person_names])
            choices = self.data.X.T[order]
        self.choices = np.nan_to_num(choices).astype(bool)

    @property
    def selected_person_index(self):
        return self.persons_model.indexOf(self.selected_person)

    def on_person_changed(self):
        self.selected_person_hint = self.selected_person
        self.update_page()

    def _init_person_column_from_net(self):
        names = self.data and sorted(var.name for var in self.data.domain.metas)
        applicable = [
            var for var in self.network.nodes.domain.metas
            if var.is_string and (
                names is None
                or sorted(self.network.nodes.get_column(var)) == names)
        ]
        if not applicable:
            self.Error.no_user_names_in_net()
            return

        if len(applicable) == 1:
            self.person_names = self.network.nodes.get_column(applicable[0])
            return

        self.person_column_model[:] = applicable
        self.column_box.setHidden(False)
        self.controls.person_column.box.setHidden(False)
        for var in applicable:
            if var.name == self.person_column_hint:
                self.person_column = var
                break
        else:
            self.person_column = applicable[0]
        self.person_names = self.network.nodes.get_column(
            self.person_column)

    def _init_person_column_from_data(self):
        nodes = self.network.nodes
        if (isinstance(nodes, Table)
                and not nodes.domain.attributes
                and len(nodes.domain.metas) == 1
                and nodes.domain.metas[0].is_string):
            person_names = nodes.metas.flatten()
        elif isinstance(nodes, np.ndarray) and (
                len(self.network.nodes.shape) == 1
                or self.network.nodes.shape[1] == 1):
            person_names = nodes.flatten()
        else:
            self.Error.invalid_node_data()
            return

        # Names of network nodes must match names of attributes from data
        if (sorted(person_names)
                != sorted(var.name for var in self.data.domain.attributes)):
            self.Error.user_names_mismatch()
            return

        self.person_names = person_names

    def _init_item_column_from_net(self):
        # tested in handleNewSignals
        assert isinstance(self.network.nodes, Table)
        self.item_names = np.array(
            [var.name for var in self.network.nodes.domain.attributes])
        if not self.item_names.size:
            self.item_names = None
            self.Error.no_item_names()
        return

    def _init_item_column_from_data(self):
        # Candidates for item names and images
        string_vars = [var for var in self.data.domain.metas if var.is_string]
        if not string_vars:
            self.Error.no_item_names()
            return

        # Find first applicable image column, either marked or heuristically
        for var in string_vars:
            if var.attributes.get("type", None) == "image":
                self.image_column = var
                break
        else:
            for var in string_vars:
                column = self.data.get_column_view(var)
                if all(os.path.splitext(v)[1] in {".png", ".jpg", ".jpeg", ".gif"}
                       for v in column):
                    self.image_column = var
                    break
            else:
                self.image_column = None

        # Exclude columns marked as images, but allow the hinted variable
        # If there are no such columns, allow any string variable
        if self.item_column_hint in [var.name for var in string_vars]:
            hinted = self.data.domain[self.item_column_hint]
        else:
            hinted = None
        applicable = [
            var for var in string_vars
            if var is hinted or var.attributes.get("type") != "image"]
        if not applicable:
            applicable = string_vars

        if len(applicable) == 1:
            self.item_names = self.data.get_column(applicable[0])
            return

        self.item_column_model[:] = applicable
        self.item_column = hinted or applicable[0]
        self.column_box.setHidden(False)
        self.controls.person_column.box.setHidden(False)

    def update_page(self):
        self.set_friends()
        self.set_recommendations()

    def set_friends(self):
        sorted_friends = self.get_friends()[0]
        if len(sorted_friends) == 0:
            self.friends_list.setText("<b>Prijatelji</b>:")
            return
        self.friends_list.setText("<b>Prijatelji</b>: " + ", ".join(self.person_names[sorted_friends]))

    def set_recommendations(self):
        if not self.is_valid:
            self.rec_model.set_data([], [], [], [], [], "")
            return
        image_data = self.image_column and self.data.get_column(self.image_column)
        image_origin = self.image_column.attributes.get("origin", ".")

        chosen_indices = np.flatnonzero(self.choices[self.selected_person_index])
        chosen_items = self.item_names[chosen_indices]
        chosen_images = image_data[chosen_indices] if image_data is not None else None

        recm_indices, recommenders = self.get_recommendations()
        recm_items = self.item_names[recm_indices]
        recm_images = image_data[recm_indices] if image_data is not None else None

        recommenders = [", ".join(recm) for recm in recommenders]
        self.rec_table.setRowHeight(1, 280 if self.image_column is not None else 80)
        self.rec_table.setRowHeight(3, 280 if self.image_column is not None else 80)
        self.rec_model.set_data(
            chosen_items, chosen_images, recm_items, recm_images, recommenders,
            image_origin)

    def get_friends(self):
        if not self.is_valid:
            return [], []
        # TODO: when https://github.com/biolab/orange3-network/pull/273
        #  is merged and released, change this function to
        # neighs, weights = self.network.outgoing(self.selected_person_index, weights=True)
        # inds = np.argsort(weights)
        # return neighs[inds], weights[inds]
        matrix = self.network.edges[0].edges
        person = self.selected_person_index
        fr, to = matrix.indptr[person], matrix.indptr[person + 1]
        neighs = matrix.indices[fr:to]
        weights = matrix.data[fr:to]
        inds = np.argsort(weights)
        return neighs[inds], weights[inds]

    def get_recommendations(self):
        neighbours_indices = self.get_friends()[0] if self.is_valid else []
        if len(neighbours_indices) == 0:
            return [], []

        neighbours_choices = self.choices[neighbours_indices]
        counts = np.sum(neighbours_choices, axis=0)
        counts[self.choices[self.selected_person_index] == 1] = -1
        sorted_items = np.argsort(counts)[::-1]
        most_freq = sorted_items[:5]
        item_indices = most_freq
        neighbours = self.person_names[neighbours_indices]
        recommenders = [
            neighbours[neighbours_choices[:, i] == 1] for i in most_freq]
        return item_indices, recommenders


def main():
    # pylint: disable=import-outside-toplevel
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.network.network.readwrite \
        import read_pajek

    dirname = os.path.join(os.path.dirname(__file__), "..", 'datasets', 'cartoons')

    items = Table(os.path.join(dirname, 'cartoons.xlsx'))
    items.domain["poster"].attributes["origin"] = dirname

    network = read_pajek(os.path.join(dirname, 'cartoons.net'))
    WidgetPreview(OWRecommendation).run(set_network=network, set_item_data=items)


if __name__ == "__main__":
    main()
