import numpy as np

from AnyQt.QtCore import Qt, QSize
from AnyQt.QtWidgets import QGridLayout, QLabel
from AnyQt.QtGui import QFontMetrics

from orangewidget.utils.itemmodels import PyListModel

from Orange.data import Table, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input
from Orange.widgets.utils.itemmodels import DomainModel

from orangecontrib.network import Network


class OWRecommendation(OWWidget):
    name = "Recommendation"
    description = "Demo for simple network-based recommendation algorithm"
    icon = "icons/recommendation.png"

    class Inputs:
        network = Input("Network", Network, default=True)

    settingsHandler = settings.DomainContextHandler()
    selected_node_hint = settings.ContextSetting(None)
    name_column = settings.ContextSetting(None)

    want_control_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()
        self.network: Network = None
        self.selected_node = None

        self.name_column_model = DomainModel(valid_types=StringVariable)
        gui.comboBox(
            self.mainArea, self, "name_column",
            label="Name column: ", box=True,
            model=self.name_column_model,
            callback=self.set_value_list,
            orientation=Qt.Horizontal)

        grid = QGridLayout()
        gui.widgetBox(self.mainArea, box=True, orientation=grid)

        self.nodes_model = PyListModel()
        combo = gui.comboBox(
            None, self, "selected_node",
            model=self.nodes_model,
            callback=self.on_node_changed)

        align_top = Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        grid.addWidget(combo, 0, 0, 1, 2)
        grid.addWidget(
            QLabel(self, text="<b>Items: </b"),
            1, 0, alignment=align_top)
        self.item_list = QLabel(self, wordWrap=True)
        grid.addWidget(self.item_list, 1, 1)
        grid.addWidget(
            QLabel(self, text="<b>Friends: </b"),
            2, 0, alignment=align_top)
        self.friends_list = QLabel(self, wordWrap=True)
        grid.addWidget(self.friends_list, 2, 1)

        fm = QFontMetrics(self.font())
        box3 = gui.hBox(self.mainArea, "Recommendations")
        self.rec = gui.widgetLabel(
            box3,
            minimumSize=QSize(40 * fm.averageCharWidth(), 5 * fm.height()),
            wordWrap=True)

    @Inputs.network
    def set_network(self, network):
        self.closeContext()
        self.clear()
        self.network = network

        if self.network is None:
            return

        self.name_column_model.set_domain(network.nodes.domain)
        if self.name_column_model:
            self.name_column = self.name_column_model[0]
        self.controls.name_column.box.setHidden(len(self.name_column_model) < 2)

        self.openContext(self.network.nodes)

        self.set_value_list()
        if self.selected_node_hint in self.nodes_model:
            self.selected_node = self.selected_node_hint

        self.update_labels()

    def clear(self):
        self.name_column_model.set_domain(None)
        self.nodes_model.clear()
        self.network = None
        self.name_column = None
        self.selected_node = None
        self.update_labels()

    def set_value_list(self):
        if self.name_column is None:
            self.nodes_model.clear()
            return

        self.nodes_model[:] = self.network.nodes.get_column(self.name_column)
        self.selected_node = self.nodes_model[0]

    @property
    def has_selection(self):
        return self.name_column is not None and self.selected_node is not None

    @property
    def selected_node_index(self):
        return self.nodes_model.indexOf(self.selected_node)

    def on_node_changed(self):
        self.selected_node_hint = self.selected_node
        self.update_labels()

    def update_labels(self):
        self.set_items()
        self.set_friends()
        self.set_recommendations()

    def set_items(self):
        self.item_list.setText(", ".join(self.get_item_names() ) or "No items")

    def set_friends(self):
        sorted_friends = self.get_friends()[0]
        if len(sorted_friends) == 0:
            self.friends_list.setText("No friends")
            return

        neighbours_names = [self.nodes_model[row] for row in sorted_friends]
        self.friends_list.setText(", ".join(neighbours_names))

    def set_recommendations(self):
        items, recommenders = self.get_recommendations()
        if len(items) == 0:
            self.rec.setText("No recommendations")
            return

        output = ""
        for item, recommenders in zip(items, recommenders):
            output += f"<b>{item}: </b>"
            output += f"{', '.join(recommenders)}<br/><br/>"
        self.rec.setText(output)

    def get_item_names(self):
        if not self.has_selection:
            return []
        node_choices = self.network.nodes.X[self.selected_node_index]
        attributes = self.network.nodes.domain.attributes
        chosen = np.flatnonzero(node_choices)
        item_names = [attributes[i].name for i in chosen]
        return item_names

    def get_friends(self):
        if not self.has_selection:
            return [], []
        # TODO: when https://github.com/biolab/orange3-network/pull/273
        #  is merged and released, change this function to
        # neighs, weights = self.network.outgoing(self.selected_node_index, weights=True)
        # inds = np.argsort(weights)
        # return neighs[inds], weights[inds]
        matrix = self.network.edges[0].edges
        node = self.selected_node_index
        fr, to = matrix.indptr[node], matrix.indptr[node + 1]
        neighs = matrix.indices[fr:to]
        weights = matrix.data[fr:to]
        inds = np.argsort(weights)
        return neighs[inds], weights[inds]

    def get_recommendations(self):
        neighbours_indices = self.get_friends()[0] if self.has_selection else []
        if len(neighbours_indices) == 0:
            return [], []

        neighbours_choices = self.network.nodes.X[neighbours_indices]
        counts = np.sum(neighbours_choices, axis=0)
        counts[self.network.nodes.X[self.selected_node_index] == 1] = -1
        sorted_items = np.argsort(counts)[::-1]
        most_freq = sorted_items[:5]
        items = [self.network.nodes.domain.attributes[i].name for i in most_freq]
        neighbours = self.network.nodes.get_column(self.name_column)[neighbours_indices]
        recommenders = [
            neighbours[neighbours_choices[:, i] == 1] for i in most_freq]
        return items, recommenders


def main():
    # pylint: disable=import-outside-toplevel
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.network.network.readwrite \
        import read_pajek
    from os.path import join, dirname

    table = Table(join(dirname(dirname(__file__)), 'networks', 'data.xlsx'))
    network = read_pajek(join(dirname(dirname(__file__)), 'networks', 'nodes_features.net'))
    network.nodes = table
    WidgetPreview(Recommendation).run(set_network=network)


if __name__ == "__main__":
    main()
