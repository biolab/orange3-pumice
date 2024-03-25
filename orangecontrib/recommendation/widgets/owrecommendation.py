import numpy as np

from AnyQt.QtCore import Qt, QSize
from AnyQt.QtWidgets import QGridLayout, QLabel
from AnyQt.QtGui import QFontMetrics

from orangewidget.utils.itemmodels import PyListModel

import Orange
from Orange.data import Table
from Orange.widgets.widget import OWWidget, Input
from Orange.widgets import gui, settings
from Orange.widgets.utils.itemmodels import DomainModel

from orangecontrib.network import Network


class Recommendation(OWWidget):
    name = "Recommendation"
    description = "Recommend feature based on selected node"
    icon = "icons/recommendation.png"
    category = "Recommendation"

    class Inputs:
        network = Input("Network", Network, default=True)

    settingsHandler = settings.DomainContextHandler()
    selected_node_hint = settings.ContextSetting(None)
    node_name = settings.ContextSetting(None)
    want_control_area = False

    resizing_enabled = False

    def __init__(self):
        super().__init__()
        self.node_name_id = None
        self.network: Network = None

        self.selected_node = None

        self.node_name_model = DomainModel(
            valid_types=Orange.data.StringVariable)

        gui.comboBox(
            self.mainArea, self, "node_name",
            label="Name column: ", box=True,
            model=self.node_name_model,
            callback=self.set_value_list,
            orientation=Qt.Horizontal)

        grid = QGridLayout()

        gui.widgetBox(self.mainArea, box=True,
                      orientation=grid)

        self.nodes_model = PyListModel()

        combo = gui.comboBox(
            None, self, "selected_node",
            model=self.nodes_model,
            callback=self.on_node_changed)

        align_top = Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        grid.addWidget(combo, 0, 0, 1, 2)
        grid.addWidget(QLabel(self,
                              text="<b>Items: </b"), 1, 0,
                       alignment=align_top)
        self.features_list_label = QLabel(self, wordWrap=True)
        grid.addWidget(self.features_list_label, 1, 1)
        grid.addWidget(QLabel(self, text="<b>Friends: </b"),
                       2, 0, alignment=align_top)
        self.friends_list_label = QLabel(self, wordWrap=True)
        grid.addWidget(self.friends_list_label, 2, 1)

        fm = QFontMetrics(self.font())
        box3 = gui.hBox(self.mainArea, "Recommendations")
        self.rec = gui.widgetLabel(box3,
                                   minimumSize=QSize(
                                       40 * fm.averageCharWidth(),
                                       5 * fm.height()), wordWrap=True)

    @Inputs.network
    def set_network(self, network):
        self.closeContext()
        self.clear()
        self.network = network

        if self.network is None:
            return

        self.node_name_model.set_domain(network.nodes.domain)

        if self.node_name_model:
            self.node_name = self.node_name_model[0]

        self.controls.node_name.box.setHidden(len(self.node_name_model) < 2)
        self.openContext(self.network.nodes)

        self.set_value_list()
        if self.selected_node_hint \
                and self.selected_node_hint in self.nodes_model:
            self.selected_node = self.selected_node_hint
        self.set_value_list()
        self.update_labels()

    def clear(self):
        self.nodes_model.clear()
        self.network = None
        self.node_name = None
        self.selected_node = None
        self.features_list_label.setText("No features")
        self.friends_list_label.setText("No friends")
        self.rec.setText("No recommendations")

    def set_value_list(self):
        if self.node_name is None:
            self.nodes_model.clear()
        else:
            self.nodes_model[:] = self.network.nodes.get_column(self.node_name)
            self.selected_node = self.nodes_model[0]

    @property
    def selected_node_index(self):
        return self.nodes_model.indexOf(self.selected_node)

    def on_node_changed(self):
        self.selected_node_hint = self.selected_node
        self.update_labels()

    def update_labels(self):
        self.set_friends()
        self.set_features()
        self.set_recommendations()

    def set_friends(self):
        if self.node_name is None or self.selected_node is None:
            self.friends_list_label.setText("No friends")
            return

        sorted_friends, _ = self.sorted_node_weights()

        # neighbours_names = [self.nodes_model[row[1]] for row in sorted_friends]
        neighbours_names = [self.nodes_model[row] if np.isscalar(row)
                            else self.nodes_model[row[1]] for row in
                            sorted_friends]

        self.friends_list_label.setText(", ".join(neighbours_names))

    def get_features_names(self):
        node_choices = self.network.nodes.X[self.selected_node_index]
        attributes = self.network.nodes.domain.attributes
        chosen = np.flatnonzero(node_choices)
        features_names = [attributes[i].name for i in chosen]
        return features_names

    def set_features(self):
        if self.node_name is None or self.selected_node is None:
            features_names = []
        else:
            features_names = self.get_features_names()

        self.features_list_label.setText(", ".join(features_names) or "No features")

    def set_recommendations(self):
        if self.node_name is None or self.selected_node is None:
            self.rec.setText("No recommendations")
            return

        neighbours_indices = self.network.outgoing(self.selected_node_index)
        if neighbours_indices is None:
            self.rec.setText("No recommendations")
            return
        neighbours_choices = self.network.nodes.X[neighbours_indices]
        counts = np.sum(neighbours_choices, axis=0)
        counts[self.network.nodes.X[self.selected_node_index] == 1] = -1
        sorted_features = np.argsort(counts)[::-1]
        most_freq = sorted_features[:5]
        recommended_names = [self.network.nodes.domain.attributes[i].name for i in most_freq]
        neighbours = self.network.nodes.get_column(self.node_name)[neighbours_indices]
        recommenders_for_feature = [
            neighbours[neighbours_choices[:, i] == 1] for i in most_freq]

        output = ""
        for feature, recommenders in zip(recommended_names, recommenders_for_feature):
            output += f"<b>{feature}: </b>"
            output += f"{', '.join(recommenders)}<br/><br/>"
        self.rec.setText(output)

    def sorted_node_weights(self):
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
