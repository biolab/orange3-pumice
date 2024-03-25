import unittest
from unittest.mock import Mock, patch
import os
from os.path import join, dirname

import numpy as np

import Orange
from Orange.data import Domain, Table, StringVariable, DiscreteVariable, ContinuousVariable
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.network import Network
from orangecontrib.network.network.readwrite import read_pajek

from Recommendation.owrecommendation import Recommendation


class TestowRecommendation(WidgetTest):

    def setUp(self):
        file_path_xlsx = join(dirname(dirname(__file__)), 'networks', 'data.xlsx')
        file_path_net = join(dirname(dirname(__file__)), 'networks', 'nodes_features.net')
        self.table = Table(file_path_xlsx)
        self.network = read_pajek(file_path_net)
        self.network.nodes = self.table
        self.widget: Recommendation = self.create_widget(Recommendation)

    def test_set_features_node_selected_domain(self):
        domain = Domain(
            [DiscreteVariable(name="Feature 1"), DiscreteVariable(name="Feature 2")],
            DiscreteVariable(name="Feature 3"),
            [StringVariable(name="Feature 4"),
             DiscreteVariable(name="Feature 4.5", values=["Value1", "Value2"]),
             StringVariable(name="Feature 5")]
        )

        self.widget.node_name_model.set_domain(domain)

        checking_types = (isinstance(variable, StringVariable) for variable in self.widget.node_name_model)

        self.assertTrue(all(checking_types))

    def test_set_features_node_not_selected(self):
        self.widget.node_name = None
        self.widget.selected_node = None
        self.widget.network = self.network
        self.widget.set_features()
        self.assertEqual(self.widget.features_list_label.text(), "No features")

    def test_set_features_node_selected(self):
        self.send_signal(self.network)
        self.widget.node_name = "Feature 1"
        self.widget.selected_node = "Ema Novak"
        self.widget.set_features()
        self.assertEqual(self.widget.features_list_label.text(), "Moana, Spirited Away, Spider-Man: Into the "
                                                                 "Spider-Verse, Kung Fu Panda, Hotel Transylvania")

    def test_set_friends_node_not_selected(self):
        self.widget.node_name = None
        self.widget.selected_node = None
        self.widget.network = self.network
        self.widget.set_friends()
        self.assertEqual(self.widget.friends_list_label.text(), "No friends")

    def test_set_recommendations_node_not_selected(self):
        self.widget.node_name = None
        self.widget.selected_node = None
        self.widget.network = self.network
        self.widget.set_recommendations()
        self.assertEqual(self.widget.rec.text(), "No recommendations")

    def test_set_recommendations_node_selected(self):
        self.send_signal(self.network)
        self.widget.node_name = "Feature 1"
        self.widget.selected_node = "Ema Novak"
        self.widget.set_recommendations()
        self.assertEqual(self.widget.rec.text(), "<b>Coco: </b>Maja Čeh, Tim Bizjak<br/><br/>"
                                                 "<b>Lilo & Stich: </b>Jan Horvat, Zoja Leban<br/><br/>"
                                                 "<b>Tangled 1: </b>Maja Čeh<br/><br/>"
                                                 "<b>Happy Feet: </b>Jan Horvat<br/><br/>"
                                                 "<b>Rango: </b>Tim Bizjak<br/><br/>")

    def test_set_recommendations_no_network(self):
        self.widget.set_recommendations()
        self.assertEqual(self.widget.rec.text(), "No recommendations")

    def test_set_recommendations_no_features(self):
        self.send_signal(self.network)
        self.widget.node_name = None
        self.widget.selected_node = "Ema Novak"
        self.widget.set_recommendations()
        self.assertEqual(self.widget.rec.text(), "No recommendations")

    def test_set_recommendations_no_selected_node(self):
        # Test scenario where no node is selected
        self.send_signal(self.network)
        self.widget.node_name = "Feature 1"
        self.widget.selected_node = None
        self.widget.set_recommendations()
        self.assertEqual(self.widget.rec.text(), "No recommendations")

    def test_on_node_changed(self):
        self.send_signal(self.network)
        self.widget.node_name = "Feature 1"
        self.widget.selected_node = "Ema Novak"
        self.widget.on_node_changed()
        self.assertEqual(self.widget.selected_node, "Ema Novak")
        self.send_signal(None)
        self.assertEqual(self.widget.selected_node, None)
        self.send_signal(self.network)
        self.assertEqual(self.widget.selected_node, "Ema Novak")

    @patch('Recommendation.owrecommendation.Recommendation.set_features')
    @patch('Recommendation.owrecommendation.Recommendation.set_friends')
    @patch('Recommendation.owrecommendation.Recommendation.set_recommendations')
    def test_update_calls_methods(self, mock_set_recommendations, mock_set_friends, mock_set_features):
        self.widget.update_labels()

        mock_set_features.assert_called_once()
        mock_set_friends.assert_called_once()
        mock_set_recommendations.assert_called_once()

    def test_set_value_list(self):
        self.send_signal(self.network)
        self.widget.set_value_list()
        self.assertEqual(self.widget.node_name, StringVariable(name='Feature 1'))

    def test_node_name_none(self):
        self.widget.set_value_list()
        self.assertEqual(self.widget.node_name, None)

    def test_clear_widget_content(self):
        self.widget.set_network(self.network)
        self.widget.set_network(None)

        self.assertEqual(self.widget.features_list_label.text(), "No features")
        self.assertEqual(self.widget.friends_list_label.text(), "No friends")
        self.assertEqual(self.widget.rec.text(), "No recommendations")

    def test_set_network(self):
        self.widget.set_network(self.network)
        self.assertEqual(self.widget.network, self.network)

    def test_set_friends_empty_sorted_data(self):
        self.send_signal(self.network)
        self.widget.node_name = "Feature 1"
        self.widget.selected_node = "Ema Novak"
        self.widget.sort_node_weights = Mock(return_value=[])
        self.widget.set_friends()
        neighb = self.network.neighbours(self.widget.selected_node_index)
        self.assertEqual(self.widget.friends_list_label.text(),
                         "Tilen Novak, Tim Bizjak, Zoja Leban, Jan Horvat, Maja Čeh")

    def test_set_network_none(self):
        self.widget.set_network(None)
        self.assertIsNone(self.widget.network)

    def test_set_network_selected_node_hint_in_nodes_model(self):
        self.widget.selected_node_hint = "Ema Novak"
        self.send_signal(self.network)
        self.assertEqual(self.widget.selected_node, "Ema Novak")




if __name__ == '__main__':
    unittest.main()
