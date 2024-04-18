import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.sparse import csr_matrix

from Orange.data import Domain, Table, StringVariable, ContinuousVariable, \
    DiscreteVariable
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.network import Network

from orangecontrib.pumice.widgets.owrecommendation import OWRecommendation


class TestOWRecommendation(WidgetTest):
    def setUp(self):
        self.widget: OWRecommendation = self.create_widget(OWRecommendation)

        edges = "AF AC BC BD BE BF BG CD CF CG DG EA EF GB GD GF"
        mat = np.zeros((7, 7))
        for edge in edges.split():
            mat[ord(edge[0]) - ord("A"), ord(edge[1]) - ord("A")] = 1
        self.edges = csr_matrix(mat)
        self.choices = np.array(
            [list(map(int, row)) for row in ("00110100",
                                             "10001010",
                                             "11101010",
                                             "01001110",
                                             "01101110",
                                             "00000000",
                                             "11111111")])
        self.names = "Cilka Ana Franz Greta Benjamin Dani Ema".split()
        self.items = list("ADBCGFEH")

        domain = Domain([ContinuousVariable(x) for x in self.items],
                        None,
                        [StringVariable("name")])
        data = Table.from_numpy(
            domain, self.choices, metas=np.array(self.names)[:, None])
        self.network_one_name = Network(data, self.edges)

        domain = Domain([ContinuousVariable(x) for x in self.items],
                        None,
                        [StringVariable("name"),
                         StringVariable("name2"),
                         ContinuousVariable("x"),
                         StringVariable("non-unique")])
        data = Table.from_numpy(
            domain, self.choices,
            metas=np.array([[n, n[::-1], 0, self.names[i % 4]]
                            for i, n in enumerate(self.names)]))
        self.network_more_names = Network(data, self.edges)

        self.network_array = Network(np.array(self.names), self.edges)

        domain = Domain([ContinuousVariable(x) for x in self.names],
                        None,
                        [StringVariable("name")])
        self.data_name = Table.from_numpy(
            domain, self.choices.T, metas=np.array(self.items)[:, None])

    def test_network_no_data(self):
        w = self.widget
        self.assertFalse(w.is_valid)
        self.assertFalse(w.Error.no_choices.is_shown())

        self.send_signal(w.Inputs.network, Network(self.names, self.edges))
        self.assertFalse(w.is_valid)
        self.assertTrue(w.Error.no_choices.is_shown())

        self.send_signal(w.Inputs.network, None)
        self.assertFalse(w.is_valid)
        self.assertFalse(w.Error.no_choices.is_shown())

    def test_init_person_column_from_net_table_one_name(self):
        w = self.widget
        self.assertFalse(w.is_valid)
        self.assertFalse(w.Error.no_choices.is_shown())

        self.send_signal(w.Inputs.network, self.network_one_name)

        for _ in range(2):
            # First, test with network that has only one name
            self.assertTrue(w.is_valid)

            self.assertEqual(list(w.person_names), sorted(self.names))
            self.assertEqual(list(w.item_names), self.items)
            np.testing.assert_equal(w.choices,
                                    self.choices[[1, 4, 0, 5, 6, 2, 3]].astype(bool))
            self.assertTrue(w.column_box.isHidden())

            # In the next iteration, repeat with data whose attributes match the column
            self.send_signal(w.Inputs.item_data, self.data_name)

        # Now remove the network to clear the widget
        self.send_signal(w.Inputs.network, None)
        self.assertIsNone(w.person_names)

        # Now try with network that has multiple attributes, but just one
        # matches the data columns, so only one attribute is applicable
        self.send_signal(w.Inputs.network, self.network_more_names)
        self.assertEqual(list(w.person_names), sorted(self.names))
        self.assertEqual(list(w.item_names), self.items)
        np.testing.assert_equal(w.choices,
                                self.choices[[1, 4, 0, 5, 6, 2, 3]].astype(bool))
        self.assertTrue(w.column_box.isHidden())

        # Remove input data - now we again have multiple applicable columns
        self.send_signal(w.Inputs.item_data, None)
        self.assertFalse(w.column_box.isHidden())

    def test_init_person_column_from_net_table_more_names(self):
        w = self.widget
        self.assertFalse(w.is_valid)
        self.assertFalse(w.Error.no_choices.is_shown())

        self.send_signal(w.Inputs.network, self.network_more_names)
        self.assertTrue(w.is_valid)

        self.assertEqual(list(w.person_names), sorted(self.names))
        self.assertEqual(list(w.item_names), self.items)
        np.testing.assert_equal(w.choices,
                                self.choices[[1, 4, 0, 5, 6, 2, 3]].astype(bool))
        self.assertFalse(w.column_box.isHidden())
        domain = self.network_more_names.nodes.domain
        self.assertEqual(tuple(w.person_column_model), domain.metas[:2])
        self.assertIs(w.person_column_model[0], domain.metas[0])

    def test_init_person_column_from_net_table_no_names(self):
        w = self.widget
        self.assertFalse(w.is_valid)
        self.assertFalse(w.Error.no_choices.is_shown())

        domain = Domain([ContinuousVariable(x) for x in self.items],
                        None)
        data = Table.from_numpy(domain, self.choices)
        self.send_signal(w.Inputs.network, Network(data, self.edges))
        self.assertFalse(w.is_valid)
        self.assertTrue(w.Error.no_user_names_in_net.is_shown())

        self.send_signal(w.Inputs.network, None)
        self.assertFalse(w.is_valid)
        self.assertFalse(w.Error.no_user_names_in_net.is_shown())

        domain = Domain([ContinuousVariable(x) for x in self.items],
                        None,
                        [ContinuousVariable("x"),
                         StringVariable("non-unique")])
        data = Table.from_numpy(
            domain, self.choices,
            metas=np.array([[0, self.names[i % 4]]
                            for i, n in enumerate(self.names)]))
        self.send_signal(w.Inputs.network, Network(data, self.edges))
        self.assertFalse(w.is_valid)
        self.assertTrue(w.Error.no_user_names_in_net.is_shown())

        self.send_signal(w.Inputs.network, None)
        self.assertFalse(w.is_valid)
        self.assertFalse(w.Error.no_user_names_in_net.is_shown())

    def test_init_person_column_from_net_table_no_applicable_names(self):
        w = self.widget
        self.assertFalse(w.is_valid)
        self.assertFalse(w.Error.no_choices.is_shown())

        self.send_signal(w.Inputs.network, self.network_more_names)
        self.assertTrue(w.is_valid)
        self.assertFalse(w.column_box.isHidden())

        self.send_signal(
            w.Inputs.item_data,
            Table.from_list(Domain([ContinuousVariable(x) for x in "abcdefgh"]),
                            [[0] * 8])
                            )
        self.assertFalse(w.is_valid)
        self.assertTrue(w.Error.no_user_names_in_net.is_shown())

    def test_init_person_column_from_net_array(self):
        w = self.widget
        self.send_signal(w.Inputs.network, self.network_array)
        self.assertFalse(w.is_valid)
        self.assertTrue(w.column_box.isHidden())
        self.assertTrue(w.Error.no_choices)

        self.send_signal(w.Inputs.item_data, self.data_name)
        self.assertTrue(w.is_valid)
        self.assertEqual(list(w.person_names), sorted(self.names))

    def test_init_person_column_from_net_array_inapplicable(self):
        w = self.widget

        self.send_signal(w.Inputs.network, self.network_array)
        self.send_signal(
            w.Inputs.item_data,
            Table.from_list(Domain([ContinuousVariable(x) for x in "abcdefgh"]),
                            [[0] * 8])
                            )
        self.assertFalse(w.is_valid)
        self.assertFalse(w.Error.no_user_names_in_net.is_shown())
        self.assertTrue(w.Error.user_names_mismatch.is_shown())

    def test_init_item_column_from_net(self):
        w = self.widget
        self.send_signal(w.Inputs.network, self.network_one_name)
        self.assertTrue(w.is_valid)
        np.testing.assert_equal(w.item_names, self.items)

        self.send_signal(w.Inputs.network, self.network_more_names)
        self.assertTrue(w.is_valid)
        np.testing.assert_equal(w.item_names, self.items)

        self.send_signal(w.Inputs.network, Network(
            Table.from_list(Domain([], None, [StringVariable("name")]),
                            [[str(n)] for n in range(self.edges.shape[0])]),
            self.edges))
        self.assertFalse(w.is_valid)

    @patch("orangecontrib.pumice.widgets.owrecommendation.OWRecommendation.set_images")
    def test_init_item_column_from_data(self, set_images):
        w = self.widget
        nnodes = self.edges.shape[0]
        nodes = np.array([f"n{i}" for i in range(nnodes)])
        attributes = [ContinuousVariable(name) for name in nodes]

        name1 = StringVariable("name")
        name2 = StringVariable("name2")
        image1 = StringVariable("image1")
        image2 = StringVariable("image2")
        image1.attributes["type"] = "image"
        image2.attributes["type"] = "image"
        x = ContinuousVariable("x")
        d = DiscreteVariable("y", values=("a", "b"))

        a = np.zeros((5, nnodes))
        m = np.array([[chr(65 + i), chr(70 + i), chr(75 + i), chr(80 + i)]
                     for i in range(5)])

        n = Network(nodes, self.edges)
        self.send_signal(w.Inputs.network, n)

        self.send_signal(w.Inputs.item_data, Table.from_numpy(
            Domain(attributes, None, [name1, name2]),
            a, metas=m[:, :2]))
        self.assertTrue(w.is_valid)
        self.assertIs(w.item_column, name1)
        self.assertEqual(list(w.item_names), list("ABCDE"))
        self.assertIsNone(w.image_column)
        self.assertFalse(w.column_box.isHidden())
        self.assertEqual(tuple(w.item_column_model), (name1, name2))
        set_images.assert_called_once()
        set_images.reset_mock()

        self.send_signal(w.Inputs.item_data, Table.from_numpy(
            Domain(attributes, None, [image1, image2, name1, name2]),
            a, metas=m))
        self.assertTrue(w.is_valid)
        self.assertIs(w.item_column, name1)
        self.assertEqual(list(w.item_names), list("KLMNO"))
        self.assertIs(w.image_column, image1)
        self.assertFalse(w.column_box.isHidden())
        self.assertEqual(tuple(w.item_column_model), (name1, name2))
        set_images.assert_called_once()
        set_images.reset_mock()

        self.send_signal(w.Inputs.item_data, Table.from_numpy(
            Domain(attributes, None, [name1, image1, image2, name2]),
            a, metas=m))
        self.assertTrue(w.is_valid)
        self.assertIs(w.item_column, name1)
        self.assertEqual(list(w.item_names), list("ABCDE"))
        self.assertIs(w.image_column, image1)
        self.assertFalse(w.column_box.isHidden())
        self.assertEqual(tuple(w.item_column_model), (name1, name2))
        set_images.assert_called_once()
        set_images.reset_mock()

        self.send_signal(w.Inputs.item_data, Table.from_numpy(
            Domain(attributes, None, [image1, image2, name2, name1]),
            a, metas=m))
        self.assertTrue(w.is_valid)
        self.assertIs(w.item_column, name2)
        self.assertEqual(list(w.item_names), list("KLMNO"))
        self.assertIs(w.image_column, image1)
        self.assertFalse(w.column_box.isHidden())
        self.assertEqual(tuple(w.item_column_model), (name2, name1))
        set_images.assert_called_once()
        set_images.reset_mock()

        self.send_signal(w.Inputs.item_data, Table.from_numpy(
            Domain(attributes, None, [image1, image2, name1]),
            a, metas=m[:, :3]))
        self.assertTrue(w.is_valid)
        self.assertIsNone(w.item_column)
        self.assertEqual(list(w.item_names), list("KLMNO"))
        self.assertIs(w.image_column, image1)
        self.assertTrue(w.column_box.isHidden())
        self.assertEqual(tuple(w.item_column_model), ())
        set_images.assert_called_once()
        set_images.reset_mock()

        self.send_signal(w.Inputs.item_data, Table.from_numpy(
            Domain(attributes, None, [image1, image2]),
            a, metas=m[:, :2]))
        self.assertTrue(w.is_valid)
        self.assertIs(w.item_column, image1)
        self.assertEqual(list(w.item_names), list("ABCDE"))
        self.assertIs(w.image_column, image1)
        self.assertFalse(w.column_box.isHidden())
        self.assertEqual(tuple(w.item_column_model), (image1, image2))
        set_images.assert_called_once()
        set_images.reset_mock()

        self.send_signal(w.Inputs.item_data, Table.from_numpy(
            Domain(attributes, None, [image1]),
            a, metas=m[:, :1]))
        self.assertTrue(w.is_valid)
        self.assertIsNone(w.item_column)
        self.assertEqual(list(w.item_names), list("ABCDE"))
        self.assertIs(w.image_column, image1)
        self.assertTrue(w.column_box.isHidden())
        self.assertEqual(tuple(w.item_column_model), ())
        set_images.assert_called_once()
        set_images.reset_mock()

        w.item_column_hint = "image2"
        self.send_signal(w.Inputs.item_data, Table.from_numpy(
            Domain(attributes, None, [image1, image2, name1]),
            a, metas=m[:, :3]))
        self.assertTrue(w.is_valid)
        self.assertIs(w.item_column, image2)
        self.assertEqual(list(w.item_names), list("FGHIJ"))
        self.assertIs(w.image_column, image1)
        self.assertFalse(w.column_box.isHidden())
        self.assertEqual(tuple(w.item_column_model), (image2, name1))
        set_images.assert_called_once()
        set_images.reset_mock()


if __name__ == '__main__':
    unittest.main()
