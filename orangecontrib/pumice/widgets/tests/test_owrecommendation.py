import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.sparse import csr_matrix

from Orange.data import Domain, Table, StringVariable, ContinuousVariable, \
    DiscreteVariable
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.network import Network
# TODO: when DirectedEdges is reexported from network, import it from there
from orangecontrib.network.network.base import DirectedEdges

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
        self.network_one_name = Network(data, DirectedEdges(self.edges))

        domain = Domain([ContinuousVariable(x) for x in self.items],
                        None,
                        [StringVariable("name"),
                         StringVariable("name2"),
                         ContinuousVariable("x"),
                         StringVariable("non-unique")])
        data = Table.from_numpy(
            domain, self.choices,
            metas=np.array([[n, self.names[-i - 1], 0, self.names[i % 4]]
                            for i, n in enumerate(self.names)]))
        self.network_more_names = Network(data, self.edges)

        self.network_array = Network(np.array(self.names), self.edges)

        domain = Domain([ContinuousVariable(x) for x in self.names],
                        None,
                        [StringVariable("name")])
        self.data_name = Table.from_numpy(
            domain, self.choices.T, metas=np.array(self.items)[:, None])

        nnodes = self.edges.shape[0]
        nodes = np.array([f"n{i}" for i in range(nnodes)])
        attributes = [ContinuousVariable(name) for name in nodes]

        name1 = StringVariable("name")
        name2 = StringVariable("name2")
        image1 = StringVariable("image1")
        image2 = StringVariable("image2")
        image1.attributes["type"] = "image"
        image2.attributes["type"] = "image"
        a = np.zeros((5, nnodes))
        m = np.array([[chr(65 + i), chr(70 + i), chr(75 + i), chr(80 + i)]
                     for i in range(5)])
        self.more_item_data = Table.from_numpy(
            Domain(attributes, None, [name1, name2, image1, image2]),
            a, metas=m)

    def test_minimum_size(self):
        pass

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

            self.assertEqual(list(w.person_names), self.names)
            self.assertEqual(list(w.item_names), self.items)
            self.assertTrue(w.column_box.isHidden())

            # In the next iteration, repeat with data whose attributes match the column
            self.send_signal(w.Inputs.item_data, self.data_name)

        # Now remove the network to clear the widget
        self.send_signal(w.Inputs.network, None)
        self.assertIsNone(w.person_names)

        # Now try with network that has multiple attributes, but just one
        # matches the data columns, so only one attribute is applicable
        data = self.network_more_names.nodes
        with data.unlocked(data.metas):
            data.metas[0, 1] = "foo"
        self.send_signal(w.Inputs.network, self.network_more_names)
        self.assertEqual(list(w.person_names), self.names)
        self.assertEqual(list(w.item_names), self.items)
        self.assertTrue(w.column_box.isHidden())

        # Remove input data - now we again have multiple applicable columns
        self.send_signal(w.Inputs.item_data, None)
        self.assertFalse(w.column_box.isHidden())
        self.assertFalse(w.controls.person_column.box.isHidden())
        self.assertTrue(w.controls.item_column.box.isHidden())

    def test_init_person_column_from_net_table_more_names(self):
        w = self.widget
        self.assertFalse(w.is_valid)
        self.assertFalse(w.Error.no_choices.is_shown())

        self.send_signal(w.Inputs.network, self.network_more_names)
        self.assertTrue(w.is_valid)

        self.assertEqual(list(w.person_names), self.names)
        self.assertEqual(list(w.item_names), self.items)
        self.assertFalse(w.column_box.isHidden())
        self.assertFalse(w.controls.person_column.box.isHidden())
        self.assertTrue(w.controls.item_column.box.isHidden())

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
        self.assertFalse(w.controls.person_column.box.isHidden())
        self.assertTrue(w.controls.item_column.box.isHidden())

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
        self.assertEqual(list(w.person_names), self.names)

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
        self.assertTrue(w.controls.person_column.box.isHidden())
        self.assertFalse(w.controls.item_column.box.isHidden())
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
        self.assertTrue(w.controls.person_column.box.isHidden())
        self.assertFalse(w.controls.item_column.box.isHidden())
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
        self.assertTrue(w.controls.person_column.box.isHidden())
        self.assertFalse(w.controls.item_column.box.isHidden())
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
        self.assertTrue(w.controls.person_column.box.isHidden())
        self.assertFalse(w.controls.item_column.box.isHidden())
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
        self.assertTrue(w.controls.person_column.box.isHidden())
        self.assertFalse(w.controls.item_column.box.isHidden())
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
        self.assertTrue(w.controls.person_column.box.isHidden())
        self.assertFalse(w.controls.item_column.box.isHidden())
        self.assertEqual(tuple(w.item_column_model), (image2, name1))
        set_images.assert_called_once()
        set_images.reset_mock()

    @patch("orangecontrib.pumice.widgets.owrecommendation.OWRecommendation.update_page")
    def test_on_person_column_changed(self, update_page):
        w = self.widget
        combo = w.controls.person_column

        self.send_signal(w.Inputs.network, self.network_more_names)
        self.send_signal(w.Inputs.item_data, self.data_name)
        self.assertTrue(w.is_valid)
        self.assertFalse(w.column_box.isHidden())
        self.assertTrue(w.controls.item_column.box.isHidden())
        self.assertFalse(w.controls.person_column.box.isHidden())
        self.assertEqual([v.name for v in w.person_column_model], ["name", "name2"])
        self.assertEqual(w.person_column.name, "name")
        np.testing.assert_equal(w.person_names, self.names)
        np.testing.assert_equal(w.choices, self.data_name.X.T)

        update_page.reset_mock()
        combo.setCurrentIndex(1)
        combo.activated.emit(1)
        np.testing.assert_equal(w.person_names, self.names[::-1])
        np.testing.assert_equal(w.choices, self.data_name.X.T[::-1])
        update_page.assert_called()

    def test_set_person_names(self):
        w = self.widget
        self.send_signal(w.Inputs.network, self.network_more_names)
        self.send_signal(w.Inputs.item_data, self.data_name)

        w._set_person_names(np.array(self.names[3:] + self.names[:3]))
        arr = self.data_name.X.T
        np.testing.assert_equal(w.choices, np.vstack((arr[3:], arr[:3])))

    def test_person_column_hint(self):
        w = self.widget
        combo = w.controls.person_column

        self.send_signal(w.Inputs.network, self.network_more_names)
        self.send_signal(w.Inputs.item_data, self.data_name)
        assert w.person_column.name == "name"

        combo.setCurrentIndex(1)
        combo.activated.emit(1)

        assert w.person_column.name == "name2"

        self.send_signal(w.Inputs.network, None)
        assert w.person_column is None

        self.send_signal(w.Inputs.network, self.network_more_names)
        self.assertEqual(w.person_column.name, "name2")

    @patch("orangecontrib.pumice.widgets.owrecommendation.OWRecommendation.update_page")
    def test_on_item_column_changed(self, update_page):
        w = self.widget
        combo = w.controls.item_column

        nnodes = self.edges.shape[0]
        nodes = np.array([f"n{i}" for i in range(nnodes)])
        n = Network(nodes, self.edges)
        self.send_signal(w.Inputs.network, n)
        self.send_signal(w.Inputs.item_data, self.more_item_data)
        assert w.is_valid

        self.assertFalse(w.column_box.isHidden())
        self.assertFalse(w.controls.item_column.box.isHidden())
        self.assertTrue(w.controls.person_column.box.isHidden())

        np.testing.assert_equal(w.item_names, self.more_item_data.metas[:, 0])

        update_page.reset_mock()
        combo.setCurrentIndex(1)
        combo.activated.emit(1)

        update_page.assert_called()
        update_page.reset_mock()
        np.testing.assert_equal(w.item_names, self.more_item_data.metas[:, 1])

        self.send_signal(w.Inputs.item_data, None)
        assert w.item_names is None
        update_page.assert_called()
        update_page.reset_mock()

        self.send_signal(w.Inputs.item_data, self.more_item_data)
        np.testing.assert_equal(w.item_names, self.more_item_data.metas[:, 1])
        update_page.assert_called()

    def test_include_image_column_if_hinted(self):
        w = self.widget
        combo = w.controls.item_column
        w.item_column_hint = "image2"

        nnodes = self.edges.shape[0]
        nodes = np.array([f"n{i}" for i in range(nnodes)])
        n = Network(nodes, self.edges)
        self.send_signal(w.Inputs.network, n)
        self.send_signal(w.Inputs.item_data, self.more_item_data)
        assert w.is_valid

        self.assertEqual(w.item_column.name, "image2")
        self.assertEqual({v.name for v in w.item_column_model},
                         {"name", "name2", "image2"})

    def test_update_page_sets_model(self):
        w = self.widget
        w.rec_model = Mock()
        self.send_signal(w.Inputs.network, self.network_array)
        self.assertFalse(w.is_valid)
        w.rec_model.reset.assert_called()
        w.rec_model.reset.reset_mock()
        w.rec_model.set_data.assert_not_called()

        self.send_signal(w.Inputs.item_data, self.data_name)
        self.assertTrue(w.is_valid)
        w.rec_model.set_data.assert_called()

    @patch("os.path.exists", new=lambda x: bool(ord(x[-1]) % 2))
    @patch("orangecontrib.pumice.widgets.owrecommendation.QPixmap")
    def test_set_images(self, _):
        w = self.widget

        w.image_column = None
        w.set_images()
        self.assertIsNone(w.images)

        nnodes = self.edges.shape[0]
        nodes = np.array([f"n{i}" for i in range(nnodes)])
        n = Network(nodes, self.edges)
        self.send_signal(w.Inputs.network, n)
        self.send_signal(w.Inputs.item_data, self.more_item_data)

        self.assertIsNotNone(w.images[0])
        self.assertIsNone(w.images[1])
        self.assertIsNotNone(w.images[2])
        self.assertIsNone(w.images[3])
        self.assertIsNotNone(w.images[4])

    def test_get_friends_one(self):
        w = self.widget
        self.network_one_name.edges[0].edges[1, 3] = 0.5
        self.send_signal(w.Inputs.network, self.network_one_name)

        np.testing.assert_equal(w._get_friends_one(0)[0], [2, 5])
        np.testing.assert_almost_equal(w._get_friends_one(0)[1], [1, 1])
        # 3 has a lower weight, so it must be last
        np.testing.assert_equal(w._get_friends_one(1)[0], [2, 4, 5, 6, 3])
        np.testing.assert_almost_equal(w._get_friends_one(1)[1], [1, 1, 1, 1, 0.5])
        np.testing.assert_equal(w._get_friends_one(2)[0], [3, 5, 6])
        np.testing.assert_equal(w._get_friends_one(5)[0], [])
        np.testing.assert_equal(w._get_friends_one(6)[0], [1, 3, 5])

    def test_get_friends(self):
        w = self.widget
        self.assertIsNone(w.get_friends())

        self.network_one_name.edges[0].edges[1, 3] = 0.5
        self.send_signal(w.Inputs.network, self.network_one_name)

        self.assertEqual([list(x[0]) for x in w.get_friends()],
                         [[2, 5], [2, 4, 5, 6, 3], [3, 5, 6], [6], [0, 5],
                          [], [1, 3, 5]])
        np.testing.assert_almost_equal(w.get_friends()[1][1], [1, 1, 1, 1, 0.5])

    def test_get_recommendations_one(self):
        w = self.widget
        self.network_one_name.edges[0].edges[1, 3] = 0.5
        self.send_signal(w.Inputs.network, self.network_one_name)

        for data_present in ("No", "Yes"):
            with self.subTest(data_present=data_present):
                recs = [tuple(w._get_recommendations_one(0, 5)[0]) for _ in range(10)]
                # Same recommendations, but in different order
                self.assertTrue(all(set(rec[:-1]) == {0, 1, 4, 6} for rec in recs))
                self.assertGreater(len(set(recs)), 1)
                self.assertTrue(all(rec[-1] == 7 for rec in recs))
                self.assertEqual(w._get_recommendations_one(0, 5)[1], [[2]] * 4 + [[]])

                self.assertEqual(w._get_recommendations_one(1, 1)[0], [1])
                self.assertEqual(set(w._get_recommendations_one(1, 1)[1][0]), {2, 3, 4, 6})
                self.assertEqual(set(w._get_recommendations_one(1, 3)[0]), {1, 2, 5})
                self.assertEqual(set(w._get_recommendations_one(1, 3)[1][0]), {2, 3, 4, 6})

                for n in (5, 6, 7, 10):
                    with self.subTest(n=n):
                        for _ in range(5):
                            recs = w._get_recommendations_one(1, 5)[0]
                            self.assertEqual(recs[0], 1)
                            self.assertEqual(set(recs[1:3]), {2, 5})
                            self.assertEqual(set(recs[3:5]), {3, 7})

                self.send_signal(w.Inputs.item_data, self.data_name)


if __name__ == '__main__':
    unittest.main()
