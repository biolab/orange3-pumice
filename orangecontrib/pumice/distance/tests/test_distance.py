import unittest
from unittest.mock import Mock, patch, call

import numpy as np

from Orange.data import Domain, Table, StringVariable, DiscreteVariable

from orangecontrib.pumice.distance.distance import \
    LevenshteinDistance, LevenshteinModel


class TestLevenshteinDistance(unittest.TestCase):
    def test_fit(self):
        def callback(i: float) -> bool:
            return True

        dist = LevenshteinDistance(callback=callback, normalize=False, delimiter="|")
        with patch("orangecontrib.pumice.distance.distance.LevenshteinModel.__call__"):
            model = dist.fit(Mock())
        self.assertIsInstance(model, LevenshteinModel)
        self.assertIs(model.callback, callback)
        self.assertFalse(model.normalize)
        self.assertEqual(model.delimiter, "|")


class TestLevenshteinModel(unittest.TestCase):
    def setUp(self):
        self.domain = Domain(
            [DiscreteVariable("foo", values=["a", "b"])],
            None,
            [DiscreteVariable("bar", values=["d", "e"])]
            + [StringVariable(name)
               for name in ("one", "two", "three", "four", "five")]
            + [DiscreteVariable("baz", values=["x", "y"])])
        self.data = Table.from_list(self.domain, [
            ["a", "d", "ena", "dva", "tri", "štiri", "pet", "x"],
            ["b", "d", "jedan", "dva", "tri", "četiri", "pet", "x"],
            ["a", "d", "jeden", "dva", "tři", "čtyři", "pět", "x"],
            ["a", "d", "eins", "zwei", "drei", "vier", "fünf", "x"],
            ["b", "d", "uno", "dos", "tres", "cuatro", "cinco", "x"]])

        self.delim_data = Table.from_list(self.domain, [
            ["a", "d", "ena | too-long-consider", "dva", "tri", "too-long-consider | too-long-consider | štiri", "pet", "x"],
            ["b", "d", "jedan", "too-long-consider| too-long-consider|dva", "tri", "četiri", "pet", "x"],
            ["a", "d", "jeden", "dva", "tři", "čtyři", "pět", "x"],
            ["a", "d", "eins", "zwei", "drei", "vier", "too-long-consider | fünf", "x"],
            ["b", "d", "uno", "dos", "tres", "cuatro", "cinco", "x"]])

    def test_call_requires_tables(self):
        model = LevenshteinModel()
        self.assertRaises(ValueError, model, self.data.metas)
        self.assertRaises(ValueError, model, self.data.X)
        self.assertRaises(ValueError, model, self.data, self.data.metas)

    def test_call_checks_string_metas(self):
        model = LevenshteinModel()
        d = self.data.domain
        a = d.attributes
        self.assertRaises(ValueError, model,
                          self.data.transform(Domain(a, None)))
        self.assertRaises(ValueError, model,
                          self.data.transform(Domain(a, None, d.metas[:1])))
        self.assertRaises(ValueError, model,
                          self.data.transform(Domain(a, None, d.metas[:2])),
                          self.data.transform(Domain(a, None, d.metas[2:])))

    def test_prepare_data(self):
        model = LevenshteinModel()
        data = self.data.transform(
            Domain(self.data.domain.attributes, None, self.data.domain.metas[1:-1]))
        e1metas, e2metas = model._prepare_data(data)
        np.testing.assert_equal(e1metas, data.metas)
        np.testing.assert_equal(e2metas, data.metas)

        e1metas, e2metas = model._prepare_data(data)
        self.assertIs(e1metas, e1metas)
        self.assertIs(e2metas, e2metas)
        self.assertTrue(np.issubdtype(e1metas.dtype, np.str_))
        self.assertTrue(np.issubdtype(e2metas.dtype, np.str_))

        e1 = self.data.transform(
            Domain(self.data.domain.attributes, None, self.data.domain.metas[:3]))
        e2 = self.data.transform(
            Domain(self.data.domain.attributes, None, self.data.domain.metas[1:]))

        e1metas, e2metas = model._prepare_data(e1, e2)
        np.testing.assert_array_equal(e1metas, e1.metas[:, 1:])
        np.testing.assert_array_equal(e2metas, e2.metas[:, :2])
        self.assertTrue(np.issubdtype(e1metas.dtype, np.str_))
        self.assertTrue(np.issubdtype(e2metas.dtype, np.str_))

    def _compute_distance_same(self, data, model):
        def levenmock(word1, word2):
            return abs(len(word1) - len(word2))

        with patch("Levenshtein.distance", side_effect=levenmock):
            smetas = data.metas[:, 1:-1].astype(str)
            dist = model.compute_distances(smetas, smetas)
            np.testing.assert_equal(dist, np.array([
                [0, 3, 2, 5, 4],
                [3, 0, 1, 6, 5],
                [2, 1, 0, 5, 6],
                [5, 6, 5, 0, 5],
                [4, 5, 6, 5, 0]]))

            smetas[0, 3] = ""
            dist = model.compute_distances(smetas, smetas)
            np.testing.assert_almost_equal(dist, np.array([
                [0, 2 + 7/6, 2 + 7/6, 4 + 7/6, 3 + 7/6],
                [2 + 7/6, 0, 1, 6, 5],
                [2 + 7/6, 1, 0, 5, 6],
                [4 + 7/6, 6, 5, 0, 5],
                [3 + 7/6, 5, 6, 5, 0]]))

            smetas[:, 3] = ""
            dist = model.compute_distances(smetas, smetas)
            np.testing.assert_equal(dist, np.array([
                [0, 2, 2, 4, 3],
                [2, 0, 0, 4, 5],
                [2, 0, 0, 4, 5],
                [4, 4, 4, 0, 3],
                [3, 5, 5, 3, 0]]))

            smetas = data.metas[:, 1:-1].astype(str)
            smetas[0] = ""
            dist = model.compute_distances(smetas, smetas)
            d = 7 / 6 + 3 / 6 + 4 / 6 + 7 / 6 + 7 / 6
            np.testing.assert_almost_equal(dist, np.array([
                [0, d, d, d, d],
                [d, 0, 1, 6, 5],
                [d, 1, 0, 5, 6],
                [d, 6, 5, 0, 5],
                [d, 5, 6, 5, 0]]))

    def test_compute_distance_same_no_delim(self):
        model = LevenshteinModel(normalize=False)
        self._compute_distance_same(self.data, model)

    def test_compute_distance_same_with_delim(self):
        model = LevenshteinModel(normalize=False, delimiter="|")
        self._compute_distance_same(self.delim_data, model)

    def _compute_distance_diff(self, data, model):
        def levenmock(word1, word2):
            return abs(len(word1) - len(word2))

        with patch("Levenshtein.distance", side_effect=levenmock):
            smetas1 = data.metas[:3, 1:-1].astype(str)
            smetas2 = data.metas[3:, 1:-1].astype(str)
            dist = model.compute_distances(smetas1, smetas2)
            np.testing.assert_equal(dist, np.array([
                [5, 4],
                [6, 5],
                [5, 6]]))
            dist = model.compute_distances(smetas2, smetas1)
            np.testing.assert_equal(dist, np.array([
                [5, 4],
                [6, 5],
                [5, 6]]).T)

            smetas1[0, 0] = ""
            dist = model.compute_distances(smetas1, smetas2)
            np.testing.assert_almost_equal(dist, np.array([
                [4 + 3 / 2, 4 + 3 / 2],
                [6, 5],
                [5, 6]]))

            dist = model.compute_distances(smetas2, smetas1)
            np.testing.assert_almost_equal(dist, np.array([
                [4 + 3 / 2, 4 + 3 / 2],
                [6, 5],
                [5, 6]]).T)

            smetas1[:, 0] = ""
            dist = model.compute_distances(smetas1, smetas2)
            np.testing.assert_equal(dist, np.array([
                [4, 4],
                [5, 3],
                [4, 4]]))

            dist = model.compute_distances(smetas2, smetas1)
            np.testing.assert_equal(dist, np.array([
                [4, 4],
                [5, 3],
                [4, 4]]).T)

            smetas1 = data.metas[:3, 1:-1].astype(str)
            smetas1[0] = ""
            dist = model.compute_distances(smetas1, smetas2)
            d = 6 / 4 + 1 / 4 + 4 / 4 + 5 / 4 + 6 / 4
            np.testing.assert_equal(dist, np.array([
                [d, d],
                [6, 5],
                [5, 6]]))

            dist = model.compute_distances(smetas2, smetas1)
            np.testing.assert_equal(dist, np.array([
                [d, d],
                [6, 5],
                [5, 6]]).T)

    def test_compute_distance_diff_no_delim(self):
        model = LevenshteinModel(normalize=False)
        self._compute_distance_diff(self.data, model)

    def test_compute_distance_diff_with_delim(self):
        model = LevenshteinModel(normalize=False, delimiter="|")
        self._compute_distance_diff(self.delim_data, model)

    def test_callback(self):
        callback = Mock()
        model = LevenshteinModel(callback=callback)
        model(self.data[:3])
        self.assertEqual(
            callback.call_args_list,
            [call(0), call(20), call(40), call(60), call(80)])

    def test_normalize(self):
        with patch("Levenshtein.distance", side_effect=lambda *_: 1), \
                patch("Levenshtein.ratio", side_effect=lambda *_: 0.5):
            model = LevenshteinModel(normalize=True)
            dist = model(self.data)
            d = np.array([
                [0, 1, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 0, 1],
                [1, 1, 1, 1, 0]])
            np.testing.assert_equal(dist, 2.5 * d)
            model = LevenshteinModel(normalize=False)
            dist = model(self.data)
            np.testing.assert_equal(dist, 5 * d)


if __name__ == '__main__':
    unittest.main()