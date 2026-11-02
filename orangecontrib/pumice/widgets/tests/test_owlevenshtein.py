import unittest
from unittest.mock import Mock, patch

import numpy as np

from AnyQt import QtCore
from AnyQt.QtTest import QTest

from Orange.data import Domain, Table, StringVariable, DiscreteVariable
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.pumice.distance.distance import LevenshteinModel

from orangecontrib.pumice.widgets.owlevenshtein import OWLevenshtein, \
    InterruptException


def levenmock(word1, word2):
    return abs(len(word1) - len(word2))

class TestLevenshteinModel(WidgetTest):
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

        self.widget: OWLevenshtein = self.create_widget(OWLevenshtein)

    def test_attr_list(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertEqual(
            list(self.widget.attr_model),
            ["one", "two", "three", "four", "five"])

    @patch("Levenshtein.distance", side_effect=levenmock)
    def test_selection_updates_output(self, *_):
        widget = self.widget
        widget.normalize = False
        widget.auto_apply = True
        widget.selection_timer.start = widget.commit.deferred

        self.send_signal(widget.Inputs.data, self.data)

        dist = self.get_output(widget.Outputs.distances)
        # Check that the output has original data, not reduced data as labels
        self.assertIs(dist.row_items, self.data)
        self.assertIs(dist.col_items, self.data)
        np.testing.assert_equal(dist, np.array([
            [0, 3, 2, 5, 4],
            [3, 0, 1, 6, 5],
            [2, 1, 0, 5, 6],
            [5, 6, 5, 0, 5],
            [4, 5, 6, 5, 0]]))

        widget.attr_list.selectionModel().select(
            widget.attr_model.index(0, 0),
            QtCore.QItemSelectionModel.Deselect)
        self.assertEqual(widget.unselected_hint, ["one"])

        dist = self.get_output(widget.Outputs.distances)
        np.testing.assert_equal(dist, np.array([
            [0, 1, 0, 4, 4],
            [1, 0, 1, 5, 3],
            [0, 1, 0, 4, 4],
            [4, 5, 4, 0, 4],
            [4, 3, 4, 4, 0]]))

        widget.select_none()
        self.assertEqual(
            set(widget.unselected_hint),
            {"one", "two", "three", "four", "five"})
        dist = self.get_output(widget.Outputs.distances)
        self.assertIsNone(dist)

        widget.select_all()
        self.assertEqual(widget.unselected_hint, [])
        dist = self.get_output(widget.Outputs.distances)
        np.testing.assert_equal(dist, np.array([
            [0, 3, 2, 5, 4],
            [3, 0, 1, 6, 5],
            [2, 1, 0, 5, 6],
            [5, 6, 5, 0, 5],
            [4, 5, 6, 5, 0]]))

    def test_commit_called_once(self):
        # This test can fail if set_data does not call unconditional commit,
        # or if commit is called twice because of setting the attribute selection
        widget = self.widget

        widget.auto_apply = False
        start = widget.start = Mock()
        self.send_signal(self.data)
        start.assert_called_once()

        self.send_signal(None)
        start.assert_called_once()
        self.assertIsNone(self.get_output(widget.Outputs.distances))

    @patch("orangecontrib.pumice.distance.distance.LevenshteinModel",
           wraps=LevenshteinModel)
    def test_model_parameters(self, model):
        widget = self.widget
        widget.normalize = False
        widget.split_words = False
        widget.delimiter = "|"

        self.send_signal(self.data)
        self.get_output()  # Wait to finish processing
        self.assertEqual(model.call_args[1], {"normalize": False, "delimiter": None})

        widget.controls.normalize.click()
        self.get_output()
        self.assertEqual(model.call_args[1], {"normalize": True, "delimiter": None})

        widget.controls.split_words.click()
        self.get_output()
        self.assertEqual(model.call_args[1], {"normalize": True, "delimiter": "|"})

        widget.controls.split_words.click()
        self.get_output()
        self.assertEqual(model.call_args[1], {"normalize": True, "delimiter": None})

        widget.delimiter = ","
        widget._on_delimiter_changed()
        self.get_output()
        self.assertEqual(model.call_args[1], {"normalize": True, "delimiter": ","})

    def test_no_string_attributes(self):
        widget = self.widget

        self.send_signal(self.data)
        self.assertIsNotNone(self.get_output())

        self.send_signal(
            self.data.transform(Domain(self.data.domain.attributes)))
        self.assertIsNone(self.get_output())
        self.assertTrue(widget.Error.no_string_attributes.is_shown())

        self.send_signal(self.data)
        self.assertIsNotNone(self.get_output())
        self.assertFalse(widget.Error.no_string_attributes.is_shown())

        self.send_signal(
            self.data.transform(
                Domain(self.data.domain.attributes,
                       None,
                       self.data.domain.metas[:1])))
        self.assertIsNone(self.get_output())
        self.assertTrue(widget.Error.no_string_attributes.is_shown())

        self.send_signal(None)
        self.assertIsNone(self.get_output())
        self.assertFalse(widget.Error.no_string_attributes.is_shown())

    def test_distance_errors(self):
        widget = self.widget
        widget.normalize = False
        self.send_signal(self.data)

        with patch("Levenshtein.distance", side_effect=MemoryError):
            widget.commit.now()
            self.assertIsNone(self.get_output())
            self.assertTrue(widget.Error.memory_error.is_shown())
        widget.commit.now()
        self.assertIsNotNone(self.get_output())
        self.assertFalse(widget.Error.memory_error.is_shown())

        with patch("Levenshtein.distance", side_effect=InterruptException):
            widget.commit.now()
            self.assertIsNone(self.get_output())
            self.assertFalse(widget.Error.memory_error.is_shown())
        widget.commit.now()
        self.assertIsNotNone(self.get_output())
        self.assertFalse(widget.Error.memory_error.is_shown())

        with patch("orangecontrib.pumice.distance.distance.LevenshteinModel.__call__", new=lambda *_: None):
            widget.commit.now()
            self.assertIsNone(self.get_output())
            self.assertFalse(widget.Error.memory_error.is_shown())


if __name__ == "__main__":
    unittest.main()