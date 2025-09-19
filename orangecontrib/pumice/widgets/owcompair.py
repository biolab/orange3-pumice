import os
import random
from functools import cache
from urllib.request import urlopen
from typing import Optional
from contextlib import contextmanager

import numpy as np

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QPixmap, QFont, QPainter, QColor
from AnyQt.QtWidgets import (
    QApplication, QHBoxLayout, QWidget, QGridLayout, QSizePolicy, QLabel, QPushButton)

from orangewidget.settings import Setting
from orangewidget.utils.signals import Output
from orangewidget.utils.widgetpreview import WidgetPreview
from orangewidget.widget import Msg

from Orange.data import Table, StringVariable, Domain, DiscreteVariable
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget, Input
from Orange.classification.logistic_regression import (
    LogisticRegressionLearner, LogisticRegressionClassifier)


cars_table = Table(
    os.path.join(os.path.dirname(__file__), "..", 'datasets', 'cars.xlsx'))

BUTTON_STYLE = """
    QPushButton {
        background-color: #007aff;
        color: white;
        border: 1px solid #a1a1a1;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 24px;
        font-weight: bold;
        margin-top: 8px;
    }
    QPushButton:pressed {
        background-color: #d0d0d0;
    }
    
    QPushButton:disabled{
        background-color: #d0d0d0;
        color: #a1a1a1;
    }
    
"""


class OutlinedLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)

    def paintEvent(self, event):
        painter = QPainter(self)
        text_rect = self.rect().adjusted(-10, -10, 10, 10)

        font = QFont()
        font.setPixelSize(48)
        painter.setFont(font)
        text = self.text()

        # Draw black outline
        pen = painter.pen()
        pen.setWidth(2)
        pen.setColor(Qt.black)
        painter.setPen(pen)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            painter.drawText(text_rect.adjusted(dx, dy, dx, dy), Qt.AlignCenter, text)

        # Draw white fill
        painter.setPen(Qt.white)
        painter.drawText(text_rect, Qt.AlignCenter, text)

class ScoreCurve(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scores = []
        self.setMinimumHeight(100)

    def set_scores(self, scores):
        self.scores = scores
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect().adjusted(1, 1, -1, -1)

        painter.setBrush(QColor(16, 16, 16))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 10, 10)

        if not self.scores:
            return

        rect = rect.adjusted(3, 3, -3, -3)

        tot_correct = sum(self.scores)
        tot_wrong = len(self.scores) - tot_correct
        font = QFont()
        font.setPixelSize(24)
        painter.setFont(font)
        twidth = max(painter.fontMetrics().horizontalAdvance(str(t))
                     for t in (tot_correct, tot_wrong))
        theight = -painter.fontMetrics().tightBoundingRect(str(tot_wrong)).top()


        available_width = rect.width() - twidth - 10
        width = min(12, available_width // (len(self.scores)))
        base = rect.height() * 3 // 4

        painter.setPen(Qt.gray)
        painter.drawLine(4, base, rect.width(), base)

        height = min([6,
                      (base - 4) / (1 + tot_correct),
                      0.33 * (base - 4) / (1 + tot_wrong)])
        hspace = 2 if width > 5 else 1 if width > 3 else 0
        width -= hspace
        if height > 2:
            height = int(height)
        else:
            height = int(height * 5) / 5
        correct = 0
        for i, score in enumerate(self.scores):
            correct += score
            painter.setPen(Qt.NoPen)
            wrong = i - correct + 1
            x = 4 + i * (width + hspace)
            if height > 2:
                painter.setBrush(Qt.green)
                for j in range(correct):
                    painter.drawRoundedRect(
                        x, base - (j + 1) * height - 1,
                        width, height - 1, 1, 1)
                painter.setBrush(Qt.red)
                for j in range(wrong):
                    painter.drawRoundedRect(
                        x, base + j * height + 2,
                        width, height - 1, 1, 1)
            else:
                painter.setBrush(Qt.green)
                painter.drawRoundedRect(
                    x, base - int(correct * height) - 1,
                    width, int(correct * height), 1, 1)
                painter.setBrush(Qt.red)
                painter.drawRoundedRect(
                    x, base + 2,
                    width, int(wrong * height), 1, 1)

        x = 4 + (len(self.scores) + 1) * (width + hspace)
        painter.setBrush(Qt.black)
        painter.setPen(Qt.green)
        painter.drawText(x, base - 4, str(tot_correct))
        painter.setPen(Qt.red)
        painter.drawText(x, base + 4 + theight, str(tot_wrong))


@cache
def get_image(url):
    try:
        with urlopen(url) as response:
            data = response.read()
        pixmap = QPixmap()
        pixmap.loadFromData(data)
    except Exception as e:
        print(e)
        return QPixmap()
    else:
        return pixmap.scaledToWidth(600, Qt.SmoothTransformation)


class OWComPair(OWWidget):
    name = "Com Pair"
    description = "Learns a model that compares pairs of items"
    icon = "icons/compair.svg"

    class Inputs:
        data = Input("Items", Table)
        model = Input("Model", LogisticRegressionClassifier)

    class Outputs:
        model = Output("Model", LogisticRegressionClassifier)

    class Error(OWWidget.Error):
        no_continuous_target = Msg(
            "If data has a target variable, it must be numeric.")
        invalid_data = Msg(
            "Data must contain numeric variables without missing values.")
        not_enough_data = Msg(
            "Not enough training data (at least 5 instances).")
        equal_values = Msg(
            "Some instances have the same value of the target variable.")
        no_image_column = Msg(
            "Data must contain a string column with image URLs.")

    want_control_area = False
    resizing_enabled = False
    exclude_seen = Setting(True)

    States = NoData, ShowingPair, Predicting, NoMore = range(4)
    ButtonTexts = ["No Data", "Make a Prediction", "Show Next pair", "No More Pairs"]
    ButtonTextGuess = "Make a Guess"

    def __init__(self):
        super().__init__()

        self.data: Optional[Table] = None
        self.image_column: Optional[StringVariable] = None
        self.state = self.NoData
        self.pairs = None
        self.model = None
        self.scores = []
        self.buttons_off = False

        layout = QGridLayout()
        self.mainArea.layout().addLayout(layout)
        self.image1 = QLabel()
        self.image2 = QLabel()
        self.image1.setFixedWidth(600)
        self.image2.setFixedWidth(600)
        self.y_left = OutlinedLabel()
        self.y_left.setFixedSize(130, 50)
        self.y_right = OutlinedLabel()
        self.y_right.setFixedSize(130, 50)
        layout.addWidget(self.image1, 1, 0)
        layout.addWidget(self.image2, 1, 2)
        layout.addWidget(self.y_left, 1, 0, Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(self.y_right, 1, 2, Qt.AlignTop | Qt.AlignRight)

        self.prediction_left = QLabel()
        prediction_mid = QLabel(":")
        self.prediction_right = QLabel()
        layout.addWidget(self.prediction_left, 2, 0, Qt.AlignRight)
        layout.addWidget(prediction_mid, 2, 1, Qt.AlignCenter)
        layout.addWidget(self.prediction_right, 2, 2)
        font = QFont()
        font.setPixelSize(20)
        for label in (self.prediction_left, prediction_mid, self.prediction_right):
            label.setFont(font)

        self.restart_button = QPushButton("Start Again", autoDefault=False)
        self.restart_button.clicked.connect(self.restart)
        layout.addWidget(self.restart_button, 4, 0, 1, 3, Qt.AlignLeft)

        hlayout = QHBoxLayout()
        self.next5_button = QPushButton("Next Five", autoDefault=False)
        self.next5_button.clicked.connect(self.next5)
        hlayout.addWidget(self.next5_button)

        self.finish_button = QPushButton("Classify All", autoDefault=False)
        self.finish_button.clicked.connect(self.finish)
        hlayout.addWidget(self.finish_button)

        layout.addLayout(hlayout, 4, 0, 1, 3, Qt.AlignRight)

        self.next_button = QPushButton()
        self.next_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.next_button.setStyleSheet(BUTTON_STYLE)
        self.next_button.clicked.connect(self.next)
        layout.addWidget(self.next_button, 4, 0, 1, 3, Qt.AlignCenter)

        layout.addWidget(
            gui.checkBox(
                None, self, "exclude_seen", "Exclude from model pairs with either car in this pair"),
            5, 0, 1, 3, Qt.AlignCenter)

        layout.setRowMinimumHeight(6, 20)
        layout.addWidget(
            QLabel("Learning Curve & Current Score"),
            6, 0, 1, 3, Qt.AlignLeft | Qt.AlignBottom)

        self.score_curve = ScoreCurve()
        self.score_curve.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.score_curve.setMinimumHeight(100)
        self.score_curve.setFixedWidth(1220)
        layout.addWidget(self.score_curve, 7, 0, 1, 3, Qt.AlignCenter)


        self.set_data(cars_table)

    def set_buttons_enabled(self, enabled):
        buttons = (self.next_button, self.restart_button,
                   self.next5_button, self.finish_button)
        for button in buttons:
            button.setEnabled(enabled)
        self.controls.exclude_seen.setEnabled(enabled)

    @contextmanager
    def disabled_buttons(self):
        try:
            self.buttons_off = True
            self.set_buttons_enabled(False)
            yield
        finally:
            self.buttons_off = False
            self.set_buttons_enabled(
                self.state not in (self.NoData, self.NoMore))
            if self.state == self.NoMore:
                self.restart_button.setEnabled(True)

    @Inputs.model
    def set_model(self, model):
        self.model = model

    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        self.data = None
        self.image_column = None

        if data is None:
            data = cars_table

        if not all(var.is_continuous for var in data.domain.attributes) \
                or data.has_missing():
            self.Error.invalid_data()
        if len(data) < 5:
            self.Error.not_enough_data()
        self.image_column = self._find_image_column(data)
        if self.image_column is None:
            self.Error.no_image_column()
        if data.domain.class_vars:
            if not data.domain.class_var \
                    and data.domain.class_var.is_continuous:
                self.Error.no_continuous_target()
            elif len(set(data.Y)) != len(data):
                self.Error.equal_values()
            elif not np.all(np.diff(order := np.argsort(data.Y)) == 2):
                data = data[order]
        if self.Error.active:
            self.set_state(self.NoData)
            return

        self.data = data
        self.restart()

    def _find_image_column(self, data):
        for var in data.domain.metas:
            if var.is_string and var.attributes.get("type", None) == "image":
                return var
        for var in data.domain.metas:
            if not var.is_string:
                continue
            column = self.data.get_column(var)
            if all(os.path.splitext(v)[1] in {".png", ".jpg", ".jpeg", ".gif"}
                   for v in column):
                return var
        return None

    def _create_pairs(self):
        if self.exclude_seen:
            idx = np.arange(len(self.data))
            np.random.shuffle(idx)
            pairs = []
            for dist in range(1, len(idx)):
                pairs += [(idx[i], idx[i + dist])[::random.choice([1, -1])]
                          for j in range(dist + 1)
                          for i in range(j, len(idx) - dist, dist + 1)]
        else:
            pairs = [(i, j) if random.randint(0, 1) else (j, i)
                     for i in range(len(self.data)) for j in range(i)]
            random.shuffle(pairs)

        pairs = np.array(pairs)
        return pairs

    def restart(self):
        if self.data is None:
            return
        self.pairs = self._create_pairs()
        self.scores = []
        self.score_curve.set_scores(self.scores)
        self.set_state(self.ShowingPair)

    def next(self):
        with self.disabled_buttons():
            self.advance_state()

    def next5(self):
        with self.disabled_buttons():
            for _ in range(10):
                self.advance_state()
                QApplication.instance().processEvents()

    def finish(self):
        with self.disabled_buttons():
            while self.state != self.NoMore:
                self.advance_state()
                QApplication.instance().processEvents()

    def advance_state(self):
        if self.state == self.ShowingPair:
            self.set_state(self.Predicting)
        elif self.state == self.Predicting:
            if len(self.scores) < len(self.pairs):
                self.set_state(self.ShowingPair)
            else:
                self.set_state(self.NoMore)

    def _show_pair(self):
        self._update_images()
        self.prediction_left.setText(f"<b>?????</b>")
        self.prediction_right.setText(f"<b>?????</b>")
        self.y_left.setText("")
        self.y_right.setText("")

    def _update_images(self):
        assert self.image_column is not None
        assert self.data is not None

        iter = len(self.scores)
        url1, url2 = self.data.get_column(self.image_column)[self.pairs[iter]]
        self.image1.setPixmap(get_image(url1))
        self.image2.setPixmap(get_image(url2))

    def _show_prediction(self):
        old_young = ["newer", "older"]
        wrong_correct = ["wrong", "correct"]

        iter = len(self.scores)
        i1, i2 = self.pairs[iter]

        if iter < 2:
            # Before seeing two instances, make random guesses
            pred = random.randint(0, 1)
        else:
            if self.model is not None:
                domain = self.model.domain
                model = self.model
            else:
                domain = Domain(
                    self.data.domain.attributes,
                    [DiscreteVariable("y", values=("0", "1"))])
                seen = self.pairs[:iter].T
                if self.exclude_seen:
                    seen = seen[:,
                           (seen[0] != i1) & (seen[1] != i1)
                            & (seen[0] != i2) & (seen[1] != i2)]
                x = self.data.X[seen[0]] - self.data.X[seen[1]]
                ys = (seen[0] < seen[1]).astype(float)
                if len(set(ys)) == 1:
                    # If all instances are from the same class, flip the first pair
                    x[0] = -x[0]
                    ys[0] = 1 - ys[0]
                data = Table.from_numpy(domain, x, ys)
                model = LogisticRegressionLearner()(data)

            self.Outputs.model.send(model)
            x = (self.data.X[i1] - self.data.X[i2]).reshape(1, -1)
            test = Table.from_numpy(domain, x, [np.nan])
            pred = model(test)[0]

        correct = pred == (i1 < i2)
        self.prediction_left.setText(f"Prediction: <b>{old_young[int(pred)]}</b>")
        self.prediction_right.setText(
            f"<b>{old_young[1 - int(pred)]}</b> ({wrong_correct[int(correct)]}!)")
        if (class_var := self.data.domain.class_var) is not None:
            left = class_var.repr_val(self.data.Y[i1])
            right = class_var.repr_val(self.data.Y[i2])
            self.y_left.setText(left)
            self.y_right.setText(right)
        self.scores.append(correct)
        self.score_curve.set_scores(self.scores)

    def set_state(self, state):
        self.state = state
        if state == self.ShowingPair and len(self.scores) < 2:
            self.next_button.setText(self.ButtonTextGuess)
        else:
            self.next_button.setText(self.ButtonTexts[state])
        if self.state == self.ShowingPair:
            self._show_pair()
        elif self.state == self.Predicting:
            self._show_prediction()

        self.set_buttons_enabled(
            not self.buttons_off and self.state not in (self.NoData, self.NoMore))
        if self.state == self.NoMore:
            self.restart_button.setEnabled(True)


if __name__ == "__main__":
    # pragma: no cover
    WidgetPreview(OWComPair).run()
