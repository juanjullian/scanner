"""Sección plegada. La casilla prende el ajuste, la flecha lo abre y el título lo selecciona."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)


class HelpMark(QWidget):
    """Signo de pregunta en un círculo. El texto va en el tooltip."""

    def __init__(self, tip: str):
        super().__init__()
        self.setFixedSize(18, 18)
        self.setToolTip(tip)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor("#c8c8c8"))
        pen.setWidthF(1.2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(1, 1, 15, 15)
        painter.setPen(QColor("#f2f2f2"))
        font = painter.font()
        font.setPixelSize(11)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "?")


class FoldSection(QWidget):
    toggled = pyqtSignal(bool)
    expanded = pyqtSignal(bool)
    picked = pyqtSignal()

    def __init__(self, title: str, help_text: str = "", selectable: bool = False):
        super().__init__()
        self._open = False
        self._picked = False
        self._selectable = bool(selectable)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 2, 0, 2)
        root.setSpacing(4)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(4)
        self.check = QCheckBox()
        self.check.toggled.connect(self.toggled.emit)
        if self._selectable:
            self.title = QPushButton(title)
            self.title.setFlat(True)
            self.title.setCursor(Qt.CursorShape.PointingHandCursor)
            self.title.clicked.connect(self.picked.emit)
        else:
            self.title = QLabel(title)
        head.addWidget(self.check)
        head.addWidget(self.title, 1)
        if help_text:
            head.addWidget(HelpMark(help_text))
        self.arrow = QToolButton()
        self.arrow.setAutoRaise(True)
        self.arrow.setArrowType(Qt.ArrowType.RightArrow)
        self.arrow.setFixedSize(22, 22)
        self.arrow.setToolTip("Abrir")
        self.arrow.clicked.connect(self._toggle)
        head.addWidget(self.arrow)
        root.addLayout(head)
        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(22, 0, 0, 0)
        self.body_layout.setSpacing(4)
        self.body.setVisible(False)
        root.addWidget(self.body, 1)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self._paint_title()

    def add_widget(self, widget, stretch: int = 0):
        self.body_layout.addWidget(widget, stretch)
        return widget

    def is_open(self) -> bool:
        return self._open

    def is_checked(self) -> bool:
        return self.check.isChecked()

    def is_picked(self) -> bool:
        return self._picked

    def set_title(self, title: str):
        self.title.setText(title)

    def set_checked(self, on: bool):
        self.check.blockSignals(True)
        self.check.setChecked(bool(on))
        self.check.blockSignals(False)

    def set_picked(self, on: bool):
        self._picked = bool(on) and self._selectable and self.isEnabled()
        self._paint_title()

    def set_open(self, open_: bool):
        open_ = bool(open_)
        if open_ == self._open:
            return
        self._open = open_
        self.body.setVisible(self._open)
        self.arrow.setArrowType(
            Qt.ArrowType.DownArrow if self._open else Qt.ArrowType.RightArrow
        )
        self.arrow.setToolTip("Cerrar" if self._open else "Abrir")
        vertical = QSizePolicy.Policy.Expanding if self._open else QSizePolicy.Policy.Maximum
        self.setSizePolicy(QSizePolicy.Policy.Preferred, vertical)
        self.updateGeometry()
        self.expanded.emit(self._open)

    def _toggle(self):
        self.set_open(not self._open)

    def _paint_title(self):
        if not self._selectable:
            return
        if self._picked:
            style = (
                "QPushButton { color: palette(highlighted-text); background: palette(highlight);"
                " border: none; text-align: left; padding: 2px 8px; border-radius: 3px; }"
            )
        else:
            style = (
                "QPushButton { color: palette(window-text); background: transparent; border: none;"
                " text-align: left; padding: 2px 8px; }"
                "QPushButton:hover { background: palette(midlight); }"
            )
        self.title.setStyleSheet(style)
