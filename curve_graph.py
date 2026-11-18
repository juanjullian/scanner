"""Gráfico de la curva de luces.

Parte en la recta, sin puntos. Clic crea uno y clic derecho lo borra.
El recuadro usa todo el ancho y crece de alto hasta quedar cuadrado.
"""
from __future__ import annotations

import tone
from PyQt6.QtCore import QPointF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget


class CurveGraph(QWidget):
    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setToolTip(
            "Clic para crear un punto y arrástralo. "
            "Clic derecho lo borra. "
            "Un punto no pasa al vecino: si se traba, mueve también el de al lado."
        )
        self._points = tone.control_points(None)
        self._drag = None
        self._hover = None

    def sizeHint(self):
        side = max(280, self.width())
        return QSize(side, self._square_height(side))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cap = self._square_height(self.width())
        if cap >= self.minimumHeight() and self.maximumHeight() != cap:
            self.setMaximumHeight(cap)
            self.updateGeometry()

    def _square_height(self, width: int) -> int:
        """Alto del widget cuando el gráfico, con sus márgenes, queda cuadrado."""
        left, top, right, bottom = 8, 8, 8, 22
        plot = max(80, int(width) - left - right)
        return plot + top + bottom

    def set_curve(self, curve):
        self._points = tone.control_points(curve)
        self._drag = None
        self.update()

    def curve(self):
        points = tone.normalize(self._points)
        if points is None:
            return None
        return [[round(x, 4), round(y, 4)] for x, y in points]

    def paintEvent(self, _event):
        left, top, width, height = self._box()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#121212"))
        painter.fillRect(int(left), int(top), int(width), int(height), QColor("#1b1b1b"))
        grid = QPen(QColor("#2c2c2c"))
        grid.setWidth(1)
        painter.setPen(grid)
        for step in range(1, 4):
            x = left + width * step / 4.0
            y = top + height * step / 4.0
            painter.drawLine(QPointF(x, top), QPointF(x, top + height))
            painter.drawLine(QPointF(left, y), QPointF(left + width, y))
        guide = QPen(QColor("#4a4a4a"))
        guide.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(guide)
        painter.drawLine(QPointF(left, top + height), QPointF(left + width, top))
        drawn = tone.samples(self._points, 160)
        pen = QPen(QColor("#e8c36a"))
        pen.setWidthF(1.8)
        painter.setPen(pen)
        previous = None
        last = len(drawn) - 1
        for index, value in enumerate(drawn):
            point = self._to_px(index / last, float(value))
            if previous is not None:
                painter.drawLine(previous, point)
            previous = point
        for index, (x, y) in enumerate(self._points):
            if index == 0 or index == len(self._points) - 1:
                continue
            center = self._to_px(x, y)
            hot = index == self._hover or index == self._drag
            radius = 7.0 if hot else 5.5
            painter.setPen(QPen(QColor("#111111"), 1))
            painter.setBrush(QColor("#f4f1e8"))
            painter.drawEllipse(center, radius, radius)
        painter.setPen(QColor("#9a9a9a"))
        painter.drawText(int(left), int(top + height) + 16, "negro")
        painter.drawText(int(left + width) - 42, int(top + height) + 16, "blanco")
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._remove_at(event.position())
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not self._inside(event.position()):
            return
        hit = self._hit(event.position())
        if hit is None:
            x, y = self._from_px(event.position())
            before = len(self._points)
            self._points = tone.insert_point(self._points, x, y)
            if len(self._points) == before:
                return
            hit = self._hit(event.position())
            if hit is None:
                hit = self._nearest(x)
        self._drag = hit
        self._move_to(event.position())

    def mouseMoveEvent(self, event):
        if self._drag is None:
            hit = self._hit(event.position())
            if hit != self._hover:
                self._hover = hit
                self.update()
            self.setCursor(
                Qt.CursorShape.SizeAllCursor if hit is not None else Qt.CursorShape.CrossCursor
            )
            return
        self._move_to(event.position())

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self._drag is None:
            return
        self._move_to(event.position())
        self._drag = None

    def _remove_at(self, pos):
        hit = self._hit(pos)
        if hit is None:
            return
        self._points = tone.remove_point(self._points, hit)
        self._drag = None
        self._hover = None
        self.changed.emit()
        self.update()

    def _move_to(self, pos):
        if self._drag is None:
            return
        x, y = self._from_px(pos)
        self._points = tone.move_point(self._points, self._drag, x, y)
        self.changed.emit()
        self.update()

    def _nearest(self, x: float) -> int | None:
        best = None
        gap = 1.0
        for index, (px, _py) in enumerate(self._points):
            if index == 0 or index == len(self._points) - 1:
                continue
            dist = abs(px - x)
            if dist < gap:
                gap = dist
                best = index
        return best

    def _box(self):
        left, top, right, bottom = 8.0, 8.0, 8.0, 22.0
        width = max(80.0, float(self.width()) - left - right)
        height = max(80.0, float(self.height()) - top - bottom)
        return left, top, width, height

    def _to_px(self, x: float, y: float) -> QPointF:
        left, top, width, height = self._box()
        return QPointF(left + x * width, top + (1.0 - y) * height)

    def _from_px(self, pos) -> tuple[float, float]:
        left, top, width, height = self._box()
        x = (float(pos.x()) - left) / width
        y = 1.0 - (float(pos.y()) - top) / height
        return x, y

    def _inside(self, pos) -> bool:
        left, top, width, height = self._box()
        return left <= float(pos.x()) <= left + width and top <= float(pos.y()) <= top + height

    def _hit(self, pos) -> int | None:
        best = None
        reach = 14.0
        for index, (x, y) in enumerate(self._points):
            if index == 0 or index == len(self._points) - 1:
                continue
            center = self._to_px(x, y)
            dx = center.x() - float(pos.x())
            dy = center.y() - float(pos.y())
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= reach and (best is None or dist < best[0]):
                best = (dist, index)
        if best is None:
            return None
        return best[1]
