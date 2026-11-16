"""Independent journey-leg graphics, screen-space hit testing and direction arrows."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QPainterPath, QPen, QPolygonF, QTransform
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsItemGroup, QGraphicsPathItem


@dataclass
class LegGeometry:
    label: str
    length_m: float | None
    points: np.ndarray

    def __post_init__(self):
        self.starts = self.points[:-1]
        self.vectors = np.diff(self.points, axis=0)
        self.squared_lengths = np.einsum("ij,ij->i", self.vectors, self.vectors)
        low, high = self.points.min(axis=0), self.points.max(axis=0)
        self.bounds = QRectF(float(low[0]), float(low[1]), float(high[0] - low[0]), float(high[1] - low[1]))

    def distance(self, point: np.ndarray) -> float:
        if not len(self.vectors):
            return float(np.linalg.norm(self.points[0] - point))
        relative = point - self.starts
        projection = np.divide(np.einsum("ij,ij->i", relative, self.vectors), self.squared_lengths,
                               out=np.zeros(len(self.vectors)), where=self.squared_lengths > 0)
        nearest = self.starts + np.clip(projection, 0, 1)[:, None] * self.vectors
        return float(np.sqrt(np.min(np.square(nearest - point).sum(axis=1))))

    def description(self) -> str:
        return self.label + (f" · {self.length_m / 1000:,.3f} km" if self.length_m is not None else "")


class RouteLayer:
    def __init__(self, points: np.ndarray, legs: list[dict]):
        self.group = QGraphicsItemGroup()
        self.group.setZValue(10)
        self.legs = [LegGeometry(leg["label"], leg.get("length_m"), points[leg["path_nodes"]]) for leg in legs]
        self.items: list[QGraphicsItem] = []
        self.focused: int | None = None
        self._arrow_key = None
        self._arrows: list[QPolygonF] = []
        for leg in self.legs:
            path = QPainterPath(QPointF(*leg.points[0]))
            for point in leg.points[1:]:
                path.lineTo(QPointF(*point))
            if np.any(leg.squared_lengths > 0):
                item = QGraphicsPathItem(path, self.group)
            else:
                # An identical pair of stops still has a selectable visual leg.
                item = QGraphicsEllipseItem(-14, -14, 28, 28, self.group)
                item.setPos(QPointF(*leg.points[0]))
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.items.append(item)
        self._style()

    def _style(self) -> None:
        for index, item in enumerate(self.items):
            selected = self.focused == index
            pen = QPen(QColor("#e45c26"))
            pen.setWidthF(5.5 if selected else 4)
            pen.setCosmetic(True)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            item.setPen(pen)
            item.setOpacity(1.0 if self.focused is None or selected else 0.25)
            item.setZValue(1 if selected else 0)

    def focus(self, index: int | None) -> None:
        if index != self.focused:
            self.focused = index
            self._arrow_key = None
            self._style()

    def hits(self, point: QPointF, scale: float) -> tuple[int, ...]:
        """Return nearby coincident/crossing legs in journey order, not paint order."""
        radius = 8.0 / scale
        coordinate = np.array([point.x(), point.y()])
        distances = []
        for index, leg in enumerate(self.legs):
            if leg.bounds.adjusted(-radius, -radius, radius, radius).contains(point):
                distance = leg.distance(coordinate) * scale
                if distance <= 8.0:
                    distances.append((index, distance))
        nearest = min((distance for _, distance in distances), default=math.inf)
        # A close parallel road should not steal focus from the nearest line.
        return tuple(index for index, distance in distances if distance <= nearest + 1.25)

    @staticmethod
    def _clip(start: np.ndarray, vector: np.ndarray, width: int, height: int):
        lower, upper = 0.0, 1.0
        for origin, delta, limit in zip(start, vector, (width, height)):
            if abs(delta) < 1e-12:
                if not 0 <= origin <= limit:
                    return None
            else:
                a, b = sorted((-origin / delta, (limit - origin) / delta))
                lower, upper = max(lower, a), min(upper, b)
                if lower > upper:
                    return None
        return lower, upper

    def direction_arrows(self, transform: QTransform, viewport: QRect) -> list[QPolygonF]:
        if self.focused is None:
            return []
        key = (self.focused, transform.m11(), transform.m12(), transform.m21(), transform.m22(),
               transform.dx(), transform.dy(), viewport.width(), viewport.height())
        if key == self._arrow_key:
            return self._arrows
        self._arrow_key, self._arrows = key, []
        points = self.legs[self.focused].points
        matrix = np.array([[transform.m11(), transform.m12()], [transform.m21(), transform.m22()]])
        screen = points @ matrix + np.array([transform.dx(), transform.dy()])
        vectors = np.diff(screen, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        cumulative = np.r_[0.0, np.cumsum(lengths)]
        if cumulative[-1] < 25:
            return []
        low, high = np.minimum(screen[:-1], screen[1:]), np.maximum(screen[:-1], screen[1:])
        visible = np.flatnonzero((lengths > 1e-9) & (high[:, 0] >= 0) & (high[:, 1] >= 0)
                                 & (low[:, 0] <= viewport.width()) & (low[:, 1] <= viewport.height()))
        used = set()
        fallback = None
        for index in visible:
            clipped = self._clip(screen[index], vectors[index], viewport.width(), viewport.height())
            if clipped is None:
                continue
            lower, upper = clipped
            first = cumulative[index] + lower * lengths[index]
            last = min(cumulative[index] + upper * lengths[index], cumulative[-1] - 25)
            if fallback is None and (upper - lower) * lengths[index] >= 25:
                fallback = (index, (lower + upper) / 2)
            for step in range(max(0, math.ceil((first - 35) / 110)), math.floor((last - 35) / 110) + 1):
                if step not in used:
                    used.add(step)
                    self._arrows.append(self._triangle(screen[index], vectors[index], lengths[index],
                                                       (35 + step * 110 - cumulative[index]) / lengths[index]))
        if not self._arrows and fallback is not None:
            index, fraction = fallback
            self._arrows.append(self._triangle(screen[index], vectors[index], lengths[index], fraction))
        return self._arrows

    @staticmethod
    def _triangle(start: np.ndarray, vector: np.ndarray, length: float, fraction: float) -> QPolygonF:
        center, unit = start + fraction * vector, vector / length
        normal = np.array([-unit[1], unit[0]])
        return QPolygonF([QPointF(*(center + unit * 6)), QPointF(*(center - unit * 5 + normal * 4)),
                          QPointF(*(center - unit * 5 - normal * 4))])
