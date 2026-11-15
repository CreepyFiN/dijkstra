"""Tiled offline roads, an overview image, and independent route/marker layers."""
from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsItemGroup, QGraphicsPathItem,
    QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
)

from .data import BACKGROUND, ROAD_COLOR, MapData


class MapView(QGraphicsView):
    nodePicked = Signal(int)
    coordinateHovered = Signal(float, float)
    pickMissed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor(BACKGROUND))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setMouseTracking(True)
        self.setAccessibleName("Offline road map")
        self.data: MapData | None = None
        self.pick_mode = False
        self._press = QPoint()
        self._vectors: QGraphicsItemGroup | None = None
        self._overview: QGraphicsPixmapItem | None = None
        self._route: QGraphicsPathItem | None = None
        self._markers: list[QGraphicsEllipseItem] = []
        self._route_bounds: QRectF | None = None
        self._overview_scale = 1.0

    def set_dataset(self, data: MapData) -> None:
        self.scene().clear()
        self.data = data
        self._route = None
        self._route_bounds = None
        self._markers = []
        self.resetTransform()
        self._vectors = QGraphicsItemGroup()
        self.scene().addItem(self._vectors)
        pen = QPen(QColor(ROAD_COLOR))
        pen.setWidthF(1.0)
        pen.setCosmetic(True)
        for tile in data.tiles:
            item = QGraphicsPathItem(tile.path, self._vectors)
            item.setPen(pen)
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._overview = self.scene().addPixmap(QPixmap.fromImage(data.overview))
        self._overview.setPos(data.bounds.topLeft())
        transform = QTransform.fromScale(
            data.bounds.width() / data.overview.width(),
            data.bounds.height() / data.overview.height())
        self._overview.setTransform(transform)
        self._overview.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._overview.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._overview_scale = data.overview.width() / data.bounds.width()
        self.scene().setSceneRect(self._padded(data.bounds, 0.04))
        self.fit_map()

    @staticmethod
    def _padded(bounds: QRectF, fraction: float) -> QRectF:
        padding = max(bounds.width(), bounds.height(), 10.0) * fraction
        return bounds.adjusted(-padding, -padding, padding, padding)

    def _update_detail(self) -> None:
        if self._overview is not None:
            overview = abs(self.transform().m11()) <= self._overview_scale * 1.5
            self._overview.setVisible(overview)
            self._vectors.setVisible(not overview)

    def fit_map(self) -> None:
        if self.data is not None:
            self.fitInView(self._padded(self.data.bounds, 0.04), Qt.AspectRatioMode.KeepAspectRatio)
            self._update_detail()

    def fit_route(self) -> None:
        if self._route_bounds is not None:
            self.fitInView(self._padded(self._route_bounds, 0.12), Qt.AspectRatioMode.KeepAspectRatio)
            self._update_detail()

    def zoom(self, factor: float) -> None:
        if self.data is None:
            return
        current = abs(self.transform().m11())
        minimum = min(self.viewport().width() / self.data.bounds.width(),
                      self.viewport().height() / self.data.bounds.height()) / 4
        new = max(minimum, min(current * factor, 20.0))
        self.scale(new / current, new / current)
        self._update_detail()

    def wheelEvent(self, event) -> None:
        self.zoom(1.2 ** (event.angleDelta().y() / 120))
        event.accept()

    def set_pick_mode(self, enabled: bool) -> None:
        self.pick_mode = enabled
        self.setDragMode(QGraphicsView.DragMode.NoDrag if enabled
                         else QGraphicsView.DragMode.ScrollHandDrag)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.OpenHandCursor)

    def mousePressEvent(self, event) -> None:
        self._press = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if (self.pick_mode and self.data is not None and event.button() == Qt.MouseButton.LeftButton
                and (event.position().toPoint() - self._press).manhattanLength() <= 5):
            point = self.mapToScene(event.position().toPoint())
            node = self.data.nearest_node(point.x(), point.y(), 18 / abs(self.transform().m11()))
            if node is None:
                self.pickMissed.emit()
            else:
                self.nodePicked.emit(node)
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.data is not None:
            point = self.mapToScene(event.position().toPoint())
            self.coordinateHovered.emit(*self.data.coordinates(point.x(), point.y()))
        super().mouseMoveEvent(event)

    def show_endpoints(self, source: int, target: int) -> None:
        for item in self._markers:
            self.scene().removeItem(item)
        self._markers = []
        if self.data is None:
            return
        for node, color, label, size in ((source, "#178365", "Start", 16),
                                         (target, "#d84454", "Destination", 12)):
            item = QGraphicsEllipseItem(-size / 2, -size / 2, size, size)
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            item.setPos(QPointF(*self.data.points[node]))
            item.setBrush(QColor(color))
            item.setPen(QPen(QColor("white"), 2))
            item.setZValue(20)
            item.setToolTip(f"{label}: node {node}")
            self.scene().addItem(item)
            self._markers.append(item)

    def clear_route(self) -> None:
        if self._route is not None:
            self.scene().removeItem(self._route)
        self._route = None
        self._route_bounds = None

    def show_route(self, nodes: list[int]) -> None:
        self.clear_route()
        if self.data is None or not nodes:
            return
        points = self.data.points[nodes]
        path = QPainterPath(QPointF(*points[0]))
        for point in points[1:]:
            path.lineTo(QPointF(*point))
        item = QGraphicsPathItem(path)
        pen = QPen(QColor("#e45c26"))
        pen.setWidthF(4)
        pen.setCosmetic(True)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        item.setPen(pen)
        item.setZValue(10)
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.scene().addItem(item)
        self._route = item
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        self._route_bounds = QRectF(float(xmin), float(ymin), max(float(xmax - xmin), 100.0),
                                    max(float(ymax - ymin), 100.0))
