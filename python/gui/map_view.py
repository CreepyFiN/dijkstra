"""Tiled offline roads, an overview image, and independent route/marker layers."""
from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsItemGroup, QGraphicsPathItem,
    QGraphicsPixmapItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView,
)

from .data import BACKGROUND, ROAD_COLOR, MapData
from .route_layer import RouteLayer


class MapView(QGraphicsView):
    nodePicked = Signal(int)
    coordinateHovered = Signal(float, float)
    pickMissed = Signal()
    legHighlighted = Signal(int, str)
    pinnedLegChanged = Signal(int)
    routeCleared = Signal()

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
        self.viewport().setMouseTracking(True)
        self.setAccessibleName("Offline road map")
        self.data: MapData | None = None
        self.pick_mode = False
        self._press = QPoint()
        self._vectors: QGraphicsItemGroup | None = None
        self._overview: QGraphicsPixmapItem | None = None
        self._route: QGraphicsItemGroup | None = None
        self._route_layer: RouteLayer | None = None
        self._pinned_leg: int | None = None
        self._hover_leg: int | None = None
        self._hover_hits: tuple[int, ...] = ()
        self._last_leg_info = None
        self._markers: list[QGraphicsEllipseItem] = []
        self._route_bounds: QRectF | None = None
        self._overview_scale = 1.0

    def set_dataset(self, data: MapData) -> None:
        self.clear_route()
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
            self._clear_hover()
            self.fitInView(self._padded(self.data.bounds, 0.04), Qt.AspectRatioMode.KeepAspectRatio)
            self._update_detail()

    def fit_route(self) -> None:
        if self._route_bounds is not None:
            self._clear_hover()
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
        self._clear_hover()
        self.setDragMode(QGraphicsView.DragMode.NoDrag if enabled
                         else QGraphicsView.DragMode.ScrollHandDrag)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.OpenHandCursor)

    def mousePressEvent(self, event) -> None:
        self._press = event.position().toPoint()
        if (event.button() == Qt.MouseButton.RightButton and not self.pick_mode
                and not event.buttons() & Qt.MouseButton.LeftButton):
            if self._cycle_overlapping_legs(self.mapToScene(self._press)):
                event.accept()
                return
        if event.button() == Qt.MouseButton.LeftButton:
            self._clear_hover()
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
            if not self.pick_mode and not event.buttons():
                self._hover_at(point)
            else:
                self._clear_hover()
        super().mouseMoveEvent(event)

    def viewportEvent(self, event) -> bool:
        if event.type() == QEvent.Type.Leave:
            self._clear_hover()
        return super().viewportEvent(event)

    def contextMenuEvent(self, event) -> None:
        # Right-click is reserved for cycling shared journey legs.
        event.accept()

    @property
    def active_leg(self) -> int | None:
        return self._route_layer.focused if self._route_layer is not None else None

    @property
    def pinned_leg(self) -> int | None:
        return self._pinned_leg

    def _hits_at(self, point: QPointF) -> tuple[int, ...]:
        if self._route_layer is None or len(self._route_layer.legs) < 2:
            return ()
        scale = math.hypot(self.transform().m11(), self.transform().m12())
        return self._route_layer.hits(point, scale)

    def _hover_at(self, point: QPointF) -> None:
        self._hover_hits = self._hits_at(point)
        if self._hover_leg not in self._hover_hits:
            self._hover_leg = self._hover_hits[0] if self._hover_hits else None
        self._update_leg_focus()

    def _clear_hover(self) -> None:
        self._hover_hits, self._hover_leg = (), None
        self._update_leg_focus()

    def _cycle_overlapping_legs(self, point: QPointF) -> bool:
        hits = self._hits_at(point)
        if len(hits) < 2:
            return False
        current = self.active_leg if self.active_leg in hits else hits[0]
        selected = hits[(hits.index(current) + 1) % len(hits)]
        self._hover_hits, self._hover_leg = hits, selected
        if self._pinned_leg is not None:
            self._pinned_leg = selected
            self.pinnedLegChanged.emit(selected)
        self._update_leg_focus()
        return True

    def set_pinned_leg(self, index: int | None) -> None:
        if self._route_layer is None or index is not None and not 0 <= index < len(self._route_layer.legs):
            index = None
        self._pinned_leg = index
        self._hover_hits, self._hover_leg = (), None
        self._update_leg_focus()
        self.pinnedLegChanged.emit(index if index is not None else -1)

    def _update_leg_focus(self) -> None:
        index = self._pinned_leg if self._pinned_leg is not None else self._hover_leg
        message = ""
        if self._route_layer is not None:
            changed = index != self.active_leg
            self._route_layer.focus(index)
            if index is not None:
                message = self._route_layer.legs[index].description()
                if len(self._hover_hits) > 1:
                    message += " · Shared by legs " + ", ".join(str(i + 1) for i in self._hover_hits)
                    message += "; right-click to switch"
            if changed:
                self.viewport().update()
        info = (index if index is not None else -1, message)
        if info != self._last_leg_info:
            self._last_leg_info = info
            self.legHighlighted.emit(*info)

    def drawForeground(self, painter, rect) -> None:
        super().drawForeground(painter, rect)
        if self._route_layer is None or self.active_leg is None:
            return
        arrows = self._route_layer.direction_arrows(self.viewportTransform(), self.viewport().rect())
        markers = [self.viewportTransform().map(item.scenePos()) for item in self._markers]
        painter.save()
        try:
            painter.resetTransform()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QPen(QColor("white"), 1))
            painter.setBrush(QColor("#963714"))
            for arrow in arrows:
                center = arrow.boundingRect().center()
                if not any((center.x() - marker.x()) ** 2 + (center.y() - marker.y()) ** 2 < 18 ** 2
                           for marker in markers):
                    painter.drawPolygon(arrow)
        finally:
            painter.restore()

    def show_endpoints(self, source: int, target: int, middle_nodes: list[int] | tuple[int, ...] = ()) -> None:
        for item in self._markers:
            self.scene().removeItem(item)
        self._markers = []
        if self.data is None:
            return
        stops = [(source, "#178365", "Start", 16, None)]
        stops.extend((node, "#286fb6", f"Middle node #{index}", max(22, 12 + 7 * len(str(index))), str(index))
                     for index, node in enumerate(middle_nodes, 1))
        stops.append((target, "#d84454", "Destination", 12, None))
        for node, color, label, size, number in stops:
            item = QGraphicsEllipseItem(-size / 2, -size / 2, size, size)
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            item.setPos(QPointF(*self.data.points[node]))
            item.setBrush(QColor(color))
            item.setPen(QPen(QColor("white"), 2))
            item.setZValue(20)
            item.setToolTip(f"{label}: node {node}")
            item.setData(0, node)
            item.setData(1, label)
            if number is not None:
                text = QGraphicsSimpleTextItem(number, item)
                font = QFont()
                font.setPointSize(9)
                font.setBold(True)
                text.setFont(font)
                text.setBrush(QColor("white"))
                bounds = text.boundingRect()
                text.setPos(-bounds.width() / 2, -bounds.height() / 2)
                text.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.scene().addItem(item)
            self._markers.append(item)

    def clear_route(self) -> None:
        if self._route is not None:
            self.scene().removeItem(self._route)
        self._route = None
        self._route_layer = None
        self._pinned_leg, self._hover_leg, self._hover_hits = None, None, ()
        self._route_bounds = None
        self._update_leg_focus()
        self.routeCleared.emit()
        self.viewport().update()

    def show_route(self, nodes: list[int], legs: list[dict] | None = None) -> None:
        self.clear_route()
        if self.data is None or not nodes:
            return
        points = self.data.points[nodes]
        self._route_layer = RouteLayer(self.data.points, legs or [dict(label="Route", path_nodes=nodes)])
        self._route = self._route_layer.group
        self.scene().addItem(self._route)
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        self._route_bounds = QRectF(float(xmin), float(ymin), max(float(xmax - xmin), 100.0),
                                    max(float(ymax - ymin), 100.0))
