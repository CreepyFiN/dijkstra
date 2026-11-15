"""Read the existing CSR/CSV pair and prepare display geometry off the UI thread."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
import struct
from typing import Callable

import numpy as np
from PySide6.QtCore import QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen

BACKGROUND = "#edf2f5"
ROAD_COLOR = "#879ba9"
EARTH_RADIUS = 6_371_000.0


class LoadCancelled(Exception):
    pass


@dataclass
class Tile:
    path: QPainterPath


@dataclass
class MapData:
    graph_path: Path
    rows: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    osm_ids: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    points: np.ndarray
    center_latitude: float
    center_longitude: float
    bounds: QRectF
    tiles: list[Tile] = field(default_factory=list)
    overview: QImage = field(default_factory=QImage)

    @property
    def node_count(self) -> int:
        return len(self.points)

    def coordinates(self, x: float, y: float) -> tuple[float, float]:
        latitude = self.center_latitude - math.degrees(y / EARTH_RADIUS)
        longitude = self.center_longitude + math.degrees(
            x / (EARTH_RADIUS * math.cos(math.radians(self.center_latitude))))
        return latitude, longitude

    def nearest_node(self, x: float, y: float, radius: float) -> int | None:
        # Vectorized search is inexpensive at regional scale; ties use node index.
        squared = np.square(self.points[:, 0] - x) + np.square(self.points[:, 1] - y)
        node = int(np.argmin(squared))
        return node if squared[node] <= radius * radius else None

    def validate_route(self, report: dict, source: int, target: int) -> None:
        """Reject mismatched/malformed output before indexing coordinates."""
        if (report.get("source") != source or report.get("target") != target
                or report.get("node_count") != self.node_count
                or report.get("directed_edge_count") != len(self.targets)):
            raise ValueError("Routing result does not match the loaded dataset/endpoints")
        nodes, edges = report.get("path_nodes"), report.get("path_edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ValueError("Routing result is missing its path arrays")
        if report.get("reachable") is False:
            if nodes or edges or report.get("length_m") is not None:
                raise ValueError("Invalid unreachable-route result")
            return
        if report.get("reachable") is not True or not nodes or len(edges) != len(nodes) - 1:
            raise ValueError("Invalid route length or reachability")
        if (any(type(node) is not int or not 0 <= node < self.node_count for node in nodes)
                or len(set(nodes)) != len(nodes) or nodes[0] != source or nodes[-1] != target):
            raise ValueError("Invalid route nodes")
        for u, v, edge in zip(nodes, nodes[1:], edges):
            if (type(edge) is not int or not int(self.rows[u]) <= edge < int(self.rows[u + 1])
                    or int(self.targets[edge]) != v):
                raise ValueError("Route contains an invalid directed edge")
        length = report.get("length_m")
        if (not isinstance(length, (float, int)) or not math.isfinite(length) or length < 0
                or not math.isclose(float(np.sum(self.weights[edges])), length,
                                    rel_tol=1e-10, abs_tol=1e-8)):
            raise ValueError("Route distance does not match its edges")


def load_dataset(graph_path: Path, nodes_path: Path | None = None,
                 progress: Callable[[str, int], None] = lambda *_: None,
                 cancelled: Callable[[], bool] = lambda: False,
                 render: bool = True) -> MapData:
    def checkpoint(message: str, percent: int) -> None:
        if cancelled():
            raise LoadCancelled()
        progress(message, percent)

    graph_path = graph_path.resolve()
    nodes_path = nodes_path or graph_path.with_name(graph_path.stem + "_nodes.csv")
    checkpoint("Reading road graph", 0)
    with graph_path.open("rb") as stream:
        header = stream.read(24)
        if len(header) != 24:
            raise ValueError("Truncated CSR header")
        magic, n, m = struct.unpack("<8sQQ", header)
        if magic != b"PSVCSR01" or not 0 < n < 2**32 - 1:
            raise ValueError("Invalid CSR header or empty graph")
        if graph_path.stat().st_size != 24 + 8 * (n + 1) + 12 * m:
            raise ValueError("CSR size does not match its header")
        rows = np.fromfile(stream, dtype="<u8", count=n + 1)
        targets = np.fromfile(stream, dtype="<u4", count=m)
        weights = np.fromfile(stream, dtype="<f8", count=m)
    if (rows[0] != 0 or rows[-1] != m or np.any(rows[1:] < rows[:-1])
            or np.any(targets >= n) or not np.all(np.isfinite(weights)) or np.any(weights < 0)):
        raise ValueError("Invalid CSR offsets, targets, or weights")
    checkpoint("Reading node coordinates", 10)
    osm_ids = np.empty(n, dtype=np.int64)
    latitude, longitude = np.empty(n), np.empty(n)
    count = 0
    with nodes_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"node_index", "osm_node_id", "latitude", "longitude"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Node CSV is missing required columns")
        for index, row in enumerate(reader):
            if index >= n or int(row["node_index"]) != index:
                raise ValueError("Node CSV indices must match the CSR order exactly")
            osm_ids[index] = int(row["osm_node_id"])
            latitude[index], longitude[index] = float(row["latitude"]), float(row["longitude"])
            count += 1
            if count % 20_000 == 0:
                checkpoint("Reading node coordinates", 10 + int(30 * count / n))
    if count != n:
        raise ValueError("Node CSV count does not match the CSR graph")
    if (not np.all(np.isfinite(latitude)) or not np.all(np.isfinite(longitude))
            or np.any(np.abs(latitude) >= 90) or np.any(np.abs(longitude) > 180)
            or np.any(osm_ids[1:] <= osm_ids[:-1])):
        raise ValueError("Node CSV must have valid coordinates and increasing OSM IDs")
    # Local equirectangular display projection. Routing still uses CSR metre weights.
    center_lat, center_lon = float(np.mean(latitude)), float(np.mean(longitude))
    x = np.radians(longitude - center_lon) * EARTH_RADIUS * math.cos(math.radians(center_lat))
    y = -np.radians(latitude - center_lat) * EARTH_RADIUS
    points = np.column_stack((x, y))
    bounds = QRectF(float(x.min()), float(y.min()),
                    max(float(np.ptp(x)), 10.0), max(float(np.ptp(y)), 10.0))
    data = MapData(graph_path, rows, targets, weights, osm_ids, latitude, longitude,
                   points, center_lat, center_lon, bounds)
    if not render:
        return data
    checkpoint("Grouping road geometry", 45)
    sources = np.repeat(np.arange(n, dtype=np.uint32), np.diff(rows).astype(np.int64))
    # Only the drawing is deduplicated; routing arrays preserve direction and parallels.
    codes = np.unique((np.minimum(sources, targets).astype(np.uint64) << np.uint64(32))
                      | np.maximum(sources, targets).astype(np.uint64))
    u, v = (codes >> np.uint64(32)).astype(np.intp), (codes & np.uint64(0xffffffff)).astype(np.intp)
    midpoint = (points[u] + points[v]) * 0.5
    tile_size = max(bounds.width(), bounds.height()) / 32
    keys = np.floor((midpoint[:, 0] - bounds.left()) / tile_size).astype(np.int64) * 64
    keys += np.floor((midpoint[:, 1] - bounds.top()) / tile_size).astype(np.int64)
    order = np.argsort(keys, kind="stable")
    divisions = np.flatnonzero(np.diff(keys[order])) + 1
    groups = np.split(order, divisions)
    for i, group in enumerate(groups):
        checkpoint("Preparing road tiles", 50 + int(35 * i / max(len(groups), 1)))
        path = QPainterPath()
        for edge in group:
            a, b = points[u[edge]], points[v[edge]]
            path.moveTo(float(a[0]), float(a[1]))
            path.lineTo(float(b[0]), float(b[1]))
        if not path.isEmpty():
            data.tiles.append(Tile(path))
    checkpoint("Preparing map overview", 90)
    factor = 3072 / max(bounds.width(), bounds.height())
    image = QImage(max(1, math.ceil(bounds.width() * factor)),
                   max(1, math.ceil(bounds.height() * factor)), QImage.Format.Format_RGB32)
    image.fill(QColor(BACKGROUND))
    painter = QPainter(image)
    try:
        painter.scale(factor, factor)
        painter.translate(-bounds.left(), -bounds.top())
        pen = QPen(QColor(ROAD_COLOR))
        pen.setWidthF(1.8)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for tile in data.tiles:
            painter.drawPath(tile.path)
    finally:
        painter.end()
    data.overview = image
    checkpoint("Map ready", 100)
    return data


class DatasetLoader(QThread):
    progress = Signal(str, int)
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, graph: Path, nodes: Path | None = None, parent=None):
        super().__init__(parent)
        self.graph, self.nodes = graph, nodes

    def run(self) -> None:
        try:
            data = load_dataset(self.graph, self.nodes, self.progress.emit, self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.loaded.emit(data)
        except LoadCancelled:
            pass
        except Exception as error:
            self.failed.emit(str(error))
