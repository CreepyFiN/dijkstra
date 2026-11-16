"""Main Qt Widgets window; routing stays in the existing executable."""
from __future__ import annotations

import math
import os
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QAction, QBrush, QCloseEvent, QKeySequence, QPalette
from PySide6.QtWidgets import (
    QAbstractButton, QAbstractSpinBox, QApplication, QButtonGroup, QCheckBox,
    QComboBox, QFileDialog, QFormLayout, QGroupBox, QMenuBar,
    QGraphicsOpacityEffect, QHBoxLayout, QLabel, QLayout, QMainWindow, QProgressBar, QPushButton, QScrollArea,
    QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from .backend import ALGORITHMS, ROOT, JsonProcess, find_executable
from .buckets import BucketControl
from .data import DatasetLoader, MapData
from .endpoints import EndpointRow
from .map_view import MapView
from .routing import JourneyRequest, RouteRunner

LABELS = {"heap": "Dijkstra — serial CPU, heap", "phased": "Dijkstra — serial CPU, phased",
          "cpu-parallel": "Dijkstra — parallel CPU, OpenMP", "cuda": "Dijkstra — parallel GPU, CUDA",
          "delta-cpu": "Delta-stepping — parallel CPU, OpenMP",
          "delta-cuda": "Delta-stepping — parallel GPU, CUDA"}
LEG_HINT = "Hover over a leg to highlight it. Right-click shared sections to switch legs."


def initial_destination(node_count: int) -> int:
    """Prefer 100000, or the largest valid lower power of ten."""
    limit = min(100_000, node_count - 1)
    if limit < 1:
        return 0
    node = 1
    while node * 10 <= limit:
        node *= 10
    return node


class MainWindow(QMainWindow):
    def __init__(self, graph: Path | None = None, nodes: Path | None = None,
                 executable: Path | None = None, *, autoload: bool = True):
        super().__init__()
        self.setWindowTitle("Pohjois-Savo | Shortest paths")
        self.resize(1320, 880)
        self.setMinimumSize(930, 680)
        self.data: MapData | None = None
        self.executable = executable if executable is not None else find_executable()
        self.loader: DatasetLoader | None = None
        self.capabilities: dict | None = None
        self.request: JourneyRequest | None = None
        self._close_requested = False
        self._pick_target: EndpointRow | None = None
        self.middle_rows: list[EndpointRow] = []
        self.route_job, self.probe_job = RouteRunner(self), JsonProcess(self)
        self._build_ui()
        self._build_menu()
        # pressed also handles keyboard/programmatic button activation. The
        # application filter covers child widgets, dropdown arrows and menus.
        for button in self.findChildren(QAbstractButton):
            if button not in self.pick_group.buttons():
                button.pressed.connect(self._cancel_picking)
        QApplication.instance().installEventFilter(self)
        self.route_job.completed.connect(self._route_completed)
        self.route_job.failed.connect(self._route_failed)
        self.route_job.cancelled.connect(self._route_cancelled)
        self.route_job.progress.connect(self.status.setText)
        self.probe_job.completed.connect(self._probe_completed)
        self.probe_job.failed.connect(self._probe_failed)
        self.probe_job.cancelled.connect(self._probe_cancelled)
        self._refresh_enabled()
        if autoload:
            QTimer.singleShot(0, self, self.probe_backends)
            if graph is None:
                graph = ROOT / "data/processed/graph.csr"
            QTimer.singleShot(0, self, lambda: self.open_dataset(graph, nodes))

    def _build_ui(self) -> None:
        self.setStyleSheet("""
            QMainWindow { background: #f5f7fa; }
            QGroupBox { font-weight: 600; margin-top: 10px; padding-top: 14px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }
            QPushButton { padding: 6px 10px; }
            QSpinBox, QComboBox { min-height: 26px; }
            QPushButton#calculate { background: #176f9b; color: white; font-weight: 600; }
            QPushButton#calculate:disabled { background: #b7c8d1; }
            QLabel#title { font-size: 22px; font-weight: 600; color: #193b50; }
            QLabel#result { font-size: 17px; font-weight: 600; }
        """)
        self.sidebar = QWidget()
        controls = QVBoxLayout(self.sidebar)
        controls.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        controls.setContentsMargins(16, 14, 16, 14)
        title = QLabel("Plan a route")
        title.setObjectName("title")
        controls.addWidget(title)
        hint = QLabel("Pick road nodes on the map or enter their indices. Drag to pan; scroll to zoom.")
        hint.setWordWrap(True)
        controls.addWidget(hint)

        self.endpoint_box = QGroupBox("Endpoints")
        endpoint_layout = QVBoxLayout(self.endpoint_box)
        self.pick_group = QButtonGroup(self)
        self.start_row = EndpointRow("Start node", "Pick start")
        self.destination_row = EndpointRow("Destination node", "Pick end")
        self.source, self.target = self.start_row.node, self.destination_row.node
        self.source_details, self.target_details = self.start_row.details, self.destination_row.details
        self.pick_source, self.pick_destination = self.start_row.pick_button, self.destination_row.pick_button
        endpoint_content = QWidget()
        endpoint_content_layout = QVBoxLayout(endpoint_content)
        endpoint_content_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        endpoint_content_layout.setContentsMargins(4, 4, 4, 4)
        endpoint_content_layout.addWidget(self.start_row)
        self.middle_container = QWidget()
        self.middle_layout = QVBoxLayout(self.middle_container)
        self.middle_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.middle_layout.setContentsMargins(0, 0, 0, 0)
        endpoint_content_layout.addWidget(self.middle_container)
        self.add_middle_button = QPushButton("Add a middle point")
        endpoint_content_layout.addWidget(self.add_middle_button)
        endpoint_content_layout.addWidget(self.destination_row)
        endpoint_content_layout.addStretch(1)
        endpoint_layout.addWidget(endpoint_content)
        self.swap = QPushButton("Reverse route")
        self.swap.setToolTip("Reverse the start, middle-point order, and destination")
        endpoint_layout.addWidget(self.swap)
        controls.addWidget(self.endpoint_box)

        self.settings_box = QGroupBox("Calculation")
        settings = QVBoxLayout(self.settings_box)
        self.algorithm = QComboBox()
        for name in ALGORITHMS:
            self.algorithm.addItem(LABELS[name], name)
        settings.addWidget(self.algorithm)
        self.early_stop = QCheckBox("Stop when destination is finalized")
        self.early_stop.setChecked(True)
        self.early_stop.setToolTip("Stop at the next requested point on each leg; uncheck for full search from each leg's start")
        settings.addWidget(self.early_stop)
        self.advanced_toggle = QCheckBox("Advanced settings")
        settings.addWidget(self.advanced_toggle)
        self.advanced = QWidget()
        advanced = QVBoxLayout(self.advanced)
        advanced.setContentsMargins(0, 0, 0, 0)
        self.bucket = BucketControl()
        advanced.addWidget(self.bucket)
        self.thread_controls = QWidget()
        thread_layout = QHBoxLayout(self.thread_controls)
        thread_layout.setContentsMargins(0, 0, 0, 0)
        self.threads_label = QLabel("OpenMP threads")
        self.threads = QSpinBox()
        self.threads.setRange(1, max(os.cpu_count() or 1, 4))
        self.threads.setValue(min(os.cpu_count() or 1, 4))
        self.threads_label.setBuddy(self.threads)
        thread_layout.addWidget(self.threads_label)
        thread_layout.addWidget(self.threads, 1)
        self.thread_opacity = QGraphicsOpacityEffect(self.thread_controls)
        self.thread_controls.setGraphicsEffect(self.thread_opacity)
        advanced.addWidget(self.thread_controls)
        self.timeout = QSpinBox()
        self.timeout.setRange(1, 3600)
        self.timeout.setValue(120)
        self.timeout.setSuffix(" s")
        self.timeout.setToolTip("Maximum duration for the entire journey, including all legs")
        timeout_layout = QFormLayout()
        timeout_layout.addRow("Timeout", self.timeout)
        advanced.addLayout(timeout_layout)
        settings.addWidget(self.advanced)
        self.advanced.hide()
        controls.addWidget(self.settings_box)

        buttons = QHBoxLayout()
        self.calculate, self.cancel = QPushButton("Calculate"), QPushButton("Cancel")
        self.calculate.setObjectName("calculate")
        buttons.addWidget(self.calculate, 1)
        buttons.addWidget(self.cancel)
        self.backend_info = QLabel("Select a routing executable from the File menu if needed.")
        self.backend_info.setWordWrap(True)
        controls.addWidget(self.backend_info)
        controls.addStretch(1)

        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sidebar_scroll.setWidget(self.sidebar)
        left_panel = QWidget()
        left_panel.setMinimumWidth(350)
        left_panel.setMaximumWidth(450)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.sidebar_scroll, 1)
        left_layout.addLayout(buttons)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        self.dataset_info = QLabel("No dataset loaded")
        toolbar.addWidget(self.dataset_info, 1)
        self.zoom_in, self.zoom_out = QPushButton("+"), QPushButton("−")
        self.fit_map_button, self.fit_route_button = QPushButton("Fit map"), QPushButton("Fit route")
        self.clear_button = QPushButton("Clear route")
        self.zoom_in.setAccessibleName("Zoom in")
        self.zoom_out.setAccessibleName("Zoom out")
        for button in (self.zoom_in, self.zoom_out, self.fit_map_button, self.fit_route_button, self.clear_button):
            toolbar.addWidget(button)
        right_layout.addLayout(toolbar)
        self.map = MapView()
        right_layout.addWidget(self.map, 1)
        credit = QLabel('Map data © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a> · ODbL 1.0')
        credit.setOpenExternalLinks(True)
        credit.setAlignment(Qt.AlignmentFlag.AlignRight)
        right_layout.addWidget(credit)
        results = QGroupBox("Route result")
        result_layout = QVBoxLayout(results)
        self.result = QLabel("Choose a start and destination")
        self.result.setObjectName("result")
        self.result.setWordWrap(True)
        self.metrics = QLabel("Calculation time and finalized-node count will appear here.")
        self.metrics.setWordWrap(True)
        self.leg_controls = QWidget()
        leg_layout = QVBoxLayout(self.leg_controls)
        leg_layout.setContentsMargins(0, 0, 0, 0)
        leg_row = QHBoxLayout()
        leg_row.addWidget(QLabel("Highlight leg"))
        self.leg_combo = QComboBox()
        self.leg_combo.setAccessibleName("Journey leg to highlight")
        self.leg_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.leg_combo.setMinimumContentsLength(16)
        self.leg_combo.addItem("All legs", None)
        self.leg_combo.setToolTip("Select a leg to keep it highlighted; All legs restores hover highlighting")
        leg_row.addWidget(self.leg_combo, 1)
        leg_layout.addLayout(leg_row)
        self.leg_info = QLabel(LEG_HINT)
        self.leg_info.setWordWrap(True)
        self.leg_info.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        # Hover text must not resize the map underneath the cursor.
        self.leg_info.setFixedHeight(self.leg_info.fontMetrics().lineSpacing() * 3)
        leg_layout.addWidget(self.leg_info)
        self.leg_controls.hide()
        self.gpu_details = QLabel()
        self.gpu_details.setWordWrap(True)
        self.gpu_toggle = QCheckBox("Show GPU timing details")
        self.gpu_toggle.hide()
        self.gpu_details.hide()
        result_layout.addWidget(self.result)
        result_layout.addWidget(self.metrics)
        result_layout.addWidget(self.leg_controls)
        result_layout.addWidget(self.gpu_toggle)
        result_layout.addWidget(self.gpu_details)
        right_layout.addWidget(results)
        splitter = QSplitter()
        splitter.addWidget(left_panel)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([350, 970])
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(splitter)
        self.setCentralWidget(central)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        self.statusBar().addWidget(self.status, 1)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(140)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.cursor_coordinates = QLabel()
        self.statusBar().addPermanentWidget(self.cursor_coordinates)

        self._connect_endpoint(self.start_row)
        self._connect_endpoint(self.destination_row)
        self.add_middle_button.clicked.connect(self.add_middle_point)
        self.swap.clicked.connect(self._swap_endpoints)
        self.map.nodePicked.connect(self._node_picked)
        self.map.pickMissed.connect(lambda: self.status.setText("No node nearby. Click closer to a road or zoom in."))
        self.map.coordinateHovered.connect(lambda lat, lon: self.cursor_coordinates.setText(f"{lat:.5f}, {lon:.5f}"))
        self.algorithm.currentIndexChanged.connect(self._settings_changed)
        self.early_stop.toggled.connect(self._settings_changed)
        self.bucket.changed.connect(self._invalidate)
        self.threads.valueChanged.connect(self._invalidate)
        self.advanced_toggle.toggled.connect(self.advanced.setVisible)
        self.calculate.clicked.connect(self.calculate_route)
        self.cancel.clicked.connect(self.route_job.cancel)
        self.zoom_in.clicked.connect(lambda: self.map.zoom(1.4))
        self.zoom_out.clicked.connect(lambda: self.map.zoom(1 / 1.4))
        self.fit_map_button.clicked.connect(self.map.fit_map)
        self.fit_route_button.clicked.connect(self.map.fit_route)
        self.clear_button.clicked.connect(self._clear_result)
        self.gpu_toggle.toggled.connect(self.gpu_details.setVisible)
        self.leg_combo.currentIndexChanged.connect(self._leg_selection_changed)
        self.map.legHighlighted.connect(self._leg_highlighted)
        self.map.pinnedLegChanged.connect(self._map_pinned_leg_changed)
        self.map.routeCleared.connect(self._clear_leg_controls)

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("File")
        self.open_action = QAction("Open dataset…", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self._choose_dataset)
        menu.addAction(self.open_action)
        self.executable_action = QAction("Select routing executable…", self)
        self.executable_action.triggered.connect(self._choose_executable)
        menu.addAction(self.executable_action)
        menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        menu.addAction(quit_action)
        escape = QAction(self)
        escape.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        escape.triggered.connect(self._cancel_picking)
        self.addAction(escape)

    def eventFilter(self, watched, event) -> bool:
        if (self._pick_target is not None and event.type() == QEvent.Type.MouseButtonPress
                and isinstance(watched, QWidget) and self.isAncestorOf(watched)
                and watched is not self.map and not self.map.isAncestorOf(watched)):
            control = watched
            while control is not None and control is not self:
                if isinstance(control, QAbstractButton) and control in self.pick_group.buttons():
                    break  # Pick buttons switch the active endpoint normally.
                if isinstance(control, (QAbstractButton, QComboBox, QAbstractSpinBox, QSlider, QMenuBar)):
                    self._cancel_picking()
                    break
                control = control.parentWidget()
        return super().eventFilter(watched, event)

    def _cancel_picking(self) -> None:
        if self._pick_target is not None:
            self._pick_mode(None)
            self.status.setText("Map picking cancelled. Choose endpoints or calculate a route.")

    def _choose_dataset(self) -> None:
        name, _ = QFileDialog.getOpenFileName(self, "Open CSR graph", str(ROOT / "data/processed"), "CSR graphs (*.csr)")
        if name:
            graph = Path(name)
            nodes = graph.with_name(graph.stem + "_nodes.csv")
            if not nodes.is_file():
                name, _ = QFileDialog.getOpenFileName(self, "Select matching node CSV", str(graph.parent), "CSV files (*.csv)")
                if not name:
                    return
                nodes = Path(name)
            self.open_dataset(graph, nodes)

    def open_dataset(self, graph: Path, nodes: Path | None = None) -> None:
        if self.loader is not None or self.route_job.busy:
            return
        self._clear_result()
        loader = DatasetLoader(graph, nodes, self)
        self.loader = loader
        loader.progress.connect(self._load_progress)
        loader.loaded.connect(self._dataset_loaded)
        loader.failed.connect(lambda message: self.status.setText("Cannot load dataset: " + message))
        loader.finished.connect(self._load_finished)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.show()
        self.status.setText("Loading dataset…")
        self._refresh_enabled()
        loader.start()

    def _load_progress(self, message: str, percent: int) -> None:
        self.status.setText(message)
        self.progress.setValue(percent)

    def _dataset_loaded(self, data: MapData) -> None:
        if self._close_requested:
            return
        self._discard_middle_rows()
        self.data = data
        self.map.set_dataset(data)
        for spin in (self.source, self.target):
            spin.blockSignals(True)
            spin.setRange(0, data.node_count - 1)
        self.source.setValue(0)
        self.target.setValue(initial_destination(data.node_count))
        for spin in (self.source, self.target):
            spin.blockSignals(False)
        self.dataset_info.setText(f"{data.graph_path.stem} · {data.node_count:,} nodes · {len(data.targets):,} edges")
        self.dataset_info.setToolTip(str(data.graph_path))
        self._endpoints_changed()
        self.status.setText("Map ready. Choose endpoints and calculate a route.")

    def _load_finished(self) -> None:
        loader = self.loader
        self.loader = None
        loader.deleteLater()
        self.progress.hide()
        self._refresh_enabled()
        self._maybe_close()

    def _choose_executable(self) -> None:
        name, _ = QFileDialog.getOpenFileName(self, "Select psv_dijkstra executable", str(ROOT),
                                              "Executables (*.exe);;All files (*)" if os.name == "nt" else "All files (*)")
        if name:
            self.executable = Path(name)
            self._clear_result()
            self.probe_backends()

    def probe_backends(self) -> None:
        if self.probe_job.busy:
            return
        self.capabilities = None
        if self.executable is None:
            self._probe_failed("Build psv_dijkstra, then select it from the File menu.")
            return
        self.backend_info.setText("Checking available backends…")
        self.probe_job.start(self.executable, ["--backends"], timeout_ms=15_000)
        self._refresh_enabled()

    def _probe_completed(self, report: dict, _elapsed: float) -> None:
        required = ("cpu-parallel", "cuda", "delta-cpu", "delta-cuda")
        if any(not isinstance(report.get(name), dict) or type(report[name].get("available")) is not bool
               for name in required):
            self._probe_failed("Executable returned invalid backend information")
            return
        self.capabilities = report
        disabled_color = self.algorithm.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
        disabled_color.setAlpha(150)
        for index, name in enumerate(ALGORITHMS):
            status = report.get(name, {"available": True})
            item = self.algorithm.model().item(index)
            item.setEnabled(status["available"])
            item.setText(LABELS[name] + ("" if status["available"] else " (unavailable)"))
            if status["available"]:
                item.setData(None, Qt.ItemDataRole.ForegroundRole)
            else:
                item.setForeground(QBrush(disabled_color))
            item.setToolTip(status.get("reason") or status.get("device") or LABELS[name])
        if not self.algorithm.model().item(self.algorithm.currentIndex()).isEnabled():
            self.algorithm.setCurrentIndex(0)
        gpu = report["cuda"]
        self.backend_info.setText("CUDA: " + (gpu.get("device", "available") if gpu["available"] else "unavailable")
                                  + "\nOpenMP: " + ("available" if report["cpu-parallel"]["available"] else "unavailable"))
        self.backend_info.setToolTip(str(self.executable) + "\n" + gpu.get("reason", ""))
        self._refresh_enabled()
        self._maybe_close()

    def _probe_failed(self, message: str) -> None:
        self.capabilities = None
        self.backend_info.setText(message)
        self._refresh_enabled()
        self._maybe_close()

    def _probe_cancelled(self) -> None:
        self._refresh_enabled()
        self._maybe_close()

    def _refresh_enabled(self) -> None:
        busy = self.route_job.busy
        loading = self.loader is not None
        self.endpoint_box.setEnabled(self.data is not None and not loading and not busy)
        self.settings_box.setEnabled(not busy)
        self.algorithm.setEnabled(self.capabilities is not None and not self.probe_job.busy)
        self._update_thread_availability()
        self.calculate.setEnabled(self.data is not None and self.capabilities is not None
                                  and not loading and not busy and not self.probe_job.busy)
        self.cancel.setEnabled(busy)
        self.open_action.setEnabled(not loading and not busy)
        self.executable_action.setEnabled(not busy and not self.probe_job.busy)
        self.clear_button.setEnabled(not busy)

    def _settings_changed(self, *_args) -> None:
        self.bucket.configure(self.algorithm.currentData(), self.early_stop.isChecked())
        self._update_thread_availability()
        self._invalidate()

    def _update_thread_availability(self) -> None:
        uses_threads = self.algorithm.currentData() in ("cpu-parallel", "delta-cpu")
        openmp_available = (self.capabilities is not None and not self.probe_job.busy
                            and self.capabilities["cpu-parallel"]["available"])
        applicable = uses_threads and openmp_available
        self.thread_controls.setEnabled(applicable)
        self.thread_opacity.setOpacity(1.0 if applicable else 0.4)
        self.thread_controls.setToolTip(
            "Worker count for the selected OpenMP CPU backend" if applicable
            else "Used only by CPU-parallel or CPU delta-stepping when OpenMP is available")

    def _endpoints_changed(self, *_args) -> None:
        if self.data is None:
            return
        for row in self._endpoint_rows():
            node = row.node.value()
            row.details.setText(f"OSM {self.data.osm_ids[node]}\n{self.data.latitude[node]:.6f}, {self.data.longitude[node]:.6f}")
        self.map.show_endpoints(self.source.value(), self.target.value(),
                                [row.node.value() for row in self.middle_rows])
        self._invalidate()

    def _endpoint_rows(self) -> list[EndpointRow]:
        return [self.start_row, *self.middle_rows, self.destination_row]

    def route_nodes(self) -> tuple[int, ...]:
        return tuple(row.node.value() for row in self._endpoint_rows())

    def _connect_endpoint(self, row: EndpointRow) -> None:
        self.pick_group.addButton(row.pick_button)
        row.node.valueChanged.connect(self._endpoints_changed)
        row.pick_button.clicked.connect(lambda checked: self._pick_mode(row if checked else None))
        row.removeRequested.connect(lambda: self.remove_middle_point(row))
        if hasattr(row, "remove_button"):
            row.remove_button.pressed.connect(self._cancel_picking)

    def add_middle_point(self) -> None:
        if self.data is None or self.route_job.busy or self.loader is not None:
            return
        row = EndpointRow(f"Middle node #{len(self.middle_rows) + 1}", removable=True)
        row.node.setRange(0, self.data.node_count - 1)
        row.node.setValue(self.middle_rows[-1].node.value() if self.middle_rows else self.source.value())
        self.middle_rows.append(row)
        self.middle_layout.addWidget(row)
        self._connect_endpoint(row)
        self._endpoints_changed()
        QTimer.singleShot(0, self, lambda: self._reveal_middle_point(row))
        row.node.setFocus()
        row.node.selectAll()
        self._pick_mode(row)

    def _reveal_middle_point(self, row: EndpointRow) -> None:
        if row in self.middle_rows:
            self.sidebar_scroll.ensureWidgetVisible(row)

    def remove_middle_point(self, row: EndpointRow) -> None:
        if row not in self.middle_rows or self.route_job.busy or self.loader is not None:
            return
        if self._pick_target is row:
            self._pick_mode(None)
        self.middle_rows.remove(row)
        self.pick_group.removeButton(row.pick_button)
        self.middle_layout.removeWidget(row)
        row.setParent(None)
        row.deleteLater()
        for index, remaining in enumerate(self.middle_rows, 1):
            remaining.set_caption(f"Middle node #{index}")
        self._endpoints_changed()

    def _discard_middle_rows(self) -> None:
        self._pick_mode(None)
        for row in self.middle_rows:
            self.pick_group.removeButton(row.pick_button)
            self.middle_layout.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self.middle_rows.clear()

    def _swap_endpoints(self) -> None:
        self._pick_mode(None)
        for row, node in zip(self._endpoint_rows(), reversed(self.route_nodes())):
            row.node.blockSignals(True)
            row.node.setValue(node)
            row.node.blockSignals(False)
        self._endpoints_changed()

    def _pick_mode(self, endpoint: EndpointRow | None) -> None:
        self._pick_target = endpoint
        self.pick_group.setExclusive(False)
        for row in self._endpoint_rows():
            row.pick_button.setChecked(endpoint is row)
        self.pick_group.setExclusive(True)
        self.map.set_pick_mode(endpoint is not None)
        if endpoint is not None:
            self.status.setText("Click near a road to choose " + endpoint.title.text().lower() + ".")

    def _node_picked(self, node: int) -> None:
        if self.route_job.busy or self.loader is not None:
            return
        if self._pick_target is not None:
            self._pick_target.node.setValue(node)
        self._pick_mode(None)
        self.status.setText(f"Selected node {node}. Ready to calculate.")

    def _invalidate(self, *_args) -> None:
        if self.request is not None and not self.route_job.busy:
            self.map.clear_route()
            self.result.setText("Settings changed — calculate again")
            self.metrics.clear()
            self.gpu_toggle.hide()
            self.gpu_details.hide()
            self.request = None

    def _clear_result(self) -> None:
        self.map.clear_route()
        self.request = None
        self.result.setText("Choose a start and destination")
        self.metrics.clear()
        self.gpu_toggle.setChecked(False)
        self.gpu_toggle.hide()
        self.gpu_details.hide()

    def calculate_route(self) -> None:
        if not self.calculate.isEnabled():
            return
        self._pick_mode(None)
        self._clear_result()
        self.request = JourneyRequest(self.data.graph_path, self.route_nodes(), self.algorithm.currentData(),
                                      self.early_stop.isChecked(), self.bucket.value, self.threads.value())
        self.route_job.start(self.executable, self.data, self.request, self.timeout.value() * 1000)
        self.result.setText("Calculating…")
        self.status.setText("Calculating shortest path…")
        self.progress.setRange(0, 0)
        self.progress.show()
        self._refresh_enabled()

    def _route_completed(self, report: dict, elapsed_ms: float) -> None:
        try:
            # RouteRunner validates each directed leg before joining its path.
            # A valid multi-leg journey can revisit vertices across legs.
            if (report.get("algorithm") != self.request.algorithm
                    or report.get("early_stop") != self.request.early_stop
                    or not math.isfinite(report["algorithm_ms"]) or report["algorithm_ms"] < 0):
                raise ValueError("Unexpected algorithm, search mode, or timing")
            if report["reachable"]:
                self.map.show_route(report["path_nodes"], report["legs"])
                self._populate_leg_controls(report["legs"])
                self.result.setText(f"{report['length_m'] / 1000:,.3f} km · {LABELS[report['algorithm']]}")
                self.status.setText("Route ready")
            else:
                self.result.setText("No directed route — " + report["failed_leg"])
                self.status.setText("Unreachable connection: " + report["failed_leg"])
            multiple = report["leg_count"] > 1
            summary = ("Stopped at unreachable leg" if not report["reachable"] else
                       "Full searches completed" if report["search_complete"] else
                       "Partial searches; requested stops finalized")
            self.metrics.setText(f"Calculation: {report['algorithm_ms']:,.2f} ms   ·   Total request: {elapsed_ms:,.2f} ms\n"
                                 + (f"Legs searched: {report['attempted_legs']}/{report['leg_count']}   ·   " if multiple else "")
                                 + ("Finalized-node visits: " if multiple else "Finalized nodes: ")
                                 + f"{report['settled_nodes']:,}   ·   "
                                 + summary
                                 + (f"   ·   Bucket: {report['delta']:,} m" if report.get("delta") is not None else ""))
            if report.get("gpu_compute_ms") is not None:
                scope = f" (sum of {report['attempted_legs']} legs)" if multiple else ""
                self.gpu_details.setText(f"GPU computation phase{scope}: {report['gpu_compute_ms']:,.2f} ms\n"
                                         f"Upload: {report['gpu_upload_ms']:,.2f} ms · Download: {report['gpu_download_ms']:,.2f} ms")
                self.gpu_toggle.show()
                self.gpu_details.setVisible(self.gpu_toggle.isChecked())
        except (ValueError, KeyError, TypeError) as error:
            self._route_failed(str(error))
            return
        self._route_finished()

    def _clear_leg_controls(self) -> None:
        previous = self.leg_combo.blockSignals(True)
        self.leg_combo.clear()
        self.leg_combo.addItem("All legs", None)
        self.leg_combo.blockSignals(previous)
        self.leg_info.setText(LEG_HINT)
        self.leg_controls.hide()

    def _populate_leg_controls(self, legs: list[dict]) -> None:
        self._clear_leg_controls()
        previous = self.leg_combo.blockSignals(True)
        for index, leg in enumerate(legs):
            label = f"{leg['label']} · {leg['length_m'] / 1000:,.3f} km"
            self.leg_combo.addItem(label, index)
            self.leg_combo.setItemData(index + 1, label, Qt.ItemDataRole.ToolTipRole)
        self.leg_combo.blockSignals(previous)
        self.leg_controls.setVisible(len(legs) > 1)

    def _leg_selection_changed(self, _index: int) -> None:
        self.map.set_pinned_leg(self.leg_combo.currentData())

    def _map_pinned_leg_changed(self, index: int) -> None:
        previous = self.leg_combo.blockSignals(True)
        self.leg_combo.setCurrentIndex(index + 1)
        self.leg_combo.blockSignals(previous)

    def _leg_highlighted(self, _index: int, message: str) -> None:
        text = message or LEG_HINT
        if self.leg_info.text() != text:
            self.leg_info.setText(text)
            self.leg_info.setToolTip(text)

    def _route_failed(self, message: str) -> None:
        self.map.clear_route()
        self.result.setText("Calculation failed")
        self.status.setText(message)
        self.status.setToolTip(message)
        self.request = None
        self._route_finished()

    def _route_cancelled(self) -> None:
        self.result.setText("Calculation cancelled")
        self.status.setText("Cancelled")
        self.request = None
        self._route_finished()

    def _route_finished(self) -> None:
        self.progress.hide()
        self._refresh_enabled()
        self._maybe_close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.loader is not None or self.route_job.busy or self.probe_job.busy:
            self._close_requested = True
            self.setEnabled(False)
            if self.loader is not None:
                self.loader.requestInterruption()
            self.route_job.cancel()
            self.probe_job.cancel()
            self.status.setText("Stopping background work…")
            event.ignore()
        else:
            event.accept()

    def _maybe_close(self) -> None:
        if (self._close_requested and self.loader is None
                and not self.route_job.busy and not self.probe_job.busy):
            QTimer.singleShot(0, self, self.close)
