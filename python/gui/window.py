"""Main Qt Widgets window; routing stays in the existing executable."""
from __future__ import annotations

import math
import os
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QBrush, QCloseEvent, QKeySequence, QPalette
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QGraphicsOpacityEffect, QHBoxLayout, QLabel, QMainWindow, QProgressBar, QPushButton, QScrollArea,
    QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from .backend import ALGORITHMS, ROOT, JsonProcess, RouteRequest, find_executable
from .buckets import BucketControl
from .data import DatasetLoader, MapData
from .map_view import MapView

LABELS = {"heap": "Heap — serial CPU", "phased": "Phased — serial CPU scan",
          "cpu-parallel": "Dijkstra — OpenMP CPU", "cuda": "Dijkstra — CUDA GPU",
          "delta-cpu": "Delta-stepping — CPU", "delta-cuda": "Delta-stepping — CUDA GPU"}


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
        self.request: RouteRequest | None = None
        self._close_requested = False
        self._pick_target: str | None = None
        self.route_job, self.probe_job = JsonProcess(self), JsonProcess(self)
        self._build_ui()
        self._build_menu()
        self.route_job.completed.connect(self._route_completed)
        self.route_job.failed.connect(self._route_failed)
        self.route_job.cancelled.connect(self._route_cancelled)
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
        controls.setContentsMargins(16, 14, 16, 14)
        title = QLabel("Plan a route")
        title.setObjectName("title")
        controls.addWidget(title)
        hint = QLabel("Pick road nodes on the map or enter their indices. Drag to pan; scroll to zoom.")
        hint.setWordWrap(True)
        controls.addWidget(hint)

        self.endpoint_box = QGroupBox("Endpoints")
        endpoint_layout = QVBoxLayout(self.endpoint_box)
        self.source, self.target = QSpinBox(), QSpinBox()
        self.source.setAccessibleName("Start node index")
        self.target.setAccessibleName("Destination node index")
        self.source_details, self.target_details = QLabel(), QLabel()
        self.source_details.setWordWrap(True)
        self.target_details.setWordWrap(True)
        self.pick_source, self.pick_destination = QPushButton("Pick start"), QPushButton("Pick end")
        self.pick_group = QButtonGroup(self)
        for caption, spin, button, detail in (
                ("Start node", self.source, self.pick_source, self.source_details),
                ("Destination node", self.target, self.pick_destination, self.target_details)):
            endpoint_layout.addWidget(QLabel(caption))
            row = QHBoxLayout()
            row.addWidget(spin, 1)
            row.addWidget(button)
            endpoint_layout.addLayout(row)
            endpoint_layout.addWidget(detail)
            button.setCheckable(True)
            self.pick_group.addButton(button)
        self.swap = QPushButton("Swap endpoints")
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
        self.early_stop.setToolTip("Uncheck to compute shortest distances to every reachable node")
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
        controls.addLayout(buttons)
        self.backend_info = QLabel("Select a routing executable from the File menu if needed.")
        self.backend_info.setWordWrap(True)
        controls.addWidget(self.backend_info)
        controls.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.sidebar)
        scroll.setMinimumWidth(330)
        scroll.setMaximumWidth(430)
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
        self.gpu_details = QLabel()
        self.gpu_details.setWordWrap(True)
        self.gpu_toggle = QCheckBox("Show GPU timing details")
        self.gpu_toggle.hide()
        self.gpu_details.hide()
        result_layout.addWidget(self.result)
        result_layout.addWidget(self.metrics)
        result_layout.addWidget(self.gpu_toggle)
        result_layout.addWidget(self.gpu_details)
        right_layout.addWidget(results)
        splitter = QSplitter()
        splitter.addWidget(scroll)
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

        self.source.valueChanged.connect(self._endpoints_changed)
        self.target.valueChanged.connect(self._endpoints_changed)
        self.swap.clicked.connect(self._swap_endpoints)
        self.pick_source.clicked.connect(lambda checked: self._pick_mode("source" if checked else None))
        self.pick_destination.clicked.connect(lambda checked: self._pick_mode("target" if checked else None))
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
        escape.triggered.connect(lambda: self._pick_mode(None))
        self.addAction(escape)

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
        self.data = data
        self.map.set_dataset(data)
        for spin in (self.source, self.target):
            spin.blockSignals(True)
            spin.setRange(0, data.node_count - 1)
        self.source.setValue(0)
        self.target.setValue(min(1, data.node_count - 1))
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
        for node, label in ((self.source.value(), self.source_details), (self.target.value(), self.target_details)):
            label.setText(f"OSM {self.data.osm_ids[node]}\n{self.data.latitude[node]:.6f}, {self.data.longitude[node]:.6f}")
        self.map.show_endpoints(self.source.value(), self.target.value())
        self._invalidate()

    def _swap_endpoints(self) -> None:
        source, target = self.source.value(), self.target.value()
        self.source.setValue(target)
        self.target.setValue(source)

    def _pick_mode(self, endpoint: str | None) -> None:
        self._pick_target = endpoint
        self.pick_group.setExclusive(False)
        self.pick_source.setChecked(endpoint == "source")
        self.pick_destination.setChecked(endpoint == "target")
        self.pick_group.setExclusive(True)
        self.map.set_pick_mode(endpoint is not None)
        if endpoint is not None:
            self.status.setText("Click near a road to choose the " + ("start." if endpoint == "source" else "destination."))

    def _node_picked(self, node: int) -> None:
        if self.route_job.busy or self.loader is not None:
            return
        if self._pick_target == "source":
            self.source.setValue(node)
        elif self._pick_target == "target":
            self.target.setValue(node)
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
        self.request = RouteRequest(self.data.graph_path, self.source.value(), self.target.value(),
                                    self.algorithm.currentData(), self.early_stop.isChecked(),
                                    self.bucket.value, self.threads.value())
        self.route_job.start(self.executable, self.request.arguments(), self.timeout.value() * 1000, self.request.threads)
        self.result.setText("Calculating…")
        self.status.setText("Calculating shortest path…")
        self.progress.setRange(0, 0)
        self.progress.show()
        self._refresh_enabled()

    def _route_completed(self, report: dict, elapsed_ms: float) -> None:
        try:
            self.data.validate_route(report, self.request.source, self.request.target)
            if (report.get("algorithm") != self.request.algorithm
                    or report.get("early_stop") != self.request.early_stop
                    or not math.isfinite(report["algorithm_ms"]) or report["algorithm_ms"] < 0):
                raise ValueError("Unexpected algorithm, search mode, or timing")
            if report["reachable"]:
                self.map.show_route(report["path_nodes"])
                self.result.setText(f"{report['length_m'] / 1000:,.3f} km · {LABELS[report['algorithm']]}")
                self.status.setText("Route ready")
            else:
                self.result.setText("No directed route between these nodes")
                self.status.setText("Destination is unreachable from the selected start")
            self.metrics.setText(f"Calculation: {report['algorithm_ms']:,.2f} ms   ·   Total request: {elapsed_ms:,.2f} ms\n"
                                 f"Finalized nodes: {report['settled_nodes']:,}   ·   "
                                 + ("Full search completed" if report["search_complete"] else "Partial search; destination finalized")
                                 + (f"   ·   Bucket: {report['delta']:,} m" if report.get("delta") is not None else ""))
            if report.get("gpu_compute_ms") is not None:
                self.gpu_details.setText(f"GPU computation phase: {report['gpu_compute_ms']:,.2f} ms\n"
                                         f"Upload: {report['gpu_upload_ms']:,.2f} ms · Download: {report['gpu_download_ms']:,.2f} ms")
                self.gpu_toggle.show()
                self.gpu_details.setVisible(self.gpu_toggle.isChecked())
        except (ValueError, KeyError, TypeError) as error:
            self._route_failed(str(error))
            return
        self._route_finished()

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
