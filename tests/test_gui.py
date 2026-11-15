"""Offscreen GUI checks; requires PySide6, but no extra pytest plugins."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from python.gui.backend import ALGORITHMS, JsonProcess, RouteRequest, find_executable
from python.gui.buckets import BUCKET_VALUES, BucketControl, default_bucket
from python.gui.data import LoadCancelled, load_dataset
from python.gui.map_view import MapView
from python.gui.window import MainWindow


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


def wait_for(app, condition, timeout=15):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail("Timed out waiting for asynchronous GUI work")
        app.processEvents()
        QTest.qWait(10)
    app.processEvents()


@pytest.fixture
def graph(tmp_path):
    path = tmp_path / "graph.csr"
    with path.open("wb") as stream:
        stream.write(struct.pack("<8sQQ", b"PSVCSR01", 6, 5))
        np.array([0, 2, 3, 4, 4, 5, 5], dtype="<u8").tofile(stream)
        np.array([1, 2, 3, 1, 5], dtype="<u4").tofile(stream)
        np.array([13, 8, 1, 3, 1], dtype="<f8").tofile(stream)
    with path.with_name("graph_nodes.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["node_index", "osm_node_id", "latitude", "longitude"])
        writer.writerows((node, 100 + node, 63 + node * 0.001, 27 + (node % 3) * 0.001)
                         for node in range(6))
    return path


@pytest.fixture
def executable():
    path = find_executable()
    if path is None or not path.is_file():
        pytest.skip("Build psv_dijkstra or set PSV_EXECUTABLE for GUI integration checks")
    return path


def test_bucket_scale_and_every_mode_reset(app):
    widget = BucketControl()
    assert not widget.isEnabled() and widget.value is None
    assert widget.opacity.opacity() == 0.4
    assert BUCKET_VALUES[:10] == tuple(range(100, 1001, 100))
    assert BUCKET_VALUES[-2:] == (9_000_000, 10_000_000)
    for algorithm, early, expected in (("delta-cpu", True, 300), ("delta-cuda", True, 10_000),
                                        ("delta-cuda", False, 1_000_000), ("delta-cpu", False, 1000),
                                        ("delta-cpu", True, 300)):
        widget.configure(algorithm, early)
        assert widget.value == expected
        assert BUCKET_VALUES[widget.slider.value()] == expected
        assert widget.isEnabled() and widget.opacity.opacity() == 1.0
        widget.slider.setValue(len(BUCKET_VALUES) - 1)
        assert widget.value == 10_000_000
    widget.reset_button.click()
    assert widget.value == 300
    widget.spin.setValue(1350)
    assert widget.value == 1350  # Numeric entry need not be a slider tick.
    widget.configure("heap", True)
    assert not widget.isEnabled() and widget.value is None
    widget.configure("delta-cpu", True)
    assert widget.value == 300


def test_request_flags_and_defaults(graph):
    for algorithm in ALGORITHMS:
        for early in (False, True):
            default = default_bucket(algorithm, early)
            args = RouteRequest(graph, 0, 3, algorithm, early, default).arguments()
            assert ("--early-stop" in args) == early
            assert ("--delta" in args) == algorithm.startswith("delta-")
            if default:
                assert args[args.index("--delta") + 1] == str(default)
    assert "--delta" not in RouteRequest(graph, 0, 3, "heap", delta=1000).arguments()


def test_dataset_coordinates_direction_and_route_validation(graph):
    data = load_dataset(graph, render=False)
    assert data.node_count == 6 and len(data.targets) == 5
    for node, point in enumerate(data.points):
        assert data.nearest_node(*point, 1) == node
        assert data.coordinates(*point) == pytest.approx((data.latitude[node], data.longitude[node]))
    assert data.nearest_node(1e9, 1e9, 10) is None
    report = dict(source=0, target=3, node_count=6, directed_edge_count=5, reachable=True,
                  path_nodes=[0, 2, 1, 3], path_edges=[1, 3, 2], length_m=12)
    data.validate_route(report, 0, 3)
    with pytest.raises(ValueError, match="directed edge"):
        data.validate_route({**report, "path_edges": [0, 3, 2]}, 0, 3)
    with pytest.raises(ValueError, match="distance"):
        data.validate_route({**report, "length_m": 13}, 0, 3)
    data.validate_route({**report, "reachable": False, "path_nodes": [], "path_edges": [], "length_m": None}, 0, 3)
    with pytest.raises(LoadCancelled):
        load_dataset(graph, cancelled=lambda: True)


@pytest.mark.parametrize("damage", ["truncated", "offset", "target", "weight", "csv_count", "csv_order", "coordinate"])
def test_invalid_dataset_rejected(graph, damage):
    raw = bytearray(graph.read_bytes())
    if damage == "truncated":
        graph.write_bytes(raw[:-1])
    elif damage == "offset":
        struct.pack_into("<Q", raw, 24, 1)
        graph.write_bytes(raw)
    elif damage == "target":
        struct.pack_into("<I", raw, 24 + 7 * 8, 999)
        graph.write_bytes(raw)
    elif damage == "weight":
        struct.pack_into("<d", raw, 24 + 7 * 8 + 5 * 4, float("nan"))
        graph.write_bytes(raw)
    else:
        nodes = graph.with_name("graph_nodes.csv")
        lines = nodes.read_text().splitlines()
        if damage == "csv_count":
            lines.pop()
        elif damage == "csv_order":
            lines[1], lines[2] = lines[2], lines[1]
        else:
            lines[1] = "0,100,nan,27"
        nodes.write_text("\n".join(lines))
    with pytest.raises(ValueError):
        load_dataset(graph)


def test_map_click_layers_and_fit(app, graph):
    data = load_dataset(graph)
    view = MapView()
    view.resize(900, 650)
    view.show()
    view.set_dataset(data)
    app.processEvents()
    assert data.tiles and not data.overview.isNull()
    picked = []
    view.nodePicked.connect(picked.append)
    view.set_pick_mode(True)
    position = view.mapFromScene(QPointF(*data.points[2]))
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=position)
    assert picked == [2]
    view.show_endpoints(0, 3)
    view.show_route([0, 2, 1, 3])
    assert len(view._markers) == 2 and view._route is not None
    view.fit_route()
    view.zoom(10)
    assert view._vectors.isVisible()
    view.clear_route()
    assert view._route is None and len(view._markers) == 2
    view.close()


@pytest.mark.parametrize("mode", ["json", "invalid", "nonzero", "start_error", "cancel", "timeout"])
def test_process_lifecycle(app, tmp_path, mode):
    job = JsonProcess()
    result, failures, cancellations = [], [], []
    job.completed.connect(lambda report, ms: result.append((report, ms)))
    job.failed.connect(failures.append)
    job.cancelled.connect(lambda: cancellations.append(True))
    if mode == "start_error":
        job.start(tmp_path / "missing.exe", [])
    else:
        code = {"json": 'print("{\\"ok\\": true}")', "invalid": 'print("broken")',
                "nonzero": 'import sys; sys.stderr.write("Expected failure"); sys.exit(2)',
                "cancel": "import time; time.sleep(30)", "timeout": "import time; time.sleep(30)"}[mode]
        job.start(Path(sys.executable), ["-c", code], timeout_ms=100 if mode == "timeout" else 5000)
        if mode == "cancel":
            wait_for(app, lambda: job.process.state().name == "Running")
            job.cancel()
    wait_for(app, lambda: not job.busy)
    assert len(result) + len(failures) + len(cancellations) == 1
    if mode == "json":
        assert result[0][0] == {"ok": True} and result[0][1] >= 0
    elif mode == "cancel":
        assert cancellations
    else:
        assert failures


def test_process_can_restart_after_failure_and_cancel(app, tmp_path):
    job = JsonProcess()
    failures, cancelled, reports = [], [], []
    job.failed.connect(failures.append)
    job.cancelled.connect(lambda: cancelled.append(True))
    job.completed.connect(lambda report, _ms: reports.append(report))
    job.start(tmp_path / "missing.exe", [])
    wait_for(app, lambda: not job.busy)
    assert len(failures) == 1
    job.start(Path(sys.executable), ["-c", "import time; time.sleep(30)"])
    wait_for(app, lambda: job.process.state().name == "Running")
    job.cancel()
    wait_for(app, lambda: not job.busy)
    assert len(cancelled) == 1
    job.start(Path(sys.executable), ["-c", 'print("{\\"ok\\": true}")'])
    wait_for(app, lambda: not job.busy)
    assert reports == [{"ok": True}]


def test_window_backend_selection_resets_and_routes(app, graph, executable):
    # Exercise startup discovery rather than manually invoking the probe.
    window = MainWindow(graph=graph, executable=executable)
    window.show()
    assert not window.algorithm.isEnabled()
    wait_for(app, lambda: window.data is not None and window.loader is None and not window.probe_job.busy)
    assert window.data is not None and window.calculate.isEnabled(), window.backend_info.text()
    assert window.algorithm.isEnabled()
    assert window.algorithm.currentData() == "heap" and not window.bucket.isEnabled()
    window.target.setValue(3)
    for algorithm in ALGORITHMS:
        index = window.algorithm.findData(algorithm)
        available = window.capabilities.get(algorithm, {"available": True})["available"]
        assert window.algorithm.model().item(index).isEnabled() == available
        if not available:
            continue
        window.algorithm.setCurrentIndex(index)
        for early in (False, True):
            window.early_stop.setChecked(early)
            assert window.bucket.value == default_bucket(algorithm, early)
            window.calculate_route()
            assert window.cancel.isEnabled() and not window.settings_box.isEnabled()
            wait_for(app, lambda: not window.route_job.busy)
            assert window.result.text().startswith("0.012 km"), window.status.text()
            assert window.map._route is not None
    window.source.setValue(4)
    assert window.map._route is None  # Changing endpoints invalidates the old route.
    window.algorithm.setCurrentIndex(0)
    window.calculate_route()
    wait_for(app, lambda: not window.route_job.busy)
    assert "No directed route" in window.result.text()
    assert window.map._route is None
    window.target.setValue(4)
    window.calculate_route()
    wait_for(app, lambda: not window.route_job.busy)
    assert window.result.text().startswith("0.000 km")
    window.close()
    app.processEvents()


def test_window_bucket_reset_after_manual_overrides(app):
    window = MainWindow(autoload=False)
    for algorithm, early in (("delta-cpu", True), ("delta-cuda", True), ("delta-cuda", False),
                              ("delta-cpu", False), ("delta-cpu", True)):
        window.algorithm.setCurrentIndex(window.algorithm.findData(algorithm))
        window.early_stop.setChecked(early)
        assert window.bucket.value == default_bucket(algorithm, early)
        window.bucket.spin.setValue(9_000_000)
    window.algorithm.setCurrentIndex(0)
    assert window.bucket.value is None and not window.bucket.isEnabled()
    window.close()


@pytest.mark.parametrize("cuda,openmp", [(False, False), (False, True), (True, False), (True, True)])
def test_advanced_controls_follow_backend_availability(app, cuda, openmp):
    window = MainWindow(autoload=False)
    assert window.timeout.value() == 120
    assert window.advanced.isHidden()
    assert window.bucket.parentWidget() is window.advanced
    window.advanced_toggle.click()
    assert not window.advanced.isHidden()
    report = {
        "cuda": {"available": cuda, "reason": "" if cuda else "No compatible GPU", "device": "GPU"},
        "delta-cuda": {"available": cuda, "reason": "" if cuda else "No compatible GPU", "device": "GPU"},
        "cpu-parallel": {"available": openmp, "reason": "" if openmp else "OpenMP not compiled", "device": "OpenMP"},
        "delta-cpu": {"available": True, "device": "OpenMP" if openmp else "serial CPU"},
    }
    window._probe_completed(report, 0)
    for index, algorithm in enumerate(ALGORITHMS):
        item = window.algorithm.model().item(index)
        available = report.get(algorithm, {"available": True})["available"]
        assert item.isEnabled() == available
        assert ("unavailable" in item.text()) == (not available)
        if not available:
            assert item.toolTip() == report[algorithm]["reason"]
            assert item.foreground().color().alpha() < 255
            continue
        window.algorithm.setCurrentIndex(index)
        threads_applicable = openmp and algorithm in ("cpu-parallel", "delta-cpu")
        assert window.threads.isEnabled() == threads_applicable
        assert window.threads_label.isEnabled() == threads_applicable
        assert window.thread_opacity.opacity() == (1.0 if threads_applicable else 0.4)
    # A replacement executable can restore previously unavailable choices.
    all_available = {name: {"available": True, "device": "test backend"} for name in report}
    window._probe_completed(all_available, 0)
    for index in range(len(ALGORITHMS)):
        assert window.algorithm.model().item(index).isEnabled()
        assert "unavailable" not in window.algorithm.itemText(index)
        assert window.algorithm.itemData(index, Qt.ItemDataRole.ForegroundRole) is None
    window.close()


def test_window_closes_after_background_work(app, graph, executable):
    window = MainWindow(executable=executable, autoload=False)
    window.show()
    window.open_dataset(graph)
    window.probe_backends()
    window.close()
    wait_for(app, lambda: window.loader is None and not window.probe_job.busy and not window.isVisible())


def test_missing_dataset_and_executable_can_be_replaced(app, graph, executable, tmp_path):
    window = MainWindow(executable=tmp_path / "missing.exe", autoload=False)
    window.probe_backends()
    window.open_dataset(tmp_path / "missing.csr")
    wait_for(app, lambda: window.loader is None and not window.probe_job.busy)
    assert not window.calculate.isEnabled()
    assert "Cannot load dataset" in window.status.text()
    window.executable = executable
    window.probe_backends()
    window.open_dataset(graph)
    wait_for(app, lambda: window.loader is None and not window.probe_job.busy)
    assert window.calculate.isEnabled()
    window.close()
