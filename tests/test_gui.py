"""Offscreen GUI checks; requires PySide6, but no extra pytest plugins."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QTransform
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QScrollArea

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from python.gui.backend import ALGORITHMS, JsonProcess, RouteRequest, find_executable
from python.gui.buckets import BUCKET_VALUES, BucketControl, default_bucket
from python.gui.data import LoadCancelled, load_dataset
from python.gui.map_view import MapView
from python.gui.route_layer import RouteLayer
from python.gui.routing import JourneyRequest, RouteRunner
from python.gui.window import MainWindow, initial_destination


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


@pytest.fixture
def loop_graph(tmp_path):
    """Directed return routes through 0..4, plus isolated node 5."""
    path = tmp_path / "trip.csr"
    with path.open("wb") as stream:
        stream.write(struct.pack("<8sQQ", b"PSVCSR01", 6, 7))
        np.array([0, 2, 3, 5, 6, 7, 7], dtype="<u8").tofile(stream)
        np.array([1, 3, 2, 0, 4, 1, 0], dtype="<u4").tofile(stream)
        np.array([4, 1, 6, 3, 2, 2, 9], dtype="<f8").tofile(stream)
    with path.with_name("trip_nodes.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["node_index", "osm_node_id", "latitude", "longitude"])
        writer.writerows((node, 100 + node, 63 + node * 0.001, 27 + (node % 3) * 0.001)
                         for node in range(6))
    return path


@pytest.fixture
def line_data(tmp_path):
    path = tmp_path / "line.csr"
    with path.open("wb") as stream:
        stream.write(struct.pack("<8sQQ", b"PSVCSR01", 5, 8))
        np.array([0, 1, 3, 5, 7, 8], dtype="<u8").tofile(stream)
        np.array([1, 0, 2, 1, 3, 2, 4, 3], dtype="<u4").tofile(stream)
        np.full(8, 50, dtype="<f8").tofile(stream)
    with path.with_name("line_nodes.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["node_index", "osm_node_id", "latitude", "longitude"])
        writer.writerows((node, 100 + node, 63, 27 + node * 0.001) for node in range(5))
    return load_dataset(path)


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


def test_middle_point_rows_picking_renumbering_and_reset(app, graph):
    window = MainWindow(autoload=False)
    window.show()
    data = load_dataset(graph)
    window._dataset_loaded(data)
    window._refresh_enabled()
    window.add_middle_button.click()
    first = window.middle_rows[0]
    first.node.setValue(2)
    window.add_middle_button.click()
    second = window.middle_rows[1]
    second.node.setValue(3)
    assert window.route_nodes() == (0, 2, 3, 1)
    assert [item.data(1) for item in window.map._markers] == ["Start", "Middle node #1", "Middle node #2", "Destination"]
    assert [window.map._markers[index].childItems()[0].text() for index in (1, 2)] == ["1", "2"]
    second.pick_button.click()
    first.remove_button.click()  # Other buttons cancel picking before their action.
    assert second.title.text() == "Middle node #1" and window._pick_target is None
    second.pick_button.click()
    app.processEvents()
    position = window.map.mapFromScene(QPointF(*data.points[1]))
    QTest.mouseClick(window.map.viewport(), Qt.MouseButton.LeftButton, pos=position)
    assert second.node.value() == 1 and not window.map.pick_mode
    second.pick_button.click()
    second.remove_button.click()
    assert not window.middle_rows and window._pick_target is None and not window.map.pick_mode
    window.add_middle_button.click()
    window.middle_rows[0].node.setValue(5)
    window._dataset_loaded(data)
    assert not window.middle_rows and window.route_nodes() == (0, 1)
    assert len(window.map._markers) == 2
    window.close()


@pytest.mark.parametrize("count,expected", [(1, 0), (2, 1), (10, 1), (11, 10), (75, 10),
                                           (100, 10), (101, 100), (800, 100),
                                           (100_000, 10_000), (100_001, 100_000), (448_192, 100_000)])
def test_initial_destination_stays_valid_and_uses_round_indices(count, expected):
    assert initial_destination(count) == expected
    assert 0 <= initial_destination(count) < count


def test_add_middle_point_enters_picking_and_map_click_still_selects(app, graph):
    window = MainWindow(autoload=False)
    window.show()
    data = load_dataset(graph)
    window._dataset_loaded(data)
    window._refresh_enabled()
    window.add_middle_button.click()
    row = window.middle_rows[0]
    assert window._pick_target is row and row.pick_button.isChecked() and window.map.pick_mode
    window.add_middle_button.click()
    row = window.middle_rows[1]
    assert window._pick_target is row and not window.middle_rows[0].pick_button.isChecked()
    app.processEvents()
    point = window.map.mapFromScene(QPointF(*data.points[2]))
    QTest.mouseClick(window.map.viewport(), Qt.MouseButton.LeftButton, pos=point)
    assert row.node.value() == 2 and window._pick_target is None and not window.map.pick_mode
    window.pick_source.click()
    window.pick_destination.click()
    assert window._pick_target is window.destination_row and window.map.pick_mode
    assert not window.pick_source.isChecked() and window.pick_destination.isChecked()
    window.close()


def test_outside_controls_cancel_picking_without_swallowing_actions(app, graph):
    window = MainWindow(autoload=False)
    window.resize(1050, 680)
    window.show()
    window._dataset_loaded(load_dataset(graph))
    window._probe_completed({name: {"available": True, "device": "test backend"}
                             for name in ("cuda", "cpu-parallel", "delta-cpu", "delta-cuda")}, 0)
    window.add_middle_button.click()
    assert window.map.pick_mode
    window.fit_map_button.click()
    assert window._pick_target is None and not window.map.pick_mode
    window.pick_source.click()
    window.advanced_toggle.click()
    assert window.advanced.isVisible() and window._pick_target is None
    window.pick_source.click()
    app.processEvents()
    QTest.mouseClick(window.source.lineEdit(), Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    assert not window.map.pick_mode
    window.sidebar_scroll.ensureWidgetVisible(window.algorithm)
    app.processEvents()
    window.pick_source.click()
    QTest.mouseClick(window.algorithm, Qt.MouseButton.LeftButton,
                     pos=QPoint(window.algorithm.width() - 10, window.algorithm.height() // 2))
    assert window._pick_target is None
    window.algorithm.hidePopup()
    window.pick_source.click()
    scrollbar = window.sidebar_scroll.verticalScrollBar()
    wait_for(app, lambda: scrollbar.maximum() > 0)
    QTest.mouseClick(scrollbar, Qt.MouseButton.LeftButton,
                     pos=QPoint(scrollbar.width() // 2, scrollbar.height() // 2))
    assert window.map.pick_mode  # Scrolling and map navigation preserve selection mode.
    window.middle_rows[0].remove_button.click()
    assert not window.map.pick_mode and not window.middle_rows
    window.close()


def test_long_endpoint_list_scrolls_with_actions_pinned(app, graph):
    window = MainWindow(autoload=False)
    window.resize(1050, 680)
    window.show()
    window._dataset_loaded(load_dataset(graph))
    window._refresh_enabled()
    for index in range(12):
        window.add_middle_point()
        window.middle_rows[-1].node.setValue(index % 6)
    window.advanced_toggle.setChecked(True)
    wait_for(app, lambda: window.sidebar_scroll.verticalScrollBar().maximum() > 0)
    assert window.findChildren(QScrollArea) == [window.sidebar_scroll]
    window.sidebar_scroll.verticalScrollBar().setValue(window.sidebar_scroll.verticalScrollBar().maximum())
    app.processEvents()
    for button in (window.calculate, window.cancel):
        assert not window.sidebar_scroll.isAncestorOf(button)
        assert button.isVisible()
        top_left = button.mapTo(window, QPoint(0, 0))
        bottom_right = button.mapTo(window, button.rect().bottomRight())
        assert window.rect().contains(top_left) and window.rect().contains(bottom_right)
    original = window.route_nodes()
    window.swap.click()
    assert window.route_nodes() == tuple(reversed(original))
    window.close()


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("early", [False, True])
def test_ordered_journey_all_backends_and_modes(app, loop_graph, executable, algorithm, early):
    capabilities = json.loads(subprocess.check_output([str(executable), "--backends"], text=True))
    if not capabilities.get(algorithm, {"available": True})["available"]:
        pytest.skip("Backend unavailable")
    data = load_dataset(loop_graph, render=False)
    runner = RouteRunner()
    results, failures = [], []
    runner.completed.connect(lambda report, ms: results.append((report, ms)))
    runner.failed.connect(failures.append)
    runner.start(executable, data, JourneyRequest(loop_graph, (0, 2, 0, 4), algorithm, early,
                                                 default_bucket(algorithm, early)))
    wait_for(app, lambda: not runner.busy)
    assert not failures, failures
    report, elapsed = results[0]
    assert report["length_m"] == 23
    assert report["path_nodes"] == [0, 3, 1, 2, 0, 3, 1, 2, 4]
    assert report["path_edges"] == [1, 5, 2, 3, 1, 5, 2, 4]
    assert len(runner.reports) == report["leg_count"] == 3
    assert [leg["path_nodes"] for leg in report["legs"]] == [r["path_nodes"] for r in runner.reports]
    assert report["algorithm_ms"] == pytest.approx(sum(r["algorithm_ms"] for r in runner.reports))
    assert report["settled_nodes"] == sum(r["settled_nodes"] for r in runner.reports)
    assert elapsed >= report["algorithm_ms"]
    if algorithm in ("cuda", "delta-cuda"):
        for field in ("gpu_compute_ms", "gpu_upload_ms", "gpu_download_ms"):
            assert report[field] == pytest.approx(sum(r[field] for r in runner.reports))


def test_journey_repeated_stops_and_unreachable_leg(app, loop_graph, executable):
    data = load_dataset(loop_graph, render=False)
    runner = RouteRunner()
    results, failures = [], []
    runner.completed.connect(lambda report, _ms: results.append(report))
    runner.failed.connect(failures.append)
    runner.start(executable, data, JourneyRequest(loop_graph, (0, 0, 2, 2, 4)))
    wait_for(app, lambda: not runner.busy)
    assert results[-1]["path_nodes"] == [0, 3, 1, 2, 4]
    assert results[-1]["length_m"] == 11
    runner.start(executable, data, JourneyRequest(loop_graph, (0, 2, 5, 4)))
    wait_for(app, lambda: not runner.busy)
    assert not failures, failures
    report = results[-1]
    assert report["reachable"] is False and report["length_m"] is None
    assert not report["path_nodes"] and not report["path_edges"]
    assert len(runner.reports) == report["attempted_legs"] == 2
    assert "Leg 2/3" in report["failed_leg"] and "Middle node #1 → Middle node #2" in report["failed_leg"]


@pytest.mark.parametrize("when", ["between", "running"])
def test_cancel_journey_stops_remaining_legs_and_can_retry(app, loop_graph, executable, when):
    data = load_dataset(loop_graph, render=False)
    runner = RouteRunner()
    cancellations, results, failures = [], [], []
    runner.completed.connect(lambda report, _ms: results.append(report))
    runner.cancelled.connect(lambda: cancellations.append(True))
    runner.failed.connect(failures.append)

    def cancel_second_leg(*_args):
        if len(runner.reports) == 1:
            runner.cancel()

    signal = runner.progress if when == "between" else runner.job.process.started
    signal.connect(cancel_second_leg)
    runner.start(executable, data, JourneyRequest(loop_graph, (0, 2, 0, 4)))
    wait_for(app, lambda: not runner.busy)
    QTest.qWait(30)
    assert cancellations == [True] and not failures and not results
    assert len(runner.reports) == 1
    signal.disconnect(cancel_second_leg)
    runner.start(executable, data, JourneyRequest(loop_graph, (0, 2, 4)))
    wait_for(app, lambda: not runner.busy)
    assert results[-1]["length_m"] == 11 and not failures


def test_journey_timeout_covers_all_legs(app, loop_graph, monkeypatch):
    # The first leg finishes quickly; a delayed second leg must not get a fresh
    # full timeout or allow the third leg to start.
    class DelayedProcess(QObject):
        completed = Signal(object, float)
        failed = Signal(str)
        cancelled = Signal()

        def __init__(self, parent):
            super().__init__(parent)
            self.busy = False
            self.calls = []
            self.timer = QTimer(self)
            self.timer.setSingleShot(True)
            self.timer.timeout.connect(self.finish)

        def start(self, executable, arguments, timeout_ms, threads):
            self.calls.append(arguments)
            self.busy = True
            self.timer.start(1 if len(self.calls) == 1 else 1000)

        def finish(self):
            self.busy = False
            self.completed.emit(dict(source=0, target=2, node_count=6, directed_edge_count=7,
                                     reachable=True, path_nodes=[0, 3, 1, 2], path_edges=[1, 5, 2],
                                     length_m=9, algorithm="heap", early_stop=True,
                                     search_complete=False, settled_nodes=4, algorithm_ms=1), 2)

        def cancel(self):
            self.timer.stop()
            self.busy = False
            self.cancelled.emit()

    monkeypatch.setattr("python.gui.routing.JsonProcess", DelayedProcess)
    runner = RouteRunner()
    failures, results = [], []
    runner.failed.connect(failures.append)
    runner.completed.connect(lambda report, _ms: results.append(report))
    runner.start(Path("unused"), load_dataset(loop_graph, render=False),
                 JourneyRequest(loop_graph, (0, 2, 0, 4)), timeout_ms=100)
    wait_for(app, lambda: not runner.busy)
    assert len(failures) == 1 and "timeout" in failures[0]
    assert not results and len(runner.job.calls) == 2


def test_window_displays_combined_middle_point_route(app, loop_graph, executable):
    window = MainWindow(graph=loop_graph, executable=executable)
    window.show()
    wait_for(app, lambda: window.calculate.isEnabled())
    window.target.setValue(4)
    for node in (2, 0):
        window.add_middle_button.click()
        window.middle_rows[-1].node.setValue(node)
    window.calculate_route()
    assert not window.add_middle_button.isEnabled()
    wait_for(app, lambda: not window.route_job.busy)
    assert window.result.text().startswith("0.023 km"), window.status.text()
    assert window.map._route is not None and len(window.map._markers) == 4
    assert "Finalized-node visits" in window.metrics.text() and "3/3" in window.metrics.text()
    assert window.leg_controls.isVisible() and window.leg_combo.count() == 4
    window.leg_combo.setCurrentIndex(2)
    assert window.map.pinned_leg == window.map.active_leg == 1
    window.middle_rows[1].node.setValue(5)
    assert window.map._route is None
    assert window.map.pinned_leg is None and window.leg_controls.isHidden()
    assert window.leg_combo.count() == 1
    window.calculate_route()
    wait_for(app, lambda: not window.route_job.busy)
    assert "No directed route" in window.result.text() and "Leg 2/3" in window.status.text()
    assert window.map._route is None
    window.close()


def test_middle_point_remove_alignment(app, graph):
    window = MainWindow(autoload=False)
    window.show()
    window._dataset_loaded(load_dataset(graph))
    window._refresh_enabled()
    window.add_middle_point()
    row = window.middle_rows[0]
    app.processEvents()
    pick, remove, details = row.pick_button.geometry(), row.remove_button.geometry(), row.details.geometry()
    assert remove.top() > pick.bottom()
    assert remove.right() == pick.right()
    assert abs(remove.center().y() - details.center().y()) <= 1
    window.close()


@pytest.mark.parametrize("scale", [0.5, 1.0, 4.0])
def test_leg_hit_tolerance_is_in_screen_pixels(app, scale):
    points = np.array([[0, 20], [100, 20], [200, 20]], dtype=float)
    layer = RouteLayer(points, [dict(label="Outbound", length_m=200, path_nodes=[0, 1, 2]),
                                dict(label="Return", length_m=200, path_nodes=[2, 1, 0])])
    assert layer.hits(QPointF(50, 20 + 7 / scale), scale) == (0, 1)
    assert layer.hits(QPointF(50, 20 + 9 / scale), scale) == ()
    assert layer.hits(QPointF(1000, 20), scale) == ()


def test_leg_arrows_follow_return_direction_and_clip_at_high_zoom(app):
    points = np.array([[0, 20], [100, 20], [200, 20]], dtype=float)
    layer = RouteLayer(points, [dict(label="Outbound", length_m=200, path_nodes=[0, 1, 2]),
                                dict(label="Return", length_m=200, path_nodes=[2, 1, 0])])
    for leg, sign in ((0, 1), (1, -1)):
        layer.focus(leg)
        arrows = layer.direction_arrows(QTransform.fromScale(2, 2), QRect(0, 0, 600, 200))
        assert arrows
        assert all(sign * (arrow[0].x() - (arrow[1].x() + arrow[2].x()) / 2) > 0 for arrow in arrows)
        assert layer.items[leg].opacity() == 1 and layer.items[1 - leg].opacity() == 0.25
        assert layer.items[leg].zValue() > layer.items[1 - leg].zValue()
    transform = QTransform()
    transform.translate(-1_000_000, -199_900)
    transform.scale(10_000, 10_000)
    assert 0 < len(layer.direction_arrows(transform, QRect(0, 0, 600, 200))) <= 10
    layer.focus(None)
    assert all(item.opacity() == 1 for item in layer.items)
    assert not layer.direction_arrows(transform, QRect(0, 0, 600, 200))


def test_map_hover_and_right_click_cycle_overlapping_legs(app, line_data):
    view = MapView()
    view.resize(900, 650)
    view.show()
    view.set_dataset(line_data)
    legs = [dict(label="Leg 1: Outbound", length_m=200, path_nodes=[0, 1, 2, 3, 4]),
            dict(label="Leg 2: Return", length_m=200, path_nodes=[4, 3, 2, 1, 0]),
            dict(label="Leg 3: Outbound", length_m=200, path_nodes=[0, 1, 2, 3, 4])]
    view.show_route([0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4], legs)
    messages = []
    view.legHighlighted.connect(lambda index, text: messages.append((index, text)))
    point = (line_data.points[1] + line_data.points[2]) / 2
    pixel = view.mapFromScene(QPointF(*point))
    app.processEvents()
    QTest.mouseMove(view.viewport(), QPoint(5, 5))
    QTest.mouseMove(view.viewport(), pixel)
    app.processEvents()
    assert view.active_leg == 0
    assert "Shared by legs 1, 2, 3" in messages[-1][1]
    for expected in (1, 2, 0):
        QTest.mouseClick(view.viewport(), Qt.MouseButton.RightButton, pos=pixel)
        assert view.active_leg == expected
        QTest.mouseMove(view.viewport(), pixel + QPoint(1, 0))
        assert view.active_leg == expected  # Stable while still on the shared section.
    QApplication.sendEvent(view.viewport(), QEvent(QEvent.Type.Leave))
    assert view.active_leg is None and all(item.opacity() == 1 for item in view._route_layer.items)
    view.set_pinned_leg(1)
    QApplication.sendEvent(view.viewport(), QEvent(QEvent.Type.Leave))
    assert view.active_leg == view.pinned_leg == 1
    pins = []
    view.pinnedLegChanged.connect(pins.append)
    QTest.mouseClick(view.viewport(), Qt.MouseButton.RightButton, pos=pixel)
    assert view.pinned_leg == view.active_leg == pins[-1] == 2
    view.set_pinned_leg(None)
    assert view.active_leg is None
    view.set_pick_mode(True)
    QTest.mouseClick(view.viewport(), Qt.MouseButton.RightButton, pos=pixel)
    QTest.mouseMove(view.viewport(), pixel)
    assert view.active_leg is None
    view.set_pick_mode(False)
    QTest.mouseMove(view.viewport(), pixel + QPoint(2, 0))
    assert view.active_leg == 0
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=pixel)
    assert view.active_leg is None
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=pixel)
    view.clear_route()
    assert view._route_layer is None and view.active_leg is None and view.pinned_leg is None
    view.close()


def test_nonoverlapping_and_zero_length_legs(app, line_data):
    view = MapView()
    view.resize(900, 650)
    view.show()
    view.set_dataset(line_data)
    legs = [dict(label="First", length_m=100, path_nodes=[0, 1, 2]),
            dict(label="Pause", length_m=0, path_nodes=[2]),
            dict(label="Last", length_m=100, path_nodes=[2, 3, 4])]
    view.show_route([0, 1, 2, 3, 4], legs)
    view._hover_at(QPointF(*((line_data.points[0] + line_data.points[1]) / 2)))
    assert view.active_leg == 0
    view._hover_at(QPointF(*((line_data.points[3] + line_data.points[4]) / 2)))
    assert view.active_leg == 2
    assert not view._cycle_overlapping_legs(QPointF(*line_data.points[4]))
    view.set_pinned_leg(1)
    assert view.active_leg == 1
    assert not view._route_layer.direction_arrows(view.viewportTransform(), view.viewport().rect())
    view.set_dataset(line_data)
    assert view._route is None and view.active_leg is None
    view.close()


def test_leg_dropdown_and_right_click_stay_in_sync(app, line_data, executable):
    window = MainWindow(graph=line_data.graph_path, executable=executable)
    window.show()
    wait_for(app, lambda: window.calculate.isEnabled())
    window.target.setValue(0)
    window.add_middle_point()
    window.middle_rows[0].node.setValue(4)
    window.calculate_route()
    wait_for(app, lambda: not window.route_job.busy)
    assert window.result.text().startswith("0.400 km"), window.status.text()
    assert window.leg_combo.count() == 3 and window.leg_controls.isVisible()
    window.map.fit_route()
    app.processEvents()
    size = window.map.viewport().size()
    window.leg_combo.setCurrentIndex(2)
    assert window.map.pinned_leg == window.map.active_leg == 1
    point = (window.data.points[1] + window.data.points[2]) / 2
    pixel = window.map.mapFromScene(QPointF(*point))
    QTest.mouseClick(window.map.viewport(), Qt.MouseButton.RightButton, pos=pixel)
    assert window.map.active_leg == window.map.pinned_leg == 0
    assert window.leg_combo.currentIndex() == 1
    assert "Shared by legs 1, 2" in window.leg_info.text()
    app.processEvents()
    assert window.map.viewport().size() == size  # No layout jump on hover labels.
    window.leg_combo.setCurrentIndex(0)
    assert window.map.active_leg is None and window.map.pinned_leg is None
    assert all(item.opacity() == 1 for item in window.map._route_layer.items)
    window.source.setValue(1)
    assert window.leg_controls.isHidden() and window.leg_combo.count() == 1
    window.close()
