from __future__ import annotations

import copy
import csv
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

import networkx as nx
import numpy as np
import osmnx as ox
import pytest
from shapely.geometry import Polygon, mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import download_osm
from convert_to_csr import MAGIC, UINT32_MAX, UINT64_MAX, binary_size, graph_to_csr, write_csr
from validate import (check_route, check_sssp, make_hand_graph, make_random_graph,
                      reference_distances, run_cpp, validate_graph)


@pytest.fixture(scope="session")
def executable() -> Path:
    path = Path(os.environ.get("PSV_EXECUTABLE", ROOT / "build/psv_dijkstra"))
    if not path.is_file():
        pytest.fail(f"Build the C++ executable first: {path}")
    return path


@pytest.fixture(scope="session")
def probe() -> Path:
    path = Path(os.environ.get("PSV_CSR_PROBE", ROOT / "build/test_csr"))
    if not path.is_file():
        pytest.fail(f"Build the C++ CSR test probe first: {path}")
    return path


def test_exact_binary_layout_and_mappings(tmp_path, probe):
    graph = make_hand_graph()
    # A parallel edge and a self-loop must remain separate CSR entries.
    graph.add_edge(0, 1, key=8, length=0.25)
    graph.add_edge(2, 2, key=9, length=0.0)
    data = graph_to_csr(graph)
    binary = write_csr(data, tmp_path)
    expected = struct.pack("<8sQQ", MAGIC, len(graph), graph.number_of_edges())
    expected += struct.pack(f"<{len(data.row_offsets)}Q", *data.row_offsets)
    expected += struct.pack(f"<{len(data.targets)}I", *data.targets)
    expected += struct.pack(f"<{len(data.weights_m)}d", *data.weights_m)
    assert binary.read_bytes() == expected
    decoded = json.loads(subprocess.check_output([str(probe), str(binary)], text=True))
    assert decoded == {"row_offsets": data.row_offsets.tolist(), "targets": data.targets.tolist(),
                       "weights_m": data.weights_m.tolist()}
    with (tmp_path / "graph_nodes.csv").open(newline="") as stream:
        nodes = list(csv.DictReader(stream))
    assert [int(row["osm_node_id"]) for row in nodes] == sorted(graph)
    for index, row in enumerate(nodes):
        node = int(row["osm_node_id"])
        assert int(row["node_index"]) == index
        assert float(row["longitude"]) == graph.nodes[node]["x"]
        assert float(row["latitude"]) == graph.nodes[node]["y"]
    with (tmp_path / "graph_edges.csv").open(newline="") as stream:
        edges = list(csv.DictReader(stream))
    assert len(edges) == graph.number_of_edges()
    assert [tuple(int(row[key]) for key in ("osm_u", "osm_v", "osm_key")) for row in edges] == data.osm_edges
    assert [int(row["csr_edge_index"]) for row in edges] == list(range(data.edge_count))
    assert data.row_offsets.dtype == np.dtype("uint64")
    assert data.targets.dtype == np.dtype("uint32")
    assert data.osm_node_ids.dtype == np.dtype("int64")
    assert all(array.dtype == np.dtype("float64") for array in (data.weights_m, data.latitude, data.longitude))


def test_numeric_order_int64_and_insertion_independence(tmp_path):
    graph = nx.MultiDiGraph(crs="epsg:4326")
    ids = [2**63 - 1, 100, 9, -(2**63)]
    for node in ids:
        graph.add_node(node, x=27, y=63)
    for key in [10, 2, 1]:
        graph.add_edge(9, 100, key=key, length=float(key))
    graph.add_edge(9, 9, key=0, length=0)
    reverse = nx.MultiDiGraph(crs="epsg:4326")
    reverse.add_nodes_from(reversed(list(graph.nodes(data=True))))
    reverse.add_edges_from(reversed(list(graph.edges(keys=True, data=True))))
    first, second = graph_to_csr(graph), graph_to_csr(reverse)
    assert first.osm_node_ids.tolist() == sorted(ids)
    assert first.osm_edges == [(9, 9, 0), (9, 100, 1), (9, 100, 2), (9, 100, 10)]
    left = write_csr(first, tmp_path / "left")
    right = write_csr(second, tmp_path / "right")
    for name in ("graph.csr", "graph_nodes.csv", "graph_edges.csv"):
        assert (left.parent / name).read_bytes() == (right.parent / name).read_bytes()


@pytest.mark.parametrize("weight", [-1, math.nan, math.inf, -math.inf])
def test_invalid_weights(weight):
    graph = make_hand_graph()
    graph[0][1][0]["length"] = weight
    with pytest.raises(ValueError, match="length"):
        graph_to_csr(graph)


@pytest.mark.parametrize("node", [2**63, -(2**63) - 1, "123", 1.5])
def test_invalid_osm_ids(node):
    graph = nx.MultiDiGraph()
    graph.add_node(node, x=27, y=63)
    with pytest.raises(ValueError, match="int64"):
        graph_to_csr(graph)


@pytest.mark.parametrize("field,value", [("x", math.nan), ("x", 181), ("y", math.inf), ("y", 91)])
def test_invalid_coordinates(field, value):
    graph = make_hand_graph()
    graph.nodes[0][field] = value
    with pytest.raises(ValueError, match="coordinates"):
        graph_to_csr(graph)


def test_reject_wrong_graph_kind_and_simplified_graph():
    for graph in [nx.Graph(), nx.MultiGraph(), nx.DiGraph()]:
        with pytest.raises(ValueError, match="MultiDiGraph"):
            graph_to_csr(graph)
    graph = make_hand_graph()
    graph.graph["simplified"] = True
    with pytest.raises(ValueError, match="simplified"):
        graph_to_csr(graph)
    graph = make_hand_graph()
    graph.add_edge(0, 1, key="a", length=1)
    with pytest.raises(ValueError, match="keys"):
        graph_to_csr(graph)


def test_count_limits_without_allocating():
    assert binary_size(0, 0) == 32
    assert binary_size(UINT32_MAX, 0) == 24 + 8 * (UINT32_MAX + 1)
    for n, m in [(UINT32_MAX + 1, 0), (2**64, 0), (1, UINT64_MAX), (1, UINT64_MAX // 12), (1, 2**64)]:
        with pytest.raises(OverflowError):
            binary_size(n, m)
    for n, m in [(-1, 0), (0, -1), (1.1, 0)]:
        with pytest.raises(ValueError):
            binary_size(n, m)


def test_invalid_csr_arrays_are_rejected_before_writing(tmp_path):
    data = graph_to_csr(make_hand_graph())
    bad = copy.deepcopy(data)
    bad.row_offsets[1] = 20
    with pytest.raises(ValueError, match="offsets"):
        write_csr(bad, tmp_path)
    bad = copy.deepcopy(data)
    bad.targets[0] = UINT32_MAX
    with pytest.raises(ValueError, match="target"):
        write_csr(bad, tmp_path)
    bad = copy.deepcopy(data)
    bad.weights_m[0] = math.nan
    with pytest.raises(ValueError, match="Weights"):
        write_csr(bad, tmp_path)
    bad = copy.deepcopy(data)
    bad.targets = bad.targets.astype(np.uint64)
    with pytest.raises(ValueError, match="layout"):
        write_csr(bad, tmp_path)
    bad = copy.deepcopy(data)
    bad.osm_edges[0] = (99, 100, 0)
    with pytest.raises(ValueError, match="mapping"):
        write_csr(bad, tmp_path)
    assert not (tmp_path / "graph.csr").exists()


@pytest.mark.parametrize("algorithm", ["heap", "phased"])
def test_hand_graph_all_sources_and_targets(tmp_path, executable, algorithm):
    graph = make_hand_graph()
    original = tmp_path / "original.graphml"
    ox.save_graphml(graph, original)
    loaded = ox.load_graphml(original)
    data = graph_to_csr(loaded)
    assert reference_distances(graph, data, 0) == [0, 2, 1, 3, math.inf]
    assert validate_graph(loaded, executable, tmp_path / "csr", list(range(5)), (algorithm,), True,
                          original_graph=graph) == 25


@pytest.mark.parametrize("case", range(12))
def test_random_graph_full_pipeline(tmp_path, executable, probe, case):
    graph = make_random_graph(case)
    ox.save_graphml(graph, tmp_path / "input.graphml")
    loaded = ox.load_graphml(tmp_path / "input.graphml")
    data = graph_to_csr(loaded)
    binary = write_csr(data, tmp_path)
    decoded = json.loads(subprocess.check_output([str(probe), str(binary)], text=True))
    assert decoded["row_offsets"] == data.row_offsets.tolist()
    assert decoded["targets"] == data.targets.tolist()
    assert decoded["weights_m"] == data.weights_m.tolist()
    sources = sorted({0, len(graph) // 2, len(graph) - 1})
    validate_graph(loaded, executable, tmp_path, sources, ("heap", "phased"), True, original_graph=graph)


@pytest.mark.parametrize("algorithm", ["heap", "phased"])
def test_zero_cycles_parallel_edges_and_one_way(tmp_path, executable, algorithm):
    graph = nx.MultiDiGraph(crs="epsg:4326")
    for node in range(7):
        graph.add_node(node, x=27, y=63)
    for u, v, weight in [(0, 0, 0), (0, 1, 2), (0, 1, 0), (1, 0, 0), (1, 2, 0),
                          (2, 1, 0), (2, 3, 1), (0, 4, 1), (4, 3, 0), (5, 6, 1)]:
        graph.add_edge(u, v, length=weight)
    validate_graph(graph, executable, tmp_path, list(range(7)), (algorithm,), True)


def test_single_node_and_empty_graph(tmp_path, executable, probe):
    graph = nx.MultiDiGraph(crs="epsg:4326")
    binary = write_csr(graph_to_csr(graph), tmp_path / "empty")
    decoded = json.loads(subprocess.check_output([str(probe), str(binary)], text=True))
    assert decoded == {"row_offsets": [0], "targets": [], "weights_m": []}
    for algorithm in ("heap", "phased"):
        result = subprocess.run([str(executable), "--graph", str(binary), "--source", "0", "--target", "0",
                                 "--algorithm", algorithm], capture_output=True, text=True)
        assert result.returncode != 0 and "outside" in result.stderr
    graph.add_node(9876543210, x=27, y=63)
    validate_graph(graph, executable, tmp_path / "single", [0], ("heap", "phased"), True)


@pytest.mark.parametrize("source", ["-1", "5", str(UINT32_MAX), str(2**64), "1x"])
def test_invalid_cli_source(tmp_path, executable, source):
    binary = write_csr(graph_to_csr(make_hand_graph()), tmp_path)
    result = subprocess.run([str(executable), "--graph", str(binary), "--source", source, "--target", "0"],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "Error:" in result.stderr


def test_cli_defaults_errors_and_input_protection(tmp_path, executable):
    binary = write_csr(graph_to_csr(make_hand_graph()), tmp_path)
    base = [str(executable), "--graph", str(binary), "--source", "0", "--target", "3"]
    report = json.loads(subprocess.check_output(base, text=True))
    assert report["algorithm"] == "heap" and report["length_m"] == 3
    before = binary.read_bytes()
    for extra in [["--algorithm", "wrong"], ["--distances", str(binary)], ["--unknown", "1"], ["--distances"]]:
        failed = subprocess.run(base + extra, capture_output=True, text=True)
        assert failed.returncode != 0 and "Error:" in failed.stderr
    assert binary.read_bytes() == before
    assert "Usage:" in subprocess.check_output([str(executable), "--help"], text=True)
    binary.write_bytes(before[:-1])
    failed = subprocess.run(base, capture_output=True, text=True)
    assert failed.returncode != 0 and "length" in failed.stderr


@pytest.mark.parametrize("algorithm", ["heap", "phased"])
def test_distance_overflow_integration(tmp_path, executable, algorithm):
    graph = nx.MultiDiGraph(crs="epsg:4326")
    for node in range(3):
        graph.add_node(node, x=27, y=63)
    graph.add_edge(0, 1, length=sys.float_info.max)
    graph.add_edge(1, 2, length=sys.float_info.max)
    binary = write_csr(graph_to_csr(graph), tmp_path)
    failed = subprocess.run([str(executable), "--graph", str(binary), "--source", "0", "--target", "2",
                             "--algorithm", algorithm], capture_output=True, text=True)
    assert failed.returncode != 0 and "overflows" in failed.stderr


def boundary_fixture():
    # A synthetic unit-test polygon only; never used as real download data.
    return {"osm_type": "relation", "osm_id": 918898, "category": "boundary", "type": "administrative",
            "name": "Pohjois-Savo", "display_name": "Pohjois-Savo, Suomi",
            "address": {"country_code": "fi"}, "extratags": {"admin_level": "4"},
            "namedetails": {"ISO3166-2": "FI-15"},
            "geojson": mapping(Polygon([(26, 62), (28, 62), (27, 64), (26, 62)]))}


def test_boundary_identity_and_geometry():
    result = boundary_fixture()
    assert download_osm.validate_boundary(result).geom_type == "Polygon"
    for changes in [{"osm_type": "way"}, {"name": "Other"}, {"type": "village"},
                    {"address": {"country_code": "se"}}, {"namedetails": {"ISO3166-2": "FI-04"}},
                    {"extratags": {"admin_level": "8"}},
                    {"geojson": {"type": "Point", "coordinates": [27, 63]}}]:
        with pytest.raises(ValueError):
            download_osm.validate_boundary({**result, **changes})
    multi = {"type": "MultiPolygon", "coordinates": [result["geojson"]["coordinates"]]}
    assert download_osm.validate_boundary({**result, "geojson": multi}).geom_type == "MultiPolygon"


def test_boundary_cache_and_download_options(tmp_path, monkeypatch):
    responses = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return [boundary_fixture()]

    def get(url, **kwargs):
        responses.append((url, kwargs))
        return Response()

    monkeypatch.setattr(download_osm.requests, "get", get)
    monkeypatch.setattr(download_osm.time, "sleep", lambda _: None)
    metadata_dir = tmp_path / "metadata"
    download_osm.fetch_boundary(metadata_dir, "https://example.test", "test-app/1.0")
    download_osm.fetch_boundary(metadata_dir, "https://example.test", "test-app/1.0")
    assert len(responses) == 1
    assert responses[0][1]["headers"]["User-Agent"] == "test-app/1.0"
    assert responses[0][1]["params"]["polygon_threshold"] == 0
    saved_boundary = json.loads((metadata_dir / "boundary.geojson").read_text())
    assert saved_boundary["features"][0]["geometry"]["type"] == "Polygon"
    calls = []

    def graph_from_polygon(polygon, **kwargs):
        calls.append((polygon, kwargs))
        return make_hand_graph()

    monkeypatch.setattr(download_osm.ox, "graph_from_polygon", graph_from_polygon)
    # Restore OSMnx's shared settings when this test finishes.
    for name in ("use_cache", "cache_folder", "http_user_agent", "http_accept_language", "overpass_rate_limit",
                 "overpass_url", "requests_timeout", "log_console", "all_oneway", "bidirectional_network_types"):
        monkeypatch.setattr(ox.settings, name, getattr(ox.settings, name))
    path = download_osm.download(tmp_path, "test-app/1.0", "https://example.test", "https://example.test/api")
    assert calls[0][1] == {"network_type": "drive", "simplify": False,
                          "retain_all": True, "truncate_by_edge": False}
    assert calls[0][0].equals(download_osm.validate_boundary(boundary_fixture()))
    assert path.exists()
    metadata = json.loads((metadata_dir / "download.json").read_text())
    assert metadata["status"] == "success"
    assert metadata["graph_sha256"] == download_osm.file_sha256(path)
    download_osm.download(tmp_path, "test-app/1.0", "https://example.test", "https://example.test/api")
    assert len(calls) == 1 and len(responses) == 1


def test_download_failure_never_creates_fake_graph(tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(download_osm, "fetch_boundary", fail)
    for name in ("use_cache", "cache_folder", "http_user_agent", "http_accept_language", "overpass_rate_limit",
                 "overpass_url", "requests_timeout", "log_console", "all_oneway", "bidirectional_network_types"):
        monkeypatch.setattr(ox.settings, name, getattr(ox.settings, name))
    with pytest.raises(RuntimeError, match="simulated"):
        download_osm.download(tmp_path, "test-app/1.0", "https://example.test", "https://example.test/api")
    assert not (tmp_path / "raw/pohjois_savo.graphml").exists()
    metadata = json.loads((tmp_path / "metadata/download.json").read_text())
    assert metadata["status"] == "failed" and "simulated network failure" in metadata["error"]


def test_validation_detects_wrong_results(tmp_path, executable):
    data = graph_to_csr(make_hand_graph())
    binary = write_csr(data, tmp_path)
    expected = [0, 2, 1, 3, math.inf]
    report, rows = run_cpp(executable, binary, 0, 3, "heap", tmp_path / "distances.csv")
    report["path_edges"][0] = 0
    with pytest.raises(AssertionError):
        check_route(data, report, expected, 0, 3)
    rows[3]["distance_m"] = "10"
    with pytest.raises(AssertionError):
        check_sssp(data, rows, expected, 0)


def test_real_osm_if_available(tmp_path, executable):
    path = ROOT / "data/raw/pohjois_savo.graphml"
    if not path.exists():
        pytest.skip("Real Pohjois-Savo GraphML not downloaded; synthetic tests are separate")
    metadata = json.loads((ROOT / "data/metadata/download.json").read_text())
    assert metadata["status"] == "success"
    assert metadata["graph_sha256"] == download_osm.file_sha256(path)
    graph = ox.load_graphml(path)
    assert len(graph) >= 3
    validate_graph(graph, executable, tmp_path, [0, len(graph) // 2, len(graph) - 1], ("heap",), False)
