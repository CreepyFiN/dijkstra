from __future__ import annotations

import copy
import csv
import gzip
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

import networkx as nx
import numpy as np
import osmium
import osmnx as ox
import pytest
from shapely.geometry import Polygon, mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from python import download_osm
from python import extract_osm

from python.convert_to_csr import (
    MAGIC,
    UINT32_MAX,
    UINT64_MAX,
    binary_size,
    graph_to_csr,
    write_csr
)

from python.validate import (
    check_route,
    check_sssp,
    make_hand_graph,
    make_random_graph,
    reference_distances,
    run_cpp,
    validate_graph
)


def find_executable(variable: str, name: str) -> Path:
    configured = os.environ.get(variable)
    if configured:
        candidates = [Path(configured)]
    else:
        filename = name + (".exe" if os.name == "nt" else "")
        candidates = [ROOT / "build" / filename]
        candidates += [ROOT / "build" / config / filename for config in ("Debug", "Release")]
    for path in candidates:
        if path.is_file():
            return path
    pytest.fail(f"Build {name} first or set {variable}; searched: {candidates}")


@pytest.fixture(scope="session")
def executable() -> Path:
    return find_executable("PSV_EXECUTABLE", "psv_dijkstra")


@pytest.fixture(scope="session")
def probe() -> Path:
    return find_executable("PSV_CSR_PROBE", "test_csr")


def test_default_heap_and_removed_auto(tmp_path, executable):
    binary = write_csr(graph_to_csr(make_random_graph(3)), tmp_path)
    command = [str(executable), "--graph", str(binary), "--source", "0", "--target", "1"]
    default = json.loads(subprocess.check_output(command, text=True))
    explicit = json.loads(subprocess.check_output(command + ["--algorithm", "heap"], text=True))
    assert default["algorithm"] == default["requested_algorithm"] == "heap"
    for field in ("length_m", "path_nodes", "path_edges", "settled_nodes"):
        assert default[field] == explicit[field]
    rejected = subprocess.run(command + ["--algorithm", "auto"], capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "Algorithm must be" in rejected.stderr
    backends = json.loads(subprocess.check_output([str(executable), "--backends"], text=True))
    assert "auto_policy" not in backends


@pytest.mark.parametrize("algorithm", ["cpu-parallel", "cuda"])
def test_parallel_backends(tmp_path, executable, monkeypatch, algorithm):
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("OMP_DYNAMIC", "FALSE")
    backends = json.loads(subprocess.check_output([str(executable), "--backends"], text=True))
    graph = make_random_graph(3)
    data = graph_to_csr(graph)
    binary = write_csr(data, tmp_path)
    if not backends[algorithm]["available"]:
        failed = subprocess.run([str(executable), "--graph", str(binary), "--source", "0",
                                 "--target", "1", "--algorithm", algorithm], capture_output=True, text=True)
        assert failed.returncode != 0
        assert backends[algorithm]["reason"] in failed.stderr
        return
    for source in (0, len(graph) // 2, len(graph) - 1):
        expected = reference_distances(graph, data, source)
        for target in (source, len(graph) - 1):
            report, rows = run_cpp(executable, binary, source, target, algorithm, tmp_path / "distances.csv")
            assert report["requested_algorithm"] == algorithm
            assert report["algorithm"] == algorithm
            check_sssp(data, rows, expected, source)
            check_route(data, report, expected, source, target)
            for field in ("gpu_compute_ms", "gpu_upload_ms", "gpu_download_ms"):
                assert report[field] >= 0 if algorithm == "cuda" else report[field] is None


@pytest.mark.parametrize("algorithm", ["delta-cpu", "delta-cuda"])
@pytest.mark.parametrize("delta", [0.25, 1, 100, 10000])
def test_delta_pipeline(tmp_path, executable, monkeypatch, algorithm, delta):
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    capabilities = json.loads(subprocess.check_output([str(executable), "--backends"], text=True))
    if algorithm == "delta-cuda" and not capabilities["cuda"]["available"]:
        pytest.skip("CUDA unavailable")
    for case in (0, 3, 7):
        graph = make_random_graph(case)
        data = graph_to_csr(graph)
        binary = write_csr(data, tmp_path)
        for source in (0, len(graph)-1):
            output = tmp_path / "delta.csv"
            result = subprocess.run([str(executable), "--graph", str(binary), "--source", str(source),
                "--target", str(len(graph)-1), "--algorithm", algorithm, "--delta", str(delta),
                "--distances", str(output)], capture_output=True, text=True, check=True, timeout=30)
            with output.open(newline="") as stream: rows = list(csv.DictReader(stream))
            expected = reference_distances(graph, data, source)
            check_sssp(data, rows, expected, source)
            check_route(data, json.loads(result.stdout), expected, source, len(graph)-1)


@pytest.mark.parametrize("algorithm", ["heap", "phased", "cpu-parallel", "cuda", "delta-cpu", "delta-cuda"])
def test_early_stop(tmp_path, executable, monkeypatch, algorithm):
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    backends = json.loads(subprocess.check_output([str(executable), "--backends"], text=True))
    if algorithm in backends and not backends[algorithm]["available"]:
        pytest.skip("Backend unavailable")
    # A first-found target distance of 13 must improve to 11. A long branch
    # must remain unfinished; zero cycles and an unreachable target are included.
    graph = nx.MultiDiGraph(crs="epsg:4326")
    for node in range(12): graph.add_node(node, x=27+node*0.001, y=63)
    for u,v,w in [(0,1,13),(0,2,8),(2,1,3),(0,3,1),(3,4,0),(4,3,0),
                  (0,5,100),(5,6,1),(6,7,1),(7,8,1),(8,9,1)]:
        graph.add_edge(u,v,length=w)
    data=graph_to_csr(graph)
    binary=write_csr(data,tmp_path)
    expected=reference_distances(graph,data,0)
    from python.benchmark import check_delta
    reference=tmp_path/"full.csv"
    run_cpp(executable,binary,0,1,"heap",reference)
    for delta in (1,10,1000):
        for target in (0,1,4,9,11):
            output=tmp_path/"early.csv"
            completed=subprocess.run([str(executable),"--early-stop","--graph",str(binary),"--source","0",
                "--target",str(target),"--algorithm",algorithm,"--delta",str(delta),"--distances",str(output)],
                capture_output=True,text=True,check=True,timeout=30)
            report=json.loads(completed.stdout)
            assert report["early_stop"]
            check_route(data,report,expected,0,target)
            check_delta(reference,output,binary,0,partial=not report["search_complete"])
            if target==11: assert report["search_complete"]
            if target==1 and not algorithm.startswith("delta-"):
                assert report["settled_nodes"] < sum(math.isfinite(x) for x in expected)
                assert not report["search_complete"]
            if not report["search_complete"]:
                with output.open(newline="") as stream: rows=list(csv.DictReader(stream))
                assert all(row["state"] in ("final","tentative","unknown") for row in rows)


@pytest.mark.parametrize("algorithm,early,expected", [
    ("delta-cpu",False,1000),("delta-cpu",True,300),
    ("delta-cuda",False,1000000),("delta-cuda",True,10000)])
def test_default_delta_widths(tmp_path, executable, algorithm, early, expected):
    available=json.loads(subprocess.check_output([str(executable),"--backends"],text=True))
    if not available[algorithm]["available"]: pytest.skip("Backend unavailable")
    binary=write_csr(graph_to_csr(make_hand_graph()),tmp_path)
    command=[str(executable),"--graph",str(binary),"--source","0","--target","3","--algorithm",algorithm]
    if early: command.append("--early-stop")
    for override in (None,17):
        report=json.loads(subprocess.check_output(command+([] if override is None else ["--delta",str(override)]),text=True))
        assert report["delta"]==(expected if override is None else override)
        assert report["length_m"]==3


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


def test_compressed_graphml_reuse_and_conversion(tmp_path, monkeypatch, probe):
    raw_dir = tmp_path / "raw"
    metadata_dir = tmp_path / "metadata"
    raw_dir.mkdir()
    metadata_dir.mkdir()
    original = tmp_path / "original.graphml"
    graph = make_hand_graph()
    ox.save_graphml(graph, original)
    compressed = raw_dir / "pohjois_savo.graphml.gz"
    with gzip.open(compressed, "wb") as stream:
        stream.write(original.read_bytes())
    digest = download_osm.file_sha256(original)
    assert download_osm.file_sha256(compressed) == digest
    download_osm.write_json(metadata_dir / "download.json", {"status": "success", "graph_sha256": digest})

    def unexpected_network(*args, **kwargs):
        pytest.fail("An existing verified GraphML archive must not trigger a network request")

    monkeypatch.setattr(download_osm, "fetch_boundary", unexpected_network)
    assert download_osm.download(tmp_path, "test/1.0", "https://example.test", "https://example.test/api") == compressed
    uncompressed_path = raw_dir / "pohjois_savo.graphml"
    assert download_osm.resolve_graphml(uncompressed_path) == compressed
    assert download_osm.resolve_graphml(original) == original
    subprocess.run([sys.executable, str(ROOT / "python/convert_to_csr.py"), "--graphml", str(uncompressed_path),
                    "--output-dir", str(tmp_path / "processed")], capture_output=True, text=True, check=True)
    binary = tmp_path / "processed/graph.csr"
    decoded = json.loads(subprocess.check_output([str(probe), str(binary)], text=True))
    data = graph_to_csr(graph)
    assert decoded == {"row_offsets": data.row_offsets.tolist(), "targets": data.targets.tolist(),
                       "weights_m": data.weights_m.tolist()}
    download_osm.write_json(metadata_dir / "download.json", {"status": "success", "graph_sha256": "wrong"})
    with pytest.raises(ValueError, match="metadata"):
        download_osm.download(tmp_path, "test/1.0", "https://example.test", "https://example.test/api")


@pytest.mark.parametrize("error", [download_osm.ResponseStatusCodeError("'server' responded: 502 Bad Gateway"),
                                  download_osm.requests.Timeout("read timed out"),
                                  download_osm.requests.ConnectionError("connection closed")])
def test_graph_download_retries_transient_failures(monkeypatch, error):
    calls, delays = [], []
    expected = make_hand_graph()
    polygon = download_osm.validate_boundary(boundary_fixture())

    def fetch(received_polygon, **options):
        assert received_polygon.equals(polygon)
        assert options == download_osm.GRAPH_OPTIONS
        calls.append(options)
        if len(calls) < 3:
            raise error
        return expected

    monkeypatch.setattr(download_osm.ox, "graph_from_polygon", fetch)
    monkeypatch.setattr(download_osm.time, "sleep", delays.append)
    assert download_osm.fetch_graph(polygon, max_attempts=3) is expected
    assert len(calls) == 3 and delays == [60, 120]


@pytest.mark.parametrize("message,retries", [("'server' responded: 503 Service Unavailable", 3),
                                            ("'server' responded: 403 Forbidden", 1),
                                            ("'server' responded: 406 Not Acceptable", 1)])
def test_graph_download_retry_limit_and_nontransient_errors(monkeypatch, message, retries):
    calls, delays = [], []

    def fail(*args, **kwargs):
        calls.append(1)
        raise download_osm.ResponseStatusCodeError(message)

    monkeypatch.setattr(download_osm.ox, "graph_from_polygon", fail)
    monkeypatch.setattr(download_osm.time, "sleep", delays.append)
    with pytest.raises(download_osm.ResponseStatusCodeError):
        download_osm.fetch_graph(None, max_attempts=3)
    assert len(calls) == retries and len(delays) == retries - 1


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


def test_local_extract_preserves_topology_and_uses_osmnx(tmp_path, executable, monkeypatch):
    # Synthetic test records, separate from the downloaded regional data.
    xml = tmp_path / "fixture.osm"
    xml.write_text('''<osm version="0.6" generator="unit-test">
      <node id="1" lat="63.0" lon="27.0"/>
      <node id="2" lat="63.0" lon="27.001"><tag k="highway" v="traffic_signals"/></node>
      <node id="3" lat="63.0" lon="27.002"/>
      <node id="4" lat="63.0" lon="27.02"/>
      <node id="11" lat="62.9995" lon="27.0005"/>
      <node id="12" lat="63.0005" lon="27.0005"/>
      <node id="21" lat="64.0" lon="28.0"/>
      <node id="22" lat="64.001" lon="28.0"/>
      <way id="101"><nd ref="1"/><nd ref="2"/><nd ref="3"/>
        <tag k="highway" v="residential"/><tag k="oneway" v="yes"/></way>
      <way id="102"><nd ref="1"/><nd ref="2"/>
        <tag k="highway" v="residential"/><tag k="oneway" v="yes"/></way>
      <way id="103"><nd ref="11"/><nd ref="12"/><tag k="highway" v="residential"/></way>
      <way id="104"><nd ref="3"/><nd ref="4"/><tag k="highway" v="residential"/></way>
      <way id="105"><nd ref="21"/><nd ref="22"/><tag k="highway" v="residential"/></way>
      <way id="201"><nd ref="2"/><nd ref="12"/><tag k="highway" v="footway"/></way>
      <way id="202"><nd ref="2"/><nd ref="12"/>
        <tag k="highway" v="residential"/><tag k="access" v="private"/></way>
      <way id="203"><nd ref="2"/><nd ref="12"/>
        <tag k="highway" v="residential"/><tag k="motor_vehicle" v="no"/></way>
      <way id="204"><nd ref="2"/><nd ref="12"/><tag k="highway" v="service"/></way>
    </osm>''', encoding="utf-8")
    pbf = tmp_path / "fixture.osm.pbf"
    with osmium.SimpleWriter(pbf) as writer:
        for entity in osmium.FileProcessor(xml):
            writer.add(entity)
    polygon = Polygon([(26.999, 62.999), (27.003, 62.999),
                       (27.003, 63.001), (26.999, 63.001)])
    original_supplier = ox._overpass._download_overpass_network
    original_builder = ox.graph_from_polygon
    calls = []

    def builder(boundary, **options):
        calls.append(options)
        assert boundary.equals(polygon)
        return original_builder(boundary, **options)

    def unexpected_network(*args, **kwargs):
        pytest.fail("A local OSM extract must not request roads from the network")

    monkeypatch.setattr(ox, "graph_from_polygon", builder)
    monkeypatch.setattr(download_osm.requests, "post", unexpected_network)
    graph, metadata = extract_osm.graph_from_extract(pbf, polygon, tmp_path, download_osm.GRAPH_OPTIONS)
    assert calls == [download_osm.GRAPH_OPTIONS]
    assert ox._overpass._download_overpass_network is original_supplier
    assert set(graph) == {1, 2, 3, 11, 12}
    assert graph.number_of_edges() == 5
    assert len(graph[1][2]) == 2 and not graph.has_edge(2, 1)
    assert graph.has_edge(11, 12) and graph.has_edge(12, 11)
    assert not nx.has_path(graph, 1, 11)
    assert graph.nodes[2]["highway"] == "traffic_signals"
    assert 45 < graph[1][2][0]["length"] < 55
    assert metadata["selected_osm_ways"] == 4 and metadata["selected_osm_nodes"] == 6
    assert metadata["type"] == "osm_extract"
    assert metadata["sha256"] == download_osm.file_sha256(pbf)
    assert not list(tmp_path.glob("osm-locations-*"))
    validate_graph(graph, executable, tmp_path / "csr", list(range(5)), ("heap", "phased"), True)


def test_local_extract_missing_location_preserves_error(tmp_path):
    xml = tmp_path / "missing-node.osm"
    xml.write_text('''<osm version="0.6" generator="unit-test">
      <node id="1" lat="63.0" lon="27.0"/>
      <way id="101"><nd ref="1"/><nd ref="2"/>
        <tag k="highway" v="residential"/></way>
    </osm>''', encoding="utf-8")
    pbf = tmp_path / "missing-node.osm.pbf"
    with osmium.SimpleWriter(pbf) as writer:
        for entity in osmium.FileProcessor(xml):
            writer.add(entity)
    polygon = Polygon([(26.999, 62.999), (27.003, 62.999),
                       (27.003, 63.001), (26.999, 63.001)])
    original = ox._overpass._download_overpass_network
    with pytest.raises(ValueError, match="lacks location for node 2 of way 101"):
        extract_osm.graph_from_extract(pbf, polygon, tmp_path, download_osm.GRAPH_OPTIONS)
    assert ox._overpass._download_overpass_network is original
    assert not list(tmp_path.glob("osm-locations-*"))


def test_local_extract_restores_supplier_on_failure(tmp_path, monkeypatch):
    original = ox._overpass._download_overpass_network
    invalid = tmp_path / "invalid.osm.pbf"
    invalid.write_bytes(b"invalid PBF")
    polygon = download_osm.validate_boundary(boundary_fixture())
    with pytest.raises(RuntimeError):
        extract_osm.graph_from_extract(invalid, polygon, tmp_path, download_osm.GRAPH_OPTIONS)
    assert ox._overpass._download_overpass_network is original
    monkeypatch.setattr(ox._overpass, "_get_network_filter", lambda _: '["highway"="residential"]')
    with pytest.raises(ValueError, match="Unsupported"):
        extract_osm.drive_filter()


def test_real_osm_if_available(tmp_path, executable):
    path = download_osm.resolve_graphml(ROOT / "data/raw/pohjois_savo.graphml")
    if not path.exists():
        pytest.skip("Real Pohjois-Savo GraphML not downloaded; synthetic tests are separate")
    metadata = json.loads((ROOT / "data/metadata/download.json").read_text())
    assert metadata["status"] == "success"
    assert metadata["graph_sha256"] == download_osm.file_sha256(path)
    graph = ox.load_graphml(path)
    assert len(graph) >= 3
    validate_graph(graph, executable, tmp_path, [0, len(graph) // 2, len(graph) - 1], ("heap",), False)
