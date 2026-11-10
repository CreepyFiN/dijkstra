"""Validate CPU SSSP against NetworkX Bellman-Ford and keep reproducible fixtures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import networkx as nx
import numpy as np
import osmnx as ox

from convert_to_csr import CsrData, UINT32_MAX, UINT64_MAX, graph_to_csr, write_csr
from download_osm import resolve_graphml

ROOT = Path(__file__).resolve().parents[1]
SEED = 918898


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def same_distance(actual: float, expected: float) -> bool:
    if math.isinf(expected):
        return actual == math.inf
    return math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8)


def reference_distances(graph: nx.MultiDiGraph, data: CsrData, source: int) -> list[float]:
    # Deliberately independent of both C++ Dijkstra implementations and the CSR.
    reference = nx.single_source_bellman_ford_path_length(graph, int(data.osm_node_ids[source]), weight="length")
    return [reference.get(int(node), math.inf) for node in data.osm_node_ids]


def check_original_graph(data: CsrData, original: nx.MultiDiGraph) -> None:
    require(data.node_count == len(original), "Original graph node count changed")
    require(data.edge_count == original.number_of_edges(), "Original graph edge count changed")
    require(data.osm_node_ids.tolist() == sorted(original), "Original node IDs changed")
    for index, node in enumerate(data.osm_node_ids):
        require(data.latitude[index] == original.nodes[int(node)]["y"], "Original latitude changed")
        require(data.longitude[index] == original.nodes[int(node)]["x"], "Original longitude changed")
    for index, (u, v, key) in enumerate(data.osm_edges):
        require(original.has_edge(u, v, key), "Original directed edge missing")
        require(data.weights_m[index] == original[u][v][key]["length"], "Original edge weight changed")


def run_cpp(executable: Path, binary: Path, source: int, target: int,
            algorithm: str, distances_path: Path) -> tuple[dict, list[dict]]:
    completed = subprocess.run(
        [str(executable.resolve()), "--graph", str(binary), "--source", str(source),
         "--target", str(target), "--algorithm", algorithm, "--distances", str(distances_path)],
        text=True, capture_output=True, check=True,
    )
    with distances_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return json.loads(completed.stdout), rows


def check_sssp(data: CsrData, rows: list[dict], expected: list[float], source: int) -> None:
    n = data.node_count
    require(len(rows) == n and len(expected) == n, "SSSP array size mismatch")
    distances, predecessors = [], []
    for node, (row, reference) in enumerate(zip(rows, expected)):
        require(int(row["node_index"]) == node, "Unexpected SSSP row order")
        distance = float(row["distance_m"])
        predecessor, edge = int(row["predecessor_node"]), int(row["predecessor_edge"])
        distances.append(distance)
        predecessors.append(predecessor)
        require(same_distance(distance, reference), f"Distance mismatch at {node}: {distance} != {reference}")
        reachable = math.isfinite(reference)
        require(int(row["settled"]) == int(reachable), f"Incorrect settled state at {node}")
        if not reachable or node == source:
            require(predecessor == UINT32_MAX and edge == UINT64_MAX, "Unexpected predecessor")
            continue
        require(0 <= predecessor < n and 0 <= edge < data.edge_count, "Invalid predecessor index")
        require(int(data.row_offsets[predecessor]) <= edge < int(data.row_offsets[predecessor + 1]),
                "Predecessor edge has wrong source")
        require(int(data.targets[edge]) == node, "Predecessor edge has wrong target")
        require(same_distance(float(rows[predecessor]["distance_m"]) + float(data.weights_m[edge]), distance),
                "Predecessor edge and distance disagree")
    require(distances[source] == 0, "Source distance is not zero")
    # Validate all predecessor chains and their source-to-target sums in O(N).
    # Local edge checks alone would miss a zero-weight predecessor cycle.
    state = np.zeros(n, dtype=np.uint8)
    path_lengths = np.full(n, np.inf)
    state[source], path_lengths[source] = 2, 0.0
    for target in range(n):
        if not math.isfinite(expected[target]) or state[target] == 2:
            continue
        chain = []
        node = target
        while state[node] != 2:
            require(state[node] != 1, "Cycle in predecessor forest")
            state[node] = 1
            chain.append(node)
            node = predecessors[node]
            require(0 <= node < n, "Predecessor chain does not reach source")
        for node in reversed(chain):
            edge = int(rows[node]["predecessor_edge"])
            path_lengths[node] = path_lengths[predecessors[node]] + data.weights_m[edge]
            require(same_distance(float(path_lengths[node]), expected[node]), "Path weight sum mismatch")
            state[node] = 2


def check_route(data: CsrData, report: dict, expected: list[float], source: int, target: int) -> None:
    require(report["node_count"] == data.node_count, "Reported N mismatch")
    require(report["directed_edge_count"] == data.edge_count, "Reported M mismatch")
    require(report["source"] == source and report["target"] == target, "Reported endpoints mismatch")
    require(math.isfinite(report["algorithm_ms"]) and report["algorithm_ms"] >= 0, "Invalid algorithm time")
    reachable = math.isfinite(expected[target])
    require(report["reachable"] == reachable, "Route reachability mismatch")
    require(report["status"] == ("reachable" if reachable else "unreachable"), "Route status mismatch")
    nodes, edges = report["path_nodes"], report["path_edges"]
    if not reachable:
        require(not nodes and not edges and report["length_m"] is None, "Unreachable route must be empty")
        return
    require(nodes[0] == source and nodes[-1] == target, "Route endpoints mismatch")
    require(len(nodes) == len(edges) + 1 and len(nodes) == len(set(nodes)), "Invalid route shape or cycle")
    length = 0.0
    for index, edge in enumerate(edges):
        u, v = nodes[index:index + 2]
        require(0 <= u < data.node_count and 0 <= v < data.node_count, "Route node out of range")
        require(int(data.row_offsets[u]) <= edge < int(data.row_offsets[u + 1]), "Route edge source mismatch")
        require(int(data.targets[edge]) == v, "Route edge direction mismatch")
        length += float(data.weights_m[edge])
    require(same_distance(length, expected[target]), "Route is not shortest")
    require(same_distance(report["length_m"], length), "Reported route length mismatch")
    if source == target:
        require(nodes == [source] and edges == [] and length == 0, "Incorrect source-equals-target route")


def make_hand_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(crs="epsg:4326")
    for node in reversed(range(5)):
        graph.add_node(node, x=27.0 + node * 0.01, y=63.0 + node * 0.01)
    for u, v, weight in [(0, 1, 10), (0, 2, 1), (2, 1, 1), (1, 3, 1), (2, 3, 10)]:
        graph.add_edge(u, v, length=float(weight))
    return graph


def make_random_graph(case: int) -> nx.MultiDiGraph:
    rng = random.Random(SEED + case)
    n = rng.randint(2, 16)
    ids = [5_000_000_000 + 37 * node for node in range(n)]
    insertion = ids.copy()
    rng.shuffle(insertion)
    graph = nx.MultiDiGraph(crs="epsg:4326")
    for node in insertion:
        graph.add_node(node, x=26.0 + rng.random(), y=62.0 + rng.random())
    for u in ids:
        for v in ids:
            if rng.random() < 0.18:
                for key in range(rng.randint(1, 3)):
                    weight = rng.choice([0.0, 1.0, 5.0, rng.random() * 100])
                    graph.add_edge(u, v, key=key, length=weight)
    return graph


def save_reference(path: Path, distances: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("node_index", "bellman_ford_distance_m"))
        writer.writerows(enumerate(distances))


def validate_graph(graph: nx.MultiDiGraph, executable: Path, directory: Path,
                   sources: list[int], algorithms: tuple[str, ...], all_targets: bool,
                   *, original_graph: nx.MultiDiGraph | None = None) -> int:
    data = graph_to_csr(graph)
    original = graph if original_graph is None else original_graph
    check_original_graph(data, original)
    binary = write_csr(data, directory)
    runs = 0
    for source in sources:
        expected = reference_distances(original, data, source)
        save_reference(directory / f"reference_source_{source}.csv", expected)
        if all_targets:
            targets = range(data.node_count)
        else:
            farthest = max((v for v in range(data.node_count) if math.isfinite(expected[v])),
                           key=lambda v: (expected[v], -v))
            unreachable = next((v for v, distance in enumerate(expected) if math.isinf(distance)), source)
            targets = sorted({source, farthest, unreachable})
        for algorithm in algorithms:
            first_rows = None
            for target in targets:
                report, rows = run_cpp(executable, binary, source, target, algorithm,
                                       directory / f"{algorithm}_source_{source}.csv")
                require(report["algorithm"] == algorithm, "Reported algorithm mismatch")
                check_route(data, report, expected, source, target)
                if first_rows is None:
                    check_sssp(data, rows, expected, source)
                    first_rows = rows
                else:
                    require(rows == first_rows, "SSSP depends on requested target or is nondeterministic")
                runs += 1
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=ROOT / "build/psv_dijkstra")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/validation")
    parser.add_argument("--real-graphml", type=Path, default=ROOT / "data/raw/pohjois_savo.graphml")
    args = parser.parse_args()
    try:
        runs = 0
        graphs = [("hand_example", make_hand_graph())]
        graphs += [(f"random_{case:03d}", make_random_graph(case)) for case in range(12)]
        for name, graph in graphs:
            directory = args.output_dir / name
            directory.mkdir(parents=True, exist_ok=True)
            ox.save_graphml(graph, directory / "input.graphml")
            # Validate the complete GraphML -> Python -> binary -> C++ chain.
            reloaded = ox.load_graphml(directory / "input.graphml")
            sources = sorted({0, len(graph) // 2, len(graph) - 1})
            runs += validate_graph(reloaded, args.executable, directory, sources, ("heap", "phased"), True,
                                   original_graph=graph)
            print(f"PASS {name}: both CPU algorithms vs Bellman-Ford", flush=True)
        real_validated = False
        real_graphml = resolve_graphml(args.real_graphml)
        if real_graphml.exists():
            graph = ox.load_graphml(real_graphml)
            require(len(graph) >= 3, "Real region graph has fewer than three nodes")
            sources = [0, len(graph) // 2, len(graph) - 1]
            runs += validate_graph(graph, args.executable, args.output_dir / "real_osm",
                                   sources, ("heap",), False)
            real_validated = True
            print("PASS real Pohjois-Savo: heap vs Bellman-Ford, three sources", flush=True)
        else:
            print("Real OSM validation skipped: GraphML is not available", flush=True)
        summary = {"seed": SEED, "synthetic_graphs": len(graphs), "cpp_runs": runs,
                   "reference_algorithm": "NetworkX single_source_bellman_ford_path_length",
                   "networkx_version": nx.__version__, "real_osm_validated": real_validated}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"PASS {runs} C++ executions; fixtures and reference distances saved in {args.output_dir}")
    except Exception as error:
        print(f"Validation failed: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            print(error.stderr, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
