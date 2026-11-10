"""Write deterministic directed CSR and separate OSM node/edge CSV mappings."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from numbers import Integral
from pathlib import Path
import struct
import sys

import networkx as nx
import numpy as np
import osmnx as ox

ROOT = Path(__file__).resolve().parents[1]
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
MAGIC = b"PSVCSR01"


def binary_size(n: int, m: int) -> int:
    if not isinstance(n, Integral) or not isinstance(m, Integral) or n < 0 or m < 0:
        raise ValueError("CSR counts must be nonnegative integers")
    if n > UINT32_MAX:
        raise OverflowError("Node count exceeds uint32 indices; UINT32_MAX is reserved")
    if m > UINT64_MAX:
        raise OverflowError("Edge count exceeds uint64 indices")
    size = 24 + 8 * (int(n) + 1) + 12 * int(m)
    if size > UINT64_MAX or size > np.iinfo(np.intp).max:
        raise OverflowError("CSR byte size exceeds uint64 or addressable memory")
    return size


@dataclass
class CsrData:
    row_offsets: np.ndarray
    targets: np.ndarray
    weights_m: np.ndarray
    osm_node_ids: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    # Each tuple corresponds to the CSR edge at the same index.
    osm_edges: list[tuple[int, int, int]]

    @property
    def node_count(self) -> int:
        return len(self.osm_node_ids)

    @property
    def edge_count(self) -> int:
        return len(self.targets)

    def validate(self) -> None:
        n, m = self.node_count, self.edge_count
        binary_size(n, m)
        for name, dtype, count in (
            ("row_offsets", "<u8", n + 1), ("targets", "<u4", m),
            ("weights_m", "<f8", m), ("osm_node_ids", "<i8", n),
            ("latitude", "<f8", n), ("longitude", "<f8", n),
        ):
            array = getattr(self, name)
            if array.dtype != np.dtype(dtype) or array.shape != (count,) or not array.flags.c_contiguous:
                raise ValueError(f"Invalid type, length or layout of {name}")
        if (self.row_offsets[0] != 0 or self.row_offsets[-1] != m
                or np.any(self.row_offsets[1:] < self.row_offsets[:-1])
                or np.any(self.row_offsets > m)):
            raise ValueError("Invalid row offsets")
        if np.any(self.targets >= n) or np.any(self.targets == UINT32_MAX):
            raise ValueError("Invalid target index")
        if np.any(~np.isfinite(self.weights_m)) or np.any(self.weights_m < 0):
            raise ValueError("Weights must be finite and nonnegative")
        if len(self.osm_edges) != m:
            raise ValueError("Edge mapping length does not equal M")
        if np.any(self.osm_node_ids[1:] <= self.osm_node_ids[:-1]):
            raise ValueError("OSM IDs must be unique and sorted numerically")
        if (np.any(~np.isfinite(self.latitude)) or np.any(np.abs(self.latitude) > 90)
                or np.any(~np.isfinite(self.longitude)) or np.any(np.abs(self.longitude) > 180)):
            raise ValueError("Invalid node coordinates")
        for u in range(n):
            previous = None
            for edge in range(int(self.row_offsets[u]), int(self.row_offsets[u + 1])):
                original = self.osm_edges[edge]
                if (original[0] != self.osm_node_ids[u]
                        or original[1] != self.osm_node_ids[self.targets[edge]]
                        or not isinstance(original[2], Integral)
                        or (previous is not None and original <= previous)):
                    raise ValueError("Invalid or unordered OSM edge mapping")
                previous = original


def graph_to_csr(graph: nx.MultiDiGraph) -> CsrData:
    if not graph.is_directed() or not graph.is_multigraph():
        raise ValueError("Expected a directed MultiDiGraph, preserving every parallel edge")
    if graph.graph.get("simplified", False):
        raise ValueError("A simplified OSM graph is not accepted")
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    binary_size(n, m)
    for node in graph:
        if not isinstance(node, Integral) or isinstance(node, bool) or not -(1 << 63) <= node < (1 << 63):
            raise ValueError(f"OSM node ID is not an int64: {node}")
    ordered_nodes = sorted(graph.nodes)
    node_index = {node: index for index, node in enumerate(ordered_nodes)}
    row_offsets = np.empty(n + 1, dtype="<u8")
    targets = np.empty(m, dtype="<u4")
    weights = np.empty(m, dtype="<f8")
    latitudes = np.empty(n, dtype="<f8")
    longitudes = np.empty(n, dtype="<f8")
    osm_edges = []
    edge_index = 0
    for u, node in enumerate(ordered_nodes):
        attributes = graph.nodes[node]
        longitudes[u], latitudes[u] = float(attributes["x"]), float(attributes["y"])
        row_offsets[u] = edge_index
        edges = list(graph.out_edges(node, keys=True, data=True))
        if any(not isinstance(key, Integral) or isinstance(key, bool) for _, _, key, _ in edges):
            raise ValueError("OSMnx edge keys must be integers")
        edges.sort(key=lambda edge: (int(edge[1]), int(edge[2])))
        for _, target, key, attributes in edges:
            weight = float(attributes["length"])
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"Invalid length at OSM edge {(node, target, key)}")
            targets[edge_index] = node_index[target]
            weights[edge_index] = weight
            osm_edges.append((int(node), int(target), int(key)))
            edge_index += 1
    row_offsets[n] = edge_index
    result = CsrData(row_offsets, targets, weights, np.asarray(ordered_nodes, dtype="<i8"),
                     latitudes, longitudes, osm_edges)
    result.validate()
    return result


def write_csr(data: CsrData, directory: Path, stem: str = "graph") -> Path:
    data.validate()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.csr"
    temporary = path.with_suffix(".csr.tmp")
    with temporary.open("wb") as stream:
        stream.write(struct.pack("<8sQQ", MAGIC, data.node_count, data.edge_count))
        data.row_offsets.tofile(stream)
        data.targets.tofile(stream)
        data.weights_m.tofile(stream)
    if temporary.stat().st_size != binary_size(data.node_count, data.edge_count):
        raise IOError("Written CSR file has an unexpected length")
    temporary.replace(path)
    with (directory / f"{stem}_nodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("node_index", "osm_node_id", "latitude", "longitude"))
        for i in range(data.node_count):
            writer.writerow((i, int(data.osm_node_ids[i]), repr(float(data.latitude[i])), repr(float(data.longitude[i]))))
    with (directory / f"{stem}_edges.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("csr_edge_index", "osm_u", "osm_v", "osm_key"))
        writer.writerows((i, *edge) for i, edge in enumerate(data.osm_edges))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphml", type=Path, default=ROOT / "data/raw/pohjois_savo.graphml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()
    try:
        graph = ox.load_graphml(args.graphml)
        data = graph_to_csr(graph)
        path = write_csr(data, args.output_dir)
        print(f"Saved {data.node_count} nodes and {data.edge_count} directed edges to {path}")
    except Exception as error:
        print(f"CSR conversion failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
