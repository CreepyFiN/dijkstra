"""Compare Dijkstra backends in separate processes, checking every SSSP field.

Defaults to a synthetic road-like grid. Use --graph for an existing CSR file.
Each invocation loads/uploads the graph again: these are cold-process timings,
not persistent-GPU or GUI timings. algorithm_ms excludes file I/O; process_ms
also includes startup, loading, route reconstruction and writing distances.
"""

from __future__ import annotations

import argparse
import csv
import mmap
import shutil
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import tempfile
import time


def write_grid(path: Path, nodes: int) -> None:
    width = max(1, math.isqrt(nodes))
    rows, targets, weights = [0], [], []
    for node in range(nodes):
        neighbors = []
        if node % width: neighbors.append(node - 1)
        if node % width + 1 < width and node + 1 < nodes: neighbors.append(node + 1)
        if node >= width: neighbors.append(node - width)
        if node + width < nodes: neighbors.append(node + width)
        for target in sorted(neighbors):
            targets.append(target)
            weights.append(1.0)
        rows.append(len(targets))
    with path.open("wb") as stream:
        stream.write(struct.pack("<8sQQ", b"PSVCSR01", nodes, len(targets)))
        stream.write(struct.pack(f"<{len(rows)}Q", *rows))
        stream.write(struct.pack(f"<{len(targets)}I", *targets))
        stream.write(struct.pack(f"<{len(weights)}d", *weights))


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_delta(reference: Path, actual: Path, graph: Path, source: int, partial: bool = False,
                target: int | None = None) -> None:
    distances, parents, edges, states = array('d'), array('Q'), array('Q'), bytearray()
    with reference.open(newline='') as ref, actual.open(newline='') as output:
        expected = csv.DictReader(ref)
        for node, row in enumerate(csv.DictReader(output)):
            other = next(expected)
            distance, baseline = float(row['distance_m']), float(other['distance_m'])
            if row['settled'] not in ('0','1') or (row['settled']=='1' and not math.isfinite(distance)):
                raise ValueError('Invalid finalized state')
            if (int(row['node_index']) != node or
                ((not partial or row['settled'] == '1') and
                 not (distance == baseline or math.isclose(distance, baseline, rel_tol=1e-10, abs_tol=1e-8))) or
                (not partial and row['settled'] != other['settled'])):
                raise ValueError(f'Delta distance/settled mismatch at {node}')
            distances.append(distance); parents.append(int(row['predecessor_node']))
            edges.append(int(row['predecessor_edge'])); states.append(int(row['settled']))
        if next(expected, None) is not None: raise ValueError('Missing result rows')
    n = len(distances)
    visited = bytearray(n)
    visited[source] = 2
    if distances[source] != 0 or not states[source]: raise ValueError('Invalid source distance/state')
    if target is not None and not states[target]: raise ValueError('Requested target is not finalized')
    with graph.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        _, count, m = struct.unpack_from('<8sQQ', data)
        if count != n: raise ValueError('Incorrect result node count')
        target_offset, weight_offset = 24 + 8*(n+1), 24 + 8*(n+1) + 4*m
        for node in range(n):
            if partial and not states[node]: continue
            if node == source or math.isinf(distances[node]):
                if parents[node] != 2**32-1 or edges[node] != 2**64-1: raise ValueError('Invalid sentinel')
                continue
            parent, edge = parents[node], edges[node]
            if parent >= n or edge >= m or (partial and not states[parent]): raise ValueError('Invalid predecessor')
            begin, end = struct.unpack_from('<QQ', data, 24 + 8*parent)
            target = struct.unpack_from('<I', data, target_offset + 4*edge)[0]
            weight = struct.unpack_from('<d', data, weight_offset + 8*edge)[0]
            if not (begin <= edge < end and target == node and
                    distances[parent] + weight == distances[node]): raise ValueError('Invalid predecessor edge')
        for node in range(n):
            if (partial and not states[node]) or not math.isfinite(distances[node]) or visited[node] == 2: continue
            chain, current = [], node
            while visited[current] != 2:
                if visited[current] == 1: raise ValueError('Predecessor cycle')
                visited[current] = 1; chain.append(current); current = parents[current]
                if current >= n: raise ValueError('Predecessor chain does not reach source')
            for item in chain: visited[item] = 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--nodes", type=int, default=2048)
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--target", type=int)
    parser.add_argument("--early-stop", action="store_true", help="Stop at finalized target; full heap reference is untimed")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--delta", type=float, default=None, help="Override backend/mode-specific bucket default")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--algorithms", nargs="+", default=["heap", "phased", "cpu-parallel", "cuda"],
                        choices=["heap", "phased", "cpu-parallel", "cuda", "delta-cpu", "delta-cuda"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.nodes, args.repeats, args.threads, args.timeout) <= 0 or not math.isfinite(args.timeout) or (args.delta is not None and (not math.isfinite(args.delta) or args.delta <= 0)):
        parser.error("nodes, repeats, threads and timeout must be positive and finite")
    executable = args.executable.resolve()
    environment = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE")
    capabilities = json.loads(subprocess.check_output([str(executable), "--backends"],
                                                     text=True, env=environment, timeout=args.timeout))
    results = []
    failed = False
    reference_digest = None
    with tempfile.TemporaryDirectory(prefix="psv-benchmark-") as directory:
        work = Path(directory)
        graph = args.graph.resolve() if args.graph else work / "grid.csr"
        if not args.graph: write_grid(graph, args.nodes)
        with graph.open("rb") as stream:
            magic, nodes, edges = struct.unpack("<8sQQ", stream.read(24))
        if magic != b"PSVCSR01" or not nodes:
            parser.error("benchmark requires a nonempty PSVCSR01 graph")
        target = nodes - 1 if args.target is None else args.target
        if not (0 <= args.source < nodes and 0 <= target < nodes):
            parser.error("source and target must be internal indices inside the graph")
        baseline_report = None
        if args.early_stop:
            reference = subprocess.run([str(executable), "--graph", str(graph), "--source", str(args.source),
                "--target", str(target), "--algorithm", "heap", "--distances", str(work / "heap-reference.csv")],
                capture_output=True, text=True, check=True, env=environment, timeout=args.timeout)
            baseline_report = json.loads(reference.stdout)
            reference_digest = digest_file(work / "heap-reference.csv")
        # Always establish a serial reference, even when benchmarking only CUDA.
        algorithms = ["heap"] + list(dict.fromkeys(a for a in args.algorithms if a != "heap"))
        for algorithm in algorithms:
            entry = {"requested_algorithm": algorithm, "runs": []}
            results.append(entry)
            if algorithm in capabilities and not capabilities[algorithm]["available"]:
                entry.update(status="unavailable", reason=capabilities[algorithm]["reason"])
                continue
            print(f"Benchmarking {algorithm}: {nodes} nodes, {args.repeats} runs", file=sys.stderr, flush=True)
            for _ in range(args.repeats):
                distances = work / "distances.csv"
                start = time.perf_counter()
                try:
                    process = subprocess.run([str(executable), "--graph", str(graph),
                        "--source", str(args.source), "--target", str(target),
                        "--algorithm", algorithm, "--distances", str(distances)] +
                        (["--delta", str(args.delta)] if args.delta is not None else []) +
                        (["--early-stop"] if args.early_stop else []),
                        capture_output=True, text=True, check=True, env=environment, timeout=args.timeout)
                    process_ms = (time.perf_counter() - start) * 1000
                    report = json.loads(process.stdout)
                    digest = digest_file(distances)
                    if reference_digest is None:
                        reference_digest = digest
                        shutil.copyfile(distances, work / "heap-reference.csv")
                    if args.early_stop:
                        if (report["reachable"] != baseline_report["reachable"] or
                            (report["reachable"] and not math.isclose(report["length_m"], baseline_report["length_m"], rel_tol=1e-10, abs_tol=1e-8)) or
                            (not report["reachable"] and not report["search_complete"])):
                            raise ValueError("Early-stop target does not match full heap reference")
                        check_delta(work / "heap-reference.csv", distances, graph, args.source,
                                    partial=not report["search_complete"], target=target if report["reachable"] else None)
                    elif algorithm.startswith("delta-"):
                        check_delta(work / "heap-reference.csv", distances, graph, args.source)
                    elif digest != reference_digest:
                        raise ValueError("SSSP distances/predecessors/settled flags differ from heap reference")
                    entry["runs"].append({key: report[key] for key in (
                        "algorithm", "delta", "algorithm_ms", "gpu_compute_ms", "gpu_upload_ms", "gpu_download_ms", "early_stop", "search_complete", "settled_nodes")})
                    entry["runs"][-1]["process_ms"] = process_ms
                except subprocess.TimeoutExpired:
                    entry.update(status="timeout", reason=f"Exceeded {args.timeout}s per process")
                    break
                except (subprocess.CalledProcessError, ValueError) as error:
                    entry.update(status="failed", reason=getattr(error, "stderr", None) or str(error))
                    failed = True
                    break
            else:
                entry.update(status="passed", exact_sssp_match=not args.early_stop and not algorithm.startswith("delta-"),
                             validated_shortest_paths=True,
                             median_algorithm_ms=statistics.median(run["algorithm_ms"] for run in entry["runs"]))
            if algorithm == "heap" and entry.get("status") != "passed":
                failed = True
                break
        summary = dict(graph=str(args.graph) if args.graph else "synthetic grid", nodes=nodes, edges=edges,
                       source=args.source, target=target, early_stop=args.early_stop, reference="untimed full heap" if args.early_stop else "heap trial", threads=args.threads, delta=args.delta, capabilities=capabilities,
                       timing="Separate processes; algorithm_ms excludes I/O, process_ms includes I/O",
                       results=results)
    output = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
