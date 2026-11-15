"""Measure delta widths for full SSSP and near/medium/far target-only queries.

Screening validates the requested route. The two best widths per backend/mode
are then repeated and their finalized predecessor forests checked against full
heap results. No runtime defaults are changed. Requires only the standard library.
"""
from __future__ import annotations

import argparse
from array import array
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import struct
import subprocess
import tempfile

if __package__:
    from .benchmark import check_delta
else:
    from benchmark import check_delta


def scores(rows: list[dict], backend: str, mode: str) -> list[dict]:
    groups = {}
    for row in rows:
        if row["backend"] == backend and row["mode"] == mode:
            groups.setdefault((row["case"], row["delta"]), []).append(row["algorithm_ms"])
    medians = {key: statistics.median(values) for key, values in groups.items()}
    cases = {key[0] for key in medians}
    best = {case: min(value for (name, _), value in medians.items() if name == case) for case in cases}
    ranked = []
    for delta in sorted({key[1] for key in medians}):
        values = [medians[(case, delta)] for case in cases if (case, delta) in medians]
        if len(values) != len(cases): continue
        slowdown = math.exp(sum(math.log(max(medians[(case, delta)], 1e-9) / max(best[case], 1e-9))
                                for case in cases) / len(cases))
        ranked.append(dict(delta=delta, normalized_score=slowdown,
                           mean_case_median_ms=statistics.mean(values), cases=len(cases)))
    return sorted(ranked, key=lambda row: (row["normalized_score"], row["delta"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", type=int)
    parser.add_argument("--backends", nargs="+", choices=["delta-cpu","delta-cuda"], default=["delta-cpu","delta-cuda"])
    parser.add_argument("--modes", nargs="+", choices=["full","early"], default=["full","early"])
    parser.add_argument("--deltas", nargs="+", type=float, default=[25,100,300,1000,3000,10000,30000,100000])
    parser.add_argument("--distances", nargs="+", type=float, default=[1000,25000,100000])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if (args.repeats < 2 or args.threads < 1 or not math.isfinite(args.timeout) or args.timeout <= 0
            or any(not math.isfinite(x) or x <= 0 for x in args.deltas + args.distances)):
        parser.error("positive finite widths/distances/timeout, threads >=1, repeats >=2 required")
    executable, graph = args.executable.resolve(), args.graph.resolve()
    if args.output.resolve() in (executable, graph): parser.error("output must not overwrite executable or graph")
    with graph.open("rb") as stream:
        magic,n,m = struct.unpack("<8sQQ", stream.read(24))
    if magic != b"PSVCSR01" or n == 0: parser.error("nonempty CSR graph required")
    sources = list(dict.fromkeys(args.sources or [0,n//2,n-1]))
    if any(source < 0 or source >= n for source in sources): parser.error("source outside graph")
    environment = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE")
    def run(algorithm, source, target, delta=100, early=False, output=None):
        command = [str(executable),"--graph",str(graph),"--source",str(source),"--target",str(target),
                   "--algorithm",algorithm,"--delta",str(delta)]
        if early: command.append("--early-stop")
        if output: command += ["--distances",str(output)]
        process = subprocess.run(command, capture_output=True, text=True, check=True,
                                 env=environment, timeout=args.timeout)
        return json.loads(process.stdout)
    capabilities = json.loads(subprocess.check_output([str(executable),"--backends"],
                                text=True, env=environment, timeout=args.timeout))
    backends = [name for name in dict.fromkeys(args.backends) if capabilities[name]["available"]]
    if not backends: parser.error("None of the requested backends is available")
    summary = dict(graph=str(graph), nodes=n, edges=m, threads=args.threads,
                   executable=str(executable), capabilities=capabilities, cases=[], screening=[], confirmation=[],
                   recommendations={}, seed=918898,
                   deltas=sorted(set(args.deltas)), modes=args.modes, repeats=args.repeats,
                   policy="Equal weight per case: geometric mean of relative case-median slowdown; smaller is better",
                   timing="Cold separate processes; algorithm_ms includes setup/transfers/forest construction, excludes disk I/O")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save(): args.output.write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    rng = random.Random(summary["seed"])
    with tempfile.TemporaryDirectory(prefix="delta-tuning-", dir=args.output.parent) as directory:
        work = Path(directory)
        references = {}
        expected = {}
        for source in sources:
            reference = work/f"heap-{source}.csv"
            run("heap", source, source, output=reference)
            distances = array("d")
            with reference.open(newline="") as stream:
                for row in csv.DictReader(stream): distances.append(float(row["distance_m"]))
            references[source] = reference
            positive = [node for node,distance in enumerate(distances) if math.isfinite(distance) and distance > 0]
            if not positive: raise ValueError(f"source {source} has no nontrivial reachable destinations")
            farthest = max(positive, key=lambda node: distances[node])
            cases = [dict(case=f"full-{source}", mode="full", source=source, target=farthest,
                          route_m=distances[farthest])]
            targets = set()
            for desired in args.distances:
                target = min(positive, key=lambda node: abs(distances[node]-desired))
                if target in targets: continue
                targets.add(target)
                cases.append(dict(case=f"early-{source}-{target}", mode="early", source=source, target=target,
                                  desired_route_m=desired, route_m=distances[target]))
            for case in cases:
                if case["mode"] not in args.modes: continue
                summary["cases"].append(case)
                expected[case["case"]] = distances[case["target"]]
        print("Cases: "+json.dumps(summary["cases"]), flush=True)
        def trial(case, backend, delta, validate=False):
            output = work/"trial.csv" if validate else None
            report = run(backend, case["source"], case["target"], delta, case["mode"]=="early", output)
            if (not report["reachable"] or not math.isclose(report["length_m"], expected[case["case"]],
                                                         rel_tol=1e-10, abs_tol=1e-8)):
                raise ValueError("Target does not match full heap reference")
            if validate:
                check_delta(references[case["source"]], output, graph, case["source"],
                            partial=not report["search_complete"], target=case["target"])
            return dict(case=case["case"], backend=backend, mode=case["mode"], delta=delta,
                        algorithm_ms=report["algorithm_ms"], gpu_compute_ms=report["gpu_compute_ms"],
                        settled_nodes=report["settled_nodes"], search_complete=report["search_complete"],
                        validated_forest=validate)
        jobs = [(case,backend,delta) for case in summary["cases"] for backend in backends for delta in sorted(set(args.deltas))]
        rng.shuffle(jobs)
        for i,(case,backend,delta) in enumerate(jobs):
            summary["screening"].append(trial(case,backend,delta))
            save()
            if (i+1)%8==0: print(f"Screening {i+1}/{len(jobs)}", flush=True)
        finalists = {}
        for backend in backends:
            for mode in dict.fromkeys(args.modes):
                ranked = scores(summary["screening"], backend, mode)
                finalists[(backend,mode)] = [row["delta"] for row in ranked[:2]]
                print(f"Finalists {backend}/{mode}: {finalists[(backend,mode)]}", flush=True)
        jobs = [(case,backend,delta,repeat) for case in summary["cases"] for backend in backends
                for delta in finalists[(backend,case["mode"])] for repeat in range(args.repeats)]
        rng.shuffle(jobs)
        for i,(case,backend,delta,repeat) in enumerate(jobs):
            row=trial(case,backend,delta,validate=repeat==0)
            row["repeat"]=repeat
            summary["confirmation"].append(row)
            save()
            if (i+1)%8==0: print(f"Confirmation {i+1}/{len(jobs)}", flush=True)
        for backend in backends:
            for mode in dict.fromkeys(args.modes):
                ranked=scores(summary["confirmation"],backend,mode)
                summary["recommendations"][backend+"/"+mode]=dict(best=ranked[0], finalists=ranked)
        with graph.open("rb") as stream:
            summary["graph_sha256"]=hashlib.file_digest(stream,"sha256").hexdigest()
        summary["status"]="completed"
        save()
        print(json.dumps(summary["recommendations"],indent=2),flush=True)
    return 0


if __name__=="__main__": raise SystemExit(main())
