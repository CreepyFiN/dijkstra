# Testing and benchmarking guide

Detailed build options, correctness checks, performance measurements, and bucket
width tuning for Pohjois-Savo Dijkstra. See the [project README](../README.md)
for the overview, available algorithms, and quick start.

**Run all commands from the repository root**, not from the `tests` directory.
Examples use PowerShell and Windows executable paths. Replace `build-cuda` with
your configured build directory when appropriate.

- [Build options](#build)
- [Runtime behavior](#run)
- [Tests](#tests)
- [Benchmarks and timing definitions](#benchmark)
- [Early stopping and partial results](#target-only--early-stop-benchmarking)
- [Delta-stepping and bucket tuning](#delta-stepping)
- [Automatic bucket defaults](#automatic-bucket-defaults)

For a fresh Python test environment (Python 3.11+):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Build

Requires CMake 3.24+ and a C++17 compiler. CUDA and OpenMP are optional and
detected at configuration time. Building CUDA requires a compatible NVIDIA
CUDA toolkit and host compiler; executing it requires a supported GPU/driver.

```powershell
cmake -S . -B build
cmake --build build --config Release
```

For a serial-only build:

```powershell
cmake -S . -B build-cpu-only -DPSV_ENABLE_CUDA=OFF -DPSV_ENABLE_OPENMP=OFF
cmake --build build-cpu-only --config Release
```

On Windows, if CMake reports missing CUDA integration in Visual Studio but the
toolkit is installed, use the toolkit's integration files explicitly in a
**new build directory**:

```powershell
cmake -S . -B build-cuda -T "cuda=$env:CUDA_PATH"
cmake --build build-cuda --config Release
```

This selects the installed toolkit without copying files into Visual Studio.
`CUDA_PATH` must point to the toolkit root. This command is for the Visual
Studio generator; Ninja/Unix generators do not use `-T`. CMake generator/toolset
settings cannot be changed in an already configured build directory.

CUDA builds default to architectures 75 and 86, including PTX. Override
`CMAKE_CUDA_ARCHITECTURES` when preparing a build for other supported hardware.
Runtime detection launches a tiny probe kernel, checks initialization and free
memory, and selects a usable GPU. CUDA runtime linkage is static; a CPU-only
build has no CUDA dependency at all.

## Run

Examples below use the Windows CUDA-enabled Release build:

```powershell
.\build-cuda\Release\psv_dijkstra.exe --backends
.\build-cuda\Release\psv_dijkstra.exe --graph data/processed/graph.csr --source 0 --target 1 --algorithm cuda
.\build-cuda\Release\psv_dijkstra.exe --graph data/processed/graph.csr --source 0 --target 1 --algorithm cpu-parallel
.\build-cuda\Release\psv_dijkstra.exe --graph data/processed/graph.csr --source 0 --target 1 --algorithm auto
```

Source and target are **internal node indices**, mapped to OSM IDs and
coordinates in `data/processed/graph_nodes.csv`. Add `--distances output.csv`
to save all distances, predecessors, and settled flags.

Set OpenMP worker count through its standard environment variable:

```powershell
$env:OMP_NUM_THREADS = "4"
```

JSON reports both `requested_algorithm` and the actual `algorithm`, plus a
`fallback_reason`. Explicit `cuda` or `cpu-parallel` requests fail clearly if
unavailable. Auto falls back on availability or GPU-memory allocation failures;
it does not conceal arithmetic overflow or calculation/kernel errors.

**Auto chooses by availability, not measured speed.** Classic Dijkstra backends settle
one global minimum node at a time. OpenMP keeps per-worker priority queues,
avoiding full-node scans. CUDA caches minima for 256-node tiles and recomputes
only changed tiles. It runs up to 1024 ordered rounds per kernel launch before
copying a control result to the host, and uses warp shuffles for reductions.
The batched kernel uses one block, so block barriers provide ordering without
requiring cooperative-grid support. It trades whole-GPU occupancy for lower
synchronization and transfer overhead. Graphs and results stay on the GPU
between batches; they are still allocated/uploaded for each new calculation.
Remaining candidate searches and coordination can make parallel execution
slower than CPU heap Dijkstra, especially on sparse road networks.

## Tests

Python 3.11+ and `requirements.txt` are required for pipeline tests, not for
running the C++ executable against an existing CSR graph.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cmake --build build-cuda --config Debug
ctest --test-dir build-cuda -C Debug --output-on-failure -V
```

CTest includes the serial algorithms, parallel backend tests, CSR reader, and
Python integration. Backend tests also check explicit unavailability and Auto
fallback. They exercise parallel edges, unsorted rows, deterministic ties,
zero-weight cycles, disconnected nodes, multiple CUDA blocks, and overflow.
CPU-only configurations exercise fallback without requiring a GPU.

The full-region Python reference test can take several minutes. For a quicker
run against a particular executable:

```powershell
$env:PSV_EXECUTABLE = (Resolve-Path build-cuda/Debug/psv_dijkstra.exe).Path
$env:PSV_CSR_PROBE = (Resolve-Path build-cuda/Debug/test_csr.exe).Path
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -k "not real_osm" -q
```

The existing real-OSM pytest case checks heap against NetworkX Bellman-Ford.
Parallel implementations are checked against Bellman-Ford on synthetic inputs;
the benchmark below can compare their complete regional results against heap.

## Benchmark

Use **Release** binaries. The benchmark uses only Python's standard library,
compares Dijkstra distance/predecessor/settled fields with heap via CSV hashes,
and checks delta distances (relative tolerance 1e-10, absolute 1e-8), reachability,
predecessor-edge weights/direction, and an acyclic rooted predecessor forest,
and imposes a timeout per subprocess. It always includes a heap reference.

```powershell
.\.venv\Scripts\python.exe python/benchmark.py --executable build-cuda/Release/psv_dijkstra.exe --nodes 4096 --repeats 3 --output build-cuda/benchmark-grid.json
.\.venv\Scripts\python.exe python/benchmark.py --executable build-cuda/Release/psv_dijkstra.exe --graph data/processed/graph.csr --repeats 1 --timeout 60 --output build-cuda/benchmark-region.json
```

`--threads` defaults to 4. Use `--algorithms heap cpu-parallel cuda` to omit
the slow serial scan. A timeout is reported as `timeout`, not as a verified
result or a speed measurement.

- `algorithm_ms`: host elapsed time including validation, result initialization,
  backend selection/initialization, allocations, transfers, GPU synchronization,
  and resource cleanup. Excludes disk I/O and route reconstruction.
- `gpu_compute_ms`: CUDA-event time covering the Dijkstra phase loop, including
  control transfers, synchronization and gaps between kernel launches. It is
  not a sum of kernel execution times alone.
- `gpu_upload_ms` / `gpu_download_ms`: event timings for initial graph/result
  upload and final result download. They do not include allocation time.
- Benchmark `process_ms`: full subprocess duration including startup, graph
  loading, route reconstruction, and distance-CSV output.

Each benchmark trial uses a separate process and uploads the graph again.
Persistent-device/GUI workloads need separate measurements. Distinct hardware,
CPU thread counts, system load, and graph structure affect results.

### Target-only / early-stop benchmarking

Use the same flag in the executable or benchmark script:

```powershell
.\build-cuda\Release\psv_dijkstra.exe --graph data/processed/graph.csr --source 0 --target 1 --algorithm heap --early-stop
.\.venv\Scripts\python.exe python/benchmark.py --executable build-cuda/Release/psv_dijkstra.exe --graph data/processed/graph.csr --source 0 --target 1 --algorithms heap phased cpu-parallel cuda delta-cpu delta-cuda --delta 100 --early-stop --repeats 3 --output build-cuda/benchmark-early-region.json
```

Omit `--early-stop` to benchmark full SSSP with the same endpoints. Dijkstra
stops when the target is settled, before relaxing its outgoing edges. CUDA
stops inside the device batch, not after an extra 1024 rounds. Delta-stepping
waits until no bucket at or below the target's bucket remains pending; a large
delta can therefore process much more of the graph. If the target is unreachable,
all reachable nodes are processed and the search completes normally.

JSON includes `early_stop`, `search_complete`, and `settled_nodes`. A false
`search_complete` conservatively identifies a partial result; it does not imply
that the requested route is uncertain. For partial results, `--distances` adds
a `state` column: `final`, `tentative`, or `unknown`. Unsettled infinity is
**unknown**, not proof of unreachability. Reconstructing an unfinalized node
from a partial result raises an error. Full-source CSV output stays compatible.

The benchmark computes an **untimed full-source heap reference** first in
early-stop mode. Each timed trial must match its target reachability/distance;
every finalized node's distance and predecessor forest are also checked. The
reference cost is excluded from trial times, but still contributes to script
wall time. Reported timings include full CSR validation, full-size working-array
allocation, and GPU setup/transfers, so early stopping mainly reduces search
work; it does not remove those costs. Results are endpoint-dependent: selecting
the source itself is trivial and a farthest target can require nearly full SSSP.

### Local optimization results

On the RTX 3070/MSVC Release setup, regional source 0 produced these indicative
calculation times (disk I/O excluded):

| Backend | Initial implementation, one run | Optimized, median of three runs |
| --- | ---: | ---: |
| CPU heap | 47 ms | 34 ms |
| OpenMP, 4 workers | 46.9 s | 0.53 s |
| CUDA | 13.4 s | 2.71 s |

The initial and optimized runs were not a controlled simultaneous comparison;
system load varied. Every optimized trial matched the complete heap distance,
predecessor, and settled-state CSV exactly. Heap is still the fastest backend
for this single-source workload. The optimized report is saved locally as
`build-cuda/benchmark-region-optimized-clean.json`.

## Delta-stepping

Delta-stepping groups distance labels into buckets of width `--delta` (metres;
default selected by backend and search mode). Within a bucket it repeatedly relaxes light edges (weight <= delta)
until closure, then relaxes heavy edges for the processed vertices. It can revisit
vertices and is a separate shortest-path algorithm, not classic Dijkstra.

```powershell
.\build-cuda\Release\psv_dijkstra.exe --graph data/processed/graph.csr --source 0 --target 1 --algorithm delta-cpu --delta 10000
.\build-cuda\Release\psv_dijkstra.exe --graph data/processed/graph.csr --source 0 --target 1 --algorithm delta-cuda --delta 100000
.\.venv\Scripts\python.exe python/benchmark.py --executable build-cuda/Release/psv_dijkstra.exe --graph data/processed/graph.csr --algorithms heap cpu-parallel cuda delta-cpu delta-cuda --delta 10000 --repeats 3 --output build-cuda/delta-comparison.json
```

CPU expansion uses OpenMP on frontiers of at least 128 vertices, with worker-local
requests and deterministic serial merging; it also works serially without OpenMP.
The CUDA baseline uses full-node frontier scans, device snapshots, and atomic CAS
minimum updates. Separate prepare/relax kernels avoid mixed read/write distance
snapshots. It still synchronizes with the host for bucket and closure decisions;
it is not an optimized compact-frontier implementation.

Both build deterministic predecessor trees on the CPU after distances converge,
traversing tight edges from the source to prevent zero-weight predecessor cycles.
Different equally short paths from Dijkstra are valid. This tree-building cost
is included in `algorithm_ms`, but not GPU event time. GPU event time includes
device snapshots, scans, host-control gaps and synchronization, not just kernels.

Delta must be finite and positive. Extreme bucket indices saturate rather than
overflowing an integer conversion. Negative/nonfinite weights and distance-sum
overflow retain the existing error behavior. `--backends` reports delta support;
explicit CUDA delta requests fail if unavailable. Auto retains its existing
Dijkstra-only policy. Bucket width changes performance and must be benchmarked;
larger is not uniformly better. Tests include shared destinations, zero cycles,
disconnected graphs, overflow, multiple widths, and Bellman-Ford comparisons.

### Reproduce bucket-width tuning

```powershell
.\.venv\Scripts\python.exe python/tune_delta.py --executable build-cuda/Release/psv_dijkstra.exe --graph data/processed/graph.csr --output build-cuda/delta-tuning.json
```

This screens eight widths (25 through 100000 metres) at three starting nodes,
with full SSSP and early-stop routes near 1, 25, and 100 km. It randomizes trial
order, repeats the top two widths three times per case, and validates each
finalist's finalized predecessor forest against full heap results. Screening
checks target distance; confirmation adds the full finalized-result checks.
Each case has equal weight: ranking uses the geometric mean of the ratios to
that case's best median, so long routes cannot drown out short routes.

`--sources`, `--distances`, `--deltas`, `--threads`, `--repeats`, and `--timeout`
allow custom experiments. Measurements use separate cold processes. The JSON
is checkpointed during the run; only `status="completed"` means all confirmation
checks passed. Recommendations describe the best **tested** widths, not a
universal optimum, and do not alter CLI defaults. Ranking unit tests can be run
with `python -m pytest tests/test_tuning.py -q`.

The expanded local sweep (363 timed trials, including an upper-range CUDA probe)
used sources 0, 224096 and 448191, with nine nontrivial early-stop routes. All 54
finalist predecessor-forest checks passed. Suggested explicit settings for this
dataset, RTX 3070, and four CPU workers are:

| Backend | Full-source `--delta` | Early-stop `--delta` |
| --- | ---: | ---: |
| `delta-cpu` | 1000 | 300 |
| `delta-cuda` | 1000000 | 10000 |

CPU full widths 1000 and 3000 were effectively tied (about 112 ms mean of case
medians). CPU early widths 300 and 1000 were also close; the recommendation
favors the narrower bucket. CUDA full width 1000000 averaged about 288 ms across
the three sources, versus 305 ms at 300000 and 316 ms at 100000 in separate
sweeps. Most reachable distances fit in one bucket at the largest width.

For early CUDA, 10000 scored better under equal-case relative-time weighting;
30000 had a slightly lower raw mean (184 versus 186 ms) and can suit a workload
dominated by longer routes. Short-query GPU times include substantial cold
initialization costs, making small differences noisy. These are best-tested
starting settings, not universal optima or statistically established winners.
Unreachable targets and all other graph components were not sampled separately.
**These are now the CLI and benchmark defaults when --delta is omitted.**
An explicit --delta overrides them. The low-level C++ APIs retain their existing
explicit/default width behavior; the CLI applies the backend/mode policy.

Reports are local ignored `build-cuda/delta-tuning.json` and
`build-cuda/delta-tuning-upper.json`. The latter is reproducible with:

```powershell
.\.venv\Scripts\python.exe python/tune_delta.py --executable build-cuda/Release/psv_dijkstra.exe --graph data/processed/graph.csr --backends delta-cuda --modes full --deltas 100000 300000 1000000 --output build-cuda/delta-tuning-upper.json
```

`--backends` and `--modes` allow focused follow-up sweeps. Heap Dijkstra is still
faster for these single-source queries; bucket tuning does not change that result.

Local full-region, source-0 Release medians over three separate processes:

| Backend | Delta width (metres) | Calculation time |
| --- | ---: | ---: |
| Heap Dijkstra | n/a | 35 ms |
| OpenMP Dijkstra, 4 threads | n/a | 540 ms |
| CUDA Dijkstra | n/a | 2927 ms |
| CPU delta, 4 threads | 10000 | 111 ms |
| CUDA delta | 100000 | 418 ms |

The CPU-delta value comes from a separate width-tuning run; in the common
100000-width comparison it took 350 ms. Delta speeds up the parallel baselines
in these experiments, but heap remains fastest. Full results are in ignored
`build-cuda/benchmark-delta-comparison.json` and `benchmark-delta10000.json`.
These measurements include initialization, transfers and predecessor-tree
construction but exclude disk I/O. They are not GUI/persistent-device timings.

### Automatic bucket defaults

| Backend | Full-source default | With `--early-stop` |
| --- | ---: | ---: |
| `delta-cpu` | 1000 m | 300 m |
| `delta-cuda` | 1000000 m | 10000 m |

Both the executable and benchmark use this table when `--delta` is absent.
Pass `--delta 500` (for example) to override the width in either mode. Benchmark
trial records include the resolved width; a null top-level delta means no global
width override was requested. These are provisional dataset/hardware-tuned defaults.
