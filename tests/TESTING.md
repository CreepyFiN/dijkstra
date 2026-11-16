# Testing and benchmarking guide

Build, GUI controls, validation, and performance details. See the
[README](../README.md) for the quick start.

**Run commands from the repository root.** PowerShell examples use
`build-cuda/Release`; **users of the standard `build` directory substitute `build/Release` and
`--test-dir build`**. Keep the build directory and configuration consistent.

- [Build and environment](#build)
- [Data preparation](#data-preparation)
- [Desktop GUI](#desktop-gui)
- [Command-line runtime](#run)
- [Tests](#tests)
- [Benchmarks and timing](#benchmark)
- [Early stopping](#target-only--early-stop-benchmarking)
- [Regional results](#regional-benchmark-results)
- [Delta-stepping and tuning](#delta-stepping)
- [Automatic bucket defaults](#automatic-bucket-defaults)

## Build

Requires CMake 3.24+ and a C++17 compiler. On Windows, install Visual Studio or
Build Tools with Desktop development with C++ and a Windows SDK. The GUI/pipeline
also need Python 3.11+; Python is optional for standalone C++ execution.
Install dependencies **before configuring CMake** so it finds the environment
and registers the Python/GUI tests:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Standard C++ build (CUDA and OpenMP are detected when available):

```powershell
cmake -S . -B build
cmake --build build --config Release
```

For a build without CUDA:

```powershell
cmake -S . -B build-cpu-only -DPSV_ENABLE_CUDA=OFF
cmake --build build-cpu-only --config Release
```

OpenMP remains available in that build; add `-DPSV_ENABLE_OPENMP=OFF` to the
configuration command for a fully serial build. Existing build directories
retain cached option values. To re-enable previously disabled CUDA, configure
with `-DPSV_ENABLE_CUDA=ON` or use a new directory.

CUDA requires a compatible NVIDIA toolkit/host compiler and supported GPU/driver.
With Visual Studio, select the toolkit's integration files in a **new build
directory** if installed integration is missing:

```powershell
cmake -S . -B build-cuda -T "cuda=$env:CUDA_PATH" -DPSV_ENABLE_CUDA=ON
cmake --build build-cuda --config Release
```

`CUDA_PATH` must point to the toolkit root. No integration files need copying.
Generator/toolset choices cannot change within an existing build directory.
Ninja/Unix generators omit `-T`; CUDA detection defaults to ON. Architectures
default to 75 and 86 with PTX; override `-DCMAKE_CUDA_ARCHITECTURES=...` for
other toolkit-supported hardware. Runtime detection probes a kernel and checks
initialization/free memory. CUDA runtime linkage is static; CPU-only builds have
no CUDA dependency. `PSV_ENABLE_SANITIZERS=ON` supports GCC/Clang only; use a
separate Debug build for sanitizer checks.

On Linux/macOS, create the environment with `python3 -m venv .venv` and replace
Windows Python paths with `.venv/bin/python`. For single-config Release builds,
use `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`, then
`cmake --build build`. Executables sit directly in `build/`, without `.exe`.
CUDA additionally depends on platform/toolkit support.

## Data preparation

Regional CSR/node/edge mappings and `data/raw/pohjois_savo.graphml.gz` are
included. Existing-data GUI/tests need no download. To regenerate mappings,
use a separate output folder:

```powershell
.\.venv\Scripts\python.exe python/convert_to_csr.py --graphml data/raw/pohjois_savo.graphml.gz `
  --output-dir build-data/processed
```

For a fresh network download, then conversion:

```powershell
.\.venv\Scripts\python.exe python/download_osm.py --data-dir build-data
.\.venv\Scripts\python.exe python/convert_to_csr.py --graphml build-data/raw/pohjois_savo.graphml `
  --output-dir build-data/processed
```

Nominatim verifies the boundary; Overpass supplies roads. Add
`--pbf C:/path/to/extract.osm.pbf` to download_osm.py for local PBF roads
(boundary verification still needs network access). `--user-agent` or
`OSM_USER_AGENT` sets the application identifier. Outputs record provenance and
checksums. Point GUI/benchmark commands at the resulting CSR and matching CSV.
Both GraphML and `.graphml.gz` are supported. Original regional datasets and
some validation artifacts are tracked; the examples use ignored build folders.

## Desktop GUI

```powershell
.\.venv\Scripts\python.exe -m python.gui
# Override the executable and data files:
.\.venv\Scripts\python.exe -m python.gui --executable build-cuda/Release/psv_dijkstra.exe `
  --graph data/processed/graph.csr --nodes data/processed/graph_nodes.csv
```

The default graph is `data/processed/graph.csr`; its node CSV is
`<graph stem>_nodes.csv` beside it. The File menu selects other files.
`PSV_EXECUTABLE` overrides discovery; otherwise the GUI searches
`build-cuda/Release`, `build/Release`, `build-cuda`, `build`, `build-cuda/Debug`,
then `build/Debug`, using the platform's suffix. Startup/reselection probes
backends and disables unavailable choices with reason tooltips. Heap stays the
initial algorithm even with a CUDA executable.

- **Endpoints:** initially 0 and 100000. Smaller graphs use the largest valid
  lower power of ten (75 nodes: 10; 800: 100); one-node graphs use 0.
  Enter internal indices or pick within 18 screen pixels of a node. Rows show
  OSM IDs/coordinates. Snapped nodes may belong to disconnected components.
- **Middle points:** Add numbers a new row and starts picking immediately.
  Remove renumbers later rows; the button is below Pick beside OSM/coordinates.
  Reverse route reverses all stops. Dataset reload clears middles and resets
  endpoints. One sidebar scrollbar covers endpoints/settings; Calculate/Cancel
  remain pinned below it.
- **Navigation:** drag to pan, scroll or use +/- to zoom, and Fit map / Fit route
  to frame the view. Markers are green (start), red (end), numbered blue (middle).
  Other button/dropdown/numeric-field/bucket-slider clicks cancel picking and
  perform their normal action. Pick switches endpoint; sidebar scrolling and
  map-wheel zoom preserve picking. Escape exits.
- **Calculation:** sequential C++ processes route through the displayed stops.
  Each directed leg is validated before joining; repeated stops and return trips
  are valid. Failed/unreachable legs identify the connection, stop remaining
  work, and leave no completed-route overlay. Cancel stops current/remaining
  work; closing waits for worker cleanup. Early stop is initially enabled;
  disabling it runs full SSSP separately from each leg's start.
- **Advanced:** timeout is 120 seconds for the **whole journey**, including
  startups/graph loads. OpenMP threads are dimmed outside cpu-parallel/delta-cpu
  or without OpenMP; CPU delta can still run serially. Bucket controls are dimmed
  outside delta modes. Changing delta backend or full/early mode resets manual
  widths to the [tuned defaults](#automatic-bucket-defaults). Slider steps are
  100 m through 1000 m, 1000 through 10000, 10000 through 100000, 100000 through
  1000000, then 1000000 through 10000000 m. Exact integer entry spans
  100..10000000 m; Reset restores the preset. Classic modes receive no `--delta`.
- **Results:** distance and C++/GPU times sum over legs. Request time additionally
  includes startup, graph I/O, reconstruction and JSON. Summed finalized-node
  visits are not unique nodes. C++ reloads the graph per leg; no persistent graph
  or predecessor cache is used. The GUI writes no distances CSV. Changes clear
  the old route. Calculation progress is indeterminate (CLI reports no
  intermediate progress); status identifies the active leg.
- **Highlighting:** hover within eight screen pixels to thicken a leg and fade
  others to 25%. Arrows follow travel direction. Shared sections initially pick
  the earliest matching leg, retaining it without flicker; **right-click cycles
  overlapping legs**. A fixed-height label shows connection/distance/shared-leg
  numbers without shifting the map. Highlight leg pins a choice; All legs
  restores hover. Right-click updates an existing pin/dropdown together. Hover
  pauses during dragging/picking; pins survive navigation. Route edits/reload
  clear both. Zero-geometry legs use rings.

Background loading validates CSR layout, CSV count/index order, and coordinates.
Use the CSV from the exact conversion: CSR has no OSM IDs to prove provenance.
Local planar projection affects display only; distances use CSR metre weights.
Display-only segment deduplication preserves directed/parallel routing edges;
unsimplified node chains retain curvature. Tiles and a cached overview reduce
redraw cost. The map shows roads/routes, without addresses, street labels,
buildings, or online background tiles.

Initial local UI checks prepared 448192 nodes/896165 edges in 466 tiles in about
7-9 seconds. Full heap/CPU-delta/GPU-delta source 0 to 448191 showed 68.644 km;
early source 0 to 1 showed 0.503 km. These are UI smoke observations, not new
algorithm benchmarks or a repeat of the regional Bellman-Ford test.

## Run

Source/target are internal indices mapped by graph_nodes.csv. Heap is default;
explicit unavailable requests fail without fallback. JSON uses `null` for
unreachable length. Check and run the selected build:

```powershell
.\build-cuda\Release\psv_dijkstra.exe --backends
.\build-cuda\Release\psv_dijkstra.exe --graph data/processed/graph.csr --source 0 --target 40 `
  --algorithm heap --early-stop
```

Other `--algorithm` values are phased, cpu-parallel, cuda, delta-cpu, delta-cuda;
see [README's table](../README.md#run). Omit `--early-stop` for full SSSP, add
`--distances output.csv` for arrays, or `--delta 500` to override bucket width.
Set workers with `$env:OMP_NUM_THREADS = "4"`.

Classic backends settle one global minimum per round. OpenMP uses worker-owned
heaps. CUDA caches minima per 256-node tile, refreshes dirty tiles using warp
reductions, and batches up to 1024 rounds per launch in one block. This avoids
cooperative-grid requirements but limits occupancy. Data stays on-device within
a calculation; each invocation allocates/uploads again. Coordination costs can
outweigh parallelism on sparse road networks.

## Tests

After setup/configuration, build and run the full suite:

```powershell
cmake --build build-cuda --config Release
ctest --test-dir build-cuda -C Release --output-on-failure
```

CTest registers dijkstra_cpu, dijkstra_backends, delta_stepping, early_stop,
csr_reader, python_pipeline, and optional python_gui. GUI registration requires
PySide6 at configuration time; reconfigure after installing it. Add `-V` to
stream output. The regional NetworkX Bellman-Ford case can take minutes and skips
without GraphML. It checks heap; synthetic cases cover all available backends,
ties, parallel/unsorted edges, zero cycles, disconnected nodes, invalid inputs,
overflow, bucket widths, and CUDA batches. CPU-only configurations also check
explicit backend unavailability.

Quick C++/GUI checks and Python checks excluding the long case:

```powershell
ctest --test-dir build-cuda -C Release -L cpp --output-on-failure
$env:PSV_EXECUTABLE = (Resolve-Path build-cuda/Release/psv_dijkstra.exe).Path
$env:PSV_CSR_PROBE = (Resolve-Path build-cuda/Release/test_csr.exe).Path
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -k "not real_osm" -q
.\.venv\Scripts\python.exe -m pytest tests/test_gui.py tests/test_tuning.py -q
```

CTest sets PYTHONPATH and the two executable variables itself. Direct pipeline
runs otherwise search build/, build/Debug/, build/Release/; set both variables
for another build. GUI checks use QT_QPA_PLATFORM=offscreen and need no pytest-qt
plugin. They cover corruption, mapping/picking, bucket resets, availability,
process cancellation/timeouts, ordered/repeated stops, aggregate results,
scrolling, hover/return directions, and pinned-leg synchronization. Use `-L gui`
for GUI-only CTest. test_tuning.py runs separately from CTest.

Standalone Bellman-Ford fixture/reference regeneration:

```powershell
.\.venv\Scripts\python.exe python/validate.py --executable build-cuda/Release/psv_dijkstra.exe `
  --real-graphml data/raw/pohjois_savo.graphml.gz --output-dir build-cuda/validation
```

This runs 13 synthetic graphs (heap/phased) plus three regional heap sources if
GraphML exists. Its default output is the tracked data/processed/validation;
the explicit build-folder output above keeps this run separate. No download is needed.

## Benchmark

Use Release binaries. The standard-library-only script always includes a heap
reference and reports unavailable backends/timeouts separately from verified
results. Specify all six algorithms explicitly: the default list excludes delta.
Use the [regional commands](#regional-benchmark-results) for full and target-40
runs. A small grid check is:

```powershell
.\.venv\Scripts\python.exe python/benchmark.py --executable build-cuda/Release/psv_dijkstra.exe `
  --nodes 4096 --algorithms heap phased cpu-parallel cuda delta-cpu delta-cuda --threads 4 `
  --repeats 3 --timeout 300 --output build-cuda/benchmark-grid.json
```

Defaults are four workers and a 60-second timeout **per subprocess**, not for
the whole benchmark. A timeout is not a speed measurement. Full classic arrays
(distances/predecessors/settled flags) match heap via CSV hashes. Delta permits
equally short paths: distance tolerances are relative 1e-10/absolute 1e-8, with
reachability, edge direction/tightness, sentinel-state, and rooted acyclic
predecessor-forest checks.

- `algorithm_ms`: host time for validation, result initialization, backend setup,
  allocation, transfers, synchronization, and cleanup; excludes disk I/O/route
  reconstruction.
- `gpu_compute_ms`: CUDA-event phase time including snapshots, control transfers,
  synchronization, host-control gaps, and gaps between kernels; not kernel-only.
- `gpu_upload_ms` / `gpu_download_ms`: initial graph/result upload and final
  download event times; excludes allocation.
- `process_ms`: subprocess duration including startup, loading, reconstruction,
  and distance-CSV output.

Each trial starts a fresh process and reuploads the graph. Persistent-device/GUI
workloads require separate measurements. Hardware, graph structure, thread
count, and system load affect results; parallelism need not beat heap.

### Target-only / early-stop benchmarking

Add `--target 40 --early-stop` to benchmark a route. Classic Dijkstra stops
immediately after target settlement, including inside CUDA's batch. Delta
finalizes its whole target bucket, potentially doing more work. Unreachable
targets exhaust all reachable nodes.

`early_stop`, `search_complete`, and `settled_nodes` describe results. False
`search_complete` means partial, not an uncertain target distance. Partial CSVs
add `state=final/tentative/unknown`; unsettled infinity is **unknown**, not known-
unreachable. Reconstructing an unfinalized node raises an error. Full CSV layout
stays compatible, with unreachable distance `inf`.

An **untimed full heap reference** precedes early-stop trials. Target reachability/
distance and every finalized node's distance/forest are checked. Reference and
validation add script wall time but not trial time. Setup and full-size arrays
still cost time on short queries. Source-equals-target is trivial; a far target
may require nearly full SSSP.

### Regional benchmark results

Benchmark output from the CUDA-enabled Windows Release build,
using an NVIDIA GeForce RTX 3070 and four OpenMP workers. Both runs loaded
`data/processed/graph.csr`: **448,192 nodes and 896,165 directed edges**.
Each algorithm ran three times in separate processes, with a **300-second
per-trial timeout** and no explicit bucket override.

The early-stop run used internal nodes **0 to 40**. The full run
computed **full SSSP from node 0**, with node 448191 selected for route output;
that target does not limit the full search. Times below are median
`algorithm_ms`, in milliseconds, including setup and GPU transfers but excluding
disk I/O and route reconstruction. Rows are ordered by full-search time.

| Algorithm | Early stop (ms) | Full SSSP (ms) | Early-stop finalized nodes |
| --- | ---: | ---: | ---: |
| `heap` | 4.3197 | 32.2947 | 24,648 |
| `delta-cpu` | 10.4924 | 107.9911 | 24,652 |
| `delta-cuda` | 171.2569 | 295.0090 | 30,408 |
| `cpu-parallel` | 33.3188 | 520.3173 | 24,648 |
| `cuda` | 298.0523 | 2829.6450 | 24,648 |
| `phased` | 6635.7295 | 109800.1630 | 24,648 |

All full searches completed and finalized **440,079 reachable nodes**; the
remaining 8,113 nodes were unreachable from source 0. All early-stop searches
reported partial results. Default bucket widths were 300 m / 1000 m for
`delta-cpu` and 10000 m / 1000000 m for `delta-cuda` (early stop / full).
Delta-stepping finalizes a whole bucket, explaining its larger early-stop
node counts.

**All six algorithms passed shortest-path validation in both modes.** Full
classic Dijkstra outputs matched heap exactly. Delta-stepping outputs passed
distance and predecessor-forest checks; `exact_sssp_match: false` means exact
CSV equality was not required. Early-stop results were checked against an
untimed full heap reference, including every finalized node.

Heap was fastest in both runs. For full SSSP, delta CPU was the fastest parallel
option, followed by delta CUDA, OpenMP Dijkstra, CUDA Dijkstra, and serial
phased Dijkstra. The early-stop ranking was heap, delta CPU, OpenMP
Dijkstra, delta CUDA, CUDA Dijkstra, then serial phased Dijkstra. These
measurements cover one source and one early-stop target; other endpoints and
hardware can change the ranking.

Reproduce with separate output files to retain both reports:

```powershell
.\.venv\Scripts\python.exe python/benchmark.py --executable build-cuda/Release/psv_dijkstra.exe `
  --graph data/processed/graph.csr --algorithms heap phased cpu-parallel cuda delta-cpu delta-cuda `
  --source 0 --target 40 --early-stop --threads 4 --repeats 3 --timeout 300 `
  --output build-cuda/benchmark-early-comparison.json
.\.venv\Scripts\python.exe python/benchmark.py --executable build-cuda/Release/psv_dijkstra.exe `
  --graph data/processed/graph.csr --algorithms heap phased cpu-parallel cuda delta-cpu delta-cuda `
  --source 0 --target 448191 --threads 4 --repeats 3 --timeout 300 `
  --output build-cuda/benchmark-full-comparison.json
```

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
default selected by backend and search mode). Each bucket repeatedly relaxes
light edges (weight <= delta)
until closure, then relaxes heavy edges for the processed vertices. It can revisit
vertices and is a separate shortest-path algorithm, not classic Dijkstra.

```powershell
.\build-cuda\Release\psv_dijkstra.exe --graph data/processed/graph.csr --source 0 --target 1 `
  --algorithm delta-cpu --delta 10000
.\build-cuda\Release\psv_dijkstra.exe --graph data/processed/graph.csr --source 0 --target 1 `
  --algorithm delta-cuda --delta 100000
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
explicit CUDA delta requests fail if unavailable. Bucket width changes performance
and must be benchmarked; larger is not uniformly better. Tests include shared destinations, zero cycles,
disconnected graphs, overflow, multiple widths, and Bellman-Ford comparisons.

### Reproduce bucket-width tuning

```powershell
.\.venv\Scripts\python.exe python/tune_delta.py --executable build-cuda/Release/psv_dijkstra.exe `
  --graph data/processed/graph.csr --output build-cuda/delta-tuning.json
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
with `.\.venv\Scripts\python.exe -m pytest tests/test_tuning.py -q`.

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
These settings became the CLI/benchmark defaults; see [the policy below](#automatic-bucket-defaults).
Low-level C++ APIs retain their existing width defaults.

Reports are local ignored `build-cuda/delta-tuning.json` and
`build-cuda/delta-tuning-upper.json`. The latter is reproducible with:

```powershell
.\.venv\Scripts\python.exe python/tune_delta.py --executable build-cuda/Release/psv_dijkstra.exe `
  --graph data/processed/graph.csr --backends delta-cuda --modes full --deltas 100000 300000 1000000 `
  --output build-cuda/delta-tuning-upper.json
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
