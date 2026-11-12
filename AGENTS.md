# Project memory

Repository map inspected on 2026-11-12. This file is persistent project context
for future coding sessions. Check the current source and working tree before
making changes; generated files and dataset details can change.

## Purpose and data flow

This is `pohjois_savo_dijkstra`: a C++17 CPU shortest-path project with a
Python 3.11+ OpenStreetMap preparation and validation pipeline for Pohjois-Savo,
Finland (FI-15; recorded OSM administrative relation 918898).

Data flow:

1. `python/download_osm.py` verifies the administrative boundary using Nominatim
   and obtains an OSMnx drive graph from Overpass or an optional local OSM PBF.
2. `python/extract_osm.py` supports the local PBF path through pyosmium. It feeds
   original OSM records into OSMnx's polygon builder, temporarily replacing its
   private record supplier and restoring it in `finally`.
3. `python/convert_to_csr.py` converts GraphML (including `.graphml.gz`) into a
   deterministic CSR binary and separate node/edge CSV mappings.
4. `psv_dijkstra` loads CSR, computes full single-source distances with the
   selected algorithm, reconstructs the requested route, and prints JSON.
5. `python/validate.py` compares C++ results with NetworkX Bellman-Ford on the
   original graph and saves reproducible fixtures and reference distances.

## File structure

```text
dijkstra/
|-- AGENTS.md                      Persistent repository context
|-- CMakeLists.txt                 C++ targets, optional sanitizers, CTest integration
|-- requirements.txt               Pinned Python dependencies
|-- .gitignore                     Build/environment/cache exclusions and data exceptions
|-- .gitattributes                 LF CSV files; binary CSR and gzip files
|-- include/
|   |-- csr.hpp                    Non-owning CsrView, index sentinels, validation API
|   |-- graph.hpp                  Owning CsrGraph, optional node metadata, binary loader
|   `-- dijkstra.hpp               Results, both algorithms, path API, shared helpers
|-- src/
|   |-- csr.cpp                    CSR size, overflow, structure, and weight checks
|   |-- graph.cpp                  Little-endian binary reader and owning/view bridge
|   |-- dijkstra_common.cpp        Initialization, relaxation, overflow, path validation
|   |-- dijkstra_heap.cpp          Priority queue algorithm with stale-entry rejection
|   |-- dijkstra_phased.cpp        Repeated minimum scan, settle, outgoing-edge relaxation
|   `-- main.cpp                   CLI parsing, timing, JSON route and optional distance CSV
|-- python/
|   |-- download_osm.py            Boundary/graph acquisition, caching, checksums, provenance
|   |-- extract_osm.py             Local PBF road filtering and original-node extraction
|   |-- convert_to_csr.py          Graph validation, deterministic ordering, binary/CSV writer
|   `-- validate.py                Independent reference checks and fixture generation
|-- tests/
|   |-- test_support.hpp           CHECK, exception assertions, small CSR graph builder
|   |-- test_dijkstra.cpp          Both algorithms, paths, ties, invalid inputs, overflow
|   |-- test_csr.cpp               Binary layout/corruption tests; JSON reader probe mode
|   `-- test_pipeline.py           Python/C++ integration, downloads, PBF, real graph checks
|-- data/
|   |-- raw/                      .gitkeep and tracked pohjois_savo.graphml.gz
|   |-- metadata/                 .gitkeep, boundary.geojson, nominatim_boundary.json,
|   |                             download.json, source_extract.json; ignored cache/
|   `-- processed/                .gitkeep, graph.csr, graph_nodes.csv, graph_edges.csv
|       `-- validation/           summary.json; hand_example/, random_000/ through
|                                 random_011/, real_osm/ with CSR/CSV/reference artifacts
|-- build/                        Ignored CMake/Visual Studio output and CTest logs
|-- .venv/                        Ignored local Python environment
|-- .pytest_cache/                Ignored pytest cache
`-- .git/                         Git metadata
```

Validation case folders contain graph binaries and mappings, synthetic input
GraphML, algorithm distance CSVs, and Bellman-Ford reference CSVs as applicable.
Some large datasets and validation artifacts are deliberately tracked; inspect
`.gitignore` and `git ls-files` before assuming they are disposable output.

## Contracts to preserve

- Graphs are unsimplified directed `MultiDiGraph` objects. OSMnx options are
  `network_type="drive"`, `simplify=False`, `retain_all=True`, and
  `truncate_by_edge=False`. Preserve parallel edges, one-way direction, loops,
  and disconnected components. Weights are finite, nonnegative lengths in metres.
- Internal node indices follow numerically sorted int64 OSM IDs. Outgoing edges
  are sorted by destination OSM ID, then integer edge key. CSV mappings preserve
  original node IDs, coordinates, and `(osm_u, osm_v, osm_key)` per CSR edge.
- CSR layout is little-endian: `PSVCSR01` (8 bytes), uint64 N, uint64 M,
  uint64 row offsets (N+1), uint32 targets (M), float64 weights (M).
  Total size is `24 + 8*(N+1) + 12*M`. Metadata is separate from the binary.
- C++ namespace is `psv`. `CsrView` borrows arrays; `CsrGraph` owns them.
  UINT32_MAX and UINT64_MAX are invalid node/edge sentinels.
- Both algorithms compute full SSSP, with deterministic minimum-distance then
  minimum-node-index selection. Relax only on strict distance improvement.
  Unreachable distances are infinity; invalid inputs and finite-sum overflow throw.
- The phased implementation is a serial CPU scan with worst-case O(N^2 + M)
  work. Existing real-region validation runs only heap; synthetic cases run both.
- CLI source/target values are internal indices. JSON represents unreachable
  length as `null`; distance CSV uses `inf`. Timing includes initialization and
  input validation, excluding I/O and route reconstruction.
- OSMnx private APIs are used by local extraction, so its pinned version matters.
  Download metadata records genuine source provenance and graph checksums.
- Use `with_locations("sparse_file_array")` without a filename for extraction's
  temporary disk index. Named file indexes retain a native file descriptor and
  block Windows directory cleanup, even after deleting the processor and running
  garbage collection. Keep the missing-location regression test: cleanup must
  not hide the original extraction error.

## Build and validation

CMake defines `psv_core` (CSR and algorithms), `psv_graph` (owning graph/loader,
linking core), and `psv_dijkstra` (CLI). There is no GPU implementation.
Python is optional for the standalone C++ build/runtime. CMake locates the local
Windows or Unix virtual environment and registers integration tests when a
Python 3.11+ interpreter is found; Python packages must also be installed.

Typical PowerShell commands from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cmake -S . -B build
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

CTest names: `dijkstra_cpu`, `csr_reader`, `python_pipeline`. Labels: `cpp`,
`python`, `integration`. CTest sets `PYTHONPATH`, `PSV_EXECUTABLE`, and
`PSV_CSR_PROBE` to the appropriate build outputs. Direct pytest runs need the
correct build outputs too: the fixtures honor those two environment variables,
then otherwise search `build/`, `build/Debug/`, and `build/Release/` with the
platform's executable suffix.

The inspected local build uses Visual Studio 18 2026, with existing Debug
executables under `build/Debug/`. Adjust configuration/path for other setups.
`PSV_ENABLE_SANITIZERS=ON` enables ASan/UBSan with GCC or Clang only.

Example CLI and standalone validation using the local Debug build:

```powershell
.\build\Debug\psv_dijkstra.exe --graph data/processed/graph.csr --source 0 --target 1 --algorithm heap
.\.venv\Scripts\python.exe python/validate.py --executable build/Debug/psv_dijkstra.exe
```

The validation command writes artifacts to `data/processed/validation` by
default; use `--output-dir` for a separate run. The real-OSM pytest case loads
the existing GraphML and skips if absent. It can be substantially slower than
synthetic tests. No download is needed for existing data validation.
Stack diagnostics confirmed the long-running stage is NetworkX Bellman-Ford on
the full regional graph. CTest runs pytest with verbose test names and durations;
use `ctest --test-dir build -C Debug -V` to stream output. The Python CTest has
a 600-second timeout. For a quick Python run, use
`.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -k "not real_osm"`.

Verified on 2026-11-12 after the Windows extraction fix: Debug build succeeded;
all three CTest tests passed, including all 57 Python cases. The full suite took
196 seconds, with 183 seconds in the real-OSM case. The 56 other Python cases
took about 11 seconds. These are observed local timings, not performance limits.

At inspection, recorded download metadata describes 448,192 nodes, 896,165
directed edges, and 333 weakly connected components. The stored validation
summary records 13 synthetic graphs, 661 C++ executions, and real-OSM validation
against NetworkX Bellman-Ford with seed 918898. These are saved artifact records,
not the results of the later Windows regression test run recorded above.
