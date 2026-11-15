# Dijkstra algorithm for Pohjois-Savo road network

Shortest-path calculations on the OpenStreetMap road network of Pohjois-Savo,
Finland. C++17 implements CPU and CUDA algorithms; Python prepares the data and
validates results against NetworkX Bellman-Ford. Graphs use compressed sparse
row (CSR) storage. A PySide6 GUI provides an offline road map and route selection.

## Build

Requires **CMake 3.24+** and a **C++17 compiler**. CUDA and OpenMP are optional;
building CUDA requires a compatible NVIDIA toolkit and host compiler, and
running it requires a supported GPU/driver.

```powershell
cmake -S . -B build
cmake --build build --config Release
```

For Visual Studio builds with missing CUDA integration, select the installed
toolkit in a new build directory (`CUDA_PATH` must point to its root):

```powershell
cmake -S . -B build-cuda -T "cuda=$env:CUDA_PATH"
cmake --build build-cuda --config Release
```

If you use `build-cuda`, substitute that directory in the examples below.
See [detailed build options](tests/TESTING.md#build) for CPU-only builds and
CUDA architecture configuration.

## Run

Windows Release examples, from the repository root:

```powershell
.\build\Release\psv_dijkstra.exe --backends
.\build\Release\psv_dijkstra.exe --graph data/processed/graph.csr --source 0 --target 1 --algorithm heap --early-stop
```

Source and target are **internal node indices**, mapped to OSM IDs and
coordinates in `data/processed/graph_nodes.csv`. Results are printed as JSON.
Omitting `--algorithm` selects `heap`; other backends require an explicit flag.
Python is not needed to run the executable with an existing CSR graph.

| `--algorithm` | Backend | Notes |
| --- | --- | --- |
| `heap` | Serial CPU heap Dijkstra | default and fastest tested baseline |
| `phased` | Serial CPU scan-based Dijkstra | reference implementation; slow on large graphs |
| `cpu-parallel` | OpenMP Dijkstra with worker-owned heaps | synchronized minimum selection; slower than heap in regional tests |
| `cuda` | CUDA Dijkstra with cached minima and batched execution | requires CUDA; slower than heap in regional tests |
| `delta-cpu` | CPU delta-stepping | using OpenMP when available |
| `delta-cuda` | CUDA delta-stepping | requires CUDA; faster than classic CUDA for full regional runs |

Useful options:

- `--early-stop`: stop once the requested target is finalized. Omit for full
  single-source calculation. Delta-stepping finalizes the target's whole bucket.
- `--distances output.csv`: save distance and predecessor arrays. Partial results
  identify unfinished nodes as tentative or unknown.
- `--delta 500`: override the delta-stepping bucket width in metres.
- `$env:OMP_NUM_THREADS = "4"`: set the OpenMP worker count in PowerShell.

Default bucket widths when `--delta` is omitted:

| Backend | Full calculation | With `--early-stop` |
| --- | ---: | ---: |
| `delta-cpu` | 1000 m | 300 m |
| `delta-cuda` | 1000000 m | 10000 m |

These are provisional settings tuned for this dataset; GPU/parallel execution
is not guaranteed to beat CPU heap Dijkstra.

## Desktop GUI

Install the Python dependencies and build the C++ executable, then launch from
the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m python.gui
```

Use **Pick start** / **Pick end** to select road nodes, then **Calculate**.
Drag to pan and scroll to zoom. Heap and early stopping are selected initially.
Startup checks available backends; unsupported algorithms are greyed out.
Advanced settings contain the bucket slider, OpenMP threads, and a 120-second
timeout. Inapplicable controls are dimmed and disabled. The bucket slider resets
to the tuned default when the delta backend or search mode changes.
The map uses the existing CSR and node CSV files; no online map service is needed.

The GUI prefers a CUDA-enabled Release executable when present. Use the File
menu or `--executable` / `--graph` / `--nodes` to select other files.
See [GUI controls and checks](tests/TESTING.md#desktop-gui) for details.

## Tests and benchmarks

Pipeline tests require Python 3.11+ and the packages in `requirements.txt`.
After setting up that environment and configuring the build:

```powershell
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

The full regional reference test can take several minutes.
See **[tests/TESTING.md](tests/TESTING.md)** for environment setup, quick test
commands, benchmark flags, timing definitions, implementation details, bucket
sweeps, and recorded performance results.

See the **[regional benchmark results](tests/TESTING.md#regional-benchmark-results-2026-11-15)**
for all six algorithms, comparing early stopping and full-source calculations.

## OpenStreetMap data attribution

Map data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright).
The OpenStreetMap extracts and derived road-network datasets included in this
repository are available under the
[Open Data Commons Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
This data license is separate from the project's source-code license and does
not apply to independently generated synthetic test fixtures.
