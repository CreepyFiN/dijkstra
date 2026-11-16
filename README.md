# Dijkstra algorithm for Pohjois-Savo road network

Shortest paths on the OpenStreetMap road network of Pohjois-Savo, Finland.
C++17 provides CPU/OpenMP and CUDA implementations; Python prepares and validates
CSR graph data and runs a PySide6 GUI with an offline road map.

## Build

Requires **CMake 3.24+** and a **C++17 compiler**. On Windows, install Visual
Studio or Build Tools with **Desktop development with C++** and a Windows SDK.
The GUI and data tools also require **Python 3.11+** (`python` below must refer
to that interpreter). Clone or download the repository, then run these
PowerShell commands from its root:

Python environment for the GUI, data tools, and tests:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Standard C++ build:

```powershell
cmake -S . -B build
cmake --build build --config Release
```

New builds detect CUDA and OpenMP automatically. CUDA requires a compatible
toolkit/host compiler and a supported GPU/driver to run. Use the configuration
flag `-DPSV_ENABLE_CUDA=OFF` to disable CUDA explicitly. See
[detailed build options](tests/TESTING.md#build) for CUDA/Visual Studio integration,
existing build settings, CPU-only builds, and other platforms.
The regional CSR and node mappings are included; no data download is needed.

On macOS, use these commands from the repository root. CUDA is disabled;
OpenMP is detected when available, and the heap and CPU delta backends work
without it:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPSV_ENABLE_CUDA=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
.venv/bin/python -m python.gui
```

## Desktop GUI

After building, launch from the repository root:

```powershell
.\.venv\Scripts\python.exe -m python.gui
```

Pick start/end nodes, add ordered middle points, then Calculate. Adding a middle
point starts map picking; other control clicks cancel it. Drag to pan and scroll
to zoom. Hover highlights a leg; right-click cycles overlapping legs, and the
results dropdown can pin one. Calculate/Cancel stay visible below the sidebar.

Defaults are heap, early stopping, and nodes 0 to 100000 (a valid lower power of
ten for smaller graphs). Startup disables unavailable backends. Advanced settings
contain threads, bucket width, and the 120-second journey timeout; bucket defaults
reset when changing delta backend or full/early mode. The map works offline.

The GUI prefers a CUDA-enabled Release executable when present. Use the File
menu or `--executable`, `--graph`, and `--nodes` to select other files. See
[GUI details](tests/TESTING.md#desktop-gui).

## Run

Windows Release examples, from the repository root:

```powershell
.\build\Release\psv_dijkstra.exe --backends
.\build\Release\psv_dijkstra.exe --graph data/processed/graph.csr --source 0 --target 1 `
  --algorithm heap --early-stop
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

## Tests and benchmarks

Use the same build directory/configuration as your executable:

```powershell
ctest --test-dir build -C Release --output-on-failure
```

The regional Bellman-Ford reference test can take several minutes. See
[tests/TESTING.md](tests/TESTING.md) for quick checks, data preparation, timing
definitions, and tuning. The
[regional results](tests/TESTING.md#regional-benchmark-results)
compare all six algorithms, including the updated early-stop target 40.

## License

The project's source code is available under the [MIT License](LICENSE).
The included OpenStreetMap data is licensed separately as described below.

## OpenStreetMap data attribution

Map data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright).
The OpenStreetMap extracts and derived road-network datasets included in this
repository are available under the
[Open Data Commons Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
This data license is separate from the project's source-code license and does
not apply to independently generated synthetic test fixtures.
