#include "execution.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <string>

namespace psv {
namespace {

constexpr unsigned threads = 256;
struct Minimum {
    double distance;
    std::uint32_t node;
};
struct Control {
    std::uint32_t selected;
    int overflow;
};

__global__ void probe_device() {}

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

template <class T> class DeviceBuffer {
public:
    T* data = nullptr;
    explicit DeviceBuffer(std::uint64_t count) {
        if (!count) return;
        const auto error = cudaMalloc(reinterpret_cast<void**>(&data), static_cast<std::size_t>(count) * sizeof(T));
        if (error == cudaErrorMemoryAllocation) {
            throw BackendUnavailable("CUDA could not allocate graph/result buffers: insufficient GPU memory");
        }
        check(error, "cudaMalloc");
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void upload(const T* host, std::uint64_t count) {
        if (count) check(cudaMemcpy(data, host, static_cast<std::size_t>(count) * sizeof(T), cudaMemcpyHostToDevice), "CUDA upload");
    }
    void download(T* host, std::uint64_t count) {
        if (count) check(cudaMemcpy(host, data, static_cast<std::size_t>(count) * sizeof(T), cudaMemcpyDeviceToHost), "CUDA download");
    }
};

class Events {
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
public:
    Events() {
        check(cudaEventCreate(&start_), "cudaEventCreate");
        const auto error = cudaEventCreate(&stop_);
        if (error != cudaSuccess) {
            cudaEventDestroy(start_);
            check(error, "cudaEventCreate");
        }
    }
    ~Events() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start() { check(cudaEventRecord(start_), "cudaEventRecord"); }
    double stop() {
        check(cudaEventRecord(stop_), "cudaEventRecord");
        check(cudaEventSynchronize(stop_), "cudaEventSynchronize");
        float elapsed = 0;
        check(cudaEventElapsedTime(&elapsed, start_, stop_), "cudaEventElapsedTime");
        return elapsed;
    }
};

__device__ Minimum better(Minimum a, Minimum b) {
    return (b.distance < a.distance || (b.distance == a.distance && b.node < a.node)) ? b : a;
}

__device__ Minimum warp_minimum(Minimum local) {
    const unsigned lane = threadIdx.x % 32;
    for (unsigned offset = 16; offset; offset /= 2) {
        const double distance = __shfl_down_sync(0xffffffffU, local.distance, offset);
        const auto node = __shfl_down_sync(0xffffffffU, local.node, offset);
        if (lane + offset < 32) local = better(local, {distance, node});
    }
    return local;
}

__device__ Minimum block_minimum(Minimum local) {
    __shared__ Minimum values[threads / 32];
    const auto best = warp_minimum(local);
    if (threadIdx.x % 32 == 0) values[threadIdx.x / 32] = best;
    __syncthreads();
    if (threadIdx.x < 32) {
        local = threadIdx.x < threads / 32 ? values[threadIdx.x] : Minimum{CUDART_INF, invalid_node};
        local = warp_minimum(local);
        if (threadIdx.x == 0) values[0] = local;
    }
    __syncthreads();
    return values[0];
}

__global__ void select_blocks(const double* distances, const std::uint8_t* settled,
                              std::uint64_t n, Minimum* minima) {
    Minimum local{CUDART_INF, invalid_node};
    for (std::uint64_t node = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         node < n; node += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        if (!settled[node] && distances[node] < CUDART_INF) {
            local = better(local, {distances[node], static_cast<std::uint32_t>(node)});
        }
    }
    const auto best = block_minimum(local);
    if (threadIdx.x == 0) minima[blockIdx.x] = best;
}

// Run bounded batches on one block: block barriers enforce exact Dijkstra
// ordering without a host round-trip per node. Cache a minimum per 256-node
// tile, recomputing only tiles changed by settling or successful relaxation.
__global__ void advance_batch(Minimum* minima, unsigned count, unsigned rounds,
                              std::uint64_t n, const std::uint64_t* rows,
                              const std::uint32_t* targets, const double* weights,
                              double* distances, std::uint32_t* predecessors,
                              std::uint64_t* predecessor_edges, std::uint8_t* settled,
                              unsigned* dirty, unsigned* dirty_list, Control* control, std::uint32_t stop_target) {
    __shared__ unsigned dirty_count;
    for (unsigned round = 0; round < rounds; ++round) {
        Minimum local{CUDART_INF, invalid_node};
        for (unsigned i = threadIdx.x; i < count; i += blockDim.x) local = better(local, minima[i]);
        const auto best = block_minimum(local);
        if (threadIdx.x == 0) {
            dirty_count = 0;
            control->selected = best.node;
            if (best.node != invalid_node) {
                settled[best.node] = 1;
                const auto tile = best.node / threads;
                dirty[tile] = 1;
                dirty_list[dirty_count++] = tile;
            }
        }
        __syncthreads();
        if (best.node == invalid_node || best.node == stop_target) return;
        for (auto edge = rows[best.node]; edge < rows[best.node + 1]; ++edge) {
            const auto target = targets[edge];
            if (target % blockDim.x != threadIdx.x) continue;
            const double candidate = __dadd_rn(best.distance, weights[edge]);
            if (!isfinite(candidate)) atomicExch(&control->overflow, 1);
            else if (candidate < distances[target]) {
                distances[target] = candidate;
                predecessors[target] = best.node;
                predecessor_edges[target] = edge;
                const auto tile = target / threads;
                if (atomicExch(&dirty[tile], 1U) == 0) {
                    dirty_list[atomicAdd(&dirty_count, 1U)] = tile;
                }
            }
        }
        __syncthreads();
        if (control->overflow) return;
        // Each warp owns one changed tile. Every lane participates, including
        // the partial final tile, and duplicate target tiles are listed once.
        const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
        for (unsigned i = warp; i < dirty_count; i += threads / 32) {
            const auto tile = dirty_list[i];
            Minimum candidate{CUDART_INF, invalid_node};
            const auto end = min(n, (static_cast<std::uint64_t>(tile) + 1) * threads);
            for (auto node = static_cast<std::uint64_t>(tile) * threads + lane; node < end; node += 32) {
                if (!settled[node] && distances[node] < CUDART_INF)
                    candidate = better(candidate, {distances[node], static_cast<std::uint32_t>(node)});
            }
            candidate = warp_minimum(candidate);
            if (lane == 0) {
                minima[tile] = candidate;
                dirty[tile] = 0;
            }
        }
        __syncthreads();
    }
}

__device__ double bucket_key(double distance, double delta) {
    return fmin(floor(distance / delta), 1.7976931348623157e308);
}
__global__ void delta_select(const double* distances, const unsigned* pending,
                             std::uint64_t n, Minimum* minima) {
    Minimum local{CUDART_INF, invalid_node};
    for (std::uint64_t node = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         node < n; node += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
        if (pending[node] && distances[node] < CUDART_INF)
            local = better(local, {distances[node], static_cast<std::uint32_t>(node)});
    local = block_minimum(local);
    if (!threadIdx.x) minima[blockIdx.x] = local;
}
__global__ void delta_choose(const Minimum* minima, unsigned count, Control* control) {
    Minimum local{CUDART_INF, invalid_node};
    for (unsigned i = threadIdx.x; i < count; i += blockDim.x) local = better(local, minima[i]);
    local = block_minimum(local);
    if (!threadIdx.x) control->selected = local.node;
}
__global__ void delta_prepare(const double* snapshot, unsigned* pending, unsigned* frontier,
                              unsigned* seen, std::uint64_t n, double bucket, double delta) {
    for (std::uint64_t node = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         node < n; node += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const bool active = pending[node] && bucket_key(snapshot[node], delta) == bucket;
        frontier[node] = active;
        if (active) { pending[node] = 0; seen[node] = 1; }
    }
}
__device__ bool delta_min(double* address, double value) {
    auto* bits = reinterpret_cast<unsigned long long*>(address);
    auto previous = atomicCAS(bits, 0ULL, 0ULL);
    while (value < __longlong_as_double(previous)) {
        const auto assumed = previous;
        previous = atomicCAS(bits, assumed, __double_as_longlong(value));
        if (previous == assumed) return true;
    }
    return false;
}
__global__ void delta_relax(const std::uint64_t* rows, const std::uint32_t* targets,
                            const double* weights, const double* snapshot, double* distances,
                            const unsigned* frontier, unsigned* pending, std::uint64_t n,
                            double bucket, double delta, bool light, Control* control) {
    for (std::uint64_t node = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         node < n; node += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        if (!frontier[node]) continue;
        for (auto edge = rows[node]; edge < rows[node + 1]; ++edge) {
            if ((weights[edge] <= delta) != light) continue;
            const double value = __dadd_rn(snapshot[node], weights[edge]);
            if (!isfinite(value)) { atomicExch(&control->overflow, 1); continue; }
            const auto target = targets[edge];
            if (delta_min(&distances[target], value)) {
                atomicExch(&pending[target], 1U);
                if (light && bucket_key(value, delta) == bucket) atomicExch(&control->selected, 1U);
            }
        }
    }
}

}  // namespace

BackendStatus cuda_status(std::uint64_t required_bytes) {
    int count = 0;
    auto error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess) return {false, std::string("CUDA unavailable: ") + cudaGetErrorString(error), ""};
    if (!count) return {false, "No CUDA GPU detected", ""};
    std::string reason = "No usable CUDA GPU";
    for (int device = 0; device < count; ++device) {
        cudaDeviceProp properties{};
        error = cudaGetDeviceProperties(&properties, device);
        if (error == cudaSuccess) error = cudaSetDevice(device);
        std::size_t free_bytes = 0, total_bytes = 0;
        if (error == cudaSuccess) error = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (error != cudaSuccess) {
            reason = std::string("CUDA initialization failed: ") + cudaGetErrorString(error);
            continue;
        }
        if (free_bytes < required_bytes) {
            reason = "Insufficient free GPU memory for the graph and working arrays";
            continue;
        }
        // Driver/device discovery alone does not prove that this executable's
        // compiled CUDA images can run on the selected GPU.
        probe_device<<<1, 1>>>();
        error = cudaGetLastError();
        if (error == cudaSuccess) error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            reason = std::string("CUDA executable is not usable on this GPU: ") + cudaGetErrorString(error);
            continue;
        }
        return {true, "", properties.name};
    }
    return {false, reason, ""};
}

DijkstraResult dijkstra_cuda(CsrView graph, std::uint32_t source) {
    CudaTimings timings;
    return dijkstra_cuda_timed(graph, source, timings);
}

DijkstraResult dijkstra_cuda_timed(CsrView graph, std::uint32_t source, CudaTimings& timings, std::uint32_t target) {
    auto result = detail::initialize_result(graph, source);
    if (target != invalid_node && target >= graph.node_count) throw std::out_of_range("Target index out of range");
    const auto status = cuda_status(cuda_required_bytes(graph));
    if (!status.available) throw BackendUnavailable(status.reason);
    const auto n = graph.node_count, m = graph.edge_count;
    const auto blocks = static_cast<unsigned>((n + threads - 1) / threads);
    DeviceBuffer<std::uint64_t> rows(n + 1), predecessor_edges(n);
    DeviceBuffer<std::uint32_t> targets(m), predecessors(n);
    DeviceBuffer<double> weights(m), distances(n);
    DeviceBuffer<std::uint8_t> settled(n);
    DeviceBuffer<Minimum> minima(blocks);
    DeviceBuffer<unsigned> dirty(blocks), dirty_list(blocks);
    DeviceBuffer<Control> control(1);
    Control host_control{invalid_node, 0};
    Events timer;
    timer.start();
    rows.upload(graph.row_offsets, n + 1);
    targets.upload(graph.targets, m);
    weights.upload(graph.weights_m, m);
    distances.upload(result.distances.data(), n);
    predecessors.upload(result.predecessor_node.data(), n);
    predecessor_edges.upload(result.predecessor_edge.data(), n);
    settled.upload(result.settled.data(), n);
    control.upload(&host_control, 1);
    timings.upload_ms = timer.stop();
    timer.start();
    check(cudaMemset(dirty.data, 0, blocks * sizeof(unsigned)), "CUDA dirty-tile initialization");
    select_blocks<<<blocks, threads>>>(distances.data, settled.data, n, minima.data);
    check(cudaGetLastError(), "CUDA initial minimum-selection launch");
    constexpr unsigned batch_size = 1024;
    for (std::uint64_t iteration = 0; iteration < n; iteration += batch_size) {
        const auto rounds = static_cast<unsigned>(std::min<std::uint64_t>(batch_size, n - iteration));
        advance_batch<<<1, threads>>>(minima.data, blocks, rounds, n, rows.data, targets.data, weights.data,
            distances.data, predecessors.data, predecessor_edges.data, settled.data,
            dirty.data, dirty_list.data, control.data, target);
        check(cudaGetLastError(), "CUDA batched Dijkstra launch");
        control.download(&host_control, 1);
        if (host_control.overflow) throw std::overflow_error("Finite distance plus edge weight overflows binary64");
        if (host_control.selected == target && target != invalid_node) { result.complete = false; break; }
        if (host_control.selected == invalid_node) break;
    }
    // Includes inter-round synchronization and launch gaps, not just busy-kernel time.
    timings.compute_ms = timer.stop();
    timer.start();
    distances.download(result.distances.data(), n);
    predecessors.download(result.predecessor_node.data(), n);
    predecessor_edges.download(result.predecessor_edge.data(), n);
    settled.download(result.settled.data(), n);
    timings.download_ms = timer.stop();
    return result;
}

DijkstraResult delta_cuda(CsrView graph, std::uint32_t source, double delta, CudaTimings& timings, std::uint32_t target) {
    if (!std::isfinite(delta) || delta <= 0) throw std::invalid_argument("Delta must be finite and positive");
    auto result = detail::initialize_result(graph, source);
    if (target != invalid_node && target >= graph.node_count) throw std::out_of_range("Target index out of range");
    const auto n = graph.node_count, m = graph.edge_count;
    const auto base = cuda_required_bytes(graph);
    if (n > (std::numeric_limits<std::uint64_t>::max() - base) / 24)
        throw std::overflow_error("Delta CUDA allocation estimate overflow");
    const auto status = cuda_status(base + n * 24);
    if (!status.available) throw BackendUnavailable(status.reason);
    const auto blocks = static_cast<unsigned>(std::min<std::uint64_t>((n + threads - 1) / threads, 1024));
    DeviceBuffer<std::uint64_t> rows(n + 1);
    DeviceBuffer<std::uint32_t> targets(m);
    DeviceBuffer<double> weights(m), distances(n), snapshot(n);
    DeviceBuffer<unsigned> pending(n), frontier(n), seen(n);
    DeviceBuffer<Minimum> minima(blocks);
    DeviceBuffer<Control> control(1);
    Control host{0, 0};
    double finalized_bucket = std::numeric_limits<double>::infinity();
    Events timer;
    timer.start();
    rows.upload(graph.row_offsets, n + 1); targets.upload(graph.targets, m); weights.upload(graph.weights_m, m);
    distances.upload(result.distances.data(), n);
    check(cudaMemset(pending.data, 0, static_cast<std::size_t>(n) * sizeof(unsigned)), "Delta pending initialization");
    const unsigned active = 1;
    check(cudaMemcpy(pending.data + source, &active, sizeof(active), cudaMemcpyHostToDevice), "Delta source initialization");
    control.upload(&host, 1);
    timings.upload_ms = timer.stop();
    timer.start();
    while (true) {
        delta_select<<<blocks, threads>>>(distances.data, pending.data, n, minima.data);
        delta_choose<<<1, threads>>>(minima.data, blocks, control.data);
        check(cudaGetLastError(), "Delta bucket selection");
        control.download(&host, 1);
        if (host.selected == invalid_node) break;
        double minimum;
        check(cudaMemcpy(&minimum, distances.data + host.selected, sizeof(double), cudaMemcpyDeviceToHost), "Delta bucket distance");
        const auto bucket = detail::delta_bucket(minimum, delta);
        if (target != invalid_node) {
            double target_distance;
            check(cudaMemcpy(&target_distance, distances.data + target, sizeof(double), cudaMemcpyDeviceToHost), "Delta target distance");
            if (std::isfinite(target_distance) && detail::delta_bucket(target_distance, delta) < bucket) {
                finalized_bucket = detail::delta_bucket(target_distance, delta);
                result.complete = false;
                break;
            }
        }
        check(cudaMemset(seen.data, 0, static_cast<std::size_t>(n) * sizeof(unsigned)), "Delta processed set");
        do {
            check(cudaMemcpy(snapshot.data, distances.data, static_cast<std::size_t>(n) * sizeof(double), cudaMemcpyDeviceToDevice), "Delta snapshot");
            check(cudaMemset(control.data, 0, sizeof(Control)), "Delta control reset");
            delta_prepare<<<blocks, threads>>>(snapshot.data, pending.data, frontier.data, seen.data, n, bucket, delta);
            delta_relax<<<blocks, threads>>>(rows.data, targets.data, weights.data, snapshot.data, distances.data,
                frontier.data, pending.data, n, bucket, delta, true, control.data);
            check(cudaGetLastError(), "Delta light closure");
            control.download(&host, 1);
            if (host.overflow) throw std::overflow_error("Finite distance plus edge weight overflows binary64");
        } while (host.selected);
        check(cudaMemcpy(snapshot.data, distances.data, static_cast<std::size_t>(n) * sizeof(double), cudaMemcpyDeviceToDevice), "Delta heavy snapshot");
        delta_relax<<<blocks, threads>>>(rows.data, targets.data, weights.data, snapshot.data, distances.data,
            seen.data, pending.data, n, bucket, delta, false, control.data);
        check(cudaGetLastError(), "Delta heavy relaxation");
        control.download(&host, 1);
        if (host.overflow) throw std::overflow_error("Finite distance plus edge weight overflows binary64");
    }
    timings.compute_ms = timer.stop();
    timer.start(); distances.download(result.distances.data(), n); timings.download_ms = timer.stop();
    detail::delta_predecessors(graph, source, result, delta, finalized_bucket);
    return result;
}

}  // namespace psv
