#include "execution.hpp"

#include <limits>
#include <cmath>

namespace psv {

std::uint64_t cuda_required_bytes(CsrView graph) {
    const auto csr = csr_binary_size(graph.node_count, graph.edge_count);
    // Device result fields, one minimum/dirty flag/list entry per 256-node
    // tile, and control data. Keep this in sync with the CUDA allocations.
    const auto extra = graph.node_count * 21 + ((graph.node_count + 255) / 256) * 24 + 256;
    if (csr > std::numeric_limits<std::uint64_t>::max() - extra) {
        throw std::overflow_error("CUDA allocation size overflows uint64");
    }
    return csr + extra;
}

Execution run_dijkstra(CsrView graph, std::uint32_t source, const std::string& algorithm, double delta, std::uint32_t target) {
    if (target != invalid_node && target >= graph.node_count) throw std::out_of_range("Target index out of range");
    Execution execution;
    execution.algorithm = algorithm;
    if (!std::isfinite(delta) || delta <= 0) throw std::invalid_argument("Delta must be finite and positive");
    if (algorithm == "heap") execution.result = dijkstra_heap_target(graph, source, target);
    else if (algorithm == "phased") execution.result = dijkstra_phased_target(graph, source, target);
    else if (algorithm == "cpu-parallel") execution.result = dijkstra_parallel_target(graph, source, target);
    else if (algorithm == "cuda") execution.result = dijkstra_cuda_timed(graph, source, execution.cuda_timings, target);
    else if (algorithm == "delta-cpu") execution.result = delta_cpu(graph, source, delta, target);
    else if (algorithm == "delta-cuda") execution.result = delta_cuda(graph, source, delta, execution.cuda_timings, target);
    else throw std::invalid_argument("Unknown Dijkstra algorithm: " + algorithm);
    return execution;
}

}  // namespace psv
