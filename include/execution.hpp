#pragma once

#include "dijkstra.hpp"

#include <stdexcept>
#include <string>

namespace psv {

struct BackendStatus {
    bool available = false;
    std::string reason;
    std::string device;
};

// Reports an unavailable backend or insufficient device memory.
class BackendUnavailable : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct CudaTimings {
    double compute_ms = 0;
    double upload_ms = 0;
    double download_ms = 0;
};

BackendStatus cuda_status(std::uint64_t required_bytes = 0);
BackendStatus openmp_status();
int parallel_cpu_threads();
std::uint64_t cuda_required_bytes(CsrView graph);
DijkstraResult dijkstra_parallel_target(CsrView graph, std::uint32_t source, std::uint32_t target);
DijkstraResult dijkstra_parallel_cpu(CsrView graph, std::uint32_t source);
DijkstraResult dijkstra_cuda(CsrView graph, std::uint32_t source);
DijkstraResult dijkstra_cuda_timed(CsrView graph, std::uint32_t source, CudaTimings& timings, std::uint32_t target = invalid_node);
DijkstraResult delta_cpu(CsrView graph, std::uint32_t source, double delta = 100, std::uint32_t target = invalid_node);
DijkstraResult delta_cuda(CsrView graph, std::uint32_t source, double delta, CudaTimings& timings, std::uint32_t target = invalid_node);
namespace detail {
double delta_bucket(double distance, double delta);
void delta_predecessors(CsrView graph, std::uint32_t source, DijkstraResult& result, double delta = 100,
                        double max_bucket = std::numeric_limits<double>::infinity());
}

struct Execution {
    DijkstraResult result;
    std::string algorithm;
    CudaTimings cuda_timings;
};

// Explicit backend requests never fall back. Heap is the default.
Execution run_dijkstra(CsrView graph, std::uint32_t source, const std::string& algorithm = "heap", double delta = 100, std::uint32_t target = invalid_node);

}  // namespace psv
