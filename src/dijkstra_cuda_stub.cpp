#include "execution.hpp"

namespace psv {

DijkstraResult delta_cuda(CsrView, std::uint32_t, double, CudaTimings&, std::uint32_t) {
    throw BackendUnavailable(cuda_status().reason);
}

BackendStatus cuda_status(std::uint64_t) {
    return {false, "CUDA support was not compiled into this build", ""};
}

DijkstraResult dijkstra_cuda(CsrView graph, std::uint32_t source) {
    CudaTimings timings;
    return dijkstra_cuda_timed(graph, source, timings);
}

DijkstraResult dijkstra_cuda_timed(CsrView, std::uint32_t, CudaTimings&, std::uint32_t) {
    throw BackendUnavailable(cuda_status().reason);
}

}  // namespace psv
