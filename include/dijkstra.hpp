#pragma once

#include "csr.hpp"

#include <vector>

namespace psv {

struct DijkstraResult {
    bool complete = true; // false: unsettled nodes are unknown/tentative, not unreachable
    std::vector<double> distances;
    std::vector<std::uint32_t> predecessor_node;
    std::vector<std::uint64_t> predecessor_edge;
    std::vector<std::uint8_t> settled;
};

DijkstraResult dijkstra_heap_cpu(CsrView graph, std::uint32_t source);
DijkstraResult dijkstra_phased_cpu(CsrView graph, std::uint32_t source);

DijkstraResult dijkstra_heap_target(CsrView graph, std::uint32_t source, std::uint32_t target);
DijkstraResult dijkstra_phased_target(CsrView graph, std::uint32_t source, std::uint32_t target);

enum class PathStatus { reachable, unreachable };

struct Path {
    PathStatus status = PathStatus::unreachable;
    std::vector<std::uint32_t> nodes;
    std::vector<std::uint64_t> edges;
    double length_m = std::numeric_limits<double>::infinity();
};

// Invalid inputs or corrupt predecessor chains throw; unreachable is a normal state.
Path reconstruct_path(CsrView graph, const DijkstraResult& result,
                      std::uint32_t source, std::uint32_t target);

namespace detail {
DijkstraResult initialize_result(CsrView graph, std::uint32_t source);
double checked_distance_sum(double distance, double weight);
// On an accepted relaxation, all three result fields change together.
bool relax_edge(CsrView graph, DijkstraResult& result, std::uint32_t from,
                std::uint64_t edge);
}  // namespace detail

}  // namespace psv
