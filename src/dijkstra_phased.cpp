#include "dijkstra.hpp"
#include <stdexcept>

namespace psv {
namespace {

std::uint32_t select_minimum(const DijkstraResult& result) {
    double minimum = std::numeric_limits<double>::infinity();
    std::uint32_t selected = invalid_node;
    // Ascending scan and strict comparison give the same tie order as the heap.
    for (std::size_t node = 0; node < result.distances.size(); ++node) {
        if (!result.settled[node] && result.distances[node] < minimum) {
            minimum = result.distances[node];
            selected = static_cast<std::uint32_t>(node);
        }
    }
    return selected;
}

void relax_outgoing_edges(CsrView graph, DijkstraResult& result, std::uint32_t node) {
    for (auto edge = graph.row_offsets[node]; edge < graph.row_offsets[node + 1]; ++edge) {
        detail::relax_edge(graph, result, node, edge);
    }
}

}  // namespace

DijkstraResult dijkstra_phased_cpu(CsrView graph, std::uint32_t source) {
    return dijkstra_phased_target(graph, source, invalid_node);
}

DijkstraResult dijkstra_phased_target(CsrView graph, std::uint32_t source, std::uint32_t target) {
    auto result = detail::initialize_result(graph, source);
    if (target != invalid_node && target >= graph.node_count) throw std::out_of_range("Target index out of range");
    while (true) {
        const auto node = select_minimum(result);
        if (node == invalid_node) {
            break;
        }
        result.settled[node] = 1;
        if (node == target) { result.complete = false; break; }
        relax_outgoing_edges(graph, result, node);
    }
    return result;
}

}  // namespace psv
