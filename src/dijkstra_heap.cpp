#include "dijkstra.hpp"
#include <stdexcept>

#include <functional>
#include <queue>
#include <utility>

namespace psv {

DijkstraResult dijkstra_heap_cpu(CsrView graph, std::uint32_t source) {
    return dijkstra_heap_target(graph, source, invalid_node);
}

DijkstraResult dijkstra_heap_target(CsrView graph, std::uint32_t source, std::uint32_t stop_target) {
    auto result = detail::initialize_result(graph, source);
    if (stop_target != invalid_node && stop_target >= graph.node_count) throw std::out_of_range("Target index out of range");
    // Lexicographic order gives minimum distance, then minimum internal node ID.
    using Entry = std::pair<double, std::uint32_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
    queue.emplace(0.0, source);
    while (!queue.empty()) {
        const auto [distance, node] = queue.top();
        queue.pop();
        if (result.settled[node] || distance != result.distances[node]) {
            continue;  // Discard stale entries after a better relaxation.
        }
        result.settled[node] = 1;
        if (node == stop_target) { result.complete = false; break; }
        for (auto edge = graph.row_offsets[node]; edge < graph.row_offsets[node + 1]; ++edge) {
            if (detail::relax_edge(graph, result, node, edge)) {
                const auto target = graph.targets[edge];
                queue.emplace(result.distances[target], target);
            }
        }
    }
    return result;
}

}  // namespace psv
