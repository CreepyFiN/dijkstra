#include "dijkstra.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace psv {
namespace detail {

DijkstraResult initialize_result(CsrView graph, std::uint32_t source) {
    validate_csr(graph);
    if (source == invalid_node || source >= graph.node_count) {
        throw std::out_of_range("Dijkstra source index is out of range (the graph may be empty)");
    }
    const auto n = static_cast<std::size_t>(graph.node_count);
    DijkstraResult result;
    result.distances.assign(n, std::numeric_limits<double>::infinity());
    result.predecessor_node.assign(n, invalid_node);
    result.predecessor_edge.assign(n, invalid_edge);
    result.settled.assign(n, 0);
    result.distances[source] = 0.0;
    return result;
}

double checked_distance_sum(double distance, double weight) {
    const double candidate = distance + weight;
    if (!std::isfinite(candidate)) {
        throw std::overflow_error("Finite distance plus edge weight overflows binary64");
    }
    return candidate;
}

bool relax_edge(CsrView graph, DijkstraResult& result, std::uint32_t from, std::uint64_t edge) {
    const auto target = graph.targets[edge];
    const double candidate = checked_distance_sum(result.distances[from], graph.weights_m[edge]);
    if (candidate < result.distances[target]) {
        result.distances[target] = candidate;
        result.predecessor_node[target] = from;
        result.predecessor_edge[target] = edge;
        return true;
    }
    return false;
}

}  // namespace detail

Path reconstruct_path(CsrView graph, const DijkstraResult& result,
                      std::uint32_t source, std::uint32_t target) {
    validate_csr(graph);
    const auto n = graph.node_count;
    if (source == invalid_node || target == invalid_node || source >= n || target >= n) {
        throw std::out_of_range("Path source or target index is out of range");
    }
    if (result.distances.size() != n || result.predecessor_node.size() != n
        || result.predecessor_edge.size() != n || result.settled.size() != n) {
        throw std::invalid_argument("Path result array lengths do not match the graph");
    }
    if (result.distances[source] != 0.0 || result.predecessor_node[source] != invalid_node
        || result.predecessor_edge[source] != invalid_edge || result.settled[source] != 1) {
        throw std::invalid_argument("Path result has an invalid source state");
    }
    if (!result.complete && !result.settled[target])
        throw std::invalid_argument("Target is not finalized in this partial result");
    Path path;
    if (result.distances[target] == std::numeric_limits<double>::infinity()) {
        if (result.predecessor_node[target] != invalid_node
            || result.predecessor_edge[target] != invalid_edge || result.settled[target] != 0) {
            throw std::invalid_argument("Unreachable target has corrupt predecessor/settled state");
        }
        return path;
    }
    auto current = target;
    for (std::uint64_t steps = 0; ; ++steps) {
        if (steps >= n) {
            throw std::invalid_argument("Cycle in predecessor chain");
        }
        if (!std::isfinite(result.distances[current]) || result.distances[current] < 0
            || result.settled[current] != 1) {
            throw std::invalid_argument("Invalid distance or settled state in predecessor chain");
        }
        path.nodes.push_back(current);
        if (current == source) {
            break;
        }
        const auto previous = result.predecessor_node[current];
        const auto edge = result.predecessor_edge[current];
        if (previous == invalid_node || previous >= n || edge == invalid_edge || edge >= graph.edge_count) {
            throw std::invalid_argument("Predecessor node or edge index is out of range");
        }
        if (edge < graph.row_offsets[previous] || edge >= graph.row_offsets[previous + 1]
            || graph.targets[edge] != current) {
            throw std::invalid_argument("Predecessor edge does not follow the directed graph");
        }
        if (!std::isfinite(result.distances[previous]) || result.distances[previous] < 0
            || detail::checked_distance_sum(result.distances[previous], graph.weights_m[edge])
                   != result.distances[current]) {
            throw std::invalid_argument("Predecessor distances disagree with edge weight");
        }
        path.edges.push_back(edge);
        current = previous;
    }
    std::reverse(path.nodes.begin(), path.nodes.end());
    std::reverse(path.edges.begin(), path.edges.end());
    path.length_m = 0.0;
    for (auto edge : path.edges) {
        path.length_m = detail::checked_distance_sum(path.length_m, graph.weights_m[edge]);
    }
    // Same source-to-target addition order as relaxation; no epsilon is needed.
    if (path.length_m != result.distances[target]) {
        throw std::invalid_argument("Path weight sum disagrees with shortest distance");
    }
    path.status = PathStatus::reachable;
    return path;
}

}  // namespace psv
