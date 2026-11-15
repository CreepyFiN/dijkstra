#include "execution.hpp"

#include <cmath>
#include <exception>
#include <map>
#include <utility>

namespace psv {
namespace detail {
double delta_bucket(double distance, double delta) {
    // Saturation groups extreme distances in the last bucket without integer
    // conversion overflow. Repeated light closure still converges there.
    return std::min(std::floor(distance / delta), std::numeric_limits<double>::max());
}

void delta_predecessors(CsrView graph, std::uint32_t source, DijkstraResult& result, double delta, double max_bucket) {
    // Build a deterministic rooted forest of tight edges after convergence.
    // Concurrent label updates must not create predecessor cycles on zero edges.
    std::vector<std::uint32_t> queue{source};
    result.settled[source] = 1;
    for (std::size_t i = 0; i < queue.size(); ++i) {
        const auto node = queue[i];
        for (auto edge = graph.row_offsets[node]; edge < graph.row_offsets[node + 1]; ++edge) {
            const auto target = graph.targets[edge];
            const auto candidate = result.distances[node] + graph.weights_m[edge];
            if (!std::isfinite(candidate)) {
                if (result.complete) throw std::overflow_error("Distance sum overflows binary64");
                continue;
            }
            if (!result.settled[target] && candidate == result.distances[target] &&
                delta_bucket(result.distances[target], delta) <= max_bucket) {
                result.settled[target] = 1;
                result.predecessor_node[target] = node;
                result.predecessor_edge[target] = edge;
                queue.push_back(target);
            }
        }
    }
    for (std::size_t node = 0; node < result.distances.size(); ++node) {
        if (std::isfinite(result.distances[node]) && delta_bucket(result.distances[node], delta) <= max_bucket && !result.settled[node])
            throw std::runtime_error("Delta-stepping result has no tight path to source");
    }
}
}  // namespace detail

DijkstraResult delta_cpu(CsrView graph, std::uint32_t source, double delta, std::uint32_t target) {
    if (!std::isfinite(delta) || delta <= 0) throw std::invalid_argument("Delta must be finite and positive");
    auto result = detail::initialize_result(graph, source);
    if (target != invalid_node && target >= graph.node_count) throw std::out_of_range("Target index out of range");
    double finalized_bucket = std::numeric_limits<double>::infinity();
    std::map<double, std::vector<std::uint32_t>> buckets;
    buckets[0].push_back(source);
    std::vector<std::uint8_t> in_set(graph.node_count, 0), in_frontier(graph.node_count, 0);
    using Request = std::pair<std::uint32_t, double>;
    auto relax = [&](const std::vector<std::uint32_t>& nodes, bool light) {
        std::vector<std::vector<Request>> requests(nodes.size());
        std::exception_ptr failure;
        // Snapshot distances are read-only during this phase. Each iteration
        // owns its request vector; bucket mutation happens only after joining.
#ifdef PSV_HAS_OPENMP
#pragma omp parallel for if(nodes.size() >= 128) schedule(static)
#endif
        for (std::int64_t index = 0; index < static_cast<std::int64_t>(nodes.size()); ++index) {
            try {
                const auto node = nodes[static_cast<std::size_t>(index)];
                auto& output = requests[static_cast<std::size_t>(index)];
                for (auto edge = graph.row_offsets[node]; edge < graph.row_offsets[node + 1]; ++edge) {
                    if ((graph.weights_m[edge] <= delta) != light) continue;
                    output.emplace_back(graph.targets[edge],
                        detail::checked_distance_sum(result.distances[node], graph.weights_m[edge]));
                }
            } catch (...) {
#ifdef PSV_HAS_OPENMP
#pragma omp critical(psv_delta_failure)
#endif
                { failure = std::current_exception(); }
            }
        }
        if (failure) std::rethrow_exception(failure);
        for (const auto& list : requests) for (const auto& request : list) {
            if (request.second < result.distances[request.first]) {
                result.distances[request.first] = request.second;
                buckets[detail::delta_bucket(request.second, delta)].push_back(request.first);
            }
        }
    };
    while (!buckets.empty()) {
        const auto current = buckets.begin()->first;
        if (target != invalid_node && std::isfinite(result.distances[target]) &&
            detail::delta_bucket(result.distances[target], delta) < current) {
            finalized_bucket = detail::delta_bucket(result.distances[target], delta);
            result.complete = false;
            break;
        }
        std::vector<std::uint32_t> processed;
        while (!buckets.empty() && buckets.begin()->first == current) {
            auto entries = std::move(buckets.begin()->second);
            buckets.erase(buckets.begin());
            std::vector<std::uint32_t> frontier;
            for (auto node : entries) {
                if (detail::delta_bucket(result.distances[node], delta) != current || in_frontier[node]) continue;
                in_frontier[node] = 1;
                frontier.push_back(node);
                if (!in_set[node]) { in_set[node] = 1; processed.push_back(node); }
            }
            for (auto node : frontier) in_frontier[node] = 0;
            relax(frontier, true);
        }
        relax(processed, false);
        for (auto node : processed) in_set[node] = 0;
    }
    detail::delta_predecessors(graph, source, result, delta, finalized_bucket);
    return result;
}
}  // namespace psv
