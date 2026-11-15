#include "execution.hpp"

#include <cmath>
#include <exception>
#include <functional>
#include <queue>
#include <utility>

#ifdef PSV_HAS_OPENMP
#include <omp.h>
#endif

namespace psv {

int parallel_cpu_threads() {
#ifdef PSV_HAS_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

BackendStatus openmp_status() {
#ifdef PSV_HAS_OPENMP
    return {true, "", "OpenMP"};
#else
    return {false, "OpenMP support was not compiled into this build", ""};
#endif
}

DijkstraResult dijkstra_parallel_cpu(CsrView graph, std::uint32_t source) {
    return dijkstra_parallel_target(graph, source, invalid_node);
}

DijkstraResult dijkstra_parallel_target(CsrView graph, std::uint32_t source, std::uint32_t stop_target) {
#ifndef PSV_HAS_OPENMP
    (void)graph;
    (void)source;
    (void)stop_target;
    throw BackendUnavailable(openmp_status().reason);
#else
    auto result = detail::initialize_result(graph, source);
    if (stop_target != invalid_node && stop_target >= graph.node_count) throw std::out_of_range("Target index out of range");
    struct Minimum {
        double distance;
        std::uint32_t node;
    };
    std::vector<Minimum> minima(static_cast<std::size_t>(omp_get_max_threads()));
    using Entry = std::pair<double, std::uint32_t>;
    using Queue = std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>>;
    std::vector<Queue> queues(minima.size());
    // Node ownership is fixed by node ID, independently of OpenMP scheduling.
    // Seed after the team size is known (dynamic teams may use fewer workers).
    std::exception_ptr failure;
    std::uint32_t selected = invalid_node;
    int overflow = 0;
    // One persistent team avoids creating a new team for every settled node.
#pragma omp parallel shared(selected, overflow, result, minima, queues, failure)
    {
        const auto worker = omp_get_thread_num();
        const auto workers = omp_get_num_threads();
        auto& queue = queues[worker];
        try {
            if (source % workers == static_cast<std::uint32_t>(worker)) queue.emplace(0.0, source);
        } catch (...) {
#pragma omp critical(psv_queue_failure)
            { failure = std::current_exception(); }
        }
#pragma omp barrier
        while (true) {
            Minimum local{std::numeric_limits<double>::infinity(), invalid_node};
            while (!queue.empty() && (result.settled[queue.top().second] ||
                   queue.top().first != result.distances[queue.top().second])) queue.pop();
            if (!queue.empty()) {
                local = {queue.top().first, queue.top().second};
            }
            minima[worker] = local;
#pragma omp barrier
#pragma omp single
            {
                Minimum best{std::numeric_limits<double>::infinity(), invalid_node};
                for (int thread = 0; thread < workers; ++thread) {
                    const auto candidate = minima[thread];
                    if (candidate.distance < best.distance ||
                        (candidate.distance == best.distance && candidate.node < best.node)) {
                        best = candidate;
                    }
                }
                selected = failure ? invalid_node : best.node;
                if (selected != invalid_node) result.settled[selected] = 1;
            }
            // The single construct's implicit barrier publishes selected.
            if (selected == invalid_node || selected == stop_target) break;
            try {
                for (auto edge = graph.row_offsets[selected]; edge < graph.row_offsets[selected + 1]; ++edge) {
                    const auto target = graph.targets[edge];
                    // All edges with the same target belong to one worker. This
                    // handles unsorted CSR rows and preserves first-edge tie order.
                    if (target % static_cast<std::uint32_t>(workers) != static_cast<std::uint32_t>(worker)) continue;
                    const auto candidate = result.distances[selected] + graph.weights_m[edge];
                    if (!std::isfinite(candidate)) {
#pragma omp critical(psv_distance_overflow)
                        { overflow = 1; }
                    } else if (candidate < result.distances[target]) {
                        result.distances[target] = candidate;
                        result.predecessor_node[target] = selected;
                        result.predecessor_edge[target] = edge;
                        queue.emplace(candidate, target);
                    }
                }
            } catch (...) {
#pragma omp critical(psv_queue_failure)
                { failure = std::current_exception(); }
            }
#pragma omp barrier
            if (overflow) break;
        }
    }
    if (selected != invalid_node && selected == stop_target) result.complete = false;
    if (failure) std::rethrow_exception(failure);
    if (overflow) throw std::overflow_error("Finite distance plus edge weight overflows binary64");
    return result;
#endif
}

}  // namespace psv
