#include "dijkstra.hpp"
#include "test_support.hpp"

#include <cmath>
#include <iostream>
#include <limits>

namespace {

using Algorithm = psv::DijkstraResult (*)(psv::CsrView, std::uint32_t);
constexpr double inf = std::numeric_limits<double>::infinity();

void check_case(Algorithm algorithm, const psv::CsrGraph& graph, std::uint32_t source,
                const std::vector<double>& expected) {
    const auto g = graph.view();
    const auto result = algorithm(g, source);
    CHECK(result.distances == expected);
    for (std::size_t v = 0; v < expected.size(); ++v) {
        const auto path = psv::reconstruct_path(g, result, source, static_cast<std::uint32_t>(v));
        if (std::isinf(expected[v])) {
            CHECK(path.status == psv::PathStatus::unreachable);
            CHECK(path.nodes.empty() && path.edges.empty() && std::isinf(path.length_m));
            CHECK(result.settled[v] == 0);
            CHECK(result.predecessor_node[v] == psv::invalid_node);
            CHECK(result.predecessor_edge[v] == psv::invalid_edge);
        } else {
            CHECK(path.status == psv::PathStatus::reachable && result.settled[v] == 1);
            CHECK(path.nodes.front() == source && path.nodes.back() == v);
            CHECK(path.nodes.size() == path.edges.size() + 1);
            CHECK(path.length_m == expected[v]);
            double total = 0;
            for (std::size_t i = 0; i < path.edges.size(); ++i) {
                const auto edge = path.edges[i];
                CHECK(edge >= g.row_offsets[path.nodes[i]] && edge < g.row_offsets[path.nodes[i] + 1]);
                CHECK(g.targets[edge] == path.nodes[i + 1]);
                total += g.weights_m[edge];
            }
            CHECK(total == expected[v]);
        }
    }
}

void run_algorithm_tests(Algorithm algorithm) {
    check_case(algorithm, make_graph(1, {}), 0, {0});
    check_case(algorithm, make_graph(2, {{0, 1, 2}}), 0, {0, 2});
    check_case(algorithm, make_graph(2, {{0, 1, 2}}), 1, {inf, 0});
    check_case(algorithm, make_graph(3, {{0, 1, 7}, {0, 1, 2}, {1, 2, 3}}), 0, {0, 2, 5});
    check_case(algorithm, make_graph(4, {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 2, 0},
                                       {2, 1, 0}, {2, 2, 1}, {2, 3, 2}}), 0, {0, 0, 0, 2});
    check_case(algorithm, make_graph(5, {{0, 1, 10}, {0, 2, 1}, {2, 1, 1}, {1, 3, 1}, {2, 3, 10}}),
               0, {0, 2, 1, 3, inf});  // Stale heap entries at nodes 1 and 3.
    check_case(algorithm, make_graph(5, {{0, 1, 1}, {1, 0, 1}, {2, 3, 2}}), 0, {0, 1, inf, inf, inf});
    const auto tie_graph = make_graph(4, {{0, 2, 1}, {0, 1, 1}, {1, 3, 1}, {2, 3, 1}, {0, 1, 1}});
    check_case(algorithm, tie_graph, 0, {0, 1, 1, 2});
    const auto tie_result = algorithm(tie_graph.view(), 0);
    CHECK(tie_result.predecessor_node[3] == 1);
    CHECK(tie_result.predecessor_edge[1] == 0);  // Equal parallel edge must not replace the first.
    const auto tie_again = algorithm(tie_graph.view(), 0);
    CHECK(tie_again.predecessor_node == tie_result.predecessor_node);
    CHECK(tie_again.predecessor_edge == tie_result.predecessor_edge);
    const auto zero_ties = make_graph(4, {{3, 1, 0}, {3, 2, 0}, {1, 2, 0}, {2, 1, 0}});
    check_case(algorithm, zero_ties, 3, {inf, 0, 0, 0});
    const auto zero_result = algorithm(zero_ties.view(), 3);
    CHECK(zero_result.predecessor_node[1] == 3 && zero_result.predecessor_node[2] == 3);

    auto graph = make_graph(2, {{0, 1, 1}});
    expect_throw<std::out_of_range>([&] { algorithm(graph.view(), 2); });
    expect_throw<std::out_of_range>([&] { algorithm(graph.view(), psv::invalid_node); });
    expect_throw<std::out_of_range>([&] { algorithm(make_graph(0, {}).view(), 0); });
    for (const auto bad : {-1.0, inf, -inf, std::numeric_limits<double>::quiet_NaN()}) {
        graph.weights_m[0] = bad;
        expect_throw<std::invalid_argument>([&] { algorithm(graph.view(), 0); });
    }
    // Invalid edges in unreachable components are also rejected.
    const auto invalid_component = make_graph(3, {{1, 2, -1}});
    expect_throw<std::invalid_argument>([&] { algorithm(invalid_component.view(), 0); });
    const auto maximum = std::numeric_limits<double>::max();
    check_case(algorithm, make_graph(2, {{0, 1, maximum}}), 0, {0, maximum});
    const auto overflow = make_graph(3, {{0, 1, maximum}, {1, 2, maximum}});
    expect_throw<std::overflow_error>([&] { algorithm(overflow.view(), 0); });

    const auto directed = make_graph(3, {{0, 1, 1}, {1, 2, 1}});
    const auto good = algorithm(directed.view(), 0);
    auto corrupt = good;
    corrupt.predecessor_node[2] = psv::invalid_node;
    expect_throw<std::invalid_argument>([&] { psv::reconstruct_path(directed.view(), corrupt, 0, 2); });
    corrupt = good;
    corrupt.predecessor_edge[2] = psv::invalid_edge;
    expect_throw<std::invalid_argument>([&] { psv::reconstruct_path(directed.view(), corrupt, 0, 2); });
    corrupt = good;
    corrupt.predecessor_edge[2] = 0;  // Wrong direction/edge endpoints.
    expect_throw<std::invalid_argument>([&] { psv::reconstruct_path(directed.view(), corrupt, 0, 2); });
    corrupt = good;
    corrupt.distances[2] = 3;
    expect_throw<std::invalid_argument>([&] { psv::reconstruct_path(directed.view(), corrupt, 0, 2); });
    corrupt = good;
    corrupt.distances[2] = std::numeric_limits<double>::quiet_NaN();
    expect_throw<std::invalid_argument>([&] { psv::reconstruct_path(directed.view(), corrupt, 0, 2); });
    corrupt = good;
    corrupt.predecessor_edge.pop_back();
    expect_throw<std::invalid_argument>([&] { psv::reconstruct_path(directed.view(), corrupt, 0, 2); });
    expect_throw<std::out_of_range>([&] { psv::reconstruct_path(directed.view(), good, 0, 3); });
    expect_throw<std::out_of_range>([&] { psv::reconstruct_path(directed.view(), good, 3, 0); });
    const auto cycle = make_graph(3, {{0, 1, 0}, {1, 2, 0}, {2, 1, 0}});
    corrupt = algorithm(cycle.view(), 0);
    corrupt.predecessor_node[1] = 2;
    corrupt.predecessor_edge[1] = 2;
    expect_throw<std::invalid_argument>([&] { psv::reconstruct_path(cycle.view(), corrupt, 0, 2); });
}

}  // namespace

int main() {
    try {
        run_algorithm_tests(psv::dijkstra_heap_cpu);
        run_algorithm_tests(psv::dijkstra_phased_cpu);
        std::cout << "Both CPU algorithms and path reconstruction passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
