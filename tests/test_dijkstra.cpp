#include "dijkstra.hpp"
#include "execution.hpp"
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

void check_parallel_graph(Algorithm algorithm) {
    // More than one CUDA block, duplicate destinations, zero-weight ties,
    // disconnected nodes, and high-degree rows. Reverse row order to exercise
    // valid CSR that is not destination-sorted.
    std::vector<std::tuple<std::uint32_t, std::uint32_t, double>> edges;
    for (std::uint32_t node = 1; node < 1025; ++node) {
        edges.emplace_back(0, node, static_cast<double>(node % 19));
        edges.emplace_back(0, node, static_cast<double>(node % 19));
        edges.emplace_back(0, node, static_cast<double>(node % 19 + 10));
        edges.emplace_back(node, (node % 1024) + 1, 0.0);
    }
    auto graph = make_graph(1027, edges);
    for (std::size_t node = 0; node < 1027; ++node) {
        const auto begin = static_cast<std::size_t>(graph.row_offsets[node]);
        const auto end = static_cast<std::size_t>(graph.row_offsets[node + 1]);
        std::reverse(graph.targets.begin() + begin, graph.targets.begin() + end);
        std::reverse(graph.weights_m.begin() + begin, graph.weights_m.begin() + end);
    }
    const auto expected = psv::dijkstra_heap_cpu(graph.view(), 0);
    const auto actual = algorithm(graph.view(), 0);
    CHECK(actual.distances == expected.distances);
    CHECK(actual.predecessor_node == expected.predecessor_node);
    CHECK(actual.predecessor_edge == expected.predecessor_edge);
    CHECK(actual.settled == expected.settled);
    for (const auto target : {0U, 1U, 256U, 1024U, 1026U}) {
        const auto path = psv::reconstruct_path(graph.view(), actual, 0, target);
        CHECK(path.length_m == expected.distances[target]);
    }
    // Cross partial tiles, multiple device batches, and more than 256 cached
    // minima. Reverse traversal and zero edges repeatedly invalidate winners.
    for (const auto count : {257U, 2049U, 65539U}) {
        edges.clear();
        for (std::uint32_t node = 1; node < count; ++node) {
            edges.emplace_back(node, node - 1, node % 2 ? 0.125 : 0.0);
            edges.emplace_back(node - 1, node, 0.0);
        }
        const auto chain = make_graph(count + 1, edges);
        const auto reference = psv::dijkstra_heap_cpu(chain.view(), count - 1);
        const auto parallel = algorithm(chain.view(), count - 1);
        CHECK(parallel.distances == reference.distances);
        CHECK(parallel.predecessor_node == reference.predecessor_node);
        CHECK(parallel.predecessor_edge == reference.predecessor_edge);
        CHECK(parallel.settled == reference.settled);
        CHECK(psv::reconstruct_path(chain.view(), parallel, count - 1, 0).length_m == reference.distances[0]);
    }
}

void run_backend_tests() {
    const auto graph = make_graph(3, {{0, 1, 2}, {1, 2, 3}});
    const auto cpu = psv::openmp_status();
    const auto gpu = psv::cuda_status(psv::cuda_required_bytes(graph.view()));
    for (const auto& item : std::vector<std::pair<Algorithm, psv::BackendStatus>>{
             {psv::dijkstra_parallel_cpu, cpu}, {psv::dijkstra_cuda, gpu}}) {
        if (item.second.available) {
            run_algorithm_tests(item.first);
            check_parallel_graph(item.first);
            std::cout << "PASS " << item.second.device << '\n';
        } else {
            expect_throw<psv::BackendUnavailable>([&] { item.first(graph.view(), 0); });
            std::cout << "Unavailable backend checked: " << item.second.reason << '\n';
        }
    }
    const auto default_run = psv::run_dijkstra(graph.view(), 0);
    CHECK(default_run.algorithm == "heap");
    CHECK(default_run.result.distances == std::vector<double>({0, 2, 5}));
    expect_throw<std::invalid_argument>([&] { psv::run_dijkstra(graph.view(), 0, "auto"); });
    expect_throw<std::out_of_range>([&] { psv::run_dijkstra(graph.view(), 9); });
    expect_throw<std::invalid_argument>([&] { psv::run_dijkstra(graph.view(), 0, "unknown"); });
    CHECK(!psv::cuda_status(std::numeric_limits<std::uint64_t>::max()).available);
}

void run_delta_tests() {
    std::vector<psv::CsrGraph> graphs{
        make_graph(1, {}), make_graph(5, {{0,1,10},{0,2,1},{2,1,1},{1,3,1}}),
        make_graph(5, {{0,1,0},{1,0,0},{1,2,0},{2,1,0},{0,1,4},{2,3,3}})};
    std::vector<std::tuple<std::uint32_t,std::uint32_t,double>> edges;
    for (std::uint32_t node=1; node<300; ++node) {
        edges.emplace_back(0,node,1.0);
        edges.emplace_back(node,300,2.0);
        edges.emplace_back(node,300,2.0);
        edges.emplace_back(node,(node%299)+1,0.0);
    }
    graphs.push_back(make_graph(302,edges));
    for (const std::string backend : {"delta-cpu", "delta-cuda"}) {
        if (backend == "delta-cuda" && !psv::cuda_status().available) continue;
        for (const auto delta : {0.25,1.0,100.0,10000.0}) {
            for (const auto& graph : graphs) for (std::uint32_t source : {0U,static_cast<std::uint32_t>(graph.view().node_count-1)}) {
                const auto expected = psv::dijkstra_heap_cpu(graph.view(),source);
                const auto result = psv::run_dijkstra(graph.view(),source,backend,delta).result;
                CHECK(result.distances == expected.distances);
                CHECK(result.settled == expected.settled);
                for (std::uint32_t target=0; target<graph.view().node_count; ++target)
                    CHECK(psv::reconstruct_path(graph.view(),result,source,target).length_m == expected.distances[target]);
            }
        }
        const auto g = make_graph(2,{{0,1,1}});
        for (auto delta : {0.0,-1.0,inf,std::numeric_limits<double>::quiet_NaN()})
            expect_throw<std::invalid_argument>([&] {psv::run_dijkstra(g.view(),0,backend,delta);});
        expect_throw<std::out_of_range>([&] {psv::run_dijkstra(g.view(),9,backend);});
        const auto bad=make_graph(2,{{0,1,-1}});
        expect_throw<std::invalid_argument>([&] {psv::run_dijkstra(bad.view(),0,backend);});
        const auto maximum=std::numeric_limits<double>::max();
        const auto overflow=make_graph(3,{{0,1,maximum},{1,2,maximum}});
        expect_throw<std::overflow_error>([&] {psv::run_dijkstra(overflow.view(),0,backend,0.25);});
        const auto extreme=make_graph(2,{{0,1,maximum}});
        CHECK(psv::run_dijkstra(extreme.view(),0,backend,0.25).result.distances[1] == maximum);
        std::cout << "PASS " << backend << '\n';
    }
}

void run_early_tests() {
    std::vector<std::tuple<std::uint32_t,std::uint32_t,double>> edges;
    for (std::uint32_t node=0; node<2050; ++node) edges.emplace_back(node,node+1,1.0);
    const auto graph=make_graph(2053,edges);
    for (const std::string backend : {"heap","phased","cpu-parallel","cuda","delta-cpu","delta-cuda"}) {
        if ((backend=="cuda" || backend=="delta-cuda") && !psv::cuda_status().available) continue;
        if (backend=="cpu-parallel" && !psv::openmp_status().available) continue;
        for (const auto target : {0U,1U,1023U,1024U,2049U,2052U}) {
            const auto result=psv::run_dijkstra(graph.view(),0,backend,1.0,target).result;
            const auto expected=target==2052 ? inf : static_cast<double>(target);
            CHECK(psv::reconstruct_path(graph.view(),result,0,target).length_m==expected);
            if (target==2052) CHECK(result.complete);
            else {
                CHECK(!result.complete);
                CHECK(std::count(result.settled.begin(),result.settled.end(),std::uint8_t{1})==target+1);
                expect_throw<std::invalid_argument>([&] {psv::reconstruct_path(graph.view(),result,0,2052);});
            }
        }
        expect_throw<std::out_of_range>([&] {psv::run_dijkstra(graph.view(),0,backend,1.0,2053);});
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--early-stop") {
            run_early_tests();
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--delta") {
            run_delta_tests();
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--backends") {
            run_backend_tests();
            return 0;
        }
        run_algorithm_tests(psv::dijkstra_heap_cpu);
        run_algorithm_tests(psv::dijkstra_phased_cpu);
        std::cout << "Both CPU algorithms and path reconstruction passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
