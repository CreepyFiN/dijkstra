#pragma once

#include "graph.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " #condition); } while (false)

template <typename Exception = std::exception, typename Function>
void expect_throw(Function operation) {
    bool caught = false;
    try { operation(); } catch (const Exception&) { caught = true; }
    CHECK(caught);
}

inline psv::CsrGraph make_graph(std::size_t n,
        std::vector<std::tuple<std::uint32_t, std::uint32_t, double>> edges) {
    std::stable_sort(edges.begin(), edges.end(), [](const auto& a, const auto& b) {
        return std::tie(std::get<0>(a), std::get<1>(a)) < std::tie(std::get<0>(b), std::get<1>(b));
    });
    psv::CsrGraph graph;
    graph.row_offsets.assign(n + 1, 0);
    for (const auto& [u, v, w] : edges) {
        CHECK(u < n && v < n);
        ++graph.row_offsets[u + 1];
        graph.targets.push_back(v);
        graph.weights_m.push_back(w);
    }
    for (std::size_t u = 0; u < n; ++u) graph.row_offsets[u + 1] += graph.row_offsets[u];
    return graph;
}
