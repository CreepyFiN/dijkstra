#include "csr.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace psv {
namespace {

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error("CSR size addition overflow");
    }
    return a + b;
}

std::uint64_t checked_multiply(std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error("CSR size multiplication overflow");
    }
    return a * b;
}

}  // namespace

std::uint64_t csr_binary_size(std::uint64_t n, std::uint64_t m) {
    // N == UINT32_MAX is allowed: its last valid index is UINT32_MAX-1.
    if (n > invalid_node) {
        throw std::overflow_error("CSR node count exceeds uint32_t indices (UINT32_MAX is reserved)");
    }
    const auto rows = checked_add(n, 1);
    auto size = checked_add(24, checked_multiply(rows, 8));
    size = checked_add(size, checked_multiply(m, 4));
    size = checked_add(size, checked_multiply(m, 8));
    if (rows > std::vector<std::uint64_t>().max_size()
        || m > std::vector<std::uint32_t>().max_size()
        || m > std::vector<double>().max_size()
        || n > std::vector<double>().max_size()
        || size > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("CSR arrays exceed addressable memory/index limits");
    }
    return size;
}

void validate_csr(CsrView g) {
    (void)csr_binary_size(g.node_count, g.edge_count);
    if (g.row_offsets == nullptr || (g.edge_count != 0 && (g.targets == nullptr || g.weights_m == nullptr))) {
        throw std::invalid_argument("CSR contains a null array pointer");
    }
    if (g.row_offsets[0] != 0) {
        throw std::invalid_argument("CSR row_offsets[0] must be zero");
    }
    if (g.row_offsets[g.node_count] != g.edge_count) {
        throw std::invalid_argument("CSR row_offsets[N] must equal M");
    }
    for (std::uint64_t u = 0; u < g.node_count; ++u) {
        if (g.row_offsets[u] > g.row_offsets[u + 1] || g.row_offsets[u + 1] > g.edge_count) {
            throw std::invalid_argument("CSR row_offsets must be monotone and bounded by M");
        }
    }
    for (std::uint64_t e = 0; e < g.edge_count; ++e) {
        if (g.targets[e] == invalid_node || g.targets[e] >= g.node_count) {
            throw std::invalid_argument("CSR target index out of range at edge " + std::to_string(e));
        }
        if (!std::isfinite(g.weights_m[e]) || g.weights_m[e] < 0.0) {
            throw std::invalid_argument("CSR weight must be finite and nonnegative at edge " + std::to_string(e));
        }
    }
}

}  // namespace psv
