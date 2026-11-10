#pragma once

#include <cstdint>
#include <limits>

namespace psv {

inline constexpr std::uint32_t invalid_node = std::numeric_limits<std::uint32_t>::max();
inline constexpr std::uint64_t invalid_edge = std::numeric_limits<std::uint64_t>::max();

// Non-owning view. The caller keeps the arrays alive for the entire computation.
// row_offsets has N+1 entries; targets and weights_m each have M entries.
struct CsrView {
    const std::uint64_t* row_offsets = nullptr;
    const std::uint32_t* targets = nullptr;
    const double* weights_m = nullptr;
    std::uint64_t node_count = 0;
    std::uint64_t edge_count = 0;
};

// Both functions throw on invalid sizes, including reserved indices/overflow.
std::uint64_t csr_binary_size(std::uint64_t node_count, std::uint64_t edge_count);
void validate_csr(CsrView graph);

}  // namespace psv
