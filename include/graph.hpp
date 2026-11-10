#pragma once

#include "csr.hpp"

#include <filesystem>
#include <vector>

namespace psv {

struct CsrGraph {
    std::vector<std::uint64_t> row_offsets{0};
    std::vector<std::uint32_t> targets;
    std::vector<double> weights_m;

    // Optional metadata, deliberately absent when loading only the CSR binary.
    std::vector<std::int64_t> osm_node_ids;
    std::vector<double> latitude;
    std::vector<double> longitude;

    CsrView view() const;
};

CsrGraph load_csr(const std::filesystem::path& path);

}  // namespace psv
