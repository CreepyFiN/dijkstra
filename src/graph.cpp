#include "graph.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace psv {
namespace {

void read_exact(std::istream& stream, char* bytes, std::streamsize count) {
    if (!stream.read(bytes, count)) {
        throw std::runtime_error("Truncated or unreadable CSR file");
    }
}

template <typename UInt>
UInt read_le(std::istream& stream) {
    static_assert(std::is_unsigned<UInt>::value, "Unsigned integer required");
    std::array<unsigned char, sizeof(UInt)> bytes{};
    read_exact(stream, reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    UInt value = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        value |= static_cast<UInt>(bytes[i]) << (8 * i);
    }
    return value;
}

}  // namespace

CsrView CsrGraph::view() const {
    if (row_offsets.empty() || targets.size() != weights_m.size()) {
        throw std::invalid_argument("CSR owning array lengths do not agree");
    }
    const auto n = row_offsets.size() - 1;
    if ((!osm_node_ids.empty() && osm_node_ids.size() != n)
        || (!latitude.empty() && latitude.size() != n)
        || (!longitude.empty() && longitude.size() != n)) {
        throw std::invalid_argument("CSR optional node metadata lengths do not agree");
    }
    (void)csr_binary_size(n, targets.size());
    return {row_offsets.data(), targets.data(), weights_m.data(), n, targets.size()};
}

CsrGraph load_csr(const std::filesystem::path& path) {
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
                  "CSR requires IEEE 754 binary64 doubles");
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("Cannot open CSR file: " + path.string());
    }
    const auto end = stream.tellg();
    if (end < 0) {
        throw std::runtime_error("Cannot determine CSR file length");
    }
    const auto file_size = static_cast<std::uint64_t>(end);
    stream.seekg(0);
    std::array<char, 8> magic{};
    read_exact(stream, magic.data(), static_cast<std::streamsize>(magic.size()));
    if (std::string(magic.data(), magic.size()) != "PSVCSR01") {
        throw std::runtime_error("Invalid CSR magic: expected PSVCSR01");
    }
    const auto n = read_le<std::uint64_t>(stream);
    const auto m = read_le<std::uint64_t>(stream);
    const auto expected = csr_binary_size(n, m);
    if (file_size != expected) {
        throw std::runtime_error("CSR file length mismatch: expected " + std::to_string(expected)
                                 + " bytes, got " + std::to_string(file_size));
    }
    CsrGraph graph;
    try {
        graph.row_offsets.resize(static_cast<std::size_t>(n + 1));
        graph.targets.resize(static_cast<std::size_t>(m));
        graph.weights_m.resize(static_cast<std::size_t>(m));
    } catch (const std::exception&) {
        throw std::runtime_error("Cannot allocate CSR arrays for this file");
    }
    for (auto& offset : graph.row_offsets) {
        offset = read_le<std::uint64_t>(stream);
    }
    for (auto& target : graph.targets) {
        target = read_le<std::uint32_t>(stream);
    }
    for (auto& weight : graph.weights_m) {
        const auto bits = read_le<std::uint64_t>(stream);
        std::memcpy(&weight, &bits, sizeof(weight));
    }
    if (stream.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("CSR file has unexpected trailing data");
    }
    validate_csr(graph.view());
    return graph;
}

}  // namespace psv
