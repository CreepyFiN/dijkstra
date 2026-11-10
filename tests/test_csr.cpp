#include "test_support.hpp"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>

namespace {

using Bytes = std::vector<unsigned char>;

void put(Bytes& bytes, std::size_t offset, std::uint64_t value, std::size_t width = 8) {
    for (std::size_t i = 0; i < width; ++i) bytes.at(offset + i) = static_cast<unsigned char>(value >> (8 * i));
}

void put_double(Bytes& bytes, std::size_t offset, double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(value));
    put(bytes, offset, bits);
}

Bytes header(std::uint64_t n, std::uint64_t m) {
    Bytes bytes(24, 0);
    const std::string magic = "PSVCSR01";
    std::copy(magic.begin(), magic.end(), bytes.begin());
    put(bytes, 8, n);
    put(bytes, 16, m);
    return bytes;
}

Bytes valid_file() {
    auto bytes = header(3, 2);
    bytes.resize(80);
    put(bytes, 24, 0); put(bytes, 32, 2); put(bytes, 40, 2); put(bytes, 48, 2);
    put(bytes, 56, 1, 4); put(bytes, 60, 2, 4);
    put_double(bytes, 64, 2.5); put_double(bytes, 72, 4.25);
    return bytes;
}

struct TemporaryDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("psv-csr-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TemporaryDirectory() { CHECK(std::filesystem::create_directory(path)); }
    ~TemporaryDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};

void write(const std::filesystem::path& file, const Bytes& bytes) {
    std::ofstream stream(file, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    stream.close();
    CHECK(stream.good());
}

template <typename T>
void print_array(const std::vector<T>& array) {
    std::cout << '[';
    for (std::size_t i = 0; i < array.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << array[i];
    }
    std::cout << ']';
}

void run_tests() {
    TemporaryDirectory temp;
    const auto file = temp.path / "fixture.csr";
    expect_throw([&] { psv::load_csr(file); });
    write(file, valid_file());
    auto graph = psv::load_csr(file);
    CHECK(graph.row_offsets == std::vector<std::uint64_t>({0, 2, 2, 2}));
    CHECK(graph.targets == std::vector<std::uint32_t>({1, 2}));
    CHECK(graph.weights_m == std::vector<double>({2.5, 4.25}));
    CHECK(graph.osm_node_ids.empty() && graph.latitude.empty() && graph.longitude.empty());

    auto reject = [&](const Bytes& bytes) {
        write(file, bytes);
        expect_throw([&] { psv::load_csr(file); });
    };
    auto bytes = valid_file();
    bytes[0] = 'X'; reject(bytes);
    for (std::size_t length = 0; length < 80; ++length) {
        bytes = valid_file(); bytes.resize(length); reject(bytes);
    }
    bytes = valid_file(); bytes.push_back(0); reject(bytes);
    bytes = valid_file(); put(bytes, 24, 1); reject(bytes);
    bytes = valid_file(); put(bytes, 48, 1); reject(bytes);
    bytes = valid_file(); put(bytes, 40, 1); reject(bytes);  // Decreasing interior row.
    bytes = valid_file(); put(bytes, 32, 3); reject(bytes);  // Row beyond M.
    bytes = valid_file(); put(bytes, 56, 3, 4); reject(bytes);
    bytes = valid_file(); put(bytes, 56, psv::invalid_node, 4); reject(bytes);
    for (const auto weight : {-1.0, std::numeric_limits<double>::quiet_NaN(),
                              std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()}) {
        bytes = valid_file(); put_double(bytes, 64, weight); reject(bytes);
    }
    reject(header(static_cast<std::uint64_t>(psv::invalid_node) + 1, 0));
    reject(header(std::numeric_limits<std::uint64_t>::max(), 0));
    reject(header(1, std::numeric_limits<std::uint64_t>::max()));
    reject(header(1, std::numeric_limits<std::uint64_t>::max() / 12));
    reject(header(1, 1000000000));  // Length check must precede allocation.
    bytes = header(0, 0); bytes.resize(32, 0);
    write(file, bytes);
    graph = psv::load_csr(file);
    CHECK(graph.row_offsets == std::vector<std::uint64_t>({0}));
    CHECK(graph.targets.empty() && graph.weights_m.empty() && graph.view().node_count == 0);
    bytes = header(0, 1); bytes.resize(44, 0); put(bytes, 24, 1); reject(bytes);

    CHECK(psv::csr_binary_size(0, 0) == 32);
    CHECK(psv::csr_binary_size(3, 2) == 80);
    if (sizeof(std::size_t) >= 8) {
        CHECK(psv::csr_binary_size(psv::invalid_node, 0)
              == 24 + (static_cast<std::uint64_t>(psv::invalid_node) + 1) * 8);
    }
    expect_throw<std::overflow_error>([] { psv::csr_binary_size(static_cast<std::uint64_t>(psv::invalid_node) + 1, 0); });
    expect_throw<std::overflow_error>([] { psv::csr_binary_size(1, psv::invalid_edge); });
    expect_throw<std::overflow_error>([] { psv::csr_binary_size(1, psv::invalid_edge / 12); });
    expect_throw<std::invalid_argument>([] { psv::validate_csr({}); });
    expect_throw<std::overflow_error>([] {
        psv::validate_csr({nullptr, nullptr, nullptr, static_cast<std::uint64_t>(psv::invalid_node) + 1, 0});
    });
    const std::uint64_t rows[] = {0, 1};
    expect_throw<std::invalid_argument>([&] { psv::validate_csr({rows, nullptr, nullptr, 1, 1}); });
    graph = make_graph(2, {{0, 1, 1}});
    graph.weights_m.clear();
    expect_throw<std::invalid_argument>([&] { graph.view(); });
    graph = make_graph(2, {});
    graph.osm_node_ids = {123};
    expect_throw<std::invalid_argument>([&] { graph.view(); });
    graph.row_offsets.clear();
    expect_throw<std::invalid_argument>([&] { graph.view(); });
}

}  // namespace

int main(int argc, char** argv) {
    try {
        // Test-only probe lets Python compare all arrays decoded by the C++ reader.
        if (argc == 2) {
            const auto graph = psv::load_csr(argv[1]);
            std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
            std::cout << "{\"row_offsets\":"; print_array(graph.row_offsets);
            std::cout << ",\"targets\":"; print_array(graph.targets);
            std::cout << ",\"weights_m\":"; print_array(graph.weights_m);
            std::cout << "}\n";
        } else {
            run_tests();
            std::cout << "CSR loading, corruption, limits and overflow tests passed\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
