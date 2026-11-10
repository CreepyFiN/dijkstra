#include "dijkstra.hpp"
#include "graph.hpp"

#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <stdexcept>
#include <string>
#include <system_error>

namespace {

struct Options {
    std::filesystem::path graph;
    std::filesystem::path distances;
    std::uint32_t source = psv::invalid_node;
    std::uint32_t target = psv::invalid_node;
    std::string algorithm = "heap";
};

void usage() {
    std::cout << "Usage: psv_dijkstra --graph FILE --source INDEX --target INDEX\n"
                 "                    [--algorithm heap|phased] [--distances FILE.csv]\n";
}

std::uint32_t parse_node(const std::string& text) {
    std::uint64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value >= psv::invalid_node) {
        throw std::invalid_argument("Invalid internal node index: " + text);
    }
    return static_cast<std::uint32_t>(value);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; i += 2) {
        const std::string key = argv[i];
        if (i + 1 >= argc) {
            throw std::invalid_argument("Missing value for " + key);
        }
        const std::string value = argv[i + 1];
        if (key == "--graph") options.graph = value;
        else if (key == "--source") options.source = parse_node(value);
        else if (key == "--target") options.target = parse_node(value);
        else if (key == "--algorithm") options.algorithm = value;
        else if (key == "--distances") options.distances = value;
        else throw std::invalid_argument("Unknown option: " + key);
    }
    if (options.graph.empty() || options.source == psv::invalid_node || options.target == psv::invalid_node) {
        throw std::invalid_argument("--graph, --source and --target are required");
    }
    if (options.algorithm != "heap" && options.algorithm != "phased") {
        throw std::invalid_argument("Algorithm must be heap or phased");
    }
    if (!options.distances.empty()) {
        const auto input = std::filesystem::weakly_canonical(options.graph);
        const auto output = std::filesystem::weakly_canonical(options.distances);
        if (input == output || (std::filesystem::exists(output) && std::filesystem::equivalent(input, output))) {
            throw std::invalid_argument("Distance output must not overwrite the CSR input");
        }
    }
    return options;
}

void save_distances(const std::filesystem::path& path, const psv::DijkstraResult& result) {
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open distances output: " + path.string());
    stream.imbue(std::locale::classic());
    stream << std::setprecision(std::numeric_limits<double>::max_digits10);
    stream << "node_index,distance_m,predecessor_node,predecessor_edge,settled\n";
    for (std::size_t node = 0; node < result.distances.size(); ++node) {
        stream << node << ',' << result.distances[node] << ',' << result.predecessor_node[node]
               << ',' << result.predecessor_edge[node] << ',' << static_cast<unsigned>(result.settled[node]) << '\n';
    }
    stream.close();
    if (!stream) throw std::runtime_error("Failed to write distances output");
}

template <typename T>
void print_array(const std::vector<T>& values) {
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) std::cout << ',';
        std::cout << values[i];
    }
    std::cout << ']';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            usage();
            return 0;
        }
        const auto options = parse_options(argc, argv);
        const auto graph = psv::load_csr(options.graph);
        const auto view = graph.view();
        if (options.source >= view.node_count || options.target >= view.node_count) {
            throw std::out_of_range("Source or target index is outside the graph");
        }
        const auto algorithm = options.algorithm == "heap" ? psv::dijkstra_heap_cpu : psv::dijkstra_phased_cpu;
        // Includes result initialization and input validation, excludes all I/O and path reconstruction.
        const auto start = std::chrono::steady_clock::now();
        const auto result = algorithm(view, options.source);
        const auto stop = std::chrono::steady_clock::now();
        const auto path = psv::reconstruct_path(view, result, options.source, options.target);
        if (!options.distances.empty()) save_distances(options.distances, result);
        const double time_ms = std::chrono::duration<double, std::milli>(stop - start).count();
        const bool reachable = path.status == psv::PathStatus::reachable;
        std::cout.imbue(std::locale::classic());
        std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
                  << "{\n  \"node_count\": " << view.node_count
                  << ",\n  \"directed_edge_count\": " << view.edge_count
                  << ",\n  \"algorithm\": \"" << options.algorithm << '"'
                  << ",\n  \"source\": " << options.source << ",\n  \"target\": " << options.target
                  << ",\n  \"reachable\": " << (reachable ? "true" : "false")
                  << ",\n  \"status\": \"" << (reachable ? "reachable" : "unreachable") << '"'
                  << ",\n  \"length_m\": ";
        if (reachable) std::cout << path.length_m;
        else std::cout << "null";  // JSON has no infinity literal; distances CSV uses inf.
        std::cout << ",\n  \"path_nodes\": ";
        print_array(path.nodes);
        std::cout << ",\n  \"path_edges\": ";
        print_array(path.edges);
        std::cout << ",\n  \"algorithm_ms\": " << time_ms << "\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
