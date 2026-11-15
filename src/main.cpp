#include "dijkstra.hpp"
#include "graph.hpp"
#include "execution.hpp"

#include <charconv>
#include <algorithm>
#include <cmath>
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
    double delta = 100;
    bool delta_given = false;
    bool early_stop = false;
};

void usage() {
    std::cout << "Usage: psv_dijkstra --graph FILE --source INDEX --target INDEX\n"
                 "                    [--algorithm heap|phased|cpu-parallel|cuda|auto|delta-cpu|delta-cuda] [--delta METRES]\n"
                 "                    [--distances FILE.csv] [--early-stop]\n"
                 "       psv_dijkstra --backends\n";
}

void print_string(const std::string& value) {
    std::cout << '"';
    constexpr char hex[] = "0123456789abcdef";
    for (const unsigned char ch : value) {
        if (ch == '"' || ch == '\\') std::cout << '\\' << static_cast<char>(ch);
        else if (ch < 32) std::cout << "\\u00" << hex[ch >> 4] << hex[ch & 15];
        else std::cout << static_cast<char>(ch);
    }
    std::cout << '"';
}

void print_backend(const psv::BackendStatus& status) {
    std::cout << "{\"available\":" << (status.available ? "true" : "false") << ",\"reason\":";
    print_string(status.reason);
    std::cout << ",\"device\":";
    print_string(status.device);
    std::cout << '}';
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
        if (key == "--early-stop") { options.early_stop = true; --i; continue; }
        if (i + 1 >= argc) {
            throw std::invalid_argument("Missing value for " + key);
        }
        const std::string value = argv[i + 1];
        if (key == "--graph") options.graph = value;
        else if (key == "--source") options.source = parse_node(value);
        else if (key == "--target") options.target = parse_node(value);
        else if (key == "--algorithm") options.algorithm = value;
        else if (key == "--delta") {
            std::size_t consumed = 0;
            options.delta = std::stod(value, &consumed);
            options.delta_given = true;
            if (consumed != value.size() || !std::isfinite(options.delta) || options.delta <= 0)
                throw std::invalid_argument("Delta must be finite and positive");
        }
        else if (key == "--distances") options.distances = value;
        else throw std::invalid_argument("Unknown option: " + key);
    }
    if (options.graph.empty() || options.source == psv::invalid_node || options.target == psv::invalid_node) {
        throw std::invalid_argument("--graph, --source and --target are required");
    }
    if (options.algorithm != "heap" && options.algorithm != "phased" && options.algorithm != "cuda"
        && options.algorithm != "cpu-parallel" && options.algorithm != "auto" && options.algorithm != "delta-cpu" && options.algorithm != "delta-cuda") {
        throw std::invalid_argument("Algorithm must be heap, phased, cpu-parallel, cuda, auto, delta-cpu or delta-cuda");
    }
    if (!options.distances.empty()) {
        const auto input = std::filesystem::weakly_canonical(options.graph);
        const auto output = std::filesystem::weakly_canonical(options.distances);
        if (input == output || (std::filesystem::exists(output) && std::filesystem::equivalent(input, output))) {
            throw std::invalid_argument("Distance output must not overwrite the CSR input");
        }
    }
    if (!options.delta_given) {
        if (options.algorithm == "delta-cpu") options.delta = options.early_stop ? 300 : 1000;
        if (options.algorithm == "delta-cuda") options.delta = options.early_stop ? 10000 : 1000000;
    }
    return options;
}

void save_distances(const std::filesystem::path& path, const psv::DijkstraResult& result) {
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open distances output: " + path.string());
    stream.imbue(std::locale::classic());
    stream << std::setprecision(std::numeric_limits<double>::max_digits10);
    stream << "node_index,distance_m,predecessor_node,predecessor_edge,settled";
    if (!result.complete) stream << ",state";
    stream << '\n';
    for (std::size_t node = 0; node < result.distances.size(); ++node) {
        stream << node << ',' << result.distances[node] << ',' << result.predecessor_node[node]
               << ',' << result.predecessor_edge[node] << ',' << static_cast<unsigned>(result.settled[node]);
        if (!result.complete) stream << ',' << (result.settled[node] ? "final" : std::isfinite(result.distances[node]) ? "tentative" : "unknown");
        stream << '\n';
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
        if (argc == 2 && std::string(argv[1]) == "--backends") {
            const auto gpu = psv::cuda_status();
            std::cout << "{\"cuda\":";
            print_backend(gpu);
            std::cout << ",\"cpu-parallel\":";
            print_backend(psv::openmp_status());
            std::cout << ",\"delta-cuda\":";
            print_backend(gpu);
            std::cout << ",\"delta-cpu\":";
            print_backend({true, "", psv::openmp_status().available ? "OpenMP" : "serial CPU"});
            std::cout << ",\"max_cpu_threads\":" << psv::parallel_cpu_threads()
                      << ",\"auto_policy\":\"cuda > cpu-parallel > heap\"}\n";
            return 0;
        }
        const auto options = parse_options(argc, argv);
        const auto graph = psv::load_csr(options.graph);
        const auto view = graph.view();
        if (options.source >= view.node_count || options.target >= view.node_count) {
            throw std::out_of_range("Source or target index is outside the graph");
        }
        // Includes validation, backend setup and transfers; excludes disk I/O and path reconstruction.
        const auto start = std::chrono::steady_clock::now();
        const auto execution = psv::run_dijkstra(view, options.source, options.algorithm, options.delta, options.early_stop ? options.target : psv::invalid_node);
        const auto& result = execution.result;
        const auto stop = std::chrono::steady_clock::now();
        const auto path = psv::reconstruct_path(view, result, options.source, options.target);
        if (!options.distances.empty()) save_distances(options.distances, result);
        const double time_ms = std::chrono::duration<double, std::milli>(stop - start).count();
        const bool reachable = path.status == psv::PathStatus::reachable;
        std::cout.imbue(std::locale::classic());
        std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
                  << "{\n  \"node_count\": " << view.node_count
                  << ",\n  \"directed_edge_count\": " << view.edge_count
                  << ",\n  \"algorithm\": \"" << execution.algorithm << '"'
                  << ",\n  \"requested_algorithm\": \"" << options.algorithm << '"'
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
        std::cout << ",\n  \"algorithm_ms\": " << time_ms
                  << ",\n  \"fallback_reason\": ";
        print_string(execution.fallback_reason);
        std::cout << ",\n  \"early_stop\": " << (options.early_stop ? "true" : "false")
                  << ",\n  \"search_complete\": " << (result.complete ? "true" : "false")
                  << ",\n  \"settled_nodes\": " << std::count(result.settled.begin(), result.settled.end(), std::uint8_t{1});
        std::cout << ",\n  \"delta\": ";
        if (execution.algorithm == "delta-cpu" || execution.algorithm == "delta-cuda") std::cout << options.delta;
        else std::cout << "null";
        std::cout << ",\n  \"gpu_compute_ms\": ";
        if ((execution.algorithm == "cuda" || execution.algorithm == "delta-cuda")) std::cout << execution.cuda_timings.compute_ms;
        else std::cout << "null";
        std::cout << ",\n  \"gpu_upload_ms\": ";
        if ((execution.algorithm == "cuda" || execution.algorithm == "delta-cuda")) std::cout << execution.cuda_timings.upload_ms;
        else std::cout << "null";
        std::cout << ",\n  \"gpu_download_ms\": ";
        if ((execution.algorithm == "cuda" || execution.algorithm == "delta-cuda")) std::cout << execution.cuda_timings.download_ms;
        else std::cout << "null";
        std::cout << "\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
