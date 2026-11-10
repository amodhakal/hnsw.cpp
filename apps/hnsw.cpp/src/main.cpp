#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

#include "hnsw/hnsw.hpp"

int main() {
    constexpr std::size_t dim = 16;
    constexpr std::size_t count = 200;

    std::mt19937_64 rng(0);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    std::vector<float> vectors(count * dim);
    for (float& value : vectors) {
        value = distribution(rng);
    }

    hnsw::Index index(hnsw::Config{
        dim,
        hnsw::Space::Cosine,
        8,
        100,
        50,
        1,
        true,
        hnsw::VisitedMode::Bloom,
        1.0,
        1.0,
        true,
        32,
        10000,
        0.1,
    });
    index.add_items(vectors.data(), count);

    const hnsw::SearchResult result = index.search(vectors.data(), 5);
    std::cout << "indexed " << index.size() << " vectors\n";
    std::cout << "nearest labels:";
    for (const std::uint64_t label : result.labels) {
        std::cout << ' ' << label;
    }
    std::cout << "\ndistances:";
    std::cout.precision(4);
    for (const double distance : result.distances) {
        std::cout << ' ' << distance;
    }
    std::cout << '\n';

    return 0;
}
