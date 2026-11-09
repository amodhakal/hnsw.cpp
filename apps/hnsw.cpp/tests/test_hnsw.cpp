#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hnsw/hnsw.hpp"

namespace {

int failures = 0;

#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::cerr << __FILE__ << ":" << __LINE__                         \
                      << ": CHECK failed: " #condition "\n";                 \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

#define CHECK_THROW(expression, exception_type)                              \
    do {                                                                     \
        bool caught = false;                                                 \
        try {                                                                \
            (void)(expression);                                              \
        } catch (const exception_type&) {                                    \
            caught = true;                                                   \
        } catch (...) {                                                      \
        }                                                                    \
        if (!caught) {                                                       \
            std::cerr << __FILE__ << ":" << __LINE__                         \
                      << ": expected " #exception_type " from " #expression \
                      << "\n";                                               \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

std::vector<float> make_vectors(std::size_t count,
                                std::size_t dim,
                                std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> values(count * dim);
    for (float& value : values) {
        value = distribution(rng);
    }
    return values;
}

void normalize(float* values, std::size_t dim) {
    double norm_sq = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        const double value = static_cast<double>(values[i]);
        norm_sq += value * value;
    }
    if (norm_sq == 0.0) {
        std::fill(values, values + dim, 0.0f);
        return;
    }
    const float inverse = 1.0f / static_cast<float>(std::sqrt(norm_sq));
    for (std::size_t i = 0; i < dim; ++i) {
        values[i] *= inverse;
    }
}

double distance(hnsw::Space space,
                const float* left,
                const float* right,
                std::size_t dim) {
    if (space == hnsw::Space::L2) {
        double sum = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            const double difference =
                static_cast<double>(left[i]) - right[i];
            sum += difference * difference;
        }
        return sum;
    }

    double dot = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        dot += static_cast<double>(left[i]) * right[i];
    }
    return 1.0 - dot;
}

std::vector<std::size_t> brute_force(const std::vector<float>& vectors,
                                     const std::vector<float>& query,
                                     std::size_t count,
                                     std::size_t dim,
                                     hnsw::Space space,
                                     std::size_t k) {
    std::vector<float> prepared_query(query);
    if (space == hnsw::Space::Cosine) {
        normalize(prepared_query.data(), dim);
    }

    std::vector<std::pair<double, std::size_t>> scored;
    scored.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        std::vector<float> candidate(vectors.begin() + i * dim,
                                     vectors.begin() + (i + 1) * dim);
        if (space == hnsw::Space::Cosine) {
            normalize(candidate.data(), dim);
        }
        scored.push_back({distance(space, prepared_query.data(),
                                   candidate.data(), dim),
                          i});
    }

    std::sort(scored.begin(), scored.end(),
              [](const auto& left, const auto& right) {
                  if (left.first != right.first) {
                      return left.first < right.first;
                  }
                  return left.second < right.second;
              });

    std::vector<std::size_t> result;
    for (std::size_t i = 0; i < std::min(k, scored.size()); ++i) {
        result.push_back(scored[i].second);
    }
    return result;
}

void test_recall_against_brute_force() {
    constexpr std::size_t dim = 16;
    constexpr std::size_t count = 300;
    constexpr std::size_t query_count = 20;
    constexpr std::size_t k = 5;

    const std::vector<float> vectors = make_vectors(count, dim, 7);
    const std::vector<float> queries = make_vectors(query_count, dim, 8);

    for (const hnsw::Space space :
         {hnsw::Space::Cosine, hnsw::Space::L2, hnsw::Space::InnerProduct}) {
        hnsw::Index index(hnsw::Config{
            dim, space, 8, 100, 100, 42, true,
        });
        index.add_items(vectors.data(), count);

        std::size_t hits = 0;
        for (std::size_t query = 0; query < query_count; ++query) {
            const std::vector<float> query_vector(
                queries.begin() + query * dim,
                queries.begin() + (query + 1) * dim);
            const hnsw::SearchResult result =
                index.search(query_vector.data(), k);
            const std::vector<std::size_t> exact =
                brute_force(vectors, query_vector, count, dim, space, k);

            for (const std::uint64_t label : result.labels) {
                if (std::find(exact.begin(), exact.end(),
                              static_cast<std::size_t>(label)) != exact.end()) {
                    ++hits;
                }
            }
        }

        const double recall =
            static_cast<double>(hits) / static_cast<double>(query_count * k);
        CHECK(recall >= 0.9);
    }
}

void test_deterministic_with_same_seed() {
    constexpr std::size_t dim = 8;
    constexpr std::size_t count = 200;
    constexpr std::size_t query_count = 10;

    const std::vector<float> vectors = make_vectors(count, dim, 1);
    const std::vector<float> queries = make_vectors(query_count, dim, 2);

    hnsw::Index first(hnsw::Config{dim, hnsw::Space::Cosine, 8, 200, 50, 123,
                                   true});
    first.add_items(vectors.data(), count);
    hnsw::Index second(hnsw::Config{dim, hnsw::Space::Cosine, 8, 200, 50, 123,
                                    true});
    second.add_items(vectors.data(), count);

    for (std::size_t query = 0; query < query_count; ++query) {
        const std::vector<float> query_vector(
            queries.begin() + query * dim,
            queries.begin() + (query + 1) * dim);
        const hnsw::SearchResult first_result =
            first.search(query_vector.data(), 5);
        const hnsw::SearchResult second_result =
            second.search(query_vector.data(), 5);
        CHECK(first_result.labels == second_result.labels);
        CHECK(first_result.distances == second_result.distances);
    }
}

void test_custom_labels_round_trip() {
    constexpr std::size_t dim = 4;
    constexpr std::size_t count = 20;

    const std::vector<float> vectors = make_vectors(count, dim, 3);
    std::vector<std::uint64_t> labels;
    labels.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        labels.push_back(1000 + i);
    }

    hnsw::Index index(hnsw::Config{dim, hnsw::Space::Cosine, 4, 200, 50, 5,
                                   true});
    index.add_items(vectors.data(), count, labels.data());

    const hnsw::SearchResult result =
        index.search(vectors.data(), 3);
    CHECK(!result.labels.empty());
    for (const std::uint64_t label : result.labels) {
        CHECK(label >= 1000 && label < 1000 + count);
    }
    CHECK(std::find(result.labels.begin(), result.labels.end(),
                    static_cast<std::uint64_t>(1000)) != result.labels.end());
}

void test_k_larger_than_index() {
    constexpr std::size_t dim = 4;
    constexpr std::size_t count = 5;

    const std::vector<float> vectors = make_vectors(count, dim, 4);
    hnsw::Index index(hnsw::Config{dim, hnsw::Space::Cosine, 4, 200, 50, 5,
                                   true});
    index.add_items(vectors.data(), count);

    const hnsw::SearchResult result =
        index.search(vectors.data(), 10);
    CHECK(result.labels.size() == count);
    CHECK(result.distances.size() == count);
}

void test_empty_index_raises() {
    hnsw::Index index(hnsw::Config{4, hnsw::Space::Cosine, 16, 200, 50, 100,
                                   true});
    const std::vector<float> query(4, 0.0f);
    CHECK_THROW(index.search(query.data(), 10), std::runtime_error);
}

void test_dimension_mismatch_raises() {
    hnsw::Index index(hnsw::Config{4, hnsw::Space::Cosine, 16, 200, 50, 100,
                                   true});
    const std::vector<float> vector(3, 0.0f);
    CHECK_THROW(index.add_item(vector), std::invalid_argument);
}

void test_invalid_space_raises() {
    CHECK_THROW(hnsw::parse_space("manhattan"), std::invalid_argument);
    CHECK_THROW(hnsw::parse_visited_mode("cuckoo"), std::invalid_argument);
}

void test_invalid_alpha_raises() {
    hnsw::Config bad_alpha{4, hnsw::Space::Cosine, 16, 200, 50, 100, true,
                           hnsw::VisitedMode::Epoch, 0.5, 1.1};
    CHECK_THROW(hnsw::Index(bad_alpha), std::invalid_argument);
}

void test_invalid_generosity_raises() {
    hnsw::Config bad_gen{4, hnsw::Space::Cosine, 16, 200, 50, 100, true,
                         hnsw::VisitedMode::Epoch, 1.1, 0.5};
    CHECK_THROW(hnsw::Index(bad_gen), std::invalid_argument);
}

void test_visited_modes() {
    constexpr std::size_t dim = 8;
    constexpr std::size_t count = 100;
    const std::vector<float> vectors = make_vectors(count, dim, 14);

    for (const hnsw::VisitedMode mode :
         {hnsw::VisitedMode::Epoch, hnsw::VisitedMode::Bloom}) {
        hnsw::Index index(hnsw::Config{
            dim, hnsw::Space::Cosine, 8, 100, 50, 15, true, mode,
        });
        index.add_items(vectors.data(), count);
        const hnsw::SearchResult result = index.search(vectors.data(), 5);
        CHECK(result.labels.size() == 5);
    }
}

void test_invalid_parameters_raise() {
    hnsw::Config bad_m{4, hnsw::Space::Cosine, 1, 200, 50, 100, true};
    CHECK_THROW(hnsw::Index(bad_m), std::invalid_argument);

    hnsw::Config bad_ef{4, hnsw::Space::Cosine, 16, 0, 50, 100, true};
    CHECK_THROW(hnsw::Index(bad_ef), std::invalid_argument);
}

void test_set_ef() {
    constexpr std::size_t dim = 8;
    constexpr std::size_t count = 50;

    const std::vector<float> vectors = make_vectors(count, dim, 6);
    hnsw::Index index(hnsw::Config{dim, hnsw::Space::Cosine, 4, 200, 50, 7,
                                   true});
    index.add_items(vectors.data(), count);

    index.set_ef(80);
    CHECK(index.ef_search() == 80);
    const hnsw::SearchResult result =
        index.search(vectors.data(), 5);
    CHECK(result.labels.size() == 5);

    CHECK_THROW(index.set_ef(0), std::invalid_argument);
}

void test_index_size_is_positive() {
    constexpr std::size_t dim = 4;
    constexpr std::size_t count = 20;

    const std::vector<float> vectors = make_vectors(count, dim, 9);
    hnsw::Index index(hnsw::Config{dim, hnsw::Space::Cosine, 4, 200, 50, 10,
                                   true});
    index.add_items(vectors.data(), count);
    CHECK(index.index_size_bytes() > 0);
}

void test_batch_matches_single() {
    constexpr std::size_t dim = 8;
    constexpr std::size_t count = 100;
    constexpr std::size_t query_count = 5;
    constexpr std::size_t k = 5;

    const std::vector<float> vectors = make_vectors(count, dim, 11);
    const std::vector<float> queries = make_vectors(query_count, dim, 12);

    hnsw::Index index(hnsw::Config{dim, hnsw::Space::Cosine, 8, 200, 50, 13,
                                   true});
    index.add_items(vectors.data(), count);

    const std::vector<hnsw::SearchResult> batch =
        index.search_batch(queries.data(), query_count, k);
    CHECK(batch.size() == query_count);

    for (std::size_t row = 0; row < query_count; ++row) {
        const hnsw::SearchResult single =
            index.search(queries.data() + row * dim, k);
        CHECK(batch[row].labels == single.labels);
        CHECK(batch[row].distances == single.distances);
    }
}

}  // namespace

int main() {
    test_recall_against_brute_force();
    test_deterministic_with_same_seed();
    test_custom_labels_round_trip();
    test_k_larger_than_index();
    test_empty_index_raises();
    test_dimension_mismatch_raises();
    test_invalid_space_raises();
    test_invalid_alpha_raises();
    test_invalid_generosity_raises();
    test_visited_modes();
    test_invalid_parameters_raise();
    test_set_ef();
    test_index_size_is_positive();
    test_batch_matches_single();

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All tests passed\n";
    return EXIT_SUCCESS;
}
