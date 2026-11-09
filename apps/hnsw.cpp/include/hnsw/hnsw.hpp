#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <string_view>
#include <vector>

namespace hnsw {

enum class Space { Cosine, L2, InnerProduct };
enum class VisitedMode { Epoch, Bloom };

Space parse_space(std::string_view name);
const char* space_name(Space space) noexcept;
VisitedMode parse_visited_mode(std::string_view name);
const char* visited_mode_name(VisitedMode mode) noexcept;

struct Config {
    std::size_t dim;
    Space space = Space::Cosine;
    std::size_t m = 16;
    std::size_t ef_construction = 200;
    std::size_t ef_search = 50;
    std::uint64_t seed = 100;
    bool heuristic = true;
    VisitedMode visited_mode = VisitedMode::Bloom;
    double alpha = 1.1;
    double generosity = 1.1;
    bool subdist = true;
    std::size_t subdist_prefix = 32;
    std::size_t subdist_samples = 10000;
    double subdist_max_stddev = 0.1;
};

struct SearchResult {
    std::vector<std::uint64_t> labels;
    std::vector<double> distances;
};

/// Approximate nearest-neighbour index backed by an HNSW graph.
///
/// This is a C++ port of apps/hnsw.py's ``HNSWIndex``. It is intentionally
/// single-threaded and keeps the graph in readable containers rather than a
/// compact adjacency layout.
class Index {
public:
    explicit Index(Config config);

    /// Insert one vector. When no label is given the vector's insertion index
    /// is used, matching the Python implementation.
    void add_item(const float* vector, std::optional<std::uint64_t> label = std::nullopt);

    /// Convenience overload that checks the dimensionality.
    void add_item(const std::vector<float>& vector,
                  std::optional<std::uint64_t> label = std::nullopt);

    /// Insert ``count`` consecutive vectors of ``dim`` floats each. ``labels``
    /// may be null, in which case insertion indices are used.
    void add_items(const float* vectors,
                   std::size_t count,
                   const std::uint64_t* labels = nullptr);

    void set_ef(std::size_t ef);

    SearchResult search(const float* query, std::size_t k) const;
    std::vector<SearchResult> search_batch(const float* queries,
                                           std::size_t count,
                                           std::size_t k) const;

    std::size_t size() const noexcept { return count_; }
    std::size_t dim() const noexcept { return config_.dim; }
    Space space() const noexcept { return config_.space; }
    std::size_t ef_search() const noexcept { return config_.ef_search; }
    std::size_t index_size_bytes() const;

private:
    using NodeId = std::uint32_t;
    using Level = int;

    struct Candidate {
        double distance;
        NodeId node;
    };

    struct PreparedVector {
        const std::int16_t* data = nullptr;
        float scale = 0.0f;
        float abs2 = 0.0f;
    };

    struct BloomFilter {
        void reset(std::size_t expected_visits);
        bool insert_if_absent(NodeId value);

        std::vector<std::uint64_t> bits;
        std::size_t bit_count = 0;
        std::size_t hash_count = 1;
    };

    void insert_prepared(const std::int16_t* prepared,
                         float scale,
                         float abs2,
                         std::optional<std::uint64_t> label);
    void quantize_into(const float* vector,
                       std::int16_t* out,
                       float& scale,
                       float& abs2) const;

    double dot(const std::int16_t* a, const std::int16_t* b) const;
    double distance(const PreparedVector& query, NodeId node) const;
    double distance_with_threshold(const PreparedVector& query, NodeId node,
                                   double threshold) const;
    double distance(NodeId left, NodeId right) const;

    Level random_level();
    NodeId greedy(const PreparedVector& query, NodeId entry, Level layer) const;
    std::vector<Candidate> search_layer(const PreparedVector& query,
                                        const std::vector<NodeId>& entry_points,
                                        std::size_t ef,
                                        Level layer) const;
    double generous_furthest(double worst, double max_distance,
                             double generosity) const;
    std::vector<NodeId> select_neighbors(const std::vector<Candidate>& candidates,
                                         std::size_t limit) const;
    std::vector<NodeId> prune(NodeId node, Level layer, std::size_t limit) const;

    void begin_visit_epoch() const;
    bool mark_visited(NodeId node) const;

    Config config_;
    std::size_t count_ = 0;
    std::vector<std::int16_t> data_;
    std::vector<float> scales_;
    std::vector<float> abs2_;
    std::vector<std::uint64_t> labels_;
    std::vector<Level> levels_;
    std::vector<std::vector<std::vector<NodeId>>> links_;

    std::optional<NodeId> entry_point_;
    Level max_level_ = -1;
    double level_mult_ = 1.0;

    std::mt19937_64 rng_;
    mutable std::vector<std::uint32_t> visited_marks_;
    mutable std::uint32_t visit_epoch_ = 0;
    mutable BloomFilter bloom_filter_;
    mutable double diameter_ = 0.0;
    // Subdist auto-detection state (Welford statistics over prefix/full
    // distance ratios), mutable because search is const.
    mutable std::size_t subdist_samples_seen = 0;
    mutable double subdist_ratio_mean = 0.0;
    mutable double subdist_ratio_m2 = 0.0;
    mutable bool subdist_enabled = false;
    std::vector<std::int16_t> quantization_buffer_;
};

}  // namespace hnsw
