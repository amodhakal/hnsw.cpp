#include "hnsw/hnsw.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>

namespace hnsw {

namespace {

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

}  // namespace

Space parse_space(std::string_view name) {
    if (name == "cosine") {
        return Space::Cosine;
    }
    if (name == "l2") {
        return Space::L2;
    }
    if (name == "ip") {
        return Space::InnerProduct;
    }
    throw std::invalid_argument(
        "Unsupported space '" + std::string(name) +
        "'. Use one of: cosine, ip, l2");
}

const char* space_name(Space space) noexcept {
    switch (space) {
        case Space::Cosine:
            return "cosine";
        case Space::L2:
            return "l2";
        case Space::InnerProduct:
            return "ip";
    }
    return "unknown";
}

VisitedMode parse_visited_mode(std::string_view name) {
    if (name == "epoch") {
        return VisitedMode::Epoch;
    }
    if (name == "bloom") {
        return VisitedMode::Bloom;
    }
    throw std::invalid_argument(
        "Unsupported visited mode '" + std::string(name) +
        "'. Use one of: epoch, bloom");
}

const char* visited_mode_name(VisitedMode mode) noexcept {
    switch (mode) {
        case VisitedMode::Epoch:
            return "epoch";
        case VisitedMode::Bloom:
            return "bloom";
    }
    return "unknown";
}

Index::Index(Config config) : config_(config) {
    if (config_.dim == 0) {
        throw std::invalid_argument("dim must be positive");
    }
    if (config_.m < 2) {
        throw std::invalid_argument("m must be at least 2");
    }
    if (config_.ef_construction < 1) {
        throw std::invalid_argument("ef_construction must be at least 1");
    }
    if (config_.ef_search < 1) {
        throw std::invalid_argument("ef_search must be at least 1");
    }

    level_mult_ = 1.0 / std::log(static_cast<double>(config_.m));
    rng_.seed(config_.seed);
}

void Index::add_item(const float* vector, std::optional<std::uint64_t> label) {
    if (vector == nullptr) {
        throw std::invalid_argument("vector must not be null");
    }
    if (quantization_buffer_.size() < config_.dim) {
        quantization_buffer_.resize(config_.dim);
    }
    float scale = 0.0f;
    float abs2 = 0.0f;
    quantize_into(vector, quantization_buffer_.data(), scale, abs2);
    insert_prepared(quantization_buffer_.data(), scale, abs2, label);
}

void Index::add_item(const std::vector<float>& vector,
                     std::optional<std::uint64_t> label) {
    if (vector.size() != config_.dim) {
        throw std::invalid_argument(
            "Expected a " + std::to_string(config_.dim) +
            "-dimensional vector, got " + std::to_string(vector.size()));
    }
    add_item(vector.data(), label);
}

void Index::add_items(const float* vectors,
                      std::size_t count,
                      const std::uint64_t* labels) {
    if (vectors == nullptr && count > 0) {
        throw std::invalid_argument("vectors must not be null");
    }
    for (std::size_t i = 0; i < count; ++i) {
        const std::optional<std::uint64_t> label =
            labels == nullptr ? std::nullopt : std::optional<std::uint64_t>(labels[i]);
        add_item(vectors + i * config_.dim, label);
    }
}

void Index::set_ef(std::size_t ef) {
    if (ef < 1) {
        throw std::invalid_argument("ef must be at least 1");
    }
    config_.ef_search = ef;
}

void Index::quantize_into(const float* vector,
                          std::int16_t* out,
                          float& scale,
                          float& abs2) const {
    float inverse = 1.0f;
    if (config_.space == Space::Cosine) {
        double norm_sq = 0.0;
        for (std::size_t i = 0; i < config_.dim; ++i) {
            const double value = static_cast<double>(vector[i]);
            norm_sq += value * value;
        }
        inverse = norm_sq == 0.0
                      ? 0.0f
                      : 1.0f / static_cast<float>(std::sqrt(norm_sq));
    }

    float max_abs = 0.0f;
    for (std::size_t i = 0; i < config_.dim; ++i) {
        max_abs = std::max(max_abs, std::fabs(vector[i] * inverse));
    }

    if (max_abs == 0.0f) {
        std::fill(out, out + config_.dim, static_cast<std::int16_t>(0));
        scale = 0.0f;
        abs2 = 0.0f;
        return;
    }

    scale = max_abs / 32767.0f;
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < config_.dim; ++i) {
        const double scaled = static_cast<double>(vector[i] * inverse) /
                              static_cast<double>(scale);
        const long rounded = std::lround(scaled);
        const long clamped = std::max<long>(-32767, std::min<long>(32767, rounded));
        out[i] = static_cast<std::int16_t>(clamped);
        sum_sq += static_cast<double>(clamped) * static_cast<double>(clamped);
    }

    const double scale_sq =
        static_cast<double>(scale) * static_cast<double>(scale);
    abs2 = static_cast<float>(scale_sq * sum_sq);

    if (config_.space == Space::Cosine && abs2 > 0.0f) {
        // Reconstructed cosine vectors should have unit norm so that the
        // dot product alone is enough for ranking.
        scale /= std::sqrt(abs2);
        abs2 = 1.0f;
    }
}

double Index::dot(const std::int16_t* a, const std::int16_t* b) const {
    std::int64_t sum = 0;
    for (std::size_t i = 0; i < config_.dim; ++i) {
        sum += static_cast<std::int64_t>(a[i]) * static_cast<std::int64_t>(b[i]);
    }
    return static_cast<double>(sum);
}

double Index::distance(const PreparedVector& query, NodeId node) const {
    const std::size_t offset = static_cast<std::size_t>(node) * config_.dim;
    const double scale =
        static_cast<double>(query.scale) * static_cast<double>(scales_[node]);
    const double dot_product = scale * dot(query.data, data_.data() + offset);

    switch (config_.space) {
        case Space::Cosine:
        case Space::InnerProduct:
            return 1.0 - dot_product;
        case Space::L2: {
            const double squared =
                static_cast<double>(query.abs2) +
                static_cast<double>(abs2_[node]) - 2.0 * dot_product;
            return std::max(0.0, squared);
        }
    }
    return 0.0;
}

double Index::distance(NodeId left, NodeId right) const {
    const PreparedVector prepared{
        data_.data() + static_cast<std::size_t>(left) * config_.dim,
        scales_[left],
        abs2_[left],
    };
    return distance(prepared, right);
}

Index::Level Index::random_level() {
    // 53 random bits, converted to a double in [0, 1).
    const double unit =
        static_cast<double>(rng_() >> 11) * (1.0 / 9007199254740992.0);
    const double sample = std::max(unit, 1e-12);
    return static_cast<Level>(-std::log(sample) * level_mult_);
}

void Index::begin_visit_epoch() const {
    ++visit_epoch_;
    if (visit_epoch_ == 0) {
        std::fill(visited_marks_.begin(), visited_marks_.end(), 0u);
        visit_epoch_ = 1;
    }
    if (visited_marks_.size() < count_) {
        visited_marks_.resize(count_, 0u);
    }
}

bool Index::mark_visited(NodeId node) const {
    if (visited_marks_[node] == visit_epoch_) {
        return false;
    }
    visited_marks_[node] = visit_epoch_;
    return true;
}

void Index::BloomFilter::reset(std::size_t expected_visits) {
    const std::size_t desired =
        std::max<std::size_t>(64, expected_visits * 10);
    std::size_t count = 64;
    while (count < desired && count < (std::size_t{1} << 24)) {
        count <<= 1;
    }

    if (bit_count != count) {
        bit_count = count;
        bits.assign(bit_count / 64, 0);
    } else {
        std::fill(bits.begin(), bits.end(), 0);
    }
    hash_count = 7;
}

bool Index::BloomFilter::insert_if_absent(NodeId value) {
    const std::uint64_t first = splitmix64(value);
    const std::uint64_t second =
        splitmix64(static_cast<std::uint64_t>(value) ^
                   0x9e3779b97f4a7c15ULL) |
        1ULL;
    const std::size_t mask = bit_count - 1;

    bool was_absent = false;
    for (std::size_t hash = 0; hash < hash_count; ++hash) {
        const std::size_t position =
            static_cast<std::size_t>(first + hash * second) & mask;
        std::uint64_t& word = bits[position >> 6];
        const std::uint64_t bit = std::uint64_t{1} << (position & 63);
        if ((word & bit) == 0) {
            was_absent = true;
            word |= bit;
        }
    }
    return was_absent;
}

Index::NodeId Index::greedy(const PreparedVector& query,
                           NodeId entry,
                           Level layer) const {
    NodeId current = entry;
    double current_distance = distance(query, current);

    while (true) {
        NodeId best = current;
        double best_distance = current_distance;

        for (const NodeId neighbor : links_[current][static_cast<std::size_t>(layer)]) {
            const double neighbor_distance = distance(query, neighbor);
            if (neighbor_distance < best_distance) {
                best = neighbor;
                best_distance = neighbor_distance;
            }
        }

        if (best == current) {
            return current;
        }
        current = best;
        current_distance = best_distance;
    }
}

std::vector<Index::Candidate> Index::search_layer(
    const PreparedVector& query,
    const std::vector<NodeId>& entry_points,
    std::size_t requested_ef,
    Level layer) const {
    const std::size_t ef = std::max<std::size_t>(1, requested_ef);
    const bool use_bloom = config_.visited_mode == VisitedMode::Bloom;
    if (use_bloom) {
        bloom_filter_.reset(std::max<std::size_t>(64, ef * 32));
    } else {
        begin_visit_epoch();
    }
    const auto first_visit = [this, use_bloom](NodeId node) {
        return use_bloom ? bloom_filter_.insert_if_absent(node)
                         : mark_visited(node);
    };

    using Scored = std::pair<double, NodeId>;
    struct ResultLess {
        bool operator()(const Scored& left, const Scored& right) const {
            if (left.first != right.first) {
                return left.first < right.first;
            }
            return left.second > right.second;
        }
    };

    std::priority_queue<Scored, std::vector<Scored>, std::greater<Scored>>
        candidates;
    std::priority_queue<Scored, std::vector<Scored>, ResultLess> results;

    for (const NodeId node : entry_points) {
        if (!first_visit(node)) {
            continue;
        }
        const double node_distance = distance(query, node);
        candidates.push({node_distance, node});
        results.push({node_distance, node});
    }

    while (results.size() > ef) {
        results.pop();
    }

    while (!candidates.empty()) {
        const Scored candidate = candidates.top();
        candidates.pop();

        if (results.size() >= ef && candidate.first > results.top().first) {
            break;
        }

        for (const NodeId neighbor :
             links_[candidate.second][static_cast<std::size_t>(layer)]) {
            if (!first_visit(neighbor)) {
                continue;
            }

            const double neighbor_distance = distance(query, neighbor);
            if (results.size() < ef ||
                neighbor_distance < results.top().first) {
                candidates.push({neighbor_distance, neighbor});
                results.push({neighbor_distance, neighbor});
                if (results.size() > ef) {
                    results.pop();
                }
            }
        }
    }

    std::vector<Candidate> ordered;
    ordered.reserve(results.size());
    while (!results.empty()) {
        ordered.push_back({results.top().first, results.top().second});
        results.pop();
    }

    std::sort(ordered.begin(), ordered.end(),
              [](const Candidate& left, const Candidate& right) {
                  if (left.distance != right.distance) {
                      return left.distance < right.distance;
                  }
                  return left.node < right.node;
              });
    return ordered;
}

std::vector<Index::NodeId> Index::select_neighbors(
    const std::vector<Candidate>& candidates,
    std::size_t limit) const {
    std::vector<NodeId> selected;
    selected.reserve(limit);

    if (!config_.heuristic) {
        for (const Candidate& candidate : candidates) {
            if (selected.size() >= limit) {
                break;
            }
            selected.push_back(candidate.node);
        }
        return selected;
    }

    for (const Candidate& candidate : candidates) {
        if (selected.size() >= limit) {
            break;
        }
        if (selected.empty()) {
            selected.push_back(candidate.node);
            continue;
        }

        bool keep = true;
        for (const NodeId other : selected) {
            if (distance(candidate.node, other) < candidate.distance) {
                keep = false;
                break;
            }
        }
        if (keep) {
            selected.push_back(candidate.node);
        }
    }

    return selected;
}

std::vector<Index::NodeId> Index::prune(NodeId node,
                                        Level layer,
                                        std::size_t limit) const {
    std::vector<Candidate> candidates;
    candidates.reserve(links_[node][static_cast<std::size_t>(layer)].size());

    for (const NodeId neighbor : links_[node][static_cast<std::size_t>(layer)]) {
        candidates.push_back({distance(node, neighbor), neighbor});
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& left, const Candidate& right) {
                  if (left.distance != right.distance) {
                      return left.distance < right.distance;
                  }
                  return left.node < right.node;
              });

    return select_neighbors(candidates, limit);
}

void Index::insert_prepared(const std::int16_t* prepared,
                            float scale,
                            float abs2,
                            std::optional<std::uint64_t> label) {
    const NodeId node = static_cast<NodeId>(count_);
    const Level level = random_level();

    data_.insert(data_.end(), prepared, prepared + config_.dim);
    scales_.push_back(scale);
    abs2_.push_back(abs2);
    labels_.push_back(label.value_or(static_cast<std::uint64_t>(node)));
    levels_.push_back(level);
    links_.emplace_back(static_cast<std::size_t>(level) + 1);
    ++count_;

    if (!entry_point_.has_value()) {
        entry_point_ = node;
        max_level_ = level;
        return;
    }

    const PreparedVector query{prepared, scale, abs2};

    NodeId entry = *entry_point_;
    for (Level layer = max_level_; layer > level; --layer) {
        entry = greedy(query, entry, layer);
    }

    std::vector<NodeId> entry_points;
    entry_points.push_back(entry);

    const Level top = std::min(level, max_level_);
    for (Level layer = top; layer >= 0; --layer) {
        const std::vector<Candidate> candidates =
            search_layer(query, entry_points, config_.ef_construction, layer);
        const std::vector<NodeId> neighbors =
            select_neighbors(candidates, config_.m);
        links_[node][static_cast<std::size_t>(layer)] = neighbors;

        const std::size_t limit =
            layer == 0 ? 2 * config_.m : config_.m;
        for (const NodeId neighbor : neighbors) {
            std::vector<NodeId>& neighbor_links =
                links_[neighbor][static_cast<std::size_t>(layer)];
            neighbor_links.push_back(node);
            if (neighbor_links.size() > limit) {
                links_[neighbor][static_cast<std::size_t>(layer)] =
                    prune(neighbor, layer, limit);
            }
        }

        entry_points.clear();
        entry_points.reserve(candidates.size());
        for (const Candidate& candidate : candidates) {
            entry_points.push_back(candidate.node);
        }
    }

    if (level > max_level_) {
        entry_point_ = node;
        max_level_ = level;
    }
}

SearchResult Index::search(const float* query, std::size_t k) const {
    if (count_ == 0) {
        throw std::runtime_error("Cannot search an empty index");
    }
    if (k < 1) {
        throw std::invalid_argument("k must be at least 1");
    }
    if (query == nullptr) {
        throw std::invalid_argument("query must not be null");
    }

    std::vector<std::int16_t> prepared(config_.dim);
    float scale = 0.0f;
    float abs2 = 0.0f;
    quantize_into(query, prepared.data(), scale, abs2);
    const PreparedVector prepared_query{prepared.data(), scale, abs2};

    NodeId entry = *entry_point_;
    for (Level layer = max_level_; layer > 0; --layer) {
        entry = greedy(prepared_query, entry, layer);
    }

    const std::size_t ef = std::max(config_.ef_search, k);
    const std::vector<Candidate> candidates =
        search_layer(prepared_query, std::vector<NodeId>{entry}, ef, 0);

    const std::size_t take = std::min(k, candidates.size());
    SearchResult result;
    result.labels.reserve(take);
    result.distances.reserve(take);
    for (std::size_t i = 0; i < take; ++i) {
        result.labels.push_back(labels_[candidates[i].node]);
        result.distances.push_back(candidates[i].distance);
    }
    return result;
}

std::vector<SearchResult> Index::search_batch(const float* queries,
                                              std::size_t count,
                                              std::size_t k) const {
    if (queries == nullptr && count > 0) {
        throw std::invalid_argument("queries must not be null");
    }

    std::vector<SearchResult> results;
    results.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        results.push_back(search(queries + i * config_.dim, k));
    }
    return results;
}

std::size_t Index::index_size_bytes() const {
    std::size_t size = data_.capacity() * sizeof(std::int16_t);
    size += scales_.capacity() * sizeof(float);
    size += abs2_.capacity() * sizeof(float);
    size += labels_.capacity() * sizeof(std::uint64_t);
    size += levels_.capacity() * sizeof(Level);
    size += links_.capacity() * sizeof(std::vector<std::vector<NodeId>>);
    for (const std::vector<std::vector<NodeId>>& node_links : links_) {
        size += node_links.capacity() * sizeof(std::vector<NodeId>);
        for (const std::vector<NodeId>& layer_links : node_links) {
            size += layer_links.capacity() * sizeof(NodeId);
        }
    }
    size += visited_marks_.capacity() * sizeof(std::uint32_t);
    return size;
}

}  // namespace hnsw
