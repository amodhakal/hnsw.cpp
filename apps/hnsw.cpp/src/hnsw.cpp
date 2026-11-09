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
    if (prepare_buffer_.size() < config_.dim) {
        prepare_buffer_.resize(config_.dim);
    }
    prepare_into(vector, prepare_buffer_.data());
    insert_prepared(prepare_buffer_.data(), label);
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

void Index::prepare_into(const float* vector, float* out) const {
    std::memcpy(out, vector, config_.dim * sizeof(float));
    if (config_.space != Space::Cosine) {
        return;
    }

    double norm_sq = 0.0;
    for (std::size_t i = 0; i < config_.dim; ++i) {
        const double value = static_cast<double>(vector[i]);
        norm_sq += value * value;
    }
    if (norm_sq == 0.0) {
        std::fill(out, out + config_.dim, 0.0f);
        return;
    }

    const float inverse = 1.0f / static_cast<float>(std::sqrt(norm_sq));
    for (std::size_t i = 0; i < config_.dim; ++i) {
        out[i] *= inverse;
    }
}

float Index::dot(const float* a, const float* b) const {
    float sum = 0.0f;
    for (std::size_t i = 0; i < config_.dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double Index::distance(const float* a, const float* b) const {
    switch (config_.space) {
        case Space::Cosine:
        case Space::InnerProduct:
            return 1.0 - static_cast<double>(dot(a, b));
        case Space::L2: {
            float sum = 0.0f;
            for (std::size_t i = 0; i < config_.dim; ++i) {
                const float difference = a[i] - b[i];
                sum += difference * difference;
            }
            return static_cast<double>(sum);
        }
    }
    return 0.0;
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

Index::NodeId Index::greedy(const float* query, NodeId entry, Level layer) const {
    NodeId current = entry;
    double current_distance =
        distance(query, data_.data() + static_cast<std::size_t>(current) * config_.dim);

    while (true) {
        NodeId best = current;
        double best_distance = current_distance;

        for (const NodeId neighbor : links_[current][static_cast<std::size_t>(layer)]) {
            const double neighbor_distance =
                distance(query,
                         data_.data() +
                             static_cast<std::size_t>(neighbor) * config_.dim);
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
    const float* query,
    const std::vector<NodeId>& entry_points,
    std::size_t requested_ef,
    Level layer) const {
    const std::size_t ef = std::max<std::size_t>(1, requested_ef);
    begin_visit_epoch();

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
        if (!mark_visited(node)) {
            continue;
        }
        const double node_distance =
            distance(query,
                     data_.data() + static_cast<std::size_t>(node) * config_.dim);
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
            if (!mark_visited(neighbor)) {
                continue;
            }

            const double neighbor_distance =
                distance(query,
                         data_.data() +
                             static_cast<std::size_t>(neighbor) * config_.dim);
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

        const float* candidate_vector =
            data_.data() + static_cast<std::size_t>(candidate.node) * config_.dim;
        bool keep = true;
        for (const NodeId other : selected) {
            const float* other_vector =
                data_.data() + static_cast<std::size_t>(other) * config_.dim;
            if (distance(candidate_vector, other_vector) < candidate.distance) {
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
    const float* base =
        data_.data() + static_cast<std::size_t>(node) * config_.dim;
    std::vector<Candidate> candidates;
    candidates.reserve(links_[node][static_cast<std::size_t>(layer)].size());

    for (const NodeId neighbor : links_[node][static_cast<std::size_t>(layer)]) {
        const double neighbor_distance =
            distance(base,
                     data_.data() +
                         static_cast<std::size_t>(neighbor) * config_.dim);
        candidates.push_back({neighbor_distance, neighbor});
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

void Index::insert_prepared(const float* prepared,
                            std::optional<std::uint64_t> label) {
    const NodeId node = static_cast<NodeId>(count_);
    const Level level = random_level();

    data_.insert(data_.end(), prepared, prepared + config_.dim);
    labels_.push_back(label.value_or(static_cast<std::uint64_t>(node)));
    levels_.push_back(level);
    links_.emplace_back(static_cast<std::size_t>(level) + 1);
    ++count_;

    if (!entry_point_.has_value()) {
        entry_point_ = node;
        max_level_ = level;
        return;
    }

    NodeId entry = *entry_point_;
    for (Level layer = max_level_; layer > level; --layer) {
        entry = greedy(prepared, entry, layer);
    }

    std::vector<NodeId> entry_points;
    entry_points.push_back(entry);

    const Level top = std::min(level, max_level_);
    for (Level layer = top; layer >= 0; --layer) {
        const std::vector<Candidate> candidates =
            search_layer(prepared, entry_points, config_.ef_construction, layer);
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

    std::vector<float> prepared(config_.dim);
    prepare_into(query, prepared.data());

    NodeId entry = *entry_point_;
    for (Level layer = max_level_; layer > 0; --layer) {
        entry = greedy(prepared.data(), entry, layer);
    }

    const std::size_t ef = std::max(config_.ef_search, k);
    const std::vector<Candidate> candidates =
        search_layer(prepared.data(), std::vector<NodeId>{entry}, ef, 0);

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
    std::size_t size = data_.capacity() * sizeof(float);
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
