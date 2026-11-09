#include "hnsw/hnsw.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace hnsw {

namespace {

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

// M1 cache lines are 128 bytes.
constexpr std::size_t kCacheLine = 128;

inline void prefetch_bytes(const void* p, std::size_t bytes) {
    const char* c = static_cast<const char*>(p);
    for (std::size_t off = 0; off < bytes; off += kCacheLine) {
        __builtin_prefetch(c + off, 0, 3);
    }
}

// Exact int16 dot product with int64 accumulation. Bit-identical to the
// scalar version. Each |product| <= 2^30 fits in int32, and vpadalq_s32
// widens each adjacent pair to int64 before accumulating, so no overflow.
inline std::int64_t dot_i16(const std::int16_t* a, const std::int16_t* b,
                            std::size_t n) {
    std::int64_t sum = 0;
    std::size_t i = 0;
#if defined(__ARM_NEON)
    int64x2_t acc0 = vdupq_n_s64(0);
    int64x2_t acc1 = vdupq_n_s64(0);
    for (; i + 16 <= n; i += 16) {
        const int16x8_t a0 = vld1q_s16(a + i);
        const int16x8_t b0 = vld1q_s16(b + i);
        const int16x8_t a1 = vld1q_s16(a + i + 8);
        const int16x8_t b1 = vld1q_s16(b + i + 8);
        acc0 = vpadalq_s32(acc0, vmull_s16(vget_low_s16(a0), vget_low_s16(b0)));
        acc1 = vpadalq_s32(acc1, vmull_high_s16(a0, b0));
        acc0 = vpadalq_s32(acc0, vmull_s16(vget_low_s16(a1), vget_low_s16(b1)));
        acc1 = vpadalq_s32(acc1, vmull_high_s16(a1, b1));
    }
    for (; i + 8 <= n; i += 8) {
        const int16x8_t a0 = vld1q_s16(a + i);
        const int16x8_t b0 = vld1q_s16(b + i);
        acc0 = vpadalq_s32(acc0, vmull_s16(vget_low_s16(a0), vget_low_s16(b0)));
        acc1 = vpadalq_s32(acc1, vmull_high_s16(a0, b0));
    }
    sum = vaddvq_s64(vaddq_s64(acc0, acc1));
#endif
    for (; i < n; ++i) {
        sum += static_cast<std::int64_t>(a[i]) * static_cast<std::int64_t>(b[i]);
    }
    return sum;
}

// Sum of squared differences over the first n dims of two scaled int16
// vectors, computed in float32 (used only for subdist's rough estimate).
inline double prefix_sqdist(const std::int16_t* q, float qscale,
                            const std::int16_t* d, float dscale,
                            std::size_t n) {
    std::size_t i = 0;
    double total = 0.0;
#if defined(__ARM_NEON)
    const float32x4_t vqs = vdupq_n_f32(qscale);
    const float32x4_t vds = vdupq_n_f32(dscale);
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        const int16x8_t qa = vld1q_s16(q + i);
        const int16x8_t da = vld1q_s16(d + i);

        const float32x4_t fq_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(qa)));
        const float32x4_t fd_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(da)));
        const float32x4_t diff_lo = vfmsq_f32(vmulq_f32(fq_lo, vqs), fd_lo, vds);
        acc = vfmaq_f32(acc, diff_lo, diff_lo);

        const float32x4_t fq_hi = vcvtq_f32_s32(vmovl_high_s16(qa));
        const float32x4_t fd_hi = vcvtq_f32_s32(vmovl_high_s16(da));
        const float32x4_t diff_hi = vfmsq_f32(vmulq_f32(fq_hi, vqs), fd_hi, vds);
        acc = vfmaq_f32(acc, diff_hi, diff_hi);
    }
    total = static_cast<double>(vaddvq_f32(acc));
#endif
    for (; i < n; ++i) {
        const double dq = static_cast<double>(q[i]) * static_cast<double>(qscale) -
                          static_cast<double>(d[i]) * static_cast<double>(dscale);
        total += dq * dq;
    }
    return total;
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
    if (config_.alpha < 1.0) {
        throw std::invalid_argument("alpha must be at least 1.0");
    }
    if (config_.generosity < 1.0) {
        throw std::invalid_argument("generosity must be at least 1.0");
    }
    if (config_.subdist && config_.subdist_prefix == 0) {
        throw std::invalid_argument("subdist_prefix must be positive");
    }
    if (config_.subdist_prefix > config_.dim) {
        config_.subdist_prefix = config_.dim;
    }
    if (config_.subdist && config_.subdist_max_stddev < 0.0) {
        throw std::invalid_argument("subdist_max_stddev must be non-negative");
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
    const std::size_t dim = config_.dim;
    float inverse = 1.0f;
    if (config_.space == Space::Cosine) {
        double norm_sq = 0.0;
        std::size_t i = 0;
#if defined(__ARM_NEON)
        float64x2_t s0 = vdupq_n_f64(0.0);
        float64x2_t s1 = vdupq_n_f64(0.0);
        for (; i + 4 <= dim; i += 4) {
            const float32x4_t v = vld1q_f32(vector + i);
            const float64x2_t lo = vcvt_f64_f32(vget_low_f32(v));
            const float64x2_t hi = vcvt_high_f64_f32(v);
            s0 = vfmaq_f64(s0, lo, lo);
            s1 = vfmaq_f64(s1, hi, hi);
        }
        norm_sq = vaddvq_f64(vaddq_f64(s0, s1));
#endif
        for (; i < dim; ++i) {
            const double value = static_cast<double>(vector[i]);
            norm_sq += value * value;
        }
        inverse = norm_sq == 0.0
                      ? 0.0f
                      : 1.0f / static_cast<float>(std::sqrt(norm_sq));
    }

    float max_abs = 0.0f;
    {
        std::size_t i = 0;
#if defined(__ARM_NEON)
        const float32x4_t vinv = vdupq_n_f32(inverse);
        float32x4_t vmax = vdupq_n_f32(0.0f);
        for (; i + 4 <= dim; i += 4) {
            const float32x4_t v = vld1q_f32(vector + i);
            vmax = vmaxq_f32(vmax, vabsq_f32(vmulq_f32(v, vinv)));
        }
        max_abs = vmaxvq_f32(vmax);
#endif
        for (; i < dim; ++i) {
            max_abs = std::max(max_abs, std::fabs(vector[i] * inverse));
        }
    }

    if (max_abs == 0.0f) {
        std::fill(out, out + dim, static_cast<std::int16_t>(0));
        scale = 0.0f;
        abs2 = 0.0f;
        return;
    }

    scale = max_abs / 32767.0f;
    const float factor = inverse / scale;
    {
        std::size_t i = 0;
#if defined(__ARM_NEON)
        const float32x4_t vf = vdupq_n_f32(factor);
        const int32x4_t lo_clamp = vdupq_n_s32(-32767);
        const int32x4_t hi_clamp = vdupq_n_s32(32767);
        for (; i + 8 <= dim; i += 8) {
            int32x4_t r0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(vector + i), vf));
            int32x4_t r1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(vector + i + 4), vf));
            r0 = vminq_s32(vmaxq_s32(r0, lo_clamp), hi_clamp);
            r1 = vminq_s32(vmaxq_s32(r1, lo_clamp), hi_clamp);
            vst1q_s16(out + i, vcombine_s16(vmovn_s32(r0), vmovn_s32(r1)));
        }
#endif
        for (; i < dim; ++i) {
            const long rounded = std::lround(vector[i] * factor);
            const long clamped =
                std::max<long>(-32767, std::min<long>(32767, rounded));
            out[i] = static_cast<std::int16_t>(clamped);
        }
    }

    const double sum_sq = static_cast<double>(dot_i16(out, out, dim));
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
    return static_cast<double>(dot_i16(a, b, config_.dim));
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

double Index::distance_with_threshold(const PreparedVector& query,
                                      NodeId node,
                                      double threshold) const {
    if (!config_.subdist) {
        return distance(query, node);
    }

    const std::size_t dim = config_.dim;
    const std::size_t prefix =
        std::min(config_.subdist_prefix, dim);
    const std::size_t offset = static_cast<std::size_t>(node) * dim;

    // Squared-distance prefix over the first ``prefix`` dimensions, using the
    // per-vector scales to reconstruct approximate float values.
    const double prefix_sum = prefix_sqdist(query.data, query.scale,
                                            data_.data() + offset,
                                            scales_[node], prefix);

    // Extrapolate the prefix to the full dimensionality. For cosine (unit
    // norm) the distance is half the squared L2 distance.
    const double extrapolated =
        prefix_sum * (static_cast<double>(dim) / static_cast<double>(prefix));
    const double estimated =
        (config_.space == Space::Cosine) ? 0.5 * extrapolated : extrapolated;

    // Subdist is only safe for L2 and cosine. For inner product there is no
    // bounded relationship between prefix squared distance and 1 - dot, so
    // always compute the full distance.
    const bool applicable = (config_.space == Space::L2 ||
                             config_.space == Space::Cosine);

    // Auto-detection: while sampling, always return the exact distance so
    // recall is unaffected. After enough samples, enable early rejection
    // only when the prefix/full ratio is stable.
    if (applicable && subdist_samples_seen < config_.subdist_samples) {
        const double full = distance(query, node);
        if (full > 0.0) {
            const double ratio = estimated / full;
            ++subdist_samples_seen;
            const double delta = ratio - subdist_ratio_mean;
            subdist_ratio_mean += delta / static_cast<double>(subdist_samples_seen);
            const double delta2 = ratio - subdist_ratio_mean;
            subdist_ratio_m2 += delta * delta2;
        }
        if (subdist_samples_seen >= config_.subdist_samples) {
            const double variance =
                subdist_samples_seen > 1
                    ? subdist_ratio_m2 / static_cast<double>(subdist_samples_seen - 1)
                    : 0.0;
            const double stddev = std::sqrt(std::max(0.0, variance));
            subdist_enabled = (stddev < config_.subdist_max_stddev);
        }
        return full;
    }

    if (applicable && subdist_enabled && estimated > threshold) {
        return estimated;
    }
    return distance(query, node);
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

double Index::generous_furthest(double worst, double max_distance,
                                double generosity) const {
    if (generosity <= 1.0 || max_distance <= worst) {
        return worst;
    }
    const double x = generosity - 1.0;
    const double fraction = x / (1.0 + x);
    return worst + (max_distance - worst) * fraction;
}

Index::NodeId Index::greedy(const PreparedVector& query,
                           NodeId entry,
                           Level layer) const {
    NodeId current = entry;
    double current_distance = distance(query, current);
    const std::size_t vec_bytes = config_.dim * sizeof(std::int16_t);

    while (true) {
        NodeId best = current;
        double best_distance = current_distance;

        const std::vector<NodeId>& neighbors =
            links_[current][static_cast<std::size_t>(layer)];

        // Issue all loads up front so the misses overlap.
        for (const NodeId neighbor : neighbors) {
            prefetch_bytes(data_.data() + static_cast<std::size_t>(neighbor) * config_.dim,
                           vec_bytes);
            __builtin_prefetch(&scales_[neighbor], 0, 3);
        }

        for (const NodeId neighbor : neighbors) {
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

    // Furthest distance seen within this search, starting from the entry
    // points. The persistent ``diameter_`` is only updated afterward for
    // diagnostics; it must not seed the current search or a search that
    // explored widely would raise the threshold for all later searches.
    double max_distance = 0.0;

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
        max_distance = std::max(max_distance, node_distance);
        candidates.push({node_distance, node});
        results.push({node_distance, node});
    }

    while (results.size() > ef) {
        results.pop();
    }

    const auto furthest_best = [&]() {
        if (results.empty()) {
            return std::numeric_limits<double>::max();
        }
        return generous_furthest(results.top().first, max_distance,
                                 config_.generosity);
    };

    const std::size_t vec_bytes = config_.dim * sizeof(std::int16_t);
    std::vector<NodeId> fresh;
    fresh.reserve(2 * config_.m + 1);

    while (!candidates.empty()) {
        const Scored candidate = candidates.top();
        candidates.pop();

        if (results.size() >= ef && candidate.first > furthest_best()) {
            break;
        }

        // Pass 1: filter to unvisited neighbors and prefetch their vectors,
        // so the cache misses overlap instead of serializing. The set of
        // visited nodes and their order are identical to the fused loop.
        fresh.clear();
        for (const NodeId neighbor :
             links_[candidate.second][static_cast<std::size_t>(layer)]) {
            if (!first_visit(neighbor)) {
                continue;
            }
            fresh.push_back(neighbor);
            prefetch_bytes(data_.data() + static_cast<std::size_t>(neighbor) * config_.dim,
                           vec_bytes);
            __builtin_prefetch(&scales_[neighbor], 0, 3);
            __builtin_prefetch(&abs2_[neighbor], 0, 3);
        }

        // Pass 2: score.
        for (const NodeId neighbor : fresh) {
            const double bound = furthest_best();
            const double neighbor_distance =
                distance_with_threshold(query, neighbor, bound);
            if (results.size() < ef || neighbor_distance < bound) {
                candidates.push({neighbor_distance, neighbor});
                results.push({neighbor_distance, neighbor});
                if (results.size() > ef) {
                    results.pop();
                }
            }
        }
    }

    diameter_ = std::max(diameter_, max_distance);

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
        const double threshold = std::max(
            32.0 * std::numeric_limits<double>::epsilon(),
            candidate.distance / config_.alpha);
        for (const NodeId other : selected) {
            if (distance(candidate.node, other) <= threshold) {
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

    const std::size_t vec_bytes = config_.dim * sizeof(std::int16_t);
    for (const NodeId neighbor : links_[node][static_cast<std::size_t>(layer)]) {
        prefetch_bytes(data_.data() + static_cast<std::size_t>(neighbor) * config_.dim,
                       vec_bytes);
    }
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

    // NOTE: data_ may have reallocated above, so `prepared` (which points
    // into quantization_buffer_, not data_) is still valid, but anything
    // derived from data_.data() before this point would not be.
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
