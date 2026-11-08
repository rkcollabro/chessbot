#include "nnue.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#define NNUE_AVX2 1
#elif defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#define NNUE_SSE2 1
#endif

namespace nnue {
namespace {

constexpr int INPUTS = 768;  // [own/enemy][piece type - 1][square], squares mirrored for Black
constexpr int HIDDEN = NNUE_HIDDEN;
constexpr int QA = 255;      // feature-transformer quantization
constexpr int QB_DEFAULT = 64;  // output-layer quantization of "CBNN" files; later formats give their own
constexpr int SCALE = 400;   // network output 1.0 = 400 centipawns

constexpr int MAX_BUCKETS = 16;
constexpr int MAX_KBUCKETS = 32;

enum Activation { CRELU = 0, SCRELU = 1 };

struct Net {
    // King buckets ("CBNK"/"CBN2" files): each side's inputs get their own weights per region of that side's
    // king square (kingBucket, from that side's point of view). With mirror, a king on files e-h flips every
    // square left to right first, so only files a-d need weights.
    int kbuckets = 1;
    bool mirror = false;
    uint8_t kingBucket[64] = {};
    int activation = CRELU;
    std::vector<int16_t> ftWeights;  // [kbuckets][INPUTS][HIDDEN]
    alignas(64) int16_t ftBias[HIDDEN];
    int buckets = 1;  // output buckets, chosen by the number of pieces on the board
    int qb = QB_DEFAULT;
    std::vector<int16_t> outWeights;  // [buckets][2 * HIDDEN], side to move's half first
    int32_t outBias[MAX_BUCKETS];
};

Net g_net;
unsigned g_netGen = 1;  // bumped on every load, so rebuild caches built from older weights are discarded
bool g_loaded = false;
bool g_enabled = true;

// One side's view of its own king: the weight set to use (bucket * INPUTS) and the left-right flip (0 or 7).
struct KingView {
    int base, flip;
    bool operator!=(const KingView& o) const { return base != o.base || flip != o.flip; }
};

inline KingView kingView(int perspective, int ksq) {
    if (ksq < 0) return {0, 0};
    int k = perspective == WHITE ? ksq : ksq ^ 56;
    int flip = g_net.mirror && (k & 7) >= 4 ? 7 : 0;
    return {g_net.kingBucket[k ^ flip] * INPUTS, flip};
}

inline int feature(int perspective, const KingView& kv, int piece, int sq) {
    int c = piece >> 3, t = piece & 7;
    int s = (perspective == WHITE ? sq : sq ^ 56) ^ kv.flip;
    return kv.base + (c == perspective ? 0 : 384) + (t - 1) * 64 + s;
}

}  // namespace

bool load(const std::string& path) {
    g_loaded = false;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    // "CBNN" hidden: single output, QB 64.
    // "CBNB" hidden buckets qb: one output per piece-count bucket, output weights scaled by qb.
    // "CBNK" hidden buckets qb kbuckets mirror kingBucket[64]: CBNB plus king buckets.
    // "CBN2" hidden buckets qb kbuckets mirror kingBucket[64] activation: CBNK plus the activation (0 clipped
    //        ReLU, 1 squared clipped ReLU). The weights follow in the same order for every format.
    char magic[4];
    int32_t hidden = 0, buckets = 1, qb = QB_DEFAULT, kbuckets = 1, mirror = 0, activation = CRELU;
    uint8_t layout[64] = {};
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::fread(&hidden, 4, 1, f) == 1;
    bool v2 = ok && std::memcmp(magic, "CBN2", 4) == 0;
    bool kb = v2 || (ok && std::memcmp(magic, "CBNK", 4) == 0);
    if (ok && (kb || std::memcmp(magic, "CBNB", 4) == 0))
        ok = std::fread(&buckets, 4, 1, f) == 1 && std::fread(&qb, 4, 1, f) == 1 && qb >= 1 && qb <= 4096;
    else
        ok = ok && std::memcmp(magic, "CBNN", 4) == 0;
    if (ok && kb)
        ok = std::fread(&kbuckets, 4, 1, f) == 1 && std::fread(&mirror, 4, 1, f) == 1 &&
             std::fread(layout, 1, 64, f) == 64 && kbuckets >= 1 && kbuckets <= MAX_KBUCKETS;
    if (ok && v2) ok = std::fread(&activation, 4, 1, f) == 1 && (activation == CRELU || activation == SCRELU);
    for (int i = 0; ok && i < 64; i++) ok = layout[i] < kbuckets;
    ok = ok && hidden == HIDDEN && buckets >= 1 && buckets <= MAX_BUCKETS;
    if (ok) {
        g_net.buckets = buckets;
        g_net.qb = qb;
        g_net.kbuckets = kbuckets;
        g_net.mirror = mirror != 0;
        g_net.activation = activation;
        std::memcpy(g_net.kingBucket, layout, 64);
        g_net.ftWeights.resize(size_t(kbuckets) * INPUTS * HIDDEN);
        g_net.outWeights.resize(size_t(buckets) * 2 * HIDDEN);
        ok = std::fread(g_net.ftWeights.data(), 2, g_net.ftWeights.size(), f) == g_net.ftWeights.size() &&
             std::fread(g_net.ftBias, 2, HIDDEN, f) == HIDDEN &&
             std::fread(g_net.outWeights.data(), 2, g_net.outWeights.size(), f) == g_net.outWeights.size() &&
             std::fread(g_net.outBias, 4, buckets, f) == size_t(buckets);
    }
    std::fclose(f);
    g_netGen++;  // rebuild caches hold sums from the old weights
    g_loaded = ok;
    return ok;
}

bool loaded() { return g_loaded; }
void setEnabled(bool on) { g_enabled = on; }
bool active() { return g_loaded && g_enabled; }

namespace {

// Rebuild cache, one per thread: for each side and king view, the sums of the last position rebuilt with that
// view and the pieces it had. A rebuild starts from there and only applies the pieces that differ, which is
// usually a handful instead of every piece on the board. Without king buckets it is still a valid (if less
// useful) shortcut: the single view's entry is reused by every rebuild.
struct CacheEntry {
    alignas(64) int16_t v[HIDDEN];
    Bitboard pieces[2][7];
    unsigned gen = 0;  // network the entry was built with (0 = empty)
};
thread_local CacheEntry t_cache[2][MAX_KBUCKETS * 2];

inline void addWeights(int16_t* acc, const int16_t* w, int sign) {
#if defined(NNUE_AVX2)
    for (int i = 0; i < HIDDEN; i += 16) {
        __m256i x = _mm256_load_si256((const __m256i*)(acc + i)), y = _mm256_loadu_si256((const __m256i*)(w + i));
        _mm256_store_si256((__m256i*)(acc + i), sign > 0 ? _mm256_add_epi16(x, y) : _mm256_sub_epi16(x, y));
    }
#else
    if (sign > 0)
        for (int i = 0; i < HIDDEN; i++) acc[i] += w[i];
    else
        for (int i = 0; i < HIDDEN; i++) acc[i] -= w[i];
#endif
}

// Rebuilds perspective p's sums from the board.
void refresh(const Board& b, NnueAccum& a, int p) {
    const KingView kv = kingView(p, b.kingSq[p]);
    CacheEntry& e = t_cache[p][kv.base / INPUTS * 2 + (kv.flip ? 1 : 0)];
    if (e.gen != g_netGen) {
        std::memcpy(e.v, g_net.ftBias, sizeof(e.v));
        std::memset(e.pieces, 0, sizeof(e.pieces));
        e.gen = g_netGen;
    }
    for (int c = 0; c < 2; c++)
        for (int t = PAWN; t <= KING; t++) {
            const int piece = (c << 3) | t;
            for (Bitboard add = b.pieces[c][t] & ~e.pieces[c][t]; add;) {
                int s = popLsb(add);
                addWeights(e.v, &g_net.ftWeights[size_t(feature(p, kv, piece, s)) * HIDDEN], 1);
            }
            for (Bitboard rem = e.pieces[c][t] & ~b.pieces[c][t]; rem;) {
                int s = popLsb(rem);
                addWeights(e.v, &g_net.ftWeights[size_t(feature(p, kv, piece, s)) * HIDDEN], -1);
            }
            e.pieces[c][t] = b.pieces[c][t];
        }
    std::memcpy(a.v[p], e.v, sizeof(e.v));
    a.computed[p] = true;
}

// Whether a's move took perspective p's own king into a different bucket or mirror half, which changes every
// input of that side: p's sums are then rebuilt from the board instead of updated. The other side's are not.
bool kingViewChanged(const NnueAccum& a, int p) {
    if (g_net.kbuckets == 1 && !g_net.mirror) return false;
    const int king = (p << 3) | KING;
    for (int d = 0; d < a.ndirty; d++) {
        if (a.dirtyPiece[d] != king || a.dirtySign[d] > 0) continue;
        for (int e = 0; e < a.ndirty; e++)
            if (a.dirtyPiece[e] == king && a.dirtySign[e] > 0 && kingView(p, a.dirtySq[d]) != kingView(p, a.dirtySq[e]))
                return true;
    }
    return false;
}

// Perspective p of a = parent plus the pieces a's move added, minus the ones it removed, in one pass over the
// sums. kv is p's king view, the same for parent and a (moves that change it are rebuilt instead).
void update(const NnueAccum& parent, NnueAccum& a, int p, const KingView& kv) {
    const int16_t* w[4];
    for (int d = 0; d < a.ndirty; d++)
        w[d] = &g_net.ftWeights[size_t(feature(p, kv, a.dirtyPiece[d], a.dirtySq[d])) * HIDDEN];
    const int16_t* src = parent.v[p];
    int16_t* dst = a.v[p];
#if defined(NNUE_AVX2)
    // The common shapes get loops without per-lane sign tests: a quiet move (one added, one removed), a
    // capture (one added, two removed) and castling (two and two). int16 sums wrap the same in any order,
    // so the result is bit-identical to the general loop below.
    const int16_t *add[4], *sub[4];
    int na = 0, ns = 0;
    for (int d = 0; d < a.ndirty; d++) (a.dirtySign[d] > 0 ? add[na++] : sub[ns++]) = w[d];
    auto ld = [](const int16_t* q) { return _mm256_loadu_si256((const __m256i*)q); };
    if (na == 1 && ns == 1) {
        for (int i = 0; i < HIDDEN; i += 16)
            _mm256_storeu_si256((__m256i*)(dst + i),
                                _mm256_sub_epi16(_mm256_add_epi16(ld(src + i), ld(add[0] + i)), ld(sub[0] + i)));
    } else if (na == 1 && ns == 2) {
        for (int i = 0; i < HIDDEN; i += 16)
            _mm256_storeu_si256((__m256i*)(dst + i),
                                _mm256_sub_epi16(_mm256_sub_epi16(_mm256_add_epi16(ld(src + i), ld(add[0] + i)),
                                                                  ld(sub[0] + i)), ld(sub[1] + i)));
    } else if (na == 2 && ns == 2) {
        for (int i = 0; i < HIDDEN; i += 16) {
            __m256i x = _mm256_add_epi16(_mm256_add_epi16(ld(src + i), ld(add[0] + i)), ld(add[1] + i));
            _mm256_storeu_si256((__m256i*)(dst + i), _mm256_sub_epi16(_mm256_sub_epi16(x, ld(sub[0] + i)), ld(sub[1] + i)));
        }
    } else {
        for (int i = 0; i < HIDDEN; i += 16) {
            __m256i x = ld(src + i);
            for (int d = 0; d < na; d++) x = _mm256_add_epi16(x, ld(add[d] + i));
            for (int d = 0; d < ns; d++) x = _mm256_sub_epi16(x, ld(sub[d] + i));
            _mm256_storeu_si256((__m256i*)(dst + i), x);
        }
    }
#elif defined(NNUE_SSE2)
    for (int i = 0; i < HIDDEN; i += 8) {
        __m128i x = _mm_loadu_si128((const __m128i*)(src + i));
        for (int d = 0; d < a.ndirty; d++) {
            __m128i y = _mm_loadu_si128((const __m128i*)(w[d] + i));
            x = a.dirtySign[d] > 0 ? _mm_add_epi16(x, y) : _mm_sub_epi16(x, y);
        }
        _mm_storeu_si128((__m128i*)(dst + i), x);
    }
#else
    for (int i = 0; i < HIDDEN; i++) {
        int x = src[i];
        for (int d = 0; d < a.ndirty; d++) x += a.dirtySign[d] > 0 ? w[d][i] : -w[d][i];
        dst[i] = int16_t(x);
    }
#endif
    a.computed[p] = true;
}

// Clipped ReLU: sum of clamp(acc, 0, QA) * weight over the hidden units.
int32_t dotClamped(const int16_t* acc, const int16_t* w) {
#if defined(NNUE_AVX2)
    const __m256i zero = _mm256_setzero_si256(), qa = _mm256_set1_epi16(QA);
    __m256i sum = _mm256_setzero_si256();
    for (int i = 0; i < HIDDEN; i += 16) {
        __m256i x = _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256((const __m256i*)(acc + i)), zero), qa);
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(x, _mm256_loadu_si256((const __m256i*)(w + i))));
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
#elif defined(NNUE_SSE2)
    const __m128i zero = _mm_setzero_si128(), qa = _mm_set1_epi16(QA);
    __m128i sum = _mm_setzero_si128();
    for (int i = 0; i < HIDDEN; i += 8) {
        __m128i x = _mm_min_epi16(_mm_max_epi16(_mm_loadu_si128((const __m128i*)(acc + i)), zero), qa);
        sum = _mm_add_epi32(sum, _mm_madd_epi16(x, _mm_loadu_si128((const __m128i*)(w + i))));
    }
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1));
    return _mm_cvtsi128_si32(sum);
#else
    int32_t sum = 0;
    for (int i = 0; i < HIDDEN; i++) sum += int32_t(acc[i] < 0 ? 0 : acc[i] > QA ? QA : acc[i]) * w[i];
    return sum;
#endif
}

// Squared clipped ReLU: sum of clamp(acc, 0, QA)^2 * weight. The vector form computes (c * w) in 16 bits
// first, exact because |w| <= 127 and c <= 255 keep the product within int16, then multiplies by c again
// while summing pairs into 32 bits. The scalar form is the definition.
int32_t dotSquared(const int16_t* acc, const int16_t* w) {
#if defined(NNUE_AVX2)
    const __m256i zero = _mm256_setzero_si256(), qa = _mm256_set1_epi16(QA);
    __m256i sum = _mm256_setzero_si256();
    for (int i = 0; i < HIDDEN; i += 16) {
        __m256i c = _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256((const __m256i*)(acc + i)), zero), qa);
        __m256i cw = _mm256_mullo_epi16(c, _mm256_loadu_si256((const __m256i*)(w + i)));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(cw, c));
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
#else
    int32_t sum = 0;
    for (int i = 0; i < HIDDEN; i++) {
        int32_t c = acc[i] < 0 ? 0 : acc[i] > QA ? QA : acc[i];
        sum += int32_t(int16_t(c * w[i])) * c;
    }
    return sum;
#endif
}

}  // namespace

int evaluate(const Board& b) {
    static_assert(NNUE_HIDDEN == HIDDEN, "accumulator size");
    std::vector<NnueAccum>& st = b.nnueStack;
    const int top = int(b.undoStack.size());
    // Each side separately: walk back to the nearest position whose sums are known, then update forward; a long
    // walk costs more than a rebuild. A king move into another bucket only forces a rebuild of its own side.
    for (int p = 0; p < 2; p++) {
        if (st[top].computed[p]) continue;
        int k = top;
        while (k > 0 && !st[k].computed[p] && !st[k].refresh && !kingViewChanged(st[k], p) && top - k < 16) k--;
        if (st[k].computed[p]) {
            // No move after k changed p's king view, so every position on the way shares the current one.
            const KingView kv = kingView(p, b.kingSq[p]);
            for (int j = k + 1; j <= top; j++) update(st[j - 1], st[j], p, kv);
        } else {
            refresh(b, st[top], p);
        }
    }
    const NnueAccum& a = st[top];
#ifdef NNUE_VERIFY  // debug builds: the incremental sums must equal a rebuild from the board
    {
        static thread_local NnueAccum fresh;  // summed from scratch, without the rebuild cache
        for (int p = 0; p < 2; p++) {
            const KingView kv = kingView(p, b.kingSq[p]);
            for (int i = 0; i < HIDDEN; i++) fresh.v[p][i] = g_net.ftBias[i];
            for (Bitboard occ = b.occAll; occ;) {
                int s = popLsb(occ);
                const int16_t* w = &g_net.ftWeights[size_t(feature(p, kv, b.sq[s], s)) * HIDDEN];
                for (int i = 0; i < HIDDEN; i++) fresh.v[p][i] += w[i];
            }
        }
        if (std::memcmp(fresh.v, a.v, sizeof(a.v)) != 0) {
            std::fprintf(stderr, "NNUE accumulator mismatch at ply %d\n", top);
            std::abort();
        }
    }
#endif
    int bucket = (popcount(b.occAll) - 2) * g_net.buckets / 32;
    bucket = bucket < 0 ? 0 : bucket >= g_net.buckets ? g_net.buckets - 1 : bucket;  // same as the trainer
    const int16_t* w = &g_net.outWeights[size_t(bucket) * 2 * HIDDEN];
    int64_t out;
    if (g_net.activation == SCRELU)  // the squares carry an extra factor QA, divided out before the bias
        out = (int64_t(dotSquared(a.v[b.stm], w)) + dotSquared(a.v[b.stm ^ 1], w + HIDDEN)) / QA + g_net.outBias[bucket];
    else
        out = int64_t(dotClamped(a.v[b.stm], w)) + dotClamped(a.v[b.stm ^ 1], w + HIDDEN) + g_net.outBias[bucket];
    return int(out * SCALE / (QA * g_net.qb));
}

}  // namespace nnue
