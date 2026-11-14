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

// PAIRWISE ("CBN3" files): clipped ReLU, then each perspective's two halves multiplied pairwise, feeding
// further layers instead of a single output.
enum Activation { CRELU = 0, SCRELU = 1, PAIRWISE = 2 };

constexpr int MAX_L2 = 32;  // CBN3 layer sizes: HIDDEN -> l2 -> l3 -> 1, per output bucket
constexpr int MAX_L3 = 64;

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
    // CBN3 layers (l2 > 0): l1 [buckets][l2][HIDDEN] scaled by qb over the pairwise products (side to
    // move's HIDDEN/2 first), then float biases and weights.
    int l2 = 0, l3 = 0;
    std::vector<int16_t> l1w;
    int16_t* l1wPairs = nullptr;  // l1w rearranged for the sparse loop, when l2 == 16: [bucket][HIDDEN/2][16][2]
    std::vector<float> l1b, l2w, l2b, l3w, l3b;  // [b][l2], [b][l3][l2], [b][l3], [b][l3], [b]
    std::vector<float> l2wT;                     // l2w as [b][l2][l3]
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
    // "CBN3" CBN2's header (activation 2, pairwise) l2 l3: feature weights and biases as before, then
    //        l1w int16 [buckets][l2][hidden] (|w| <= 2 * qb), and float32 l1b [buckets][l2], l2w [buckets][l3][l2],
    //        l2b [buckets][l3], l3w [buckets][l3], l3b [buckets].
    char magic[4];
    int32_t hidden = 0, buckets = 1, qb = QB_DEFAULT, kbuckets = 1, mirror = 0, activation = CRELU, l2 = 0, l3 = 0;
    uint8_t layout[64] = {};
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::fread(&hidden, 4, 1, f) == 1;
    const bool v3 = ok && std::memcmp(magic, "CBN3", 4) == 0;
    bool v2 = v3 || (ok && std::memcmp(magic, "CBN2", 4) == 0);
    bool kb = v2 || (ok && std::memcmp(magic, "CBNK", 4) == 0);
    if (ok && (kb || std::memcmp(magic, "CBNB", 4) == 0))
        ok = std::fread(&buckets, 4, 1, f) == 1 && std::fread(&qb, 4, 1, f) == 1 && qb >= 1 && qb <= 4096;
    else
        ok = ok && std::memcmp(magic, "CBNN", 4) == 0;
    if (ok && kb)
        ok = std::fread(&kbuckets, 4, 1, f) == 1 && std::fread(&mirror, 4, 1, f) == 1 &&
             std::fread(layout, 1, 64, f) == 64 && kbuckets >= 1 && kbuckets <= MAX_KBUCKETS;
    if (ok && v2)
        ok = std::fread(&activation, 4, 1, f) == 1 && (v3 ? activation == PAIRWISE : activation == CRELU || activation == SCRELU);
    if (ok && v3)
        ok = std::fread(&l2, 4, 1, f) == 1 && std::fread(&l3, 4, 1, f) == 1 && l2 >= 1 && l2 <= MAX_L2 &&
             l3 >= 1 && l3 <= MAX_L3;
    for (int i = 0; ok && i < 64; i++) ok = layout[i] < kbuckets;
    ok = ok && hidden == HIDDEN && buckets >= 1 && buckets <= MAX_BUCKETS;
    if (ok) {
        g_net.buckets = buckets;
        g_net.qb = qb;
        g_net.kbuckets = kbuckets;
        g_net.mirror = mirror != 0;
        g_net.activation = activation;
        std::memcpy(g_net.kingBucket, layout, 64);
        g_net.l2 = v3 ? l2 : 0;
        g_net.l3 = v3 ? l3 : 0;
        g_net.ftWeights.resize(size_t(kbuckets) * INPUTS * HIDDEN);
        ok = std::fread(g_net.ftWeights.data(), 2, g_net.ftWeights.size(), f) == g_net.ftWeights.size() &&
             std::fread(g_net.ftBias, 2, HIDDEN, f) == HIDDEN;
        auto readAll = [f](auto& v, size_t n) {
            v.resize(n);
            return std::fread(v.data(), sizeof(v[0]), n, f) == n;
        };
        if (v3) {
            ok = ok && readAll(g_net.l1w, size_t(buckets) * l2 * HIDDEN) && readAll(g_net.l1b, size_t(buckets) * l2) &&
                 readAll(g_net.l2w, size_t(buckets) * l3 * l2) && readAll(g_net.l2b, size_t(buckets) * l3) &&
                 readAll(g_net.l3w, size_t(buckets) * l3) && readAll(g_net.l3b, size_t(buckets));
            if (ok) {
                g_net.l2wT.assign(g_net.l2w.size(), 0.0f);
                for (int bk = 0; bk < buckets; bk++)
                    for (int k = 0; k < l3; k++)
                        for (int j = 0; j < l2; j++)
                            g_net.l2wT[(size_t(bk) * l2 + j) * l3 + k] = g_net.l2w[(size_t(bk) * l3 + k) * l2 + j];
            }
            if (ok && l2 == 16) {
                static std::vector<int16_t> pairsStore;  // over-allocated by 16 values for 32-byte alignment
                pairsStore.assign(size_t(buckets) * HIDDEN * 16 + 16, 0);
                int16_t* p = pairsStore.data();
                p += (32 - reinterpret_cast<uintptr_t>(p) % 32) % 32 / 2;
                for (int bk = 0; bk < buckets; bk++)
                    for (int i = 0; i < HIDDEN / 2; i++)
                        for (int j = 0; j < 16; j++)
                            for (int k = 0; k < 2; k++)
                                p[((size_t(bk) * (HIDDEN / 2) + i) * 16 + j) * 2 + k] =
                                    g_net.l1w[(size_t(bk) * 16 + j) * HIDDEN + 2 * i + k];
                g_net.l1wPairs = p;
            }
        } else {
            ok = ok && readAll(g_net.outWeights, size_t(buckets) * 2 * HIDDEN) &&
                 std::fread(g_net.outBias, 4, buckets, f) == size_t(buckets);
        }
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

// CBN3 first step: one perspective's sums -> HIDDEN/2 values clamp(x[i], 0, QA) * clamp(x[i + HIDDEN/2], 0, QA)
// >> PAIR_SHIFT, as a fraction of QA * QA: at most 4064, so with |l1 weight| <= 507 (1.98 at qb 256) even
// 1024 maximal products stay within int32.
constexpr int PAIR_SHIFT = 4;

void pairwise(const int16_t* acc, int16_t* out) {
    constexpr int H = HIDDEN / 2;
#if defined(NNUE_AVX2)
    const __m256i zero = _mm256_setzero_si256(), qa = _mm256_set1_epi16(QA);
    auto cl = [&](const int16_t* p) {
        return _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256((const __m256i*)p), zero), qa);
    };
    // The products reach 65025: exact as unsigned 16-bit, so a logical shift.
    for (int i = 0; i < H; i += 16)
        _mm256_storeu_si256((__m256i*)(out + i), _mm256_srli_epi16(_mm256_mullo_epi16(cl(acc + i), cl(acc + H + i)), PAIR_SHIFT));
#else
    for (int i = 0; i < H; i++) {
        int a = acc[i] < 0 ? 0 : acc[i] > QA ? QA : acc[i];
        int c = acc[H + i] < 0 ? 0 : acc[H + i] > QA ? QA : acc[H + i];
        out[i] = int16_t((a * c) >> PAIR_SHIFT);
    }
#endif
}

// Dot product of HIDDEN pairwise products with l1 weights, in int32 (see PAIR_SHIFT).
int32_t dotPairs(const int16_t* x, const int16_t* w) {
#if defined(NNUE_AVX2)
    __m256i sum = _mm256_setzero_si256();
    for (int i = 0; i < HIDDEN; i += 16)
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_loadu_si256((const __m256i*)(x + i)),
                                                      _mm256_loadu_si256((const __m256i*)(w + i))));
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
#else
    int32_t sum = 0;
    for (int i = 0; i < HIDDEN; i++) sum += int32_t(x[i]) * w[i];
    return sum;
#endif
}

inline float screlu(float x) {
    x = x < 0 ? 0 : x > 1 ? 1 : x;
    return x * x;
}

#if defined(NNUE_AVX2)
// l1 for 16 outputs over only the nonzero inputs (most pairwise products are zero), taken two at a time: each
// nonzero input pair p reads 16 outputs x 2 weights from l1wPairs ([bucket][HIDDEN/2][16][2]). Integer sums,
// so the result equals the dense dotPairs loop exactly.
// For each 8-bit mask, the positions of its set bits, in order (the rest of the 8 entries unused).
struct NzTable {
    alignas(16) uint16_t idx[256][8];
    NzTable() {
        for (int m = 0; m < 256; m++) {
            int k = 0;
            for (int b = 0; b < 8; b++)
                if (m >> b & 1) idx[m][k++] = uint16_t(b);
            while (k < 8) idx[m][k++] = 0;
        }
    }
};
const NzTable g_nzTable;

void l1Sparse(const int16_t* x, const int16_t* wp, int32_t* out) {
    const int32_t* x32 = reinterpret_cast<const int32_t*>(x);
    alignas(16) uint16_t nz[HIDDEN / 2 + 8];
    int n = 0;
    // The nonzero pair indices without branches: each block of 8 writes all 8 table entries for its mask
    // (offset by the block start) and advances by the number of set bits.
    const __m256i zero = _mm256_setzero_si256();
    __m128i base = _mm_setzero_si128();
    const __m128i eight = _mm_set1_epi16(8);
    for (int i = 0; i < HIDDEN / 2; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(x32 + i));
        unsigned m = ~unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v, zero)))) & 0xFF;
        _mm_storeu_si128((__m128i*)(nz + n), _mm_add_epi16(base, _mm_load_si128((const __m128i*)g_nzTable.idx[m])));
        n += _mm_popcnt_u32(m);
        base = _mm_add_epi16(base, eight);
    }
#ifdef NNUE_NZSTAT
    {
        static long long calls = 0, total = 0;
        calls++;
        total += n;
        if ((calls & ((1 << 20) - 1)) == 0)
            std::fprintf(stderr, "nnue: %.1f%% of l1 input pairs nonzero\n", 100.0 * total / (calls * (HIDDEN / 2)));
    }
#endif
    __m256i s0 = _mm256_setzero_si256(), s1 = _mm256_setzero_si256();
    for (int k = 0; k < n; k++) {
        const int p = nz[k];
        const __m256i in = _mm256_set1_epi32(x32[p]);
        const int16_t* w = wp + size_t(p) * 32;
        s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(in, _mm256_load_si256((const __m256i*)w)));
        s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(in, _mm256_load_si256((const __m256i*)(w + 16))));
    }
    _mm256_storeu_si256((__m256i*)out, s0);
    _mm256_storeu_si256((__m256i*)(out + 8), s1);
}
#endif

// CBN3 output for one output bucket: pairwise products of both perspectives (side to move first) -> l1
// -> SCReLU -> l2 -> SCReLU -> l3, in units of the network's output (1.0 = SCALE centipawns).
float layers(const NnueAccum& a, int stm, int bucket) {
    const Net& n = g_net;
    alignas(32) int16_t x[HIDDEN];
    pairwise(a.v[stm], x);
    pairwise(a.v[stm ^ 1], x + HIDDEN / 2);
    // x = trained product * QA * QA >> PAIR_SHIFT, and the weights are scaled by qb.
    const float s1 = float(1 << PAIR_SHIFT) / (float(QA) * QA * n.qb);
    float h1[MAX_L2], h2[MAX_L3];
    int32_t sums[MAX_L2];
#if defined(NNUE_AVX2)
    if (n.l2 == 16)
        l1Sparse(x, n.l1wPairs + size_t(bucket) * (HIDDEN / 2) * 32, sums);
    else
#endif
        for (int j = 0; j < n.l2; j++) sums[j] = dotPairs(x, &n.l1w[(size_t(bucket) * n.l2 + j) * HIDDEN]);
    for (int j = 0; j < n.l2; j++) h1[j] = screlu(float(sums[j]) * s1 + n.l1b[size_t(bucket) * n.l2 + j]);
    // l2 one input at a time across all outputs (weights stored [bucket][l2][l3] for that), so the outputs'
    // sums proceed side by side in vector registers; each still adds its terms in input order.
    const float* w2 = &n.l2wT[size_t(bucket) * n.l2 * n.l3];
    for (int k = 0; k < n.l3; k++) h2[k] = n.l2b[size_t(bucket) * n.l3 + k];
    for (int j = 0; j < n.l2; j++) {
        const float hj = h1[j];
        const float* w = w2 + size_t(j) * n.l3;
        for (int k = 0; k < n.l3; k++) h2[k] += w[k] * hj;
    }
    for (int k = 0; k < n.l3; k++) h2[k] = screlu(h2[k]);
    // l3 in eight interleaved partial sums (vectorisable), added up at the end.
    const float* w = &n.l3w[size_t(bucket) * n.l3];
    float part[8] = {};
    for (int k = 0; k < n.l3; k++) part[k & 7] += w[k] * h2[k];
    float out = n.l3b[bucket];
    for (int k = 0; k < 8; k++) out += part[k];
    return out;
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
    if (g_net.l2) return int(layers(a, b.stm, bucket) * SCALE);
    const int16_t* w = &g_net.outWeights[size_t(bucket) * 2 * HIDDEN];
    int64_t out;
    if (g_net.activation == SCRELU)  // the squares carry an extra factor QA, divided out before the bias
        out = (int64_t(dotSquared(a.v[b.stm], w)) + dotSquared(a.v[b.stm ^ 1], w + HIDDEN)) / QA + g_net.outBias[bucket];
    else
        out = int64_t(dotClamped(a.v[b.stm], w)) + dotClamped(a.v[b.stm ^ 1], w + HIDDEN) + g_net.outBias[bucket];
    return int(out * SCALE / (QA * g_net.qb));
}

}  // namespace nnue
