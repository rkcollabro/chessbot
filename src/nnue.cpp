#include "nnue.h"

#include <cstdint>
#include <cstdio>
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
constexpr int QB = 64;       // output-layer quantization
constexpr int SCALE = 400;   // network output 1.0 = 400 centipawns

struct Net {
    std::vector<int16_t> ftWeights;  // [INPUTS][HIDDEN]
    int16_t ftBias[HIDDEN];
    int16_t outWeights[2 * HIDDEN];  // side to move's half first
    int32_t outBias;
};

Net g_net;
bool g_loaded = false;
bool g_enabled = true;

inline int feature(int perspective, int piece, int sq) {
    int c = piece >> 3, t = piece & 7;
    int s = perspective == WHITE ? sq : sq ^ 56;
    return (c == perspective ? 0 : 384) + (t - 1) * 64 + s;
}

}  // namespace

bool load(const std::string& path) {
    g_loaded = false;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4];
    int32_t hidden = 0;
    g_net.ftWeights.resize(size_t(INPUTS) * HIDDEN);
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "CBNN", 4) == 0 &&
              std::fread(&hidden, 4, 1, f) == 1 && hidden == HIDDEN &&
              std::fread(g_net.ftWeights.data(), 2, g_net.ftWeights.size(), f) == g_net.ftWeights.size() &&
              std::fread(g_net.ftBias, 2, HIDDEN, f) == HIDDEN &&
              std::fread(g_net.outWeights, 2, 2 * HIDDEN, f) == 2 * HIDDEN &&
              std::fread(&g_net.outBias, 4, 1, f) == 1;
    std::fclose(f);
    g_loaded = ok;
    return ok;
}

bool loaded() { return g_loaded; }
void setEnabled(bool on) { g_enabled = on; }
bool active() { return g_loaded && g_enabled; }

namespace {

void refresh(const Board& b, NnueAccum& a) {
    for (int p = 0; p < 2; p++)
        for (int i = 0; i < HIDDEN; i++) a.v[p][i] = g_net.ftBias[i];
    for (Bitboard occ = b.occAll; occ;) {
        int s = popLsb(occ);
        int piece = b.sq[s];
        for (int p = 0; p < 2; p++) {
            const int16_t* w = &g_net.ftWeights[size_t(feature(p, piece, s)) * HIDDEN];
            for (int i = 0; i < HIDDEN; i++) a.v[p][i] += w[i];
        }
    }
    a.computed = true;
}

// a = parent plus the pieces a's move added, minus the ones it removed, done in one pass over the sums.
void update(const NnueAccum& parent, NnueAccum& a) {
    const int16_t* w[4];
    for (int p = 0; p < 2; p++) {
        for (int d = 0; d < a.ndirty; d++)
            w[d] = &g_net.ftWeights[size_t(feature(p, a.dirtyPiece[d], a.dirtySq[d])) * HIDDEN];
        const int16_t* src = parent.v[p];
        int16_t* dst = a.v[p];
#if defined(NNUE_AVX2)
        for (int i = 0; i < HIDDEN; i += 16) {
            __m256i x = _mm256_loadu_si256((const __m256i*)(src + i));
            for (int d = 0; d < a.ndirty; d++) {
                __m256i y = _mm256_loadu_si256((const __m256i*)(w[d] + i));
                x = a.dirtySign[d] > 0 ? _mm256_add_epi16(x, y) : _mm256_sub_epi16(x, y);
            }
            _mm256_storeu_si256((__m256i*)(dst + i), x);
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
    }
    a.computed = true;
}

// Sum of clamp(acc, 0, QA) * weight over the hidden units.
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

}  // namespace

int evaluate(const Board& b) {
    static_assert(NNUE_HIDDEN == HIDDEN, "accumulator size");
    std::vector<NnueAccum>& st = b.nnueStack;
    const int top = int(b.undoStack.size());
    // Walk back to the nearest position whose sums are known; a long walk costs more than a rebuild.
    int k = top;
    while (k > 0 && !st[k].computed && !st[k].refresh && top - k < 16) k--;
    if (st[k].computed) {
        for (int j = k + 1; j <= top; j++) update(st[j - 1], st[j]);
    } else {
        refresh(b, st[top]);
    }
    const NnueAccum& a = st[top];
    int64_t out = int64_t(dotClamped(a.v[b.stm], g_net.outWeights)) +
                  dotClamped(a.v[b.stm ^ 1], g_net.outWeights + HIDDEN) + g_net.outBias;
    return int(out * SCALE / (QA * QB));
}

}  // namespace nnue
