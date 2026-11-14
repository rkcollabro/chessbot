#include "search.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#ifdef _MSC_VER
#include <xmmintrin.h>
#endif

#include "book.h"
#include "eval.h"
#include "largepages.h"
#include "tablebase.h"

namespace {

constexpr int MAX_PLY = 128;

// ---------------------------------------------------------------- transposition table
// Shared by all search threads without locks: each entry stores its key XORed with its data,
// so an entry torn by two threads writing at once simply fails the key check.

enum : int { TT_EXACT = 0, TT_LOWER = 1, TT_UPPER = 2 };

struct TTEntry {
    std::atomic<uint64_t> keyXor{0};
    std::atomic<uint64_t> data{0};
};

struct TTData {
    Move move;
    int score, depth, flag;
    bool pv;  // the position was on a principal variation when stored (E_TTPV)
};

// The size is a power of two (the UCI "Hash" option rounds down to one); TT_DEFAULT_MB until set. Tests
// change the default at build time (-DTT_DEFAULT_MB=...) to compare table sizes without UCI options.
#ifndef TT_DEFAULT_MB
#define TT_DEFAULT_MB 64
#endif
size_t g_ttSize = (size_t(TT_DEFAULT_MB) << 20) / sizeof(TTEntry);
size_t g_ttMask = g_ttSize - 1;
bool g_ttLarge = false;
// All-zero bytes are an empty entry, which is what allocLargePages hands back.
TTEntry* g_tt = static_cast<TTEntry*>(allocLargePages(g_ttSize * sizeof(TTEntry), g_ttLarge));

inline uint64_t ttPack(Move m, int score, int depth, int flag, bool pv) {
    return uint64_t(m & 0xFFFFF) | (uint64_t(uint16_t(score + 32768)) << 20) | (uint64_t(depth & 0xFF) << 36) |
           (uint64_t(flag) << 44) | (uint64_t(pv) << 46);
}

// Start pulling a position's entry into cache as soon as its key is known, so the probe that
// follows a few hundred instructions later doesn't stall on main memory.
inline void ttPrefetch(uint64_t key) {
#ifdef _MSC_VER
    _mm_prefetch((const char*)&g_tt[key & g_ttMask], _MM_HINT_T0);
#else
    __builtin_prefetch(&g_tt[key & g_ttMask]);
#endif
}

bool ttProbe(uint64_t key, TTData& d) {
    TTEntry& e = g_tt[key & g_ttMask];
    uint64_t data = e.data.load(std::memory_order_relaxed);
    if ((e.keyXor.load(std::memory_order_relaxed) ^ data) != key || !data) return false;
    d.move = Move(data & 0xFFFFF);
    d.score = int((data >> 20) & 0xFFFF) - 32768;
    d.depth = int((data >> 36) & 0xFF);
    d.flag = int((data >> 44) & 3);
    d.pv = (data >> 46) & 1;
    return true;
}

void ttStore(uint64_t key, Move m, int score, int depth, int flag, bool pv = false) {
    TTEntry& e = g_tt[key & g_ttMask];
    uint64_t old = e.data.load(std::memory_order_relaxed);
    bool same = (e.keyXor.load(std::memory_order_relaxed) ^ old) == key;
    if (same) {
        if (int((old >> 36) & 0xFF) > depth + 2 && flag != TT_EXACT) return;  // keep deeper results
        if (!m) m = Move(old & 0xFFFFF);                                          // keep the known best move
    }
    uint64_t data = ttPack(m, score, std::max(0, depth), flag, pv);
    e.data.store(data, std::memory_order_relaxed);
    e.keyXor.store(key ^ data, std::memory_order_relaxed);
}

int scoreToTT(int s, int ply) {
    if (s > MATE_BOUND) return s + ply;
    if (s < -MATE_BOUND) return s - ply;
    return s;
}
int scoreFromTT(int s, int ply) {
    if (s > MATE_BOUND) return s - ply;
    if (s < -MATE_BOUND) return s + ply;
    return s;
}

// Tablebase value (plies to mate) -> search score at the given ply.
int tbScore(int v, int ply) {
    if (v > 0) return MATE - (ply + v);
    if (v < 0) return -(MATE - (ply + (-v - 1)));
    return 0;
}

// Search features that can be switched off at build time (-DSEARCH_FEATURES=<mask>) to test them one by one.
#ifndef SEARCH_FEATURES
#define SEARCH_FEATURES 0x1A7  // all but razoring (8), ProbCut (16) and history pruning (64), which failed their tests
#endif
constexpr int SF = SEARCH_FEATURES;
constexpr bool F_CONT = SF & 1, F_CAP = SF & 2, F_NPCORR = SF & 4, F_RAZOR = SF & 8, F_PROBCUT = SF & 16,
               F_QSFUT = SF & 32, F_HISTPRUNE = SF & 64, F_EXT = SF & 128, F_TTCAP = SF & 256;

// Search parameters, overridable at build time (-DNAME=value) so each setting can be tested against master.
// Values tuned by SPSA (2026-11-02): pass 1, 30k games, +7.2 Elo vs the hand-set values; pass 2, 20k more games,
// +8.8 Elo on top (4000-game SPRT).
// The defaults reproduce master exactly; razoring, ProbCut and history pruning only apply when enabled in
// SEARCH_FEATURES.
#ifndef RAZOR_DEPTH
#define RAZOR_DEPTH 3
#endif
#ifndef RAZOR_A
#define RAZOR_A 300  // razor margin: A + B * depth^2
#endif
#ifndef RAZOR_B
#define RAZOR_B 200
#endif
#ifndef PC_MARGIN
#define PC_MARGIN 200
#endif
#ifndef PC_DEPTH
#define PC_DEPTH 5
#endif
#ifndef HP_DEPTH
#define HP_DEPTH 4
#endif
#ifndef HP_MULT
#define HP_MULT 2000
#endif
#ifndef RFP_MARGIN
#define RFP_MARGIN 72
#endif
#ifndef FUT_BASE
#define FUT_BASE 115
#endif
#ifndef FUT_MULT
#define FUT_MULT 94
#endif
#ifndef LMR_BASE
#define LMR_BASE 95  // hundredths
#endif
#ifndef LMR_DIV
#define LMR_DIV 213  // hundredths
#endif
#ifndef NODE_TM
#define NODE_TM 0  // 1: also scale the time target by the share of nodes spent on the best root move
#endif
#ifndef NMP_EVAL
#define NMP_EVAL 200  // null-move reduction grows by one per this much eval above beta (up to 3)
#endif
#ifndef SEE_QUIET
#define SEE_QUIET 55  // SEE pruning thresholds per depth
#endif
#ifndef SEE_CAP
#define SEE_CAP 67
#endif
#ifndef DEXT_MARGIN
#define DEXT_MARGIN 30  // singular-search margin below which the TT move gets a double extension
#endif
#ifndef LMR_HIST
#define LMR_HIST 6906  // history per ply of LMR adjustment
#endif
#ifndef HIST_MULT
#define HIST_MULT 20  // history bonus: min(HIST_MULT * depth^2, HIST_MAX)
#endif
#ifndef HIST_MAX
#define HIST_MAX 1753
#endif
#ifndef ASP_DELTA
#define ASP_DELTA 22  // initial aspiration window half-width
#endif
#ifndef QS_FUT
#define QS_FUT 142  // qsearch futility margin
#endif

// Search ideas from the engines that beat us (Stockfish, Coda, PZChessBot), each off by default so the build
// matches master; each is SPRT'd on its own with -D<NAME>=<value>.
#ifndef E_CUT
#define E_CUT 0       // plies of extra LMR at expected cut nodes (cut-node tracking)
#endif
#ifndef E_CUTIIR
#define E_CUTIIR 0    // internal iterative reduction one more ply at cut nodes
#endif
#ifndef E_CUTSE
#define E_CUTSE 0     // singular search fails at a cut node: reduce the TT move by 2 plies
#endif
#ifndef E_TTPV
#define E_TTPV 0      // remember PV positions in the TT; reduce them one ply less outside the PV
#endif
#ifndef E_PAWNHIST
#define E_PAWNHIST 0  // pawn-structure history for quiet moves [pawn key][piece][to]
#endif
#ifndef E_CONT4
#define E_CONT4 0     // continuation history 4 plies back as well
#endif
#ifndef E_MINCORR
#define E_MINCORR 0   // correction history keyed by both sides' minor pieces and kings
#endif
#ifndef E_CONTCORR
#define E_CONTCORR 0  // correction history keyed by the last two moves (piece and to-square)
#endif
#ifndef E_CAPLMR
#define E_CAPLMR 0    // late move reductions for losing captures too
#endif
#ifndef E_DEEPER
#define E_DEEPER 0    // after a reduced search beats alpha, re-search one ply deeper/shallower by margin
#endif
#ifndef E_TTEVAL
#define E_TTEVAL 0    // pruning eval adjusted by a usable TT bound (RFP, null move, futility)
#endif
#ifndef E_RFPFIRM
#define E_RFPFIRM 0   // reverse futility returns (eval + beta) / 2 instead of eval
#endif
#ifndef E_IMPROVE
#define E_IMPROVE 0   // `improving` looks 4 plies back when 2 plies back was in check
#endif
#ifndef E_QMALUS
#define E_QMALUS 0    // a capture cutoff also penalises the quiet moves tried before it
#endif

// With -DTUNE the parameters become UCI options (for SPSA); otherwise they are compile-time constants.
#ifdef TUNE
std::vector<TuneParam>& tuneRegistry() {
    static std::vector<TuneParam> v;
    return v;
}
struct TuneReg {
    TuneReg(const char* n, int* p, int lo, int hi) { tuneRegistry().push_back({n, p, lo, hi}); }
};
#define TP(n, lo, hi) int P_##n = n; TuneReg reg_##n(#n, &P_##n, lo, hi);
#else
#define TP(n, lo, hi) constexpr int P_##n = n;
#endif
TP(RAZOR_A, 0, 1000)
TP(RAZOR_B, 50, 600)
TP(PC_MARGIN, 80, 400)
TP(HP_MULT, 500, 8000)
TP(RFP_MARGIN, 40, 150)
TP(FUT_BASE, 0, 250)
TP(FUT_MULT, 40, 160)
TP(LMR_BASE, 0, 150)
TP(LMR_DIV, 150, 350)
TP(NMP_EVAL, 100, 400)
TP(SEE_QUIET, 20, 120)
TP(SEE_CAP, 40, 160)
TP(DEXT_MARGIN, 0, 80)
TP(LMR_HIST, 4000, 24000)
TP(HIST_MULT, 4, 32)
TP(HIST_MAX, 600, 3200)
TP(ASP_DELTA, 10, 60)
TP(QS_FUT, 50, 300)

// Late move reduction amounts by [depth][move number].
int LMR[64][64];
void initLMR() {
    for (int d = 1; d < 64; d++)
        for (int m = 1; m < 64; m++)
            LMR[d][m] = int(P_LMR_BASE / 100.0 + std::log(d) * std::log(m) / (P_LMR_DIV / 100.0));
}
struct LmrInit {
    LmrInit() { initLMR(); }
} g_lmrInit;

// Static exchange evaluation: material outcome of the capture sequence on the target square,
// with both sides always recapturing with their least valuable piece. Works for quiet moves too
// (it then tells whether the moved piece can be won).
int see(const Board& b, Move m) {
    if (moveFlags(m) & MF_CASTLE) return 0;
    int from = moveFrom(m), to = moveTo(m);
    int gain[32], d = 0;
    int target = (moveFlags(m) & MF_EP) ? PAWN : typeOf(b.sq[to]);
    int attacker = isDrop(m) ? dropType(m) : typeOf(b.sq[from]);
    gain[0] = PIECE_VALUE[target];
    if (movePromo(m)) {
        gain[0] += PIECE_VALUE[movePromo(m)] - PIECE_VALUE[PAWN];
        attacker = movePromo(m);
    }
    // A dropped piece lands on an empty square without vacating one.
    Bitboard occ = isDrop(m) ? b.occAll | bit(to) : b.occAll ^ bit(from);
    if (moveFlags(m) & MF_EP) occ ^= bit(to + (b.stm == WHITE ? -8 : 8));
    const Bitboard diag = b.bb(0, BISHOP) | b.bb(1, BISHOP) | b.bb(0, QUEEN) | b.bb(1, QUEEN);
    const Bitboard orth = b.bb(0, ROOK) | b.bb(1, ROOK) | b.bb(0, QUEEN) | b.bb(1, QUEEN);
    Bitboard attackers = b.attackersTo(to, occ) & occ;
    int side = b.stm ^ 1;
    while (true) {
        Bitboard mine = attackers & b.occ[side];
        if (!mine) break;
        int t = PAWN;
        Bitboard pick = 0;
        for (; t <= KING; t++)
            if ((pick = mine & b.bb(side, t))) break;
        // A king may not recapture into a square the opponent still defends.
        if (t == KING && (attackers & b.occ[side ^ 1])) break;
        d++;
        gain[d] = PIECE_VALUE[attacker] - gain[d - 1];
        if (std::max(-gain[d - 1], gain[d]) < 0) break;
        occ ^= pick & (0 - pick);
        attackers |= (attacks::bishop(to, occ) & diag) | (attacks::rook(to, occ) & orth);
        attackers &= occ;
        attacker = t;
        side ^= 1;
    }
    while (d > 0) {
        gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
        d--;
    }
    return gain[0];
}

// Static exchange evaluation assumes ordinary captures that trade material: not antichess (where losing
// material is the aim) or atomic (where a capture blows up everything around it).
inline bool exchangeRules(const Board& b) { return b.variant != ANTICHESS && b.variant != ATOMIC; }

// Endgame tables and the opening book only know the standard rules.
inline bool tablebaseRules(const Board& b) { return b.variant == STANDARD || b.variant == CHESS960; }

// A move that wins on the spot by a variant rule: a third check, or the king stepping onto the hill.
// Lets the quiescence search see these wins, since it otherwise only looks at captures.
bool hasVariantWinNow(Board& b) {
    if (b.variant == THREE_CHECK && b.checks[b.stm] == 2) {
        MoveList ml;
        b.genMoves(ml);
        for (int i = 0; i < ml.count; i++) {
            if (!b.makeMove(ml.moves[i])) continue;
            bool win = b.inCheck();
            b.unmakeMove();
            if (win) return true;
        }
    } else if (b.variant == KING_OF_THE_HILL && b.kingSq[b.stm] >= 0) {
        int from = b.kingSq[b.stm];
        for (Bitboard to = attacks::king(from) & CENTER & ~b.occ[b.stm]; to;) {
            int t = popLsb(to);
            if (b.makeMove(encodeMove(from, t, 0, b.sq[t] ? MF_CAPTURE : 0))) {
                b.unmakeMove();
                return true;
            }
        }
    }
    return false;
}

// A position the variant rules have already decided: sets its score (side to move's view) and returns true.
bool variantDecided(const Board& b, int ply, int& score) {
    int o = b.variantOutcome();
    // Racing kings: White is home and Black, to move, can still get there. Black will (anything else loses),
    // so the position is a draw already.
    if (!o && b.variant == RACING_KINGS && b.stm == BLACK && rankOf(b.kingSq[WHITE]) == 7) o = OUTCOME_DRAW;
    if (!o) return false;
    score = o == OUTCOME_LOSS ? -MATE + ply : o == OUTCOME_WIN ? MATE - ply : 0;
    return true;
}

// Racing kings: a king stepping home is as forcing as a capture, so the quiescence search looks at it too.
inline bool isRaceFinish(const Board& b, Move m) {
    return b.variant == RACING_KINGS && !isCapture(m) && typeOf(b.sq[moveFrom(m)]) == KING && rankOf(moveTo(m)) == 7;
}

using Clock = std::chrono::steady_clock;

// State shared by all threads of one search.
struct Shared {
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> nodes{0};
};

class Searcher {
public:
    int id = 0;  // 0 = main thread (controls time and reports), others are helpers
    Board b;
    Shared* shared = nullptr;
    std::atomic<bool>* stopFlag = nullptr;  // external "move now" request
    Clock::time_point start;
    int limitMs = 1000;  // stop the search mid-iteration after this long
    int hardMs = 1000;   // absolute limit, used while the root move is failing low
    int softMs = 0;      // clock games: target time, scaled by how unsettled the search is (0 = fixed time)
    bool failLow = false;
    std::atomic<bool>* ponderFlag = nullptr;  // set while pondering; "ponderhit" clears it
    bool pondering = false;                   // main thread: no time limits until the ponderhit
    bool stopped = false;
    bool useBook = false;
    int iterDepth = 0;
    uint64_t nodes = 0;
    uint64_t maxNodes = 0;     // weaker levels: stop after this many nodes (single thread)
    int noise = 0;             // weaker levels: evaluation noise in centipawns
    uint64_t noiseSeed = 0;

    // Move ordering memory, kept between moves of a game (per thread).
    Move killers[MAX_PLY][2];
    Move counterMoves[16][64];  // [previous moved piece][previous to-square] -> refutation
    int history[2][64][64];     // [color][from][to], bounded by gravity to +-16384
    // Continuation history: how a quiet move has fared after a given move 1 and 2 plies earlier.
    int16_t contHist[16][64][16][64];  // [earlier piece][earlier to][piece][to]
    using ContTable = int16_t[16][64];
    int16_t capHist[16][64][8];        // capture history: [moving piece][to][captured type]
    static constexpr int PAWN_HIST_SIZE = 1024;
    int16_t pawnHist[PAWN_HIST_SIZE][16][64];  // E_PAWNHIST: [pawn key][piece][to]

    // Correction history: how far the static eval has been from the search result in positions with the same
    // pawn structure, as a running average in 1/CORR_GRAIN centipawns. Added to the static eval, so pruning
    // decisions in a structure the network misjudges lean on what deeper searches have already found there.
    static constexpr int CORR_SIZE = 16384;
    static constexpr int CORR_GRAIN = 256;
    static constexpr int CORR_WEIGHT_SCALE = 256;
    static constexpr int CORR_MAX = CORR_GRAIN * 64;  // at most 64 centipawns either way
    int pawnCorr[2][CORR_SIZE];  // [side to move][pawn key]
    int nonPawnCorr[2][2][CORR_SIZE];  // [side to move][piece color][that color's non-pawn piece key]
    int minorCorr[2][CORR_SIZE];       // E_MINCORR: [side to move][minor pieces + kings key]
    int16_t contCorr[16][64][16][64];  // E_CONTCORR: [piece][to] two plies back, [piece][to] one ply back
    int doubleExt[MAX_PLY];  // double extensions taken on the path to this ply
    uint64_t rootNodes[64][64];  // nodes spent below each root move this search, by from/to (NODE_TM)

    Move pv[MAX_PLY][MAX_PLY];
    int pvLen[MAX_PLY];
    int evalStack[MAX_PLY];  // static eval with the correction applied
    int rawEvalStack[MAX_PLY];  // static eval as the network gave it
    Move moveStack[MAX_PLY];
    int pieceStack[MAX_PLY];
    Move excluded[MAX_PLY];
    bool ttPvStack[MAX_PLY];
    Move rootBest = NO_MOVE;
    Move pvMove = NO_MOVE, pvReply = NO_MOVE;  // first two moves of the last completed iteration's PV
    int rootScore = 0;

    Searcher() { clearMemory(); }

    void clearMemory() {
        std::memset(killers, 0, sizeof(killers));
        std::memset(counterMoves, 0, sizeof(counterMoves));
        std::memset(history, 0, sizeof(history));
        std::memset(pawnCorr, 0, sizeof(pawnCorr));
        std::memset(contHist, 0, sizeof(contHist));
        std::memset(capHist, 0, sizeof(capHist));
        std::memset(nonPawnCorr, 0, sizeof(nonPawnCorr));
        std::memset(pawnHist, 0, sizeof(pawnHist));
        std::memset(minorCorr, 0, sizeof(minorCorr));
        std::memset(contCorr, 0, sizeof(contCorr));
    }

    int elapsed() const {
        return int(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count());
    }

    // On "ponderhit" the normal time limits start to apply, counted from that moment. If pondering already
    // took longer than the move's target time, move at once.
    void checkPonderhit() {
        if (!pondering || ponderFlag->load()) return;
        pondering = false;
        if (softMs && elapsed() >= softMs) shared->stop = true;
        start = Clock::now();
    }

    void checkTime() {
        shared->nodes.fetch_add(1024, std::memory_order_relaxed);
        if (id == 0) checkPonderhit();
        if (id == 0 && iterDepth > 1 &&
            ((stopFlag && stopFlag->load()) || (!pondering && elapsed() >= (failLow ? hardMs : limitMs))))
            shared->stop = true;
        if (iterDepth > 1 && shared->stop.load(std::memory_order_relaxed)) stopped = true;
    }

    // The static evaluation, blurred for the weaker levels. The noise depends only on the position and a
    // per-search seed, so a position gets the same value wherever it shows up in the tree.
    int eval() {
        int e = evaluate(b);
        if (!noise) return e;
        uint64_t z = (b.hash ^ noiseSeed) * 0x9E3779B97F4A7C15ULL;
        z ^= z >> 29;
        return e + int(z % uint64_t(2 * noise + 1)) - noise;
    }

    void updateHistory(int& h, int bonus) { h += bonus - h * std::abs(bonus) / 16384; }

    uint64_t pawnKey() const {
        uint64_t k = (b.pieces[WHITE][PAWN] * 0x9E3779B97F4A7C15ULL) ^ (b.pieces[BLACK][PAWN] * 0xC2B2AE3D27D4EB4FULL);
        return k ^ (k >> 31);
    }

    int& pawnCorrEntry() { return pawnCorr[b.stm][pawnKey() & (CORR_SIZE - 1)]; }

    uint64_t minorKey() const {
        uint64_t k = 0;
        for (int c = 0; c < 2; c++)
            for (int t : {KNIGHT, BISHOP, KING}) k = (k + b.pieces[c][t]) * 0x9E3779B97F4A7C15ULL;
        k ^= k >> 29;
        k *= 0xBF58476D1CE4E5B9ULL;
        return k ^ (k >> 32);
    }

    // E_CONTCORR entry for this node (nullptr when either of the last two plies was a null move or the root).
    int16_t* contCorrEntry(int ply) {
        if (!E_CONTCORR || ply < 2 || !moveStack[ply - 1] || !moveStack[ply - 2]) return nullptr;
        return &contCorr[pieceStack[ply - 2]][moveTo(moveStack[ply - 2])][pieceStack[ply - 1]][moveTo(moveStack[ply - 1])];
    }

    // Key of one color's non-pawn pieces (kings included), for the second and third correction tables.
    uint64_t nonPawnKey(int c) const {
        uint64_t k = 0;
        for (int t = KNIGHT; t <= KING; t++) k = (k + b.pieces[c][t]) * 0x9E3779B97F4A7C15ULL;
        k ^= k >> 31;
        k *= 0xD6E8FEB86659FD93ULL;
        return k ^ (k >> 32);
    }

    // The three correction entries for this position: pawn structure, then each color's non-pawn pieces.
    void corrEntries(int* e[3]) {
        e[0] = &pawnCorrEntry();
        e[1] = &nonPawnCorr[b.stm][WHITE][nonPawnKey(WHITE) & (CORR_SIZE - 1)];
        e[2] = &nonPawnCorr[b.stm][BLACK][nonPawnKey(BLACK) & (CORR_SIZE - 1)];
    }

    int corrected(int raw, int ply = -1) {
        int* e[3];
        corrEntries(e);
        // Every table learns the whole gap, so the weights must add up to 1: pawn 1/2, each non-pawn table 1/4.
        int c;
        if (!E_MINCORR && !E_CONTCORR) {
            c = F_NPCORR ? raw + (2 * *e[0] + *e[1] + *e[2]) / (4 * CORR_GRAIN) : raw + *e[0] / CORR_GRAIN;
        } else {  // extra tables get weight 1 each next to pawn 2 and non-pawn 1 + 1
            int sum = 2 * *e[0] + *e[1] + *e[2], n = 4;
            if (E_MINCORR) sum += minorCorr[b.stm][minorKey() & (CORR_SIZE - 1)], n++;
            if (E_CONTCORR) {
                if (int16_t* cc = ply >= 0 ? contCorrEntry(ply) : nullptr) sum += *cc;
                n++;
            }
            c = raw + sum / (n * CORR_GRAIN);
        }
        return std::max(-MATE_BOUND + 1, std::min(MATE_BOUND - 1, c));
    }

    // Move the entries toward the gap between the search result and the raw static eval, more for deeper searches.
    void updateCorrection(int raw, int score, int depth, int ply = -1) {
        int* e[3];
        corrEntries(e);
        int weight = std::min(depth + 1, 16);
        int target = (score - raw) * CORR_GRAIN;
        auto blend = [&](int v) {
            v = (v * (CORR_WEIGHT_SCALE - weight) + target * weight) / CORR_WEIGHT_SCALE;
            return std::max(-CORR_MAX, std::min(CORR_MAX, v));
        };
        for (int* c : e) *c = blend(*c);
        if (E_MINCORR) {
            int& m = minorCorr[b.stm][minorKey() & (CORR_SIZE - 1)];
            m = blend(m);
        }
        if (E_CONTCORR)
            if (int16_t* cc = ply >= 0 ? contCorrEntry(ply) : nullptr) *cc = int16_t(blend(*cc));
    }

    Move counterFor(int ply) const {
        if (ply == 0 || !moveStack[ply - 1]) return NO_MOVE;
        return counterMoves[pieceStack[ply - 1]][moveTo(moveStack[ply - 1])];
    }

    int movingPiece(Move m) const { return isDrop(m) ? makePiece(b.stm, dropType(m)) : b.sq[moveFrom(m)]; }
    int capturedType(Move m) const { return (moveFlags(m) & MF_EP) ? PAWN : typeOf(b.sq[moveTo(m)]); }

    // Continuation table for the move `back` plies before this node, or null if there was none (or a null move).
    ContTable* contFor(int ply, int back) {
        if (!F_CONT || ply < back || !moveStack[ply - back]) return nullptr;
        return &contHist[pieceStack[ply - back]][moveTo(moveStack[ply - back])];
    }

    // Combined history of a quiet move: butterfly + continuation 1 and 2 plies back.
    int quietStat(Move m, int ply) {
        int p = movingPiece(m), to = moveTo(m);
        int s = history[b.stm][moveFrom(m)][to];
        if (ContTable* c = contFor(ply, 1)) s += (*c)[p][to];
        if (ContTable* c = contFor(ply, 2)) s += (*c)[p][to];
        if (E_CONT4)
            if (ContTable* c = contFor(ply, 4)) s += (*c)[p][to];
        if (E_PAWNHIST) s += pawnHist[pawnKey() & (PAWN_HIST_SIZE - 1)][p][to];
        return s;
    }

    static void updateHist16(int16_t& h, int bonus) {
        int v = h;
        v += bonus - v * std::abs(bonus) / 16384;
        h = int16_t(v);
    }

    void updateQuiet(Move m, int ply, int bonus) {
        int p = movingPiece(m), to = moveTo(m);
        updateHistory(history[b.stm][moveFrom(m)][to], bonus);
        if (ContTable* c = contFor(ply, 1)) updateHist16((*c)[p][to], bonus);
        if (ContTable* c = contFor(ply, 2)) updateHist16((*c)[p][to], bonus);
        if (E_CONT4)
            if (ContTable* c = contFor(ply, 4)) updateHist16((*c)[p][to], bonus);
        if (E_PAWNHIST) updateHist16(pawnHist[pawnKey() & (PAWN_HIST_SIZE - 1)][p][to], bonus);
    }

    void updateCapture(Move m, int bonus) { updateHist16(capHist[movingPiece(m)][moveTo(m)][capturedType(m)], bonus); }

    int scoreMove(Move m, Move ttMove, int ply, Move counter) {
        if (m == ttMove) return 2000000;
        int promo = movePromo(m);
        if (isCapture(m)) {
            int victim = capturedType(m);
            int attacker = typeOf(b.sq[moveFrom(m)]);
            int mvvLva = PIECE_VALUE[victim] * 10 - PIECE_VALUE[attacker] / 100 + (promo == QUEEN ? 9000 : 0) +
                         (F_CAP ? capHist[movingPiece(m)][moveTo(m)][victim] / 16 : 0);
            // Captures that lose material in the exchange go after the quiet moves.
            if (attacker != PAWN && PIECE_VALUE[victim] < PIECE_VALUE[attacker] && exchangeRules(b) && see(b, m) < 0)
                return -200000 + mvvLva;
            return 1000000 + mvvLva;
        }
        if (promo == QUEEN) return 950000;
        if (promo) return -100000;  // underpromotions last
        if (m == killers[ply][0]) return 900000;
        if (m == killers[ply][1]) return 890000;
        if (m == counter) return 880000;
        return quietStat(m, ply);
    }

    // Moves the highest-scored remaining move to slot i (the first such move on ties). With AVX2 the maximum
    // is found 8 scores at a time and then the first slot holding it, which picks exactly the same move.
    static void pickMove(MoveList& ml, int* scores, int i) {
        int best = i;
#if defined(__AVX2__)
        const int n = ml.count;
        if (n - i >= 16) {
            __m256i mx = _mm256_loadu_si256((const __m256i*)(scores + i));
            int j = i + 8;
            for (; j + 8 <= n; j += 8) mx = _mm256_max_epi32(mx, _mm256_loadu_si256((const __m256i*)(scores + j)));
            __m128i m4 = _mm_max_epi32(_mm256_castsi256_si128(mx), _mm256_extracti128_si256(mx, 1));
            m4 = _mm_max_epi32(m4, _mm_shuffle_epi32(m4, 0x4E));
            m4 = _mm_max_epi32(m4, _mm_shuffle_epi32(m4, 0xB1));
            int top = _mm_cvtsi128_si32(m4);
            for (; j < n; j++) top = std::max(top, scores[j]);
            best = i;
            while (scores[best] != top) best++;
        } else
#endif
        for (int j = i + 1; j < ml.count; j++)
            if (scores[j] > scores[best]) best = j;
        std::swap(ml.moves[i], ml.moves[best]);
        std::swap(scores[i], scores[best]);
    }

    // Antichess: captures are compulsory, so while one is available there is no standing pat - every capture
    // is searched. Without one the position is quiet and the evaluation stands (with no move at all, the side
    // to move is stalemated, which wins).
    int qsearchAntichess(int alpha, int beta, int ply) {
        MoveList ml;
        b.genMoves(ml);
        if (!ml.count) return MATE - ply;
        if (!isCapture(ml.moves[0])) return corrected(eval(), ply);
        int scores[MAX_MOVES];
        for (int i = 0; i < ml.count; i++) scores[i] = scoreMove(ml.moves[i], NO_MOVE, ply, NO_MOVE);
        int best = -INF;
        for (int i = 0; i < ml.count; i++) {
            pickMove(ml, scores, i);
            if (!b.makeMove(ml.moves[i])) continue;
            int s = -qsearch(-beta, -alpha, ply + 1);
            b.unmakeMove();
            if (stopped) return 0;
            if (s > best) {
                best = s;
                if (s > alpha) alpha = s;
                if (s >= beta) break;
            }
        }
        return best;
    }

    int qsearch(int alpha, int beta, int ply) {
        pvLen[ply] = ply;
        if ((++nodes & 1023) == 0) checkTime();
        if (maxNodes && nodes >= maxNodes && iterDepth > 1) stopped = true;
        if (stopped) return 0;
        if (int vs; b.variant != STANDARD && variantDecided(b, ply, vs)) return vs;
        if (b.pieceCount <= 4 && tablebaseRules(b)) {
            int v;
            if (tb::probe(b, v)) return tbScore(v, ply);
        }
        if (ply >= MAX_PLY - 1) return eval();
        if (b.variant != STANDARD && hasVariantWinNow(b)) return MATE - ply - 1;
        if (b.variant == ANTICHESS) return qsearchAntichess(alpha, beta, ply);

        TTData tt;
        Move ttMove = NO_MOVE;
        if (ttProbe(b.hash, tt)) {
            int s = scoreFromTT(tt.score, ply);
            if (tt.flag == TT_EXACT || (tt.flag == TT_LOWER && s >= beta) || (tt.flag == TT_UPPER && s <= alpha))
                return s;
            ttMove = tt.move;
        }

        bool inCheck = b.inCheck();
        int best = -INF;
        const int origAlpha = alpha;
        if (!inCheck) {
            best = corrected(eval(), ply);
            if (best >= beta) return best;
            if (best > alpha) alpha = best;
        }
        MoveList ml;
        b.genMoves(ml, !inCheck);
        int scores[MAX_MOVES];
        for (int i = 0; i < ml.count; i++) scores[i] = scoreMove(ml.moves[i], ttMove, ply, NO_MOVE);
        if (b.variant == RACING_KINGS)
            for (int i = 0; i < ml.count; i++)
                if (isRaceFinish(b, ml.moves[i])) scores[i] = 2000000;
        int legal = 0;
        Move bestMove = NO_MOVE;
        for (int i = 0; i < ml.count; i++) {
            pickMove(ml, scores, i);
            Move m = ml.moves[i];
            if (!inCheck && isRaceFinish(b, m)) {
                // searched like a capture, whatever its history score
            } else if (!inCheck) {
                if (!isCapture(m) && !movePromo(m)) continue;  // a stale TT move may be quiet
                // Losing captures and underpromotions are sorted last: nothing useful remains.
                if (scores[i] < 0) break;
                // Delta pruning: even winning this piece cannot lift the score to alpha.
                int victim = (moveFlags(m) & MF_EP) ? PAWN : typeOf(b.sq[moveTo(m)]);
                if (!movePromo(m) && best + PIECE_VALUE[victim] + 200 < alpha && exchangeRules(b)) continue;
                // Futility: well short of alpha, only captures that win material can help.
                if (F_QSFUT && !movePromo(m) && best + P_QS_FUT <= alpha && exchangeRules(b) && see(b, m) <= 0) continue;
            }
            if (!b.makeMove(m)) continue;
            ttPrefetch(b.hash);
            legal++;
            moveStack[ply] = m;
            pieceStack[ply] = b.sq[moveTo(m)];
            int s = -qsearch(-beta, -alpha, ply + 1);
            b.unmakeMove();
            if (stopped) return 0;
            if (s > best) {
                best = s;
                if (s > alpha) {
                    alpha = s;
                    bestMove = m;
                    if (s >= beta) break;
                }
            }
        }
        if (inCheck && !legal) return -MATE + ply;
        int flag = best >= beta ? TT_LOWER : (best > origAlpha ? TT_EXACT : TT_UPPER);
        ttStore(b.hash, bestMove, scoreToTT(best, ply), 0, flag);
        return best;
    }

    int negamax(int depth, int alpha, int beta, int ply, bool nullOk, bool cutNode = false) {
        pvLen[ply] = ply;
        if ((++nodes & 1023) == 0) checkTime();
        if (maxNodes && nodes >= maxNodes && iterDepth > 1) stopped = true;
        if (stopped) return 0;
        const bool root = ply == 0;
        const bool pvNode = beta - alpha > 1;
        const Move excludedMove = excluded[ply];

        if (!root) {
            if (int vs; b.variant != STANDARD && variantDecided(b, ply, vs)) return vs;
            if (b.halfmove >= 100 || b.isRepetition() || b.insufficientMaterial()) return 0;
            if (b.pieceCount <= 4 && tablebaseRules(b)) {
                int v;
                if (tb::probe(b, v)) return tbScore(v, ply);
            }
            alpha = std::max(alpha, -MATE + ply);
            beta = std::min(beta, MATE - ply - 1);
            if (alpha >= beta) return alpha;
        }
        if (ply >= MAX_PLY - 1) return eval();

        const bool inCheck = b.inCheck();
        if (inCheck) depth++;
        if (depth <= 0) return qsearch(alpha, beta, ply);

        TTData tt;
        const bool ttHit = !excludedMove && ttProbe(b.hash, tt);
        const Move ttMove = ttHit ? tt.move : NO_MOVE;
        const int ttScore = ttHit ? scoreFromTT(tt.score, ply) : 0;
        if (!excludedMove) ttPvStack[ply] = pvNode || (ttHit && tt.pv);  // a singular search keeps its node's flag
        const bool ttPv = E_TTPV && ttPvStack[ply];
        if (ttHit && !pvNode && tt.depth >= depth &&
            (tt.flag == TT_EXACT || (tt.flag == TT_LOWER && ttScore >= beta) ||
             (tt.flag == TT_UPPER && ttScore <= alpha)))
            return ttScore;

        int staticEval, rawEval;
        if (inCheck) {
            staticEval = rawEval = -INF;
        } else if (excludedMove) {
            staticEval = evalStack[ply];  // same position, already evaluated
            rawEval = rawEvalStack[ply];
        } else {
            rawEval = eval();
            staticEval = corrected(rawEval, ply);
        }
        evalStack[ply] = staticEval;
        rawEvalStack[ply] = rawEval;
        bool improving = !inCheck && ply >= 2 && staticEval > evalStack[ply - 2];
        if (E_IMPROVE && !inCheck && ply >= 2 && evalStack[ply - 2] == -INF)
            improving = ply >= 4 && evalStack[ply - 4] != -INF && staticEval > evalStack[ply - 4];
        // Eval used for pruning decisions: the static eval, or the TT score when its bound says it is more accurate.
        int pEval = staticEval;
        if (E_TTEVAL && !inCheck && ttHit && std::abs(ttScore) < MATE_BOUND &&
            (tt.flag == TT_EXACT || (tt.flag == TT_LOWER && ttScore > staticEval) ||
             (tt.flag == TT_UPPER && ttScore < staticEval)))
            pEval = ttScore;
        if (root) doubleExt[0] = 0;
        doubleExt[ply + 1] = doubleExt[ply];
        // Razoring and ProbCut lean on the material-exchange model, so only in standard chess and Chess960.
        const bool plainChess = b.variant == STANDARD || b.variant == CHESS960;

        // Internal iterative reduction: without a known best move, search this node a bit shallower.
        if (!ttMove && depth >= 4 && !excludedMove) depth--;
        if (E_CUTIIR && cutNode && !ttMove && depth >= 6 && !excludedMove) depth--;

        if (!pvNode && !inCheck && !excludedMove) {
            // Reverse futility pruning: far above beta at low depth.
            if (depth <= 6 && std::abs(beta) < MATE_BOUND && pEval - P_RFP_MARGIN * (depth - improving) >= beta)
                return E_RFPFIRM ? (pEval + beta) / 2 : pEval;
            // Razoring: far below alpha at low depth, check that captures cannot save it.
            if (F_RAZOR && plainChess && depth <= RAZOR_DEPTH && std::abs(alpha) < MATE_BOUND &&
                staticEval + P_RAZOR_A + P_RAZOR_B * depth * depth <= alpha) {
                int s = qsearch(alpha, alpha + 1, ply);
                if (stopped) return 0;
                if (s <= alpha) return s;
            }
            // Null-move pruning.
            // (Not in antichess: zugzwang is the whole game there.)
            if (nullOk && depth >= 3 && pEval >= beta && b.hasNonPawnMaterial(b.stm) && b.variant != ANTICHESS) {
                int R = 3 + depth / 4 + std::min(3, (pEval - beta) / P_NMP_EVAL);
                moveStack[ply] = NO_MOVE;
                b.makeNull();
                ttPrefetch(b.hash);
                int s = -negamax(depth - 1 - R, -beta, -beta + 1, ply + 1, false, !cutNode);
                b.unmakeNull();
                if (stopped) return 0;
                if (s >= beta) return s >= MATE_BOUND ? beta : s;
            }
            // ProbCut: a capture that beats beta by a margin in a reduced search almost surely beats beta.
            const int pcBeta = beta + P_PC_MARGIN;
            if (F_PROBCUT && plainChess && depth >= PC_DEPTH && std::abs(beta) < MATE_BOUND &&
                !(ttHit && tt.depth >= depth - 3 && ttScore < pcBeta)) {
                MoveList pml;
                b.genMoves(pml, true);
                int psc[MAX_MOVES];
                for (int i = 0; i < pml.count; i++) psc[i] = scoreMove(pml.moves[i], ttMove, ply, NO_MOVE);
                for (int i = 0; i < pml.count; i++) {
                    pickMove(pml, psc, i);
                    Move m = pml.moves[i];
                    if (see(b, m) < pcBeta - staticEval) continue;
                    if (!b.makeMove(m)) continue;
                    ttPrefetch(b.hash);
                    moveStack[ply] = m;
                    pieceStack[ply] = b.sq[moveTo(m)];
                    int s = -qsearch(-pcBeta, -pcBeta + 1, ply + 1);
                    if (s >= pcBeta) s = -negamax(depth - 4, -pcBeta, -pcBeta + 1, ply + 1, true, !cutNode);
                    b.unmakeMove();
                    if (stopped) return 0;
                    if (s >= pcBeta) {
                        ttStore(b.hash, m, scoreToTT(s, ply), depth - 3, TT_LOWER);
                        return s;
                    }
                }
            }
        }

        MoveList ml;
        b.genMoves(ml);
        const Move counter = counterFor(ply);
        int scores[MAX_MOVES];
        for (int i = 0; i < ml.count; i++) scores[i] = scoreMove(ml.moves[i], ttMove, ply, counter);
        const bool ttCapture = ttMove && isCapture(ttMove);

        int legal = 0, best = -INF;
        Move bestMove = NO_MOVE;
        Move quiets[64], caps[32];
        int nQuiets = 0, nCaps = 0;
        const int origAlpha = alpha;
        for (int i = 0; i < ml.count; i++) {
            pickMove(ml, scores, i);
            Move m = ml.moves[i];
            ttPrefetch(b.keyAfterApprox(m));  // start the child's TT fetch before the pruning work below
            if (m == excludedMove) continue;
            const bool quiet = !isCapture(m) && !movePromo(m);
            const int stat = quiet ? quietStat(m, ply) : 0;

            // Move-level pruning once we have a move that avoids being mated.
            if (!root && legal > 0 && best > -MATE_BOUND) {
                if (quiet && !inCheck) {
                    // Late move pruning: at low depth, later quiet moves rarely matter.
                    int lmpLimit = (3 + depth * depth) / (improving ? 1 : 2);
                    if (depth <= 7 && nQuiets >= lmpLimit) continue;
                    // Futility pruning: a quiet move cannot lift a hopeless eval to alpha.
                    if (depth <= 6 && pEval + P_FUT_BASE + P_FUT_MULT * depth <= alpha) continue;
                    // History pruning: quiet moves that keep failing here, at low depth.
                    if (F_HISTPRUNE && depth <= HP_DEPTH && scores[i] < 880000 && stat < -P_HP_MULT * depth) continue;
                }
                // SEE pruning: skip moves that lose too much material at low depth.
                if (depth <= 6 && scores[i] < 900000 && exchangeRules(b) &&
                    see(b, m) < (quiet ? -P_SEE_QUIET * depth : -P_SEE_CAP * depth))
                    continue;
            }

            // Singular extension: if the TT move is much better than every alternative, search it deeper.
            int ext = 0;
            if (!root && m == ttMove && !excludedMove && depth >= 8 && tt.depth >= depth - 3 &&
                tt.flag != TT_UPPER && std::abs(ttScore) < MATE_BOUND) {
                int sBeta = ttScore - 2 * depth;
                excluded[ply] = m;
                int s = negamax((depth - 1) / 2, sBeta - 1, sBeta, ply, false, cutNode);
                excluded[ply] = NO_MOVE;
                if (stopped) return 0;
                if (s < sBeta)
                    ext = (F_EXT && !pvNode && s < sBeta - P_DEXT_MARGIN && doubleExt[ply] < 6) ? 2 : 1;  // clearly singular: twice
                else if (sBeta >= beta)
                    return sBeta;  // multi-cut: several moves beat beta
                else if (F_EXT && ttScore >= beta)
                    ext = -1;  // not singular and others look good too: the TT move needs less depth
                else if (E_CUTSE && cutNode)
                    ext = -2;  // expected cut node with good alternatives: the TT move needs less depth
            }
            doubleExt[ply + 1] = doubleExt[ply] + (ext >= 2);

            if (!b.makeMove(m)) continue;
            ttPrefetch(b.hash);
            legal++;
            if (quiet && nQuiets < 64) quiets[nQuiets++] = m;
            if (isCapture(m) && nCaps < 32) caps[nCaps++] = m;
            moveStack[ply] = m;
            pieceStack[ply] = b.sq[moveTo(m)];
            const bool givesCheck = b.inCheck();
            const int newDepth = depth - 1 + ext;

            const uint64_t nodesBefore = nodes;
            int s;
            if (legal == 1) {
                s = -negamax(newDepth, -beta, -alpha, ply + 1, true, !pvNode && !cutNode);
            } else {
                int r = 0;
                if (depth >= 3 && legal > 1 + 2 * pvNode && (quiet || (E_CAPLMR && scores[i] < 0))) {
                    r = LMR[std::min(depth, 63)][std::min(legal, 63)];
                    if (pvNode) r--;
                    if (ttPv && !pvNode) r--;
                    if (cutNode) r += E_CUT;
                    if (!improving) r++;
                    if (m == killers[ply][0] || m == killers[ply][1] || m == counter) r--;
                    if (givesCheck || inCheck) r--;
                    if (F_TTCAP && ttCapture) r++;
                    r -= stat / (F_CONT ? P_LMR_HIST : 8000);  // without continuation tables: butterfly history as before
                    r = std::max(0, std::min(r, newDepth - 1));
                }
                s = -negamax(newDepth - r, -alpha - 1, -alpha, ply + 1, true, r > 0 || !cutNode);
                if (s > alpha && r > 0) {
                    int nd = newDepth;
                    if (E_DEEPER) nd += (s > best + 40 + 2 * newDepth) - (s < best + newDepth);
                    s = -negamax(nd, -alpha - 1, -alpha, ply + 1, true, !cutNode);
                    if (E_DEEPER && nd < newDepth && s > alpha)
                        s = -negamax(newDepth, -alpha - 1, -alpha, ply + 1, true, !cutNode);
                }
                if (s > alpha && s < beta) s = -negamax(newDepth, -beta, -alpha, ply + 1, true, false);
            }
            b.unmakeMove();
            if (NODE_TM && root) rootNodes[moveFrom(m)][moveTo(m)] += nodes - nodesBefore;
            if (stopped) return 0;

            if (s > best) {
                best = s;
                bestMove = m;
                if (s > alpha) {
                    alpha = s;
                    pv[ply][ply] = m;
                    for (int k = ply + 1; k < pvLen[ply + 1]; k++) pv[ply][k] = pv[ply + 1][k];
                    pvLen[ply] = std::max(pvLen[ply + 1], ply + 1);
                    if (root) {
                        rootBest = m;
                        rootScore = s;
                    }
                    if (s >= beta) {
                        const int bonus = std::min(P_HIST_MULT * depth * depth, P_HIST_MAX);
                        if (quiet) {
                            if (killers[ply][0] != m) {
                                killers[ply][1] = killers[ply][0];
                                killers[ply][0] = m;
                            }
                            if (ply > 0 && moveStack[ply - 1])
                                counterMoves[pieceStack[ply - 1]][moveTo(moveStack[ply - 1])] = m;
                            for (int k = 0; k < nQuiets; k++) updateQuiet(quiets[k], ply, quiets[k] == m ? bonus : -bonus);
                        } else if (E_QMALUS) {
                            for (int k = 0; k < nQuiets; k++) updateQuiet(quiets[k], ply, -bonus);
                        }
                        // The cutoff capture is rewarded; captures tried before it (or before a quiet cutoff) lose.
                        for (int k = 0; k < nCaps; k++) updateCapture(caps[k], caps[k] == m ? bonus : -bonus);
                        break;
                    }
                }
            }
        }
        if (!legal) {
            if (excludedMove) return alpha;
            if (b.variant == ANTICHESS) return MATE - ply;  // stalemated: the side to move wins
            return inCheck ? -MATE + ply : 0;
        }

        if (!excludedMove) {
            int flag = best >= beta ? TT_LOWER : (best > origAlpha ? TT_EXACT : TT_UPPER);
            ttStore(b.hash, bestMove, scoreToTT(best, ply), depth, flag, E_TTPV && ttPvStack[ply]);
            // Learn from the result only when it says something about the eval: a quiet best move (a capture's
            // gain is material the eval could not see coming), a non-mate score, and a bound on the
            // informative side (a fail-high below the eval, or a fail-low above it, gives no direction).
            const bool quietBest = !bestMove || (!isCapture(bestMove) && !movePromo(bestMove));
            if (!inCheck && quietBest && std::abs(best) < MATE_BOUND && !(flag == TT_LOWER && best <= staticEval) &&
                !(flag == TT_UPPER && best >= staticEval))
                updateCorrection(rawEval, best, depth, ply);
        }
        return best;
    }

    std::string pvToSAN(const Move* moves, int n) {
        Board copy = b;
        std::string out;
        for (int i = 0; i < n; i++) {
            Move m = copy.parseUCI(copy.moveToUCI(moves[i]));
            if (!m) break;
            if (!out.empty()) out += ' ';
            out += copy.moveToSAN(m);
            copy.makeMove(m);
        }
        return out;
    }

    // With 4 or fewer pieces, pick the move with the best distance to mate straight from the tables.
    bool tablebaseRoot(Move& best, const InfoFn& info) {
        if (b.pieceCount > 4 || b.ep >= 0 || b.castle || !tablebaseRules(b)) return false;
        MoveList ml;
        b.genLegal(ml);
        if (!ml.count) return false;
        int bestScore = -INF;
        for (int i = 0; i < ml.count; i++) {
            b.makeMove(ml.moves[i]);
            int v;
            bool ok = tb::probe(b, v);
            int s = ok ? -tbScore(v, 1) : 0;
            // Prefer not to repeat in drawn positions so the game keeps making progress.
            if (ok && v == 0 && b.isRepetition()) s -= 1;
            b.unmakeMove();
            if (!ok) return false;
            if (s > bestScore) bestScore = s, best = ml.moves[i];
        }
        if (info) {
            SearchInfo si;
            si.depth = 0;
            si.score = bestScore;
            si.ms = elapsed();
            si.pv = pvToSAN(&best, 1);
            int plies = MATE - std::abs(bestScore);
            if (bestScore > MATE_BOUND)
                si.note = "Tablebase: win, mate in " + std::to_string((plies + 1) / 2);
            else if (bestScore < -MATE_BOUND)
                si.note = "Tablebase: loss, mated in " + std::to_string((plies + 1) / 2);
            else
                si.note = "Tablebase: draw";
            info(si);
        }
        return true;
    }

    void prepare(const Board& root, Shared* sh, int ms, uint64_t nodeLimit = 0, int evalNoise = 0, int soft = 0) {
        b = root;
        shared = sh;
        limitMs = hardMs = ms;
        softMs = soft;
        failLow = false;
        pondering = false;
        pvMove = pvReply = NO_MOVE;
        maxNodes = nodeLimit;
        noise = evalNoise;
        noiseSeed = uint64_t(Clock::now().time_since_epoch().count()) * 0xD1B54A32D192ED03ULL;
        stopped = false;
        nodes = 0;
        iterDepth = 0;
        start = Clock::now();
        std::memset(killers, 0, sizeof(killers));
        std::memset(excluded, 0, sizeof(excluded));
        std::memset(moveStack, 0, sizeof(moveStack));
        for (auto& side : history)
            for (auto& row : side)
                for (auto& h : row) h /= 2;
    }

    // One aspiration-window iteration. Returns false if the search was stopped.
    bool searchDepth(int depth, int& score) {
        iterDepth = depth;
        int delta = P_ASP_DELTA;
        int alpha = -INF, beta = INF;
        if (depth >= 5) {
            alpha = std::max(score - delta, -INF);
            beta = std::min(score + delta, INF);
        }
        while (true) {
            rootBest = NO_MOVE;
            int s = negamax(depth, alpha, beta, 0, false);
            if (stopped) return false;
            if (s <= alpha) {
                // The expected move is turning out worse: allow the main thread up to the hard limit.
                if (id == 0 && depth >= 5) failLow = true;
                beta = (alpha + beta) / 2;
                alpha = std::max(s - delta, -INF);
            } else if (s >= beta) {
                beta = std::min(s + delta, INF);
            } else {
                score = s;
                failLow = false;
                return true;
            }
            delta += delta;
            if (delta > 1000) alpha = -INF, beta = INF;
        }
    }

    // Helper threads search the same position to fill the shared TT; staggered depths diversify them.
    void helperLoop() {
        int score = 0;
        for (int depth = 1 + (id & 1); depth < MAX_PLY - 8 && !stopped; depth++)
            if (!searchDepth(depth, score)) break;
    }

    Move think(const InfoFn& info, const std::vector<Searcher*>& helpers) {
        Move best = NO_MOVE;
        if (tablebaseRoot(best, info)) return best;
        if (useBook && b.variant == STANDARD && book::loaded() && (best = book::probe(b))) {
            if (info) {
                SearchInfo si;
                si.pv = pvToSAN(&best, 1);
                si.note = "Opening book";
                info(si);
            }
            return best;
        }

        MoveList legal;
        b.genLegal(legal);
        if (!legal.count) return NO_MOVE;
        best = legal.moves[0];
        if (legal.count == 1) limitMs = hardMs = std::min(limitMs, 50), softMs = std::min(softMs, 50);

        std::vector<std::thread> threads;
        for (Searcher* h : helpers) threads.emplace_back([h] { h->helperLoop(); });

        int score = 0, prevScore = 0, stableIters = 0;
        double instability = 0;
        Move prevBest = NO_MOVE;
        if (NODE_TM) std::memset(rootNodes, 0, sizeof(rootNodes));
        const uint64_t nodesAtStart = nodes;
        if (softMs) limitMs = std::min(hardMs, softMs * 2);
        for (int depth = 1; depth < MAX_PLY - 8; depth++) {
            bool done = searchDepth(depth, score);
            if (rootBest) best = rootBest;  // a move fully searched this iteration is trustworthy
            if (!done) break;
            if (pvLen[0] >= 2) pvMove = pv[0][0], pvReply = pv[0][1];
            if (info) {
                SearchInfo si;
                si.depth = depth;
                si.score = score;
                si.nodes = shared->nodes.load() + nodes % 1024;
                si.ms = elapsed();
                si.pv = pvToSAN(pv[0], pvLen[0]);
                info(si);
            }
            if (maxNodes && nodes >= maxNodes) break;
            checkPonderhit();
            if (shared->stop) break;
            if (!softMs) {
                if (!pondering && elapsed() > limitMs / 2) break;
            } else {
                // Clock games: spend more time while the best move keeps changing or the score is dropping,
                // less once the same move has held for several iterations.
                bool changed = best != prevBest;
                instability = instability * 0.5 + (changed && depth >= 5 ? 1.0 : 0.0);
                stableIters = changed ? 0 : stableIters + 1;
                double scale = 1.0 + 1.5 * instability;
                if (depth >= 5) scale *= 1.0 + std::clamp((prevScore - score) / 100.0, 0.0, 1.0);
                if (stableIters >= 6) scale *= 0.7;
                // Most of the effort going into the best move means the alternatives were refuted quickly.
                if (NODE_TM && depth >= 6 && nodes > nodesAtStart) {
                    double f = double(rootNodes[moveFrom(best)][moveTo(best)]) / double(nodes - nodesAtStart);
                    scale *= std::clamp((1.5 - f) * 1.3, 0.5, 1.6);
                }
                double target = softMs * scale;
                limitMs = std::min(hardMs, int(target * 2));
                prevBest = best;
                prevScore = score;
                if (!pondering && elapsed() > target) break;
            }
            if (std::abs(score) > MATE_BOUND && depth > MATE - std::abs(score) + 2) break;
        }
        shared->stop = true;
        for (auto& t : threads) t.join();
        return best;
    }
};

}  // namespace

// Elo relative to level 6, spaced evenly at 500 per level. Self-play put levels 1-5 about 450-630 apart
// (README, "Levels"); the step to level 6 and to MAX was not measured, so these are approximate.
const EngineLevel LEVELS[NUM_LEVELS] = {
    {"Level 1", 3000, 1000, 150, false, -2500},
    {"Level 2", 3000, 4000, 60, false, -2000},
    {"Level 3", 3000, 20000, 25, true, -1500},
    {"Level 4", 3000, 120000, 0, true, -1000},
    {"Level 5", 3000, 800000, 0, true, -500},
    {"Level 6", 3000, 0, 0, true, 0},
    {"MAX", 30000, 0, 0, true, 500},
};

int defaultThreads() {
    unsigned n = std::thread::hardware_concurrency();
    return n ? int(std::min(n, 16u)) : 4;
}

namespace {

// Searchers persist between moves so their move-ordering history carries over.
std::vector<std::unique_ptr<Searcher>> g_pool;

}  // namespace

std::vector<TuneParam>& tuneParams() {
#ifdef TUNE
    return tuneRegistry();
#else
    static std::vector<TuneParam> none;
    return none;
#endif
}
void tuneChanged() { initLMR(); }

Move findBestMove(const Board& root, int timeMs, std::atomic<bool>* stop, const InfoFn& info, bool useBook,
                  int threads, int maxNodes, int evalNoise, int softMs, std::atomic<bool>* ponder,
                  Move* ponderMove) {
    threads = maxNodes ? 1 : std::max(1, std::min(threads, 64));  // node limits are per thread
    while (int(g_pool.size()) < threads) {
        g_pool.push_back(std::make_unique<Searcher>());
        g_pool.back()->id = int(g_pool.size()) - 1;
    }
    Shared shared;
    std::vector<Searcher*> helpers;
    for (int i = 0; i < threads; i++) {
        g_pool[i]->prepare(root, &shared, timeMs, uint64_t(std::max(0, maxNodes)), evalNoise, softMs);
        if (i) helpers.push_back(g_pool[i].get());
    }
    Searcher& main = *g_pool[0];
    main.stopFlag = stop;
    main.useBook = useBook;
    main.ponderFlag = ponder;
    main.pondering = ponder && ponder->load();
    Move best = main.think(info, helpers);
    if (ponderMove) *ponderMove = main.pvMove == best ? main.pvReply : NO_MOVE;
    return best;
}

int quiescenceScore(const Board& pos) {
    auto s = std::make_unique<Searcher>();
    Shared shared;
    s->prepare(pos, &shared, 1 << 30);
    return s->qsearch(-INF, INF, 0);
}

bool hashOnLargePages() { return g_ttLarge; }

void setHashSize(int mb) {
    const size_t bytes = size_t(std::max(1, mb)) << 20;
    size_t entries = 1;
    while (entries * 2 * sizeof(TTEntry) <= bytes) entries *= 2;
    if (entries == g_ttSize) return;
    // Free first: large pages need contiguous physical memory, and the old table would fragment it.
    freeLargePages(g_tt);
    g_tt = static_cast<TTEntry*>(allocLargePages(entries * sizeof(TTEntry), g_ttLarge));
    if (!g_tt) {  // out of memory: fall back to the default size
        entries = size_t(1) << 22;
        g_tt = static_cast<TTEntry*>(allocLargePages(entries * sizeof(TTEntry), g_ttLarge));
    }
    g_ttSize = entries;
    g_ttMask = entries - 1;
}

size_t hashSizeMB() { return (g_ttSize * sizeof(TTEntry)) >> 20; }

void clearSearchMemory() {
    for (size_t i = 0; i < g_ttSize; i++) {
        g_tt[i].keyXor.store(0, std::memory_order_relaxed);
        g_tt[i].data.store(0, std::memory_order_relaxed);
    }
    for (auto& s : g_pool) s->clearMemory();
}

std::string scoreToString(int score) {
    if (score > MATE_BOUND) return "M" + std::to_string((MATE - score + 1) / 2);
    if (score < -MATE_BOUND) return "-M" + std::to_string((MATE + score + 1) / 2);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%+.2f", score / 100.0);
    return buf;
}
