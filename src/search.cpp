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
};

constexpr size_t TT_SIZE = size_t(1) << 22;  // 64 MB
bool g_ttLarge = false;
// All-zero bytes are an empty entry, which is what allocLargePages hands back.
TTEntry* const g_tt = static_cast<TTEntry*>(allocLargePages(TT_SIZE * sizeof(TTEntry), g_ttLarge));

inline uint64_t ttPack(Move m, int score, int depth, int flag) {
    return uint64_t(m & 0xFFFFF) | (uint64_t(uint16_t(score + 32768)) << 20) | (uint64_t(depth & 0xFF) << 36) |
           (uint64_t(flag) << 44);
}

// Start pulling a position's entry into cache as soon as its key is known, so the probe that
// follows a few hundred instructions later doesn't stall on main memory.
inline void ttPrefetch(uint64_t key) {
#ifdef _MSC_VER
    _mm_prefetch((const char*)&g_tt[key & (TT_SIZE - 1)], _MM_HINT_T0);
#else
    __builtin_prefetch(&g_tt[key & (TT_SIZE - 1)]);
#endif
}

bool ttProbe(uint64_t key, TTData& d) {
    TTEntry& e = g_tt[key & (TT_SIZE - 1)];
    uint64_t data = e.data.load(std::memory_order_relaxed);
    if ((e.keyXor.load(std::memory_order_relaxed) ^ data) != key || !data) return false;
    d.move = Move(data & 0xFFFFF);
    d.score = int((data >> 20) & 0xFFFF) - 32768;
    d.depth = int((data >> 36) & 0xFF);
    d.flag = int((data >> 44) & 3);
    return true;
}

void ttStore(uint64_t key, Move m, int score, int depth, int flag) {
    TTEntry& e = g_tt[key & (TT_SIZE - 1)];
    uint64_t old = e.data.load(std::memory_order_relaxed);
    bool same = (e.keyXor.load(std::memory_order_relaxed) ^ old) == key;
    if (same) {
        if (int((old >> 36) & 0xFF) > depth + 2 && flag != TT_EXACT) return;  // keep deeper results
        if (!m) m = Move(old & 0xFFFFF);                                          // keep the known best move
    }
    uint64_t data = ttPack(m, score, std::max(0, depth), flag);
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

// Late move reduction amounts by [depth][move number].
int LMR[64][64];
struct LmrInit {
    LmrInit() {
        for (int d = 1; d < 64; d++)
            for (int m = 1; m < 64; m++) LMR[d][m] = int(0.75 + std::log(d) * std::log(m) / 2.25);
    }
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

    // Correction history: how far the static eval has been from the search result in positions with the same
    // pawn structure, as a running average in 1/CORR_GRAIN centipawns. Added to the static eval, so pruning
    // decisions in a structure the network misjudges lean on what deeper searches have already found there.
    static constexpr int CORR_SIZE = 16384;
    static constexpr int CORR_GRAIN = 256;
    static constexpr int CORR_WEIGHT_SCALE = 256;
    static constexpr int CORR_MAX = CORR_GRAIN * 64;  // at most 64 centipawns either way
    int pawnCorr[2][CORR_SIZE];  // [side to move][pawn key]

    Move pv[MAX_PLY][MAX_PLY];
    int pvLen[MAX_PLY];
    int evalStack[MAX_PLY];  // static eval with the correction applied
    int rawEvalStack[MAX_PLY];  // static eval as the network gave it
    Move moveStack[MAX_PLY];
    int pieceStack[MAX_PLY];
    Move excluded[MAX_PLY];
    Move rootBest = NO_MOVE;
    Move pvMove = NO_MOVE, pvReply = NO_MOVE;  // first two moves of the last completed iteration's PV
    int rootScore = 0;

    Searcher() { clearMemory(); }

    void clearMemory() {
        std::memset(killers, 0, sizeof(killers));
        std::memset(counterMoves, 0, sizeof(counterMoves));
        std::memset(history, 0, sizeof(history));
        std::memset(pawnCorr, 0, sizeof(pawnCorr));
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

    int& pawnCorrEntry() {
        uint64_t k = (b.pieces[WHITE][PAWN] * 0x9E3779B97F4A7C15ULL) ^ (b.pieces[BLACK][PAWN] * 0xC2B2AE3D27D4EB4FULL);
        k ^= k >> 31;
        return pawnCorr[b.stm][k & (CORR_SIZE - 1)];
    }

    int corrected(int raw) {
        int e = raw + pawnCorrEntry() / CORR_GRAIN;
        return std::max(-MATE_BOUND + 1, std::min(MATE_BOUND - 1, e));
    }

    // Move the entry toward the gap between the search result and the raw static eval, more for deeper searches.
    void updateCorrection(int raw, int score, int depth) {
        int& c = pawnCorrEntry();
        int weight = std::min(depth + 1, 16);
        int target = (score - raw) * CORR_GRAIN;
        c = (c * (CORR_WEIGHT_SCALE - weight) + target * weight) / CORR_WEIGHT_SCALE;
        c = std::max(-CORR_MAX, std::min(CORR_MAX, c));
    }

    Move counterFor(int ply) const {
        if (ply == 0 || !moveStack[ply - 1]) return NO_MOVE;
        return counterMoves[pieceStack[ply - 1]][moveTo(moveStack[ply - 1])];
    }

    int scoreMove(Move m, Move ttMove, int ply, Move counter) {
        if (m == ttMove) return 2000000;
        int promo = movePromo(m);
        if (isCapture(m)) {
            int victim = (moveFlags(m) & MF_EP) ? PAWN : typeOf(b.sq[moveTo(m)]);
            int attacker = typeOf(b.sq[moveFrom(m)]);
            int mvvLva = PIECE_VALUE[victim] * 10 - PIECE_VALUE[attacker] / 100 + (promo == QUEEN ? 9000 : 0);
            // Captures that lose material in the exchange go after the quiet moves.
            if (attacker != PAWN && PIECE_VALUE[victim] < PIECE_VALUE[attacker] && see(b, m) < 0)
                return -200000 + mvvLva;
            return 1000000 + mvvLva;
        }
        if (promo == QUEEN) return 950000;
        if (promo) return -100000;  // underpromotions last
        if (m == killers[ply][0]) return 900000;
        if (m == killers[ply][1]) return 890000;
        if (m == counter) return 880000;
        return history[b.stm][moveFrom(m)][moveTo(m)];
    }

    static void pickMove(MoveList& ml, int* scores, int i) {
        int best = i;
        for (int j = i + 1; j < ml.count; j++)
            if (scores[j] > scores[best]) best = j;
        std::swap(ml.moves[i], ml.moves[best]);
        std::swap(scores[i], scores[best]);
    }

    int qsearch(int alpha, int beta, int ply) {
        pvLen[ply] = ply;
        if ((++nodes & 1023) == 0) checkTime();
        if (maxNodes && nodes >= maxNodes && iterDepth > 1) stopped = true;
        if (stopped) return 0;
        if (b.variantOutcome()) return -MATE + ply;
        if (b.pieceCount <= 4 && tablebaseRules(b)) {
            int v;
            if (tb::probe(b, v)) return tbScore(v, ply);
        }
        if (ply >= MAX_PLY - 1) return eval();
        if (b.variant != STANDARD && hasVariantWinNow(b)) return MATE - ply - 1;

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
            best = corrected(eval());
            if (best >= beta) return best;
            if (best > alpha) alpha = best;
        }
        MoveList ml;
        b.genMoves(ml, !inCheck);
        int scores[MAX_MOVES];
        for (int i = 0; i < ml.count; i++) scores[i] = scoreMove(ml.moves[i], ttMove, ply, NO_MOVE);
        int legal = 0;
        Move bestMove = NO_MOVE;
        for (int i = 0; i < ml.count; i++) {
            pickMove(ml, scores, i);
            Move m = ml.moves[i];
            if (!inCheck) {
                if (!isCapture(m) && !movePromo(m)) continue;  // a stale TT move may be quiet
                // Losing captures and underpromotions are sorted last: nothing useful remains.
                if (scores[i] < 0) break;
                // Delta pruning: even winning this piece cannot lift the score to alpha.
                int victim = (moveFlags(m) & MF_EP) ? PAWN : typeOf(b.sq[moveTo(m)]);
                if (!movePromo(m) && best + PIECE_VALUE[victim] + 200 < alpha) continue;
            }
            if (!b.makeMove(m)) continue;
            ttPrefetch(b.hash);
            legal++;
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

    int negamax(int depth, int alpha, int beta, int ply, bool nullOk) {
        pvLen[ply] = ply;
        if ((++nodes & 1023) == 0) checkTime();
        if (maxNodes && nodes >= maxNodes && iterDepth > 1) stopped = true;
        if (stopped) return 0;
        const bool root = ply == 0;
        const bool pvNode = beta - alpha > 1;
        const Move excludedMove = excluded[ply];

        if (!root) {
            if (b.variantOutcome()) return -MATE + ply;
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
            staticEval = corrected(rawEval);
        }
        evalStack[ply] = staticEval;
        rawEvalStack[ply] = rawEval;
        const bool improving = !inCheck && ply >= 2 && staticEval > evalStack[ply - 2];

        // Internal iterative reduction: without a known best move, search this node a bit shallower.
        if (!ttMove && depth >= 4 && !excludedMove) depth--;

        if (!pvNode && !inCheck && !excludedMove) {
            // Reverse futility pruning: far above beta at low depth.
            if (depth <= 6 && std::abs(beta) < MATE_BOUND && staticEval - 85 * (depth - improving) >= beta)
                return staticEval;
            // Null-move pruning.
            if (nullOk && depth >= 3 && staticEval >= beta && b.hasNonPawnMaterial(b.stm)) {
                int R = 3 + depth / 4 + std::min(3, (staticEval - beta) / 200);
                moveStack[ply] = NO_MOVE;
                b.makeNull();
                ttPrefetch(b.hash);
                int s = -negamax(depth - 1 - R, -beta, -beta + 1, ply + 1, false);
                b.unmakeNull();
                if (stopped) return 0;
                if (s >= beta) return s >= MATE_BOUND ? beta : s;
            }
        }

        MoveList ml;
        b.genMoves(ml);
        const Move counter = counterFor(ply);
        int scores[MAX_MOVES];
        for (int i = 0; i < ml.count; i++) scores[i] = scoreMove(ml.moves[i], ttMove, ply, counter);

        int legal = 0, best = -INF;
        Move bestMove = NO_MOVE;
        Move quiets[64];
        int nQuiets = 0;
        const int origAlpha = alpha;
        for (int i = 0; i < ml.count; i++) {
            pickMove(ml, scores, i);
            Move m = ml.moves[i];
            if (m == excludedMove) continue;
            const bool quiet = !isCapture(m) && !movePromo(m);

            // Move-level pruning once we have a move that avoids being mated.
            if (!root && legal > 0 && best > -MATE_BOUND) {
                if (quiet && !inCheck) {
                    // Late move pruning: at low depth, later quiet moves rarely matter.
                    int lmpLimit = (3 + depth * depth) / (improving ? 1 : 2);
                    if (depth <= 7 && nQuiets >= lmpLimit) continue;
                    // Futility pruning: a quiet move cannot lift a hopeless eval to alpha.
                    if (depth <= 6 && staticEval + 100 + 90 * depth <= alpha) continue;
                }
                // SEE pruning: skip moves that lose too much material at low depth.
                if (depth <= 6 && scores[i] < 900000 && see(b, m) < (quiet ? -60 * depth : -90 * depth)) continue;
            }

            // Singular extension: if the TT move is much better than every alternative, search it deeper.
            int ext = 0;
            if (!root && m == ttMove && !excludedMove && depth >= 8 && tt.depth >= depth - 3 &&
                tt.flag != TT_UPPER && std::abs(ttScore) < MATE_BOUND) {
                int sBeta = ttScore - 2 * depth;
                excluded[ply] = m;
                int s = negamax((depth - 1) / 2, sBeta - 1, sBeta, ply, false);
                excluded[ply] = NO_MOVE;
                if (stopped) return 0;
                if (s < sBeta)
                    ext = 1;
                else if (sBeta >= beta)
                    return sBeta;  // multi-cut: several moves beat beta
            }

            if (!b.makeMove(m)) continue;
            ttPrefetch(b.hash);
            legal++;
            if (quiet && nQuiets < 64) quiets[nQuiets++] = m;
            moveStack[ply] = m;
            pieceStack[ply] = b.sq[moveTo(m)];
            const bool givesCheck = b.inCheck();
            const int newDepth = depth - 1 + ext;

            int s;
            if (legal == 1) {
                s = -negamax(newDepth, -beta, -alpha, ply + 1, true);
            } else {
                int r = 0;
                if (depth >= 3 && legal > 1 + 2 * pvNode && quiet) {
                    r = LMR[std::min(depth, 63)][std::min(legal, 63)];
                    if (pvNode) r--;
                    if (!improving) r++;
                    if (m == killers[ply][0] || m == killers[ply][1] || m == counter) r--;
                    if (givesCheck || inCheck) r--;
                    r -= history[b.stm ^ 1][moveFrom(m)][moveTo(m)] / 8000;
                    r = std::max(0, std::min(r, newDepth - 1));
                }
                s = -negamax(newDepth - r, -alpha - 1, -alpha, ply + 1, true);
                if (s > alpha && r > 0) s = -negamax(newDepth, -alpha - 1, -alpha, ply + 1, true);
                if (s > alpha && s < beta) s = -negamax(newDepth, -beta, -alpha, ply + 1, true);
            }
            b.unmakeMove();
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
                        if (quiet) {
                            if (killers[ply][0] != m) {
                                killers[ply][1] = killers[ply][0];
                                killers[ply][0] = m;
                            }
                            if (ply > 0 && moveStack[ply - 1])
                                counterMoves[pieceStack[ply - 1]][moveTo(moveStack[ply - 1])] = m;
                            int bonus = std::min(16 * depth * depth, 1600);
                            for (int k = 0; k < nQuiets; k++) {
                                Move q = quiets[k];
                                updateHistory(history[b.stm][moveFrom(q)][moveTo(q)], q == m ? bonus : -bonus);
                            }
                        }
                        break;
                    }
                }
            }
        }
        if (!legal) {
            if (excludedMove) return alpha;
            return inCheck ? -MATE + ply : 0;
        }

        if (!excludedMove) {
            int flag = best >= beta ? TT_LOWER : (best > origAlpha ? TT_EXACT : TT_UPPER);
            ttStore(b.hash, bestMove, scoreToTT(best, ply), depth, flag);
            // Learn from the result only when it says something about the eval: a quiet best move (a capture's
            // gain is material the eval could not see coming), a non-mate score, and a bound on the
            // informative side (a fail-high below the eval, or a fail-low above it, gives no direction).
            const bool quietBest = !bestMove || (!isCapture(bestMove) && !movePromo(bestMove));
            if (!inCheck && quietBest && std::abs(best) < MATE_BOUND && !(flag == TT_LOWER && best <= staticEval) &&
                !(flag == TT_UPPER && best >= staticEval))
                updateCorrection(rawEval, best, depth);
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
        int delta = 25;
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

void clearSearchMemory() {
    for (size_t i = 0; i < TT_SIZE; i++) {
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
