#include "eval.h"

#include "nnue.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

Score PARAMS[NUM_PARAMS];

namespace {

#if __has_include("eval_weights.inc")
const Score TUNED[NUM_PARAMS] = {
#include "eval_weights.inc"
};
constexpr bool HAVE_TUNED = true;
#else
const Score TUNED[1] = {};
constexpr bool HAVE_TUNED = false;
#endif

Bitboard PASSED_MASK[2][64];  // squares ahead on the same and adjacent files
Bitboard ADJ_FILES[8];

struct Init {
    Init() {
        initTables();
        for (int f = 0; f < 8; f++)
            ADJ_FILES[f] = (f > 0 ? fileMask(f - 1) : 0) | (f < 7 ? fileMask(f + 1) : 0);
        for (int s = 0; s < 64; s++) {
            Bitboard files = fileMask(fileOf(s)) | ADJ_FILES[fileOf(s)];
            Bitboard ahead = 0, behind = 0;
            for (int r = rankOf(s) + 1; r < 8; r++) ahead |= rankMask(r);
            for (int r = rankOf(s) - 1; r >= 0; r--) behind |= rankMask(r);
            PASSED_MASK[WHITE][s] = files & ahead;
            PASSED_MASK[BLACK][s] = files & behind;
        }
        if (HAVE_TUNED)
            std::memcpy(PARAMS, TUNED, sizeof(PARAMS));
        else
            resetDefaultParams();
    }
} g_init;

// Accumulates the weighted score.
struct ScoreAcc {
    int mg = 0, eg = 0;
    void add(int idx, int color, int n) {
        if (color == BLACK) n = -n;
        mg += PARAMS[idx].mg * n;
        eg += PARAMS[idx].eg * n;
    }
};

// Records feature counts for the tuner.
struct TraceAcc {
    int16_t counts[NUM_PARAMS] = {};
    void add(int idx, int color, int n) { counts[idx] += int16_t(color == WHITE ? n : -n); }
};

inline int relSquare(int c, int s) { return c == WHITE ? s : s ^ 56; }

template <class Acc>
int evalCore(const Board& b, Acc& acc) {
    const Bitboard occ = b.occAll;
    Bitboard pawnAtt[2];
    {
        Bitboard wp = b.bb(WHITE, PAWN), bp = b.bb(BLACK, PAWN);
        const Bitboard notA = ~FILE_A, notH = ~fileMask(7);
        pawnAtt[WHITE] = ((wp & notA) << 7) | ((wp & notH) << 9);
        pawnAtt[BLACK] = ((bp & notA) >> 9) | ((bp & notH) >> 7);
    }

    // First pass: piece attacks for both sides, needed for "safe" squares.
    struct PieceAtt {
        int t, s;
        Bitboard a;
    };
    PieceAtt list[2][32];
    int n[2] = {0, 0};
    Bitboard attackedBy[2];
    int phase = 0;
    for (int c = 0; c < 2; c++) {
        attackedBy[c] = pawnAtt[c] | (b.kingSq[c] >= 0 ? attacks::king(b.kingSq[c]) : 0);
        for (int t = KNIGHT; t <= QUEEN; t++)
            for (Bitboard bb = b.bb(c, t); bb;) {
                int s = popLsb(bb);
                Bitboard a = t == KNIGHT   ? attacks::knight(s)
                             : t == BISHOP ? attacks::bishop(s, occ)
                             : t == ROOK   ? attacks::rook(s, occ)
                                           : attacks::queen(s, occ);
                list[c][n[c]++] = {t, s, a};
                attackedBy[c] |= a;
                phase += t == QUEEN ? 4 : t == ROOK ? 2 : 1;
            }
    }

    for (int c = 0; c < 2; c++) {
        const int e = c ^ 1;
        const int ek = b.kingSq[e];
        const Bitboard ownPawns = b.bb(c, PAWN), enemyPawns = b.bb(e, PAWN);

        // Material and piece-square tables.
        for (int t = PAWN; t <= KING; t++)
            for (Bitboard bb = b.bb(c, t); bb;) {
                int s = popLsb(bb);
                if (t != KING) acc.add(OFF_MATERIAL + t, c, 1);
                acc.add(OFF_PST + t * 64 + relSquare(c, s), c, 1);
            }

        // Mobility, king attacks, safe checks.
        const Bitboard mobArea = ~b.occ[c] & ~pawnAtt[e];
        const bool hasKing = ek >= 0;
        const Bitboard kingZone = hasKing ? attacks::king(ek) | bit(ek) : 0;
        const Bitboard safe = ~b.occ[c] & ~attackedBy[e];
        const Bitboard checkB = hasKing ? attacks::bishop(ek, occ) : 0, checkR = hasKing ? attacks::rook(ek, occ) : 0;
        const Bitboard checkSq[6] = {0, 0, hasKing ? attacks::knight(ek) : 0, checkB, checkR, checkB | checkR};
        static const int MOB_OFF[6] = {0, 0, OFF_MOB_N, OFF_MOB_B, OFF_MOB_R, OFF_MOB_Q};
        for (int i = 0; i < n[c]; i++) {
            const PieceAtt& p = list[c][i];
            acc.add(MOB_OFF[p.t] + popcount(p.a & mobArea), c, 1);
            if (int k = popcount(p.a & kingZone)) acc.add(OFF_KING_ATTACK + p.t, c, k);
            if (int k = popcount(p.a & checkSq[p.t] & safe)) acc.add(OFF_SAFE_CHECK + p.t, c, k);
        }

        // Pawn structure.
        for (Bitboard bb = ownPawns; bb;) {
            int s = popLsb(bb);
            if (!(PASSED_MASK[c][s] & enemyPawns)) acc.add(OFF_PASSED + rankOf(relSquare(c, s)), c, 1);
            if (!(ADJ_FILES[fileOf(s)] & ownPawns)) acc.add(OFF_ISOLATED, c, 1);
        }
        for (int f = 0; f < 8; f++) {
            int k = popcount(ownPawns & fileMask(f));
            if (k > 1) acc.add(OFF_DOUBLED, c, k - 1);
        }

        if (popcount(b.bb(c, BISHOP)) >= 2) acc.add(OFF_BISHOP_PAIR, c, 1);
        for (Bitboard bb = b.bb(c, ROOK); bb;) {
            Bitboard file = fileMask(fileOf(popLsb(bb)));
            if (!(file & ownPawns)) acc.add((file & enemyPawns) ? OFF_ROOK_SEMI_OPEN : OFF_ROOK_OPEN, c, 1);
        }

        // King shelter.
        if (b.kingSq[c] >= 0) {
            int k = b.kingSq[c], kf = fileOf(k), kr = rankOf(relSquare(c, k));
            Bitboard files = fileMask(kf) | ADJ_FILES[kf];
            if (kr < 6) {
                Bitboard r1 = rankMask(c == WHITE ? rankOf(k) + 1 : rankOf(k) - 1);
                acc.add(OFF_SHIELD + 0, c, popcount(ownPawns & files & r1));
                if (kr < 5) {
                    Bitboard r2 = rankMask(c == WHITE ? rankOf(k) + 2 : rankOf(k) - 2);
                    acc.add(OFF_SHIELD + 1, c, popcount(ownPawns & files & r2));
                }
            }
            int open = 0;
            for (int f = std::max(0, kf - 1); f <= std::min(7, kf + 1); f++)
                if (!(ownPawns & fileMask(f))) open++;
            acc.add(OFF_SHIELD + 2, c, open);
        }

        // Threats: pawns attacking pieces.
        Bitboard enemyPieces = b.occ[e] & ~enemyPawns & ~b.bb(e, KING);
        if (int k = popcount(pawnAtt[c] & enemyPieces)) acc.add(OFF_PAWN_THREAT, c, k);
    }
    acc.add(OFF_TEMPO, b.stm, 1);
    return std::min(phase, MAX_PHASE);
}

inline int centerDistance(int s) {
    int f = fileOf(s), r = rankOf(s);
    return std::max(3 - f, f - 4) + std::max(3 - r, r - 4);
}

int nonPawnMaterial(const Board& b, int c) {
    int v = 0;
    for (int t = KNIGHT; t <= QUEEN; t++) v += popcount(b.bb(c, t)) * PIECE_VALUE[t];
    return v;
}

// Adjustments outside the tuned linear model. Returns true if any applied.
bool adjust(const Board& b, int& score) {
    bool applied = false;
    int mat[2] = {nonPawnMaterial(b, WHITE), nonPawnMaterial(b, BLACK)};
    int pawns[2] = {popcount(b.bb(WHITE, PAWN)), popcount(b.bb(BLACK, PAWN))};
    // Mop-up: with a lone enemy king, drive it to the edge and bring our king close.
    for (int c = 0; c < 2; c++) {
        int o = c ^ 1;
        if (mat[o] == 0 && pawns[o] == 0 && mat[c] >= 500 && b.kingSq[0] >= 0 && b.kingSq[1] >= 0) {
            int lk = b.kingSq[o], wk = b.kingSq[c];
            int dist = std::abs(fileOf(lk) - fileOf(wk)) + std::abs(rankOf(lk) - rankOf(wk));
            int bonus = 10 * centerDistance(lk) + 4 * (14 - dist);
            score += c == WHITE ? bonus : -bonus;
            applied = true;
        }
    }
    // Without pawns, being only a minor piece ahead is usually a draw.
    int strong = score > 0 ? WHITE : BLACK;
    if (pawns[strong] == 0 && mat[strong] - mat[strong ^ 1] < 400) {
        score /= 4;
        applied = true;
    }
    return applied;
}

// Kings: steps (king moves) to the nearest of d4/e4/d5/e5.
inline int hillDistance(int s) {
    int f = fileOf(s), r = rankOf(s);
    return std::max(std::max(3 - f, f - 4), std::max(3 - r, r - 4));
}

// Hand-set terms for the variants, outside the tuned model. White's point of view.
int variantTerms(const Board& b) {
    int score = 0;
    for (int c = 0; c < 2; c++) {
        const int sign = c == WHITE ? 1 : -1, e = c ^ 1;
        const int k = b.kingSq[c];
        switch (b.variant) {
            case CRAZYHOUSE: {
                // Pieces in hand are full material, and pawns are worth more since they drop anywhere.
                static const int HAND[7] = {0, 120, 320, 330, 480, 900, 0};
                for (int t = PAWN; t <= QUEEN; t++) score += sign * HAND[t] * b.pocket[c][t];
                // Empty squares next to the king are landing spots for the opponent's drops.
                static const int DROP_THREAT[7] = {0, 1, 2, 2, 3, 4, 0};
                int threat = 0;
                for (int t = PAWN; t <= QUEEN; t++) threat += DROP_THREAT[t] * b.pocket[e][t];
                int holes = popcount(attacks::king(k) & ~b.occAll);
                score -= sign * holes * std::min(threat, 12) * 3;
                break;
            }
            case THREE_CHECK: {
                static const int CHECK_BONUS[4] = {0, 110, 320, 0};
                score += sign * CHECK_BONUS[std::min(b.checks[c], 3)];
                break;
            }
            case KING_OF_THE_HILL: {
                static const int HILL_BONUS[5] = {0, 160, 70, 30, 10};
                if (k >= 0) score += sign * HILL_BONUS[std::min(hillDistance(k), 4)];
                break;
            }
            case HORDE:
                // The horde's pawns protect each other far better than ordinary material counting assumes;
                // without this the 36 pawns look about 7 pawns worse than Black's army at the start.
                if (c == WHITE) score += 20 * popcount(b.bb(WHITE, PAWN));
                break;
            case ATOMIC: {
                // A capture next to a king blows it up, so pieces beside our king that the opponent can take
                // are deadly - unless the kings touch (that capture would destroy the capturer's own king too).
                const int ek = b.kingSq[e];
                if (k >= 0 && !(ek >= 0 && (attacks::king(k) & bit(ek)))) {
                    int danger = 0;
                    for (Bitboard n = attacks::king(k) & b.occAll; n;)
                        if (b.attacked(popLsb(n), e)) danger++;
                    score -= sign * 90 * danger;
                }
                break;
            }
            case RACING_KINGS: {
                // Both kings race up the board, so progress counts by rank for both colors. Squares on the
                // rank ahead that the opponent controls slow the king down; if all of them are covered it is stuck.
                static const int RACE[8] = {0, 30, 75, 140, 230, 350, 520, 0};
                const int r = rankOf(k);
                score += sign * RACE[r];
                if (r < 7) {
                    Bitboard ahead = attacks::king(k) & rankMask(r + 1) & ~b.occ[c];
                    int open = 0, total = popcount(ahead);
                    for (Bitboard a = ahead; a;)
                        if (!b.attacked(popLsb(a), e)) open++;
                    score -= sign * (total - open) * (15 + 5 * r);
                    if (!open) score -= sign * (40 + 15 * r);
                }
                break;
            }
            default: break;
        }
    }
    return score;
}

}  // namespace

namespace {
// Fade the score toward a draw as the 50-move counter climbs (full value at 0, half at 100). The network
// cannot see the counter, so without this a won position scores the same whether the engine makes progress
// or shuffles, and the search only notices once the forced draw at 100 is inside its horizon.
int fiftyMoveScale(const Board& b, int score) {
    return score * (200 - b.halfmove) / 200;
}
// Antichess, from the side to move's view. Giving pieces away is the aim, so having fewer of them is good,
// and mobility helps: spare quiet moves are tempo, and a side with few moves is easier to force into captures.
// Deliberately simple: the search, with compulsory captures making play very forcing, does the real work.
int antichessEval(const Board& b) {
    const int us = b.stm, them = us ^ 1;
    int score = 60 * (popcount(b.occ[them]) - popcount(b.occ[us]));
    for (int c = 0; c < 2; c++) {
        int mob = 0;
        for (int t = KNIGHT; t <= KING; t++)
            for (Bitboard p = b.bb(c, t); p;) {
                int s = popLsb(p);
                Bitboard a = t == KNIGHT ? attacks::knight(s) : t == BISHOP ? attacks::bishop(s, b.occAll)
                           : t == ROOK   ? attacks::rook(s, b.occAll)
                           : t == QUEEN  ? attacks::queen(s, b.occAll)
                                         : attacks::king(s);
                mob += popcount(a & ~b.occAll);
            }
        score += (c == us ? 2 : -2) * mob;
    }
    return score;
}

}  // namespace

// Material scaling of the network's score: the fewer pieces (not pawns) left, the closer to zero. A side that is
// ahead then prefers keeping pieces on, and one that is behind prefers trading down, which the network alone
// does not capture well. 1.0 with all pieces on; MS_BASE sets how strongly it fades (larger = weaker).
#ifndef MS_BASE
#define MS_BASE 26000
#endif
static int materialScale(const Board& b, int score) {
    const int material = 450 * popcount(b.pieces[WHITE][KNIGHT] | b.pieces[BLACK][KNIGHT]) +
                         450 * popcount(b.pieces[WHITE][BISHOP] | b.pieces[BLACK][BISHOP]) +
                         650 * popcount(b.pieces[WHITE][ROOK] | b.pieces[BLACK][ROOK]) +
                         1250 * popcount(b.pieces[WHITE][QUEEN] | b.pieces[BLACK][QUEEN]);
    constexpr int FULL = 4 * 450 + 4 * 450 + 4 * 650 + 2 * 1250;
    return int(int64_t(score) * (MS_BASE + material) / (MS_BASE + FULL));
}

int evaluate(const Board& b) {
    if (b.variant == ANTICHESS) return fiftyMoveScale(b, antichessEval(b));
    if ((b.variant == STANDARD || b.variant == CHESS960) && nnue::active())
        return fiftyMoveScale(b, materialScale(b, nnue::evaluate(b)));
    ScoreAcc acc;
    int phase = evalCore(b, acc);
    int score = (acc.mg * phase + acc.eg * (MAX_PHASE - phase)) / MAX_PHASE;
    // Mop-up and drawish scaling assume standard rules.
    if (b.variant == STANDARD || b.variant == CHESS960)
        adjust(b, score);
    else
        score += variantTerms(b);
    return fiftyMoveScale(b, b.stm == WHITE ? score : -score);
}

void traceEval(const Board& b, EvalTrace& t) {
    TraceAcc acc;
    t.phase = evalCore(b, acc);
    t.features.clear();
    for (int i = 0; i < NUM_PARAMS; i++)
        if (acc.counts[i]) t.features.push_back({int16_t(i), acc.counts[i]});
    ScoreAcc sa;
    evalCore(b, sa);
    int raw = (sa.mg * t.phase + sa.eg * (MAX_PHASE - t.phase)) / MAX_PHASE;
    t.linear = !adjust(b, raw);
}

// Hand-written starting values; the tuner refines these.
void resetDefaultParams() {
    // Visual tables (rank 8 first) from the original evaluation.
    static const int P[64] = {
         0,  0,  0,  0,  0,  0,  0,  0,  50, 50, 50, 50, 50, 50, 50, 50,  10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,   0,  0,  0, 20, 20,  0,  0,  0,   5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,   0,  0,  0,  0,  0,  0,  0,  0};
    static const int N[64] = {
        -50,-40,-30,-30,-30,-30,-40,-50, -40,-20,  0,  0,  0,  0,-20,-40, -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30, -30,  0, 15, 20, 20, 15,  0,-30, -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40, -50,-40,-30,-30,-30,-30,-40,-50};
    static const int B[64] = {
        -20,-10,-10,-10,-10,-10,-10,-20, -10,  0,  0,  0,  0,  0,  0,-10, -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10, -10,  0, 10, 10, 10, 10,  0,-10, -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10, -20,-10,-10,-10,-10,-10,-10,-20};
    static const int R[64] = {
         0,  0,  0,  0,  0,  0,  0,  0,   5, 10, 10, 10, 10, 10, 10,  5,  -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,  -5,  0,  0,  0,  0,  0,  0, -5,  -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,   0,  0,  0,  5,  5,  0,  0,  0};
    static const int Q[64] = {
        -20,-10,-10, -5, -5,-10,-10,-20, -10,  0,  0,  0,  0,  0,  0,-10, -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,   0,  0,  5,  5,  5,  5,  0, -5, -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10, -20,-10,-10, -5, -5,-10,-10,-20};
    static const int KMG[64] = {
        -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30, -20,-30,-30,-40,-40,-30,-30,-20, -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,  20, 30, 10,  0,  0, 10, 30, 20};
    static const int KEG[64] = {
        -50,-40,-30,-20,-20,-30,-40,-50, -30,-20,-10,  0,  0,-10,-20,-30, -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30, -30,-10, 30, 40, 40, 30,-10,-30, -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-30,  0,  0,  0,  0,-30,-30, -50,-30,-30,-30,-30,-30,-30,-50};
    for (auto& p : PARAMS) p = {0, 0};
    const int VAL_MG[7] = {0, 100, 320, 330, 500, 900, 0}, VAL_EG[7] = {0, 110, 320, 330, 500, 900, 0};
    for (int t = PAWN; t <= QUEEN; t++) PARAMS[OFF_MATERIAL + t] = {VAL_MG[t], VAL_EG[t]};
    for (int s = 0; s < 64; s++) {
        int v = (7 - rankOf(s)) * 8 + fileOf(s);  // visual index for White square s
        PARAMS[OFF_PST + PAWN * 64 + s] = {P[v], P[v] / 2};
        PARAMS[OFF_PST + KNIGHT * 64 + s] = {N[v], N[v]};
        PARAMS[OFF_PST + BISHOP * 64 + s] = {B[v], B[v]};
        PARAMS[OFF_PST + ROOK * 64 + s] = {R[v], R[v]};
        PARAMS[OFF_PST + QUEEN * 64 + s] = {Q[v], Q[v]};
        PARAMS[OFF_PST + KING * 64 + s] = {KMG[v], KEG[v]};
    }
    for (int i = 0; i < 9; i++) PARAMS[OFF_MOB_N + i] = {(i - 4) * 4, (i - 4) * 4};
    for (int i = 0; i < 14; i++) PARAMS[OFF_MOB_B + i] = {(i - 6) * 5, (i - 6) * 5};
    for (int i = 0; i < 15; i++) PARAMS[OFF_MOB_R + i] = {(i - 7) * 2, (i - 7) * 4};
    for (int i = 0; i < 28; i++) PARAMS[OFF_MOB_Q + i] = {(i - 13) * 1, (i - 13) * 2};
    const int PMG[8] = {0, 5, 10, 15, 25, 40, 60, 0}, PEG[8] = {0, 10, 20, 35, 60, 90, 130, 0};
    for (int r = 0; r < 8; r++) PARAMS[OFF_PASSED + r] = {PMG[r], PEG[r]};
    PARAMS[OFF_ISOLATED] = {-10, -15};
    PARAMS[OFF_DOUBLED] = {-10, -20};
    PARAMS[OFF_BISHOP_PAIR] = {30, 50};
    PARAMS[OFF_ROOK_OPEN] = {20, 10};
    PARAMS[OFF_ROOK_SEMI_OPEN] = {10, 5};
    const int KA[7] = {0, 0, 8, 8, 6, 5, 0}, SC[7] = {0, 0, 10, 6, 10, 8, 0};
    for (int t = 0; t < 7; t++) {
        PARAMS[OFF_KING_ATTACK + t] = {KA[t], 0};
        PARAMS[OFF_SAFE_CHECK + t] = {SC[t], 0};
    }
    PARAMS[OFF_SHIELD + 0] = {10, 0};
    PARAMS[OFF_SHIELD + 1] = {5, 0};
    PARAMS[OFF_SHIELD + 2] = {-15, 0};
    PARAMS[OFF_PAWN_THREAT] = {40, 30};
    PARAMS[OFF_TEMPO] = {10, 10};
}
