#include "eval.h"

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
    void kingDanger(int color, const int* f) {
        int d = 0;
        for (int k = 0; k < KD_COUNT; k++) d += PARAMS[OFF_KING_DANGER + k].mg * f[k];
        if (d <= 0) return;
        int sign = color == WHITE ? 1 : -1;
        mg += sign * (d * d / 4096);
        eg += sign * (d / 16);
    }
};

// Records feature counts for the tuner.
struct TraceAcc {
    int16_t counts[NUM_PARAMS] = {};
    int16_t danger[2][KD_COUNT] = {};
    void add(int idx, int color, int n) { counts[idx] += int16_t(color == WHITE ? n : -n); }
    void kingDanger(int color, const int* f) {
        for (int k = 0; k < KD_COUNT; k++) danger[color][k] = int16_t(f[k]);
    }
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
    Bitboard attackedBy[2], attackedByNoKing[2];
    int phase = 0;
    for (int c = 0; c < 2; c++) {
        attackedByNoKing[c] = pawnAtt[c];
        for (int t = KNIGHT; t <= QUEEN; t++)
            for (Bitboard bb = b.bb(c, t); bb;) {
                int s = popLsb(bb);
                Bitboard a = t == KNIGHT   ? attacks::knight(s)
                             : t == BISHOP ? attacks::bishop(s, occ)
                             : t == ROOK   ? attacks::rook(s, occ)
                                           : attacks::queen(s, occ);
                list[c][n[c]++] = {t, s, a};
                attackedByNoKing[c] |= a;
                phase += t == QUEEN ? 4 : t == ROOK ? 2 : 1;
            }
        attackedBy[c] = attackedByNoKing[c] | attacks::king(b.kingSq[c]);
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

        // Mobility, and the inputs to king danger against the enemy king.
        const Bitboard mobArea = ~b.occ[c] & ~pawnAtt[e];
        Bitboard kingZone = attacks::king(ek) | bit(ek);
        kingZone |= e == WHITE ? kingZone << 8 : kingZone >> 8;  // include the rank in front of it
        const Bitboard safe = ~b.occ[c] & ~attackedBy[e];
        const Bitboard checkB = attacks::bishop(ek, occ), checkR = attacks::rook(ek, occ);
        const Bitboard checkSq[6] = {0, 0, attacks::knight(ek), checkB, checkR, checkB | checkR};
        static const int MOB_OFF[6] = {0, 0, OFF_MOB_N, OFF_MOB_B, OFF_MOB_R, OFF_MOB_Q};
        int kd[KD_COUNT] = {};
        for (int i = 0; i < n[c]; i++) {
            const PieceAtt& p = list[c][i];
            acc.add(MOB_OFF[p.t] + popcount(p.a & mobArea), c, 1);
            if (Bitboard z = p.a & kingZone) {
                kd[KD_ATTACKER_N + p.t - KNIGHT]++;
                kd[KD_ZONE_ATTACKS] += popcount(z);
            }
            kd[KD_CHECK_N + p.t - KNIGHT] += popcount(p.a & checkSq[p.t] & safe);
        }
        kd[KD_WEAK_SQUARES] = popcount(kingZone & attackedBy[c] & ~attackedByNoKing[e]);
        kd[KD_NO_QUEEN] = b.bb(c, QUEEN) ? 0 : 1;
        kd[KD_BIAS] = 1;
        // Only meaningful with at least two attackers, as a lone piece rarely mates.
        if (kd[KD_ATTACKER_N] + kd[KD_ATTACKER_B] + kd[KD_ATTACKER_R] + kd[KD_ATTACKER_Q] >= 2)
            acc.kingDanger(c, kd);

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

        // Our king's pawn shelter.
        {
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
        }

        // Pressure on the enemy king's files: pawn storms and open lines.
        {
            int ekf = fileOf(ek);
            for (int f = std::max(0, ekf - 1); f <= std::min(7, ekf + 1); f++) {
                Bitboard file = fileMask(f);
                for (Bitboard bb = ownPawns & file; bb;) {
                    int s = popLsb(bb);
                    acc.add(OFF_STORM + rankOf(relSquare(c, s)), c, 1);
                }
                if (!(enemyPawns & file)) acc.add(OFF_KING_FILE + ((ownPawns & file) ? 0 : 1), c, 1);
            }
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
        if (mat[o] == 0 && pawns[o] == 0 && mat[c] >= 500) {
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

}  // namespace

int evaluate(const Board& b) {
    ScoreAcc acc;
    int phase = evalCore(b, acc);
    int score = (acc.mg * phase + acc.eg * (MAX_PHASE - phase)) / MAX_PHASE;
    adjust(b, score);
    return b.stm == WHITE ? score : -score;
}

void traceEval(const Board& b, EvalTrace& t) {
    TraceAcc acc;
    t.phase = evalCore(b, acc);
    t.features.clear();
    for (int i = 0; i < NUM_PARAMS; i++)
        if (acc.counts[i]) t.features.push_back({int16_t(i), acc.counts[i]});
    std::memcpy(t.kingDanger, acc.danger, sizeof(acc.danger));
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
    PARAMS[OFF_SHIELD + 0] = {10, 0};
    PARAMS[OFF_SHIELD + 1] = {5, 0};
    const int STORM[8] = {0, 0, 0, 5, 10, 20, 10, 0};
    for (int r = 0; r < 8; r++) PARAMS[OFF_STORM + r] = {STORM[r], 0};
    PARAMS[OFF_KING_FILE + 0] = {10, 0};
    PARAMS[OFF_KING_FILE + 1] = {25, 0};
    // Starting king danger weights, in the spirit of Stockfish's classical evaluation.
    const int KD[KD_COUNT] = {80, 50, 45, 10, 60, 300, 250, 400, 300, 150, -800, -60};
    for (int k = 0; k < KD_COUNT; k++) PARAMS[OFF_KING_DANGER + k] = {KD[k], 0};
    PARAMS[OFF_PAWN_THREAT] = {40, 30};
    PARAMS[OFF_TEMPO] = {10, 10};
}
