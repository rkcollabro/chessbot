#include "eval.h"

#include <algorithm>
#include <cstdlib>

namespace {

// Piece-square tables, written from White's view with rank 8 on top.
const int PST_PAWN[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0};
const int PST_KNIGHT[64] = {
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50};
const int PST_BISHOP[64] = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20};
const int PST_ROOK[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0};
const int PST_QUEEN[64] = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20};
const int PST_KING_MG[64] = {
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20};
const int PST_KING_EG[64] = {
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50};

const int PASSED_MG[8] = {0, 5, 10, 15, 25, 40, 60, 0};
const int PASSED_EG[8] = {0, 10, 20, 35, 60, 90, 130, 0};

inline int pstIndex(int s, int c) {
    int f = fileOf(s), r = rankOf(s);
    return c == WHITE ? (7 - r) * 8 + f : r * 8 + f;
}

inline int centerDistance(int s) {
    int f = fileOf(s), r = rankOf(s);
    return std::max(3 - f, f - 4) + std::max(3 - r, r - 4);
}

}  // namespace

int evaluate(const Board& b) {
    int mg[2] = {0, 0}, eg[2] = {0, 0};
    int phase = 0;
    int pawnFiles[2][8] = {};
    int bishops[2] = {0, 0};
    int material[2] = {0, 0};  // non-pawn, non-king
    int pawns[2] = {0, 0};

    for (int s = 0; s < 64; s++)
        if (typeOf(b.sq[s]) == PAWN) pawnFiles[colorOf(b.sq[s])][fileOf(s)]++;

    for (int s = 0; s < 64; s++) {
        int p = b.sq[s];
        if (!p) continue;
        int c = colorOf(p), t = typeOf(p), i = pstIndex(s, c);
        int f = fileOf(s), r = rankOf(s);
        switch (t) {
            case PAWN: {
                pawns[c]++;
                mg[c] += 100 + PST_PAWN[i];
                eg[c] += 110 + PST_PAWN[i] / 2;
                // Passed pawn: no enemy pawn ahead on this or adjacent files.
                bool passed = true;
                int enemyPawn = makePiece(c ^ 1, PAWN);
                for (int ff = std::max(0, f - 1); ff <= std::min(7, f + 1) && passed; ff++) {
                    if (c == WHITE) {
                        for (int rr = r + 1; rr < 7; rr++)
                            if (b.sq[rr * 8 + ff] == enemyPawn) { passed = false; break; }
                    } else {
                        for (int rr = r - 1; rr > 0; rr--)
                            if (b.sq[rr * 8 + ff] == enemyPawn) { passed = false; break; }
                    }
                }
                if (passed) {
                    int adv = c == WHITE ? r : 7 - r;
                    mg[c] += PASSED_MG[adv];
                    eg[c] += PASSED_EG[adv];
                }
                bool isolated = (f == 0 || !pawnFiles[c][f - 1]) && (f == 7 || !pawnFiles[c][f + 1]);
                if (isolated) mg[c] -= 10, eg[c] -= 15;
                break;
            }
            case KNIGHT:
                mg[c] += 320 + PST_KNIGHT[i];
                eg[c] += 320 + PST_KNIGHT[i];
                material[c] += 320;
                phase += 1;
                break;
            case BISHOP:
                mg[c] += 330 + PST_BISHOP[i];
                eg[c] += 330 + PST_BISHOP[i];
                material[c] += 330;
                bishops[c]++;
                phase += 1;
                break;
            case ROOK: {
                mg[c] += 500 + PST_ROOK[i];
                eg[c] += 500 + PST_ROOK[i];
                material[c] += 500;
                if (!pawnFiles[c][f]) {
                    int bonus = pawnFiles[c ^ 1][f] ? 10 : 20;
                    mg[c] += bonus;
                    eg[c] += bonus / 2;
                }
                phase += 2;
                break;
            }
            case QUEEN:
                mg[c] += 900 + PST_QUEEN[i];
                eg[c] += 900 + PST_QUEEN[i];
                material[c] += 900;
                phase += 4;
                break;
            case KING:
                mg[c] += PST_KING_MG[i];
                eg[c] += PST_KING_EG[i];
                break;
        }
    }

    for (int c = 0; c < 2; c++) {
        for (int f = 0; f < 8; f++)
            if (pawnFiles[c][f] > 1) mg[c] -= 10 * (pawnFiles[c][f] - 1), eg[c] -= 20 * (pawnFiles[c][f] - 1);
        if (bishops[c] >= 2) mg[c] += 30, eg[c] += 50;
    }

    phase = std::min(phase, 24);
    int score = ((mg[WHITE] - mg[BLACK]) * phase + (eg[WHITE] - eg[BLACK]) * (24 - phase)) / 24;

    // Mop-up: with a lone enemy king, drive it to the edge and bring our king close.
    for (int c = 0; c < 2; c++) {
        int o = c ^ 1;
        if (material[o] == 0 && pawns[o] == 0 && material[c] >= 500) {
            int lk = b.kingSq[o], wk = b.kingSq[c];
            int dist = std::abs(fileOf(lk) - fileOf(wk)) + std::abs(rankOf(lk) - rankOf(wk));
            int bonus = 10 * centerDistance(lk) + 4 * (14 - dist);
            score += c == WHITE ? bonus : -bonus;
        }
    }

    // Without pawns, being only a minor piece ahead is usually a draw.
    int strong = score > 0 ? WHITE : BLACK;
    if (pawns[strong] == 0 && material[strong] - material[strong ^ 1] < 400) score /= 4;

    score += b.stm == WHITE ? 10 : -10;  // tempo
    return b.stm == WHITE ? score : -score;
}
