#pragma once
#include <cstdint>
#include <vector>

#include "board.h"

// Fixed piece values for move ordering and exchange evaluation (not the tuned evaluation weights).
constexpr int PIECE_VALUE[7] = {0, 100, 320, 330, 500, 900, 20000};

struct Score {
    int mg, eg;  // middlegame / endgame, blended by game phase
};

// Layout of the tunable parameter table. Every evaluation term is a count times one of these.
enum : int {
    OFF_MATERIAL = 0,                 // [piece type]
    OFF_PST = OFF_MATERIAL + 7,       // [piece type][square], from White's point of view
    OFF_MOB_N = OFF_PST + 7 * 64,     // [safe squares 0..8]
    OFF_MOB_B = OFF_MOB_N + 9,        // [0..13]
    OFF_MOB_R = OFF_MOB_B + 14,       // [0..14]
    OFF_MOB_Q = OFF_MOB_R + 15,       // [0..27]
    OFF_PASSED = OFF_MOB_Q + 28,      // [relative rank]
    OFF_ISOLATED = OFF_PASSED + 8,
    OFF_DOUBLED,
    OFF_BISHOP_PAIR,
    OFF_ROOK_OPEN,
    OFF_ROOK_SEMI_OPEN,
    OFF_KING_ATTACK,                  // [attacker type]: attacked squares in the enemy king zone
    OFF_SAFE_CHECK = OFF_KING_ATTACK + 7,  // [checker type]: safe checking squares
    OFF_SHIELD = OFF_SAFE_CHECK + 7,  // [0] pawn in front of king, [1] two ahead, [2] open file near king
    OFF_PAWN_THREAT = OFF_SHIELD + 3, // our pawns attacking enemy pieces
    OFF_TEMPO,
    NUM_PARAMS
};

extern Score PARAMS[NUM_PARAMS];
constexpr int MAX_PHASE = 24;

// Static evaluation in centipawns from the side to move's point of view.
int evaluate(const Board& b);

// Tuning support: the evaluation as a linear function of PARAMS.
struct EvalTrace {
    std::vector<std::pair<int16_t, int16_t>> features;  // (param index, white count - black count)
    int phase = 0;
    bool linear = true;  // false if non-tunable adjustments (mop-up, draw scaling) apply
};
void traceEval(const Board& b, EvalTrace& t);
void resetDefaultParams();
