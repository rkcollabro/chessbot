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
    OFF_SHIELD,                       // [0] own pawn in front of king, [1] two ahead
    OFF_STORM = OFF_SHIELD + 2,       // [relative rank] our pawns advancing on the enemy king's files
    OFF_KING_FILE = OFF_STORM + 8,    // [0] semi-open, [1] open file next to the enemy king
    OFF_PAWN_THREAT = OFF_KING_FILE + 2,  // our pawns attacking enemy pieces
    OFF_TEMPO,
    OFF_KING_DANGER,                  // [KD_*] inputs to the quadratic king danger (mg field only)
    NUM_PARAMS = OFF_KING_DANGER + 12
};

// King danger inputs, computed for each side attacking the enemy king. The weighted sum d adds
// d*d/4096 to the attacker's middlegame score and d/16 to its endgame score (when d > 0).
enum : int {
    KD_ATTACKER_N, KD_ATTACKER_B, KD_ATTACKER_R, KD_ATTACKER_Q,  // pieces hitting the king zone
    KD_ZONE_ATTACKS,                                              // attacked king-zone squares
    KD_CHECK_N, KD_CHECK_B, KD_CHECK_R, KD_CHECK_Q,              // safe checking squares
    KD_WEAK_SQUARES,  // king-zone squares we attack that only the king defends
    KD_NO_QUEEN,      // attacker has no queen
    KD_BIAS,
    KD_COUNT
};

extern Score PARAMS[NUM_PARAMS];
constexpr int MAX_PHASE = 24;

// Static evaluation in centipawns from the side to move's point of view.
int evaluate(const Board& b);

// Tuning support: the evaluation as a linear function of PARAMS.
struct EvalTrace {
    std::vector<std::pair<int16_t, int16_t>> features;  // (param index, white count - black count)
    int16_t kingDanger[2][KD_COUNT] = {};                // king danger inputs per attacking color
    int phase = 0;
    bool linear = true;  // false if non-tunable adjustments (mop-up, draw scaling) apply
};
void traceEval(const Board& b, EvalTrace& t);
void resetDefaultParams();
