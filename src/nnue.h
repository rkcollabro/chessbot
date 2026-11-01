#pragma once
#include <string>

#include "board.h"

// NNUE evaluation: a 768 -> 256 (per side) -> 1 network over (piece, square) inputs, trained on self-play
// positions (tools/nnue_train.py). Used for standard chess and Chess960 when a network file is loaded; the
// other variants keep the hand-written evaluation.
namespace nnue {
bool load(const std::string& path);  // false if the file is missing or malformed
bool loaded();
void setEnabled(bool on);  // UCI "UseNNUE"; has no effect without a loaded network
bool active();             // loaded and enabled
int evaluate(const Board& b);  // centipawns, side to move's point of view
}  // namespace nnue
