#pragma once
#include <string>

#include "board.h"

// NNUE evaluation over (king bucket, piece, square) inputs: per-side hidden layer (NNUE_HIDDEN_SIZE, 1024 for
// the current net) -> output bucketed by material. The current net (CBN2: 10 mirrored king buckets, SCReLU) is
// trained with Bullet (tools/bignet) on Leela-derived data. Used for standard chess and Chess960 when a network
// file is loaded; the other variants keep the hand-written evaluation.
namespace nnue {
bool load(const std::string& path);  // false if the file is missing or malformed
bool loaded();
void setEnabled(bool on);  // UCI "UseNNUE"; has no effect without a loaded network
bool active();             // loaded and enabled
bool weightsOnLargePages();  // the feature weights got large pages
int evaluate(const Board& b);  // centipawns, side to move's point of view
}  // namespace nnue
