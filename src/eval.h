#pragma once
#include "board.h"

constexpr int PIECE_VALUE[7] = {0, 100, 320, 330, 500, 900, 0};

// Static evaluation in centipawns from the side to move's point of view.
int evaluate(const Board& b);
