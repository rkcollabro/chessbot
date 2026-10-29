#pragma once
#include <cstdint>
#include <string>

#include "board.h"

// Opening book built from master-level games (see `chesscli mkbook`).
// File format: sorted records of {uint64 hash, uint16 move, uint16 weight}.
namespace book {

struct Entry {
    uint64_t key;
    uint16_t move;    // from | to << 6 | promo << 12
    uint16_t weight;
};

bool load(const std::string& path);
bool loaded();
// Picks a book move at random, weighted by how often and how well it was played. NO_MOVE if out of book.
Move probe(Board& b);

}  // namespace book
