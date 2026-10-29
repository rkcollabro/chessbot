#pragma once
#include <functional>
#include <string>
#include <vector>

#include "board.h"

// Endgame tablebases for every material combination with up to 4 pieces (kings included).
// Tables are built by retrograde analysis on first use and cached as .tb files.
//
// Values are distance-to-mate in plies from the side to move's point of view:
//   v > 0  : win, mate in v plies
//   v < 0  : loss, mated in (-v - 1) plies
//   v == 0 : draw
namespace tb {

using StatusFn = std::function<void(const std::string&)>;

void init(const std::string& directory);
// Loads every table from disk, generating any that are missing (dependencies first).
bool loadOrGenerateAll(const StatusFn& status);
bool probe(const Board& b, int& value);

int readyCount();
int totalCount();

struct TableStats {
    std::string key;
    long long wins = 0, losses = 0;  // white (stronger side) to move
    int longestMate = 0;                        // in moves
};
std::vector<TableStats> stats();

}  // namespace tb
