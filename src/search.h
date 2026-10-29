#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

#include "board.h"

constexpr int INF = 32500;
constexpr int MATE = 32000;
constexpr int MATE_BOUND = 31000;

struct SearchInfo {
    int depth = 0;
    int score = 0;  // from the side to move's point of view
    uint64_t nodes = 0;
    int ms = 0;
    std::string pv;  // SAN
    std::string note;
};

using InfoFn = std::function<void(const SearchInfo&)>;

// Iterative-deepening alpha-beta search. Stops after timeMs or when *stop becomes true.
Move findBestMove(const Board& root, int timeMs, std::atomic<bool>* stop, const InfoFn& info);
void clearSearchMemory();
std::string scoreToString(int score);  // "+0.35", "M4", "-M2"
