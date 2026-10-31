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

// Playing strength. Level 6 is the full engine at 3 s a move; MAX gives it 30 s. Lower levels cap the
// nodes searched and blur the evaluation with noise, so they see less and misjudge more.
struct EngineLevel {
    const char* name;
    int ms;     // think time per move
    int nodes;  // 0 = no limit (then all hardware threads are used)
    int noise;  // +- centipawns of evaluation noise
    bool book;  // use the opening book
    int elo;    // approximate strength relative to level 6 (see README)
};
constexpr int NUM_LEVELS = 7;
extern const EngineLevel LEVELS[NUM_LEVELS];
int defaultThreads();  // hardware threads, capped at 16

// Iterative-deepening alpha-beta search. Stops after timeMs, after maxNodes (if non-zero), or when *stop
// becomes true. Uses `threads` search threads (Lazy SMP: all threads search the same tree and share the TT).
// evalNoise adds +-evalNoise cp of noise to every evaluation, consistent within one search.
// With softMs > 0 (games on a clock) timeMs is a hard limit and softMs the target time, which the search
// stretches when the position is unsettled; otherwise it stops starting new iterations after timeMs / 2.
Move findBestMove(const Board& root, int timeMs, std::atomic<bool>* stop, const InfoFn& info, bool useBook = false,
                  int threads = 1, int maxNodes = 0, int evalNoise = 0, int softMs = 0);
// Quiescence search score (side to move's view), used to filter tuning positions.
int quiescenceScore(const Board& pos);
void clearSearchMemory();
std::string scoreToString(int score);  // "+0.35", "M4", "-M2"
