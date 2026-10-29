#include "book.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <random>
#include <vector>

namespace book {
namespace {
std::vector<Entry> g_entries;
}

bool load(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return false;
    size_t bytes = size_t(in.tellg());
    if (bytes % sizeof(Entry)) return false;
    std::vector<Entry> entries(bytes / sizeof(Entry));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(entries.data()), bytes);
    if (!in) return false;
    g_entries = std::move(entries);
    return true;
}

bool loaded() { return !g_entries.empty(); }

Move probe(Board& b) {
    auto lo = std::lower_bound(g_entries.begin(), g_entries.end(), b.hash,
                               [](const Entry& e, uint64_t k) { return e.key < k; });
    if (lo == g_entries.end() || lo->key != b.hash) return NO_MOVE;

    MoveList legal;
    b.genLegal(legal);
    std::vector<std::pair<Move, int>> options;
    int best = 0;
    for (auto it = lo; it != g_entries.end() && it->key == b.hash; ++it) {
        for (int i = 0; i < legal.count; i++) {
            Move m = legal.moves[i];
            if ((m & 0x7FFF) == it->move) {
                options.push_back({m, it->weight});
                best = std::max<int>(best, it->weight);
            }
        }
    }
    // Ignore rare sidelines: keep moves with at least 15% of the top move's weight.
    int total = 0;
    for (auto& o : options) total += o.second >= best * 15 / 100 ? o.second : 0;
    if (!total) return NO_MOVE;
    static std::mt19937 rng(uint32_t(std::chrono::steady_clock::now().time_since_epoch().count()));
    int r = int(rng() % uint32_t(total));
    for (auto& o : options) {
        if (o.second < best * 15 / 100) continue;
        if (r < o.second) return o.first;
        r -= o.second;
    }
    return NO_MOVE;
}

}  // namespace book
