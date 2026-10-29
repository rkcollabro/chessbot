// Command-line harness for testing the engine without the GUI.
//   chesscli test                 perft suite + hash consistency check
//   chesscli perft <depth> [fen]
//   chesscli tbgen                build/load all tablebases and print stats
//   chesscli tbprobe <fen>
//   chesscli search <ms> [fen]
//   chesscli uci                  UCI protocol on stdin/stdout
//   chesscli mkbook <out.bin> <maxply> <pgn>...
//   chesscli extract <out.txt> <perGame> <maxPositions> <pgn>...
//   chesscli tune <data.txt> <out.inc> [epochs]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "board.h"
#include "datagen.h"
#include "search.h"
#include "tablebase.h"
#include "uci.h"

static bool g_checkHash = false;
static bool g_hashOk = true;

static uint64_t perft(Board& b, int depth) {
    if (g_checkHash && b.hash != b.computeHash()) g_hashOk = false;
    if (depth == 0) return 1;
    MoveList ml;
    b.genMoves(ml);
    uint64_t n = 0;
    for (int i = 0; i < ml.count; i++)
        if (b.makeMove(ml.moves[i])) {
            n += perft(b, depth - 1);
            b.unmakeMove();
        }
    return n;
}

static std::string joinArgs(int argc, char** argv, int from) {
    std::string s;
    for (int i = from; i < argc; i++) s += (i > from ? " " : "") + std::string(argv[i]);
    return s;
}

static std::string exeDir(const char* argv0) {
    std::string p = argv0;
    size_t k = p.find_last_of("\\/");
    return k == std::string::npos ? "." : p.substr(0, k);
}

int main(int argc, char** argv) {
    initTables();
    std::string cmd = argc > 1 ? argv[1] : "test";
    auto t0 = std::chrono::steady_clock::now();
    auto ms = [&] {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    };

    if (cmd == "uci") {
        uciLoop(exeDir(argv[0]) + "/tb", exeDir(argv[0]) + "/book.bin");
        return 0;
    }

    if (cmd == "mkbook" && argc > 4)
        return buildBook(argv[2], std::atoi(argv[3]), std::vector<std::string>(argv + 4, argv + argc));
    if (cmd == "extract" && argc > 5)
        return extractPositions(argv[2], std::atoi(argv[3]), std::atoi(argv[4]),
                                std::vector<std::string>(argv + 5, argv + argc));
    if (cmd == "tune" && argc > 3) return tune(argv[2], argv[3], argc > 4 ? std::atoi(argv[4]) : 300);

    if (cmd == "test") {
        struct Case { const char* fen; int depth; uint64_t expected; };
        const Case cases[] = {
            {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609},
            {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603},
            {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624},
            {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333},
            {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379},
            {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4, 3894594},
        };
        g_checkHash = true;
        bool allOk = true;
        for (const auto& c : cases) {
            Board b;
            b.setFEN(c.fen);
            uint64_t n = perft(b, c.depth);
            bool ok = n == c.expected;
            allOk &= ok;
            std::printf("%s perft(%d) = %llu (expected %llu)  %s\n", ok ? "OK  " : "FAIL", c.depth,
                        (unsigned long long)n, (unsigned long long)c.expected, c.fen);
        }
        std::printf("hash consistency: %s\n", g_hashOk ? "OK" : "FAIL");
        std::printf("%s in %lld ms\n", allOk && g_hashOk ? "ALL PASSED" : "FAILURES", (long long)ms());
        return allOk && g_hashOk ? 0 : 1;
    }

    if (cmd == "perft" && argc > 2) {
        Board b;
        std::string fen = joinArgs(argc, argv, 3);
        if (fen.empty()) b.setStartPos(); else if (!b.setFEN(fen)) return std::printf("bad FEN\n"), 1;
        uint64_t n = perft(b, std::atoi(argv[2]));
        std::printf("%llu nodes in %lld ms\n", (unsigned long long)n, (long long)ms());
        return 0;
    }

    if (cmd == "tbgen" || cmd == "tbprobe" || cmd == "search") {
        tb::init(exeDir(argv[0]) + "/tb");
        tb::loadOrGenerateAll([&](const std::string& s) {
            std::printf("[%6lld ms] %s\n", (long long)ms(), s.c_str());
            std::fflush(stdout);
        });
    }

    if (cmd == "tbgen") {
        std::printf("\n%-8s %12s %12s %s\n", "table", "wins(wtm)", "losses(wtm)", "longest mate");
        for (const auto& st : tb::stats())
            std::printf("%-8s %12lld %12lld %d\n", st.key.c_str(), st.wins, st.losses, st.longestMate);
        return 0;
    }

    if (cmd == "tbprobe" && argc > 2) {
        Board b;
        if (!b.setFEN(joinArgs(argc, argv, 2))) return std::printf("bad FEN\n"), 1;
        int v;
        if (!tb::probe(b, v)) return std::printf("not in tablebase\n"), 1;
        if (v > 0) std::printf("win: mate in %d plies (%d moves)\n", v, (v + 1) / 2);
        else if (v < 0) std::printf("loss: mated in %d plies\n", -v - 1);
        else std::printf("draw\n");
        return 0;
    }

    if (cmd == "search" && argc > 2) {
        Board b;
        std::string fen = joinArgs(argc, argv, 3);
        if (fen.empty()) b.setStartPos(); else if (!b.setFEN(fen)) return std::printf("bad FEN\n"), 1;
        auto t1 = std::chrono::steady_clock::now();
        Move m = findBestMove(b, std::atoi(argv[2]), nullptr, [](const SearchInfo& si) {
            std::printf("depth %2d  score %7s  nodes %10llu  time %6d ms  %s %s\n", si.depth,
                        scoreToString(si.score).c_str(), (unsigned long long)si.nodes, si.ms, si.pv.c_str(),
                        si.note.c_str());
        });
        (void)t1;
        std::printf("bestmove %s\n", m ? b.moveToSAN(m).c_str() : "(none)");
        return 0;
    }

    std::printf("usage: chesscli test | perft <d> [fen] | tbgen | tbprobe <fen> | search <ms> [fen]\n");
    return 1;
}
