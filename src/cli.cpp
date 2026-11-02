// Command-line harness for testing the engine without the GUI.
//   chesscli test                 perft suite + hash consistency check
//   chesscli perft <depth> [fen]
//   (perft and search take --variant <chess960|3check|crazyhouse|kingofthehill|horde|racingkings|antichess> before the fen)
//   chesscli tbgen                build/load all tablebases and print stats
//   chesscli tbprobe <fen>
//   chesscli search <ms> [fen]
//   chesscli uci                  UCI protocol on stdin/stdout (also the default when stdin is a pipe)
//   chesscli mkbook <out.txt> <maxply> <minElo> <minBaseSeconds> <pgn>...
//   chesscli booktxt <vetted.txt> <book.bin>
//   chesscli extract <out.txt> <perGame> <maxPositions> <pgn>...
//   chesscli tune <data.txt> <out.inc> [epochs]
//   chesscli selfplay <out.bin> <games> <nodes> <seed>   NNUE training data from self-play
//   chesscli eval <fen>           static evaluation (NNUE when nnue.bin is next to the exe)
//   chesscli nnuecheck <games>    incremental NNUE updates vs. a full rebuild over random games
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif
#include <string>
#include <vector>

#include "board.h"
#include "eval.h"
#include "datagen.h"
#include "search.h"
#include "tablebase.h"
#include "uci.h"
#include "nnue.h"

static bool g_checkHash = false;
static bool g_hashOk = true;

static uint64_t perft(Board& b, int depth) {
    if (g_checkHash && b.hash != b.computeHash()) g_hashOk = false;
    if (depth == 0) return 1;
    if (b.variantOutcome()) return 0;  // game already won by a variant rule
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

// Reads an optional "--variant <name>" at argv[i], advancing i past it.
static Variant variantArg(int argc, char** argv, int& i) {
    if (i + 1 < argc && std::string(argv[i]) == "--variant") {
        int v = variantFromUciName(argv[i + 1]);
        i += 2;
        if (v >= 0) return Variant(v);
        std::printf("unknown variant, using standard\n");
    }
    return STANDARD;
}

static std::string exeDir(const char* argv0) {
    std::string p = argv0;
    size_t k = p.find_last_of("\\/");
    return k == std::string::npos ? "." : p.substr(0, k);
}

int main(int argc, char** argv) {
    initTables();
    // With no arguments, run as a UCI engine when stdin is a pipe (GUIs and lichess-bot launch it that way)
    // and run the test suite from a console.
    std::string cmd = argc > 1 ? argv[1] : isatty(fileno(stdin)) ? "test" : "uci";
    auto t0 = std::chrono::steady_clock::now();
    auto ms = [&] {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    };

    nnue::load(exeDir(argv[0]) + "/nnue.bin");  // optional; without it the hand-written evaluation is used
    if (cmd == "uci") {
        uciLoop(exeDir(argv[0]) + "/tb", exeDir(argv[0]) + "/book.bin");
        // Skip static destructors: the detached tablebase loader may still be using the tables.
        std::fflush(stdout);
        std::_Exit(0);
    }

    if (cmd == "mkbook" && argc > 6)
        return buildBook(argv[2], std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]),
                         std::vector<std::string>(argv + 6, argv + argc));
    if (cmd == "booktxt" && argc > 3) return bookFromText(argv[2], argv[3]);
    if (cmd == "extract" && argc > 5)
        return extractPositions(argv[2], std::atoi(argv[3]), std::atoi(argv[4]),
                                std::vector<std::string>(argv + 5, argv + argc));
    if (cmd == "selfplay" && argc > 5) {
        tb::init(exeDir(argv[0]) + "/tb");
        return selfplay(argv[2], std::atoi(argv[3]), std::atoi(argv[4]), unsigned(std::atoi(argv[5])));
    }
    if (cmd == "eval" && argc > 2) {  // static evaluation (NNUE if nnue.bin is loaded), side to move's view
        Board b;
        if (!b.setFEN(joinArgs(argc, argv, 2))) return std::printf("bad fen\n"), 1;
        std::printf("%d\n", evaluate(b));
        return 0;
    }
    if (cmd == "nnuecheck" && argc > 2) {
        if (!nnue::active()) return std::printf("no network loaded\n"), 1;
        std::srand(1);
        long checked = 0, bad = 0;
        for (int g = 0; g < std::atoi(argv[2]); g++) {
            Board b;
            b.setStartPos();
            for (int ply = 0; ply < 200; ply++) {
                MoveList ml;
                b.genLegal(ml);
                if (!ml.count) break;
                // Wander a little (make/unmake a few moves) so cached ancestors get reused, then advance.
                for (int k = 0; k < 3; k++) {
                    b.makeMove(ml.moves[std::rand() % ml.count]);
                    Board fresh;
                    fresh.setFEN(b.getFEN());
                    bad += evaluate(b) != evaluate(fresh), checked++;
                    b.unmakeMove();
                }
                b.makeMove(ml.moves[std::rand() % ml.count]);
            }
        }
        std::printf("checked %ld positions, %ld mismatches\n", checked, bad);
        return bad != 0;
    }
    if (cmd == "tune" && argc > 3) return tune(argv[2], argv[3], argc > 4 ? std::atoi(argv[4]) : 300);

    if (cmd == "test") {
        struct Case { const char* fen; int depth; uint64_t expected; Variant variant = STANDARD; };
        const Case cases[] = {
            {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609},
            {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603},
            {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624},
            {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333},
            {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379},
            {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4, 3894594},
            // Variants, verified against python-chess (tools/variant_perft.py does randomized checks).
            {"bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9", 4, 326672, CHESS960},
            {"r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3+3 3 3", 3, 28579, THREE_CHECK},
            {"rnbqkbnr/ppp2ppp/8/3pp3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1+3 0 3", 3, 33373, THREE_CHECK},
            {"r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2QK2R[Pp] w KQ - 0 9", 3, 236609, CRAZYHOUSE},
            {"rnbq1bnr/ppp1pppp/4k3/3p4/8/4K3/PPPP1PPP/RNBQ1BNR w - - 4 5", 4, 666008, KING_OF_THE_HILL},
            {"rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP w kq - 0 1", 4, 23310, HORDE},
            {"8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1", 4, 296242, RACING_KINGS},
            {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1", 4, 153299, ANTICHESS},
        };
        g_checkHash = true;
        bool allOk = true;
        for (const auto& c : cases) {
            Board b;
            b.variant = c.variant;
            if (!b.setFEN(c.fen)) std::printf("bad FEN: %s\n", c.fen);
            uint64_t n = perft(b, c.depth);
            bool ok = n == c.expected;
            allOk &= ok;
            std::printf("%s perft(%d) = %llu (expected %llu)  [%s] %s\n", ok ? "OK  " : "FAIL", c.depth,
                        (unsigned long long)n, (unsigned long long)c.expected, variantUciName(c.variant), c.fen);
        }
        std::printf("hash consistency: %s\n", g_hashOk ? "OK" : "FAIL");
        const bool magicsOk = checkMagics();
        std::printf("magic bitboards: %s\n", magicsOk ? "OK" : "FAIL");
        allOk &= magicsOk;
        std::printf("%s in %lld ms\n", allOk && g_hashOk ? "ALL PASSED" : "FAILURES", (long long)ms());
        return allOk && g_hashOk ? 0 : 1;
    }

    if (cmd == "perft" && argc > 2) {
        Board b;
        int i = 3;
        b.variant = variantArg(argc, argv, i);
        std::string fen = joinArgs(argc, argv, i);
        if (fen.empty()) b.setStartPos(); else if (!b.setFEN(fen)) return std::printf("bad FEN\n"), 1;
        uint64_t n = perft(b, std::atoi(argv[2]));
        std::printf("%llu nodes in %lld ms\n", (unsigned long long)n, (long long)ms());
        return 0;
    }

    if (cmd == "tbgen" || cmd == "tbprobe" || cmd == "search" || cmd == "bench") {
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
        int i = 3;
        b.variant = variantArg(argc, argv, i);
        std::string fen = joinArgs(argc, argv, i);
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

    // Fixed-node search over set positions, single thread: the signature (total of every position's final
    // depth, score and move) only changes if the search itself changes, so speed-only builds must match it.
    // Also the profiling workload for PGO builds.
    if (cmd == "bench") {
        const char* fens[] = {
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2NBPN2/PP3PPP/R2QK2R w KQ - 0 9",
            "r2q1rk1/1b2bppp/p2ppn2/1p6/3NP3/1BN1Q3/PPP2PPP/R4RK1 w - - 0 13",
            "2r2rk1/pp1bqppp/2n1pn2/3p4/3P4/2PBPN2/P1Q2PPP/R4RK1 w - - 0 15",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "r1bqkb1r/pp3ppp/2n1pn2/2pp4/3P4/2PBPN2/PP3PPP/RNBQK2R w KQkq - 0 6",
            "2kr3r/ppp2ppp/2n1b3/2b1p3/4P3/2N1BN2/PPP2PPP/2KR1B1R w - - 4 11",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "8/5pk1/6p1/3R4/7P/6P1/r4PK1/8 w - - 0 40",
            "4r1k1/1p3ppp/p1p5/3n4/3P4/2N2P2/PP4PP/4R1K1 b - - 0 22",
            "r2qr1k1/pb1nbppp/1p3n2/2pp4/3P4/1PN1PN2/PBQ1BPPP/R4RK1 w - - 0 12",
        };
        const int nodes = argc > 2 ? std::atoi(argv[2]) : 2000000;
        uint64_t total = 0, sig = 0;
        long long t0 = ms();
        for (const char* fen : fens) {
            Board b;
            b.setFEN(fen);
            clearSearchMemory();
            SearchInfo last;
            Move m = findBestMove(b, 1 << 30, nullptr, [&](const SearchInfo& si) { last = si; }, false, 1, nodes);
            std::printf("depth %2d  score %7s  nodes %9llu  %-6s %s\n", last.depth, scoreToString(last.score).c_str(),
                        (unsigned long long)last.nodes, b.moveToSAN(m).c_str(), fen);
            total += last.nodes;
            sig = sig * 1000003 + (uint64_t(last.depth) << 40 | uint64_t(last.score + 40000) << 20 | (m & 0xFFFFF));
        }
        long long t = std::max(1LL, ms() - t0);
        std::printf("signature %016llx  nodes %llu  time %lld ms  %llu nodes/s  large pages %s\n",
                    (unsigned long long)sig, (unsigned long long)total, t, (unsigned long long)(total * 1000 / t),
                    hashOnLargePages() ? "yes" : "no");
        return 0;
    }

    std::printf("usage: chesscli test | perft <d> [fen] | tbgen | tbprobe <fen> | search <ms> [fen] | bench [nodes]\n");
    return 1;
}
