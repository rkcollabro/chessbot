#include "uci.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "board.h"
#include "book.h"
#include "search.h"
#include "nnue.h"
#include "tablebase.h"

namespace {

// Applies "position [startpos | fen <fen>] [moves ...]".
void setPosition(Board& b, std::istringstream& in) {
    std::string tok, fen;
    in >> tok;
    if (tok == "startpos") {
        b.setStartPos();
        in >> tok;  // "moves" or nothing
    } else if (tok == "fen") {
        while (in >> tok && tok != "moves") fen += (fen.empty() ? "" : " ") + tok;
        if (!b.setFEN(fen)) b.setStartPos();
    }
    while (in >> tok) {
        Move m = b.parseUCI(tok);
        if (!m) break;
        b.makeMove(m);
    }
}

}  // namespace

void uciLoop(const std::string& tbDir, const std::string& bookPath) {
    tb::init(tbDir);
    book::load(bookPath);
    bool ownBook = true;
    int threads = 1;
    int level = 6;  // 1..6; time still comes from the "go" command
    int levelNodes = -1, levelNoise = -1;  // calibration overrides for the level's limits (-1 = level's own)
    std::thread([] { tb::loadOrGenerateAll(nullptr); }).detach();

    Board board;
    board.setStartPos();
    std::thread worker;
    std::atomic<bool> stop{false};
    std::atomic<bool> pondering{false};
    auto join = [&] {
        if (worker.joinable()) {
            stop = true;
            worker.join();
        }
    };

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string cmd;
        in >> cmd;
        if (cmd == "uci") {
            std::printf("id name ChessBot\nid author Randy\n");
            std::printf("option name OwnBook type check default true\n");
            std::printf("option name Ponder type check default false\n");
            std::printf("option name UseNNUE type check default true\n");
            std::printf("option name Threads type spin default 1 min 1 max 64\n");
            std::printf("option name Hash type spin default 64 min 1 max 8192\n");
            std::printf("option name Level type spin default 6 min 1 max 6\n");
            std::printf("option name LevelNodes type spin default -1 min -1 max 100000000\n");
            std::printf("option name LevelNoise type spin default -1 min -1 max 1000\n");
            std::printf("option name UCI_Chess960 type check default false\n");
            std::printf("option name UCI_Variant type combo default chess");
            for (int v = 0; v < NUM_VARIANTS; v++) std::printf(" var %s", variantUciName(v));
            std::printf("\n");
            for (const TuneParam& tp : tuneParams())
                std::printf("option name %s type spin default %d min %d max %d\n", tp.name, *tp.value, tp.lo, tp.hi);
            std::printf("uciok\n");
        } else if (cmd == "setoption") {
            std::string tok, name, value;
            in >> tok >> name;  // "name" <name>
            while (in >> tok && tok != "value") name += " " + tok;
            in >> value;
            for (const TuneParam& tp : tuneParams())
                if (name == tp.name) {
                    join();
                    *tp.value = std::max(tp.lo, std::min(tp.hi, std::atoi(value.c_str())));
                    tuneChanged();
                }
            if (name == "OwnBook") ownBook = value == "true";
            if (name == "UseNNUE") nnue::setEnabled(value == "true");
            if (name == "Threads") threads = std::max(1, std::atoi(value.c_str()));
            if (name == "Hash") {
                join();  // no search may be using the table while it moves
                setHashSize(std::atoi(value.c_str()));
                std::printf("info string hash %zu MB%s\n", hashSizeMB(), hashOnLargePages() ? " (large pages)" : "");
            }
            if (name == "Level") level = std::max(1, std::min(6, std::atoi(value.c_str())));
            if (name == "LevelNodes") levelNodes = std::atoi(value.c_str());
            if (name == "LevelNoise") levelNoise = std::atoi(value.c_str());
            if (name == "UCI_Variant" && variantFromUciName(value) >= 0) {
                join();
                board.variant = Variant(variantFromUciName(value));
                board.setStartPos();
            }
            if (name == "UCI_Chess960" && (board.variant == STANDARD || board.variant == CHESS960)) {
                join();
                board.variant = value == "true" ? CHESS960 : STANDARD;
            }
        } else if (cmd == "isready") {
            std::printf("readyok\n");
        } else if (cmd == "ucinewgame") {
            join();
            clearSearchMemory();
        } else if (cmd == "position") {
            join();
            setPosition(board, in);
        } else if (cmd == "go") {
            join();
            int wtime = -1, btime = -1, winc = 0, binc = 0, movetime = -1, movestogo = 0;
            bool infinite = false, ponder = false;
            std::string tok;
            while (in >> tok) {
                if (tok == "wtime") in >> wtime;
                else if (tok == "btime") in >> btime;
                else if (tok == "winc") in >> winc;
                else if (tok == "binc") in >> binc;
                else if (tok == "movetime") in >> movetime;
                else if (tok == "movestogo") in >> movestogo;
                else if (tok == "infinite") infinite = true;
                else if (tok == "ponder") ponder = true;
            }
            int ms, soft = 0;
            if (movetime > 0) {
                ms = movetime;
            } else if (infinite) {
                ms = 1 << 30;
            } else {
                // Aim for a share of the remaining clock, and allow up to five times that (but never more
                // than a third of the clock) when the search is unsettled.
                int left = board.stm == WHITE ? wtime : btime;
                int inc = board.stm == WHITE ? winc : binc;
                if (left < 0) left = 10000;
                int togo = movestogo > 0 ? std::min(movestogo, 40) : 25;
                soft = left / togo + inc * 3 / 4;
                ms = std::max(10, std::min(soft * 5, left / 3));
                soft = std::max(5, std::min(soft, ms));
            }
            stop = false;
            pondering = ponder;
            Board copy = board;
            EngineLevel lv = LEVELS[level - 1];
            if (levelNodes >= 0) lv.nodes = levelNodes;
            if (levelNoise >= 0) lv.noise = levelNoise;
            worker = std::thread([copy, ms, soft, ownBook, threads, lv, &stop, &pondering] {
                Move expected = NO_MOVE;
                Move m = findBestMove(copy, ms, &stop, [](const SearchInfo& si) {
                    std::printf("info depth %d score %s nodes %llu time %d\n", si.depth,
                                si.score > MATE_BOUND    ? ("mate " + std::to_string((MATE - si.score + 1) / 2)).c_str()
                                : si.score < -MATE_BOUND ? ("mate -" + std::to_string((MATE + si.score + 1) / 2)).c_str()
                                                         : ("cp " + std::to_string(si.score)).c_str(),
                                (unsigned long long)si.nodes, si.ms);
                    std::fflush(stdout);
                }, ownBook && lv.book, threads, lv.nodes, lv.noise, soft, &pondering, &expected);
                // UCI forbids answering a "go ponder" before "ponderhit" or "stop", even if the search ended.
                while (pondering && !stop) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                std::string reply = m ? copy.moveToUCI(m) : "0000";
                if (m && expected) {
                    Board after = copy;
                    after.makeMove(m);
                    MoveList legal;
                    after.genLegal(legal);
                    for (int i = 0; i < legal.count; i++)
                        if (legal.moves[i] == expected) reply += " ponder " + after.moveToUCI(expected);
                }
                std::printf("bestmove %s\n", reply.c_str());
                std::fflush(stdout);
            });
        } else if (cmd == "ponderhit") {
            pondering = false;
        } else if (cmd == "stop") {
            stop = true;
        } else if (cmd == "quit") {
            break;
        }
        std::fflush(stdout);
    }
    join();
}
