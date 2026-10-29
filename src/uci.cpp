#include "uci.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "board.h"
#include "book.h"
#include "search.h"
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
    std::thread([] { tb::loadOrGenerateAll(nullptr); }).detach();

    Board board;
    board.setStartPos();
    std::thread worker;
    std::atomic<bool> stop{false};
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
            std::printf("option name Threads type spin default 1 min 1 max 64\n");
            std::printf("uciok\n");
        } else if (cmd == "setoption") {
            std::string tok, name, value;
            in >> tok >> name;  // "name" <name>
            while (in >> tok && tok != "value") name += " " + tok;
            in >> value;
            if (name == "OwnBook") ownBook = value == "true";
            if (name == "Threads") threads = std::max(1, std::atoi(value.c_str()));
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
            bool infinite = false;
            std::string tok;
            while (in >> tok) {
                if (tok == "wtime") in >> wtime;
                else if (tok == "btime") in >> btime;
                else if (tok == "winc") in >> winc;
                else if (tok == "binc") in >> binc;
                else if (tok == "movetime") in >> movetime;
                else if (tok == "movestogo") in >> movestogo;
                else if (tok == "infinite") infinite = true;
            }
            int ms;
            if (movetime > 0) {
                ms = movetime;
            } else if (infinite) {
                ms = 1 << 30;
            } else {
                int left = board.stm == WHITE ? wtime : btime;
                int inc = board.stm == WHITE ? winc : binc;
                if (left < 0) left = 10000;
                int togo = movestogo > 0 ? movestogo : 30;
                ms = left / togo + inc * 3 / 4;
                ms = std::max(10, std::min(ms, left / 3));
            }
            stop = false;
            Board copy = board;
            worker = std::thread([copy, ms, ownBook, threads, &stop] {
                Move m = findBestMove(copy, ms, &stop, [](const SearchInfo& si) {
                    std::printf("info depth %d score %s nodes %llu time %d\n", si.depth,
                                si.score > MATE_BOUND    ? ("mate " + std::to_string((MATE - si.score + 1) / 2)).c_str()
                                : si.score < -MATE_BOUND ? ("mate -" + std::to_string((MATE + si.score + 1) / 2)).c_str()
                                                         : ("cp " + std::to_string(si.score)).c_str(),
                                (unsigned long long)si.nodes, si.ms);
                    std::fflush(stdout);
                }, ownBook, threads);
                std::printf("bestmove %s\n", m ? copy.moveToUCI(m).c_str() : "0000");
                std::fflush(stdout);
            });
        } else if (cmd == "stop") {
            stop = true;
        } else if (cmd == "quit") {
            break;
        }
        std::fflush(stdout);
    }
    join();
}
