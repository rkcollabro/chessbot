#include "search.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "eval.h"
#include "tablebase.h"

namespace {

constexpr int MAX_PLY = 128;

enum : uint8_t { TT_EXACT = 0, TT_LOWER = 1, TT_UPPER = 2 };

struct TTEntry {
    uint64_t key;
    Move move;
    int16_t score;
    uint8_t depth;
    uint8_t flag;
};

constexpr size_t TT_SIZE = size_t(1) << 21;  // 32 MB
std::vector<TTEntry> g_tt(TT_SIZE);
int g_history[16][64];

int scoreToTT(int s, int ply) {
    if (s > MATE_BOUND) return s + ply;
    if (s < -MATE_BOUND) return s - ply;
    return s;
}
int scoreFromTT(int s, int ply) {
    if (s > MATE_BOUND) return s - ply;
    if (s < -MATE_BOUND) return s + ply;
    return s;
}

// Tablebase value (plies to mate) -> search score at the given ply.
int tbScore(int v, int ply) {
    if (v > 0) return MATE - (ply + v);
    if (v < 0) return -(MATE - (ply + (-v - 1)));
    return 0;
}

using Clock = std::chrono::steady_clock;

class Searcher {
public:
    Board b;
    std::atomic<bool>* stopFlag = nullptr;
    Clock::time_point start;
    int limitMs = 1000;
    bool stopped = false;
    int iterDepth = 0;
    uint64_t nodes = 0;
    Move killers[MAX_PLY][2];
    Move pv[MAX_PLY][MAX_PLY];
    int pvLen[MAX_PLY];
    Move rootBest = NO_MOVE;
    int rootScore = 0;

    int elapsed() const {
        return int(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count());
    }

    void checkTime() {
        if (iterDepth <= 1) return;  // always finish depth 1
        if ((stopFlag && stopFlag->load()) || elapsed() >= limitMs) stopped = true;
    }

    int scoreMove(Move m, Move ttMove, int ply) {
        if (m == ttMove) return 2000000;
        int promo = movePromo(m);
        if (isCapture(m)) {
            int victim = (moveFlags(m) & MF_EP) ? PAWN : typeOf(b.sq[moveTo(m)]);
            int attacker = typeOf(b.sq[moveFrom(m)]);
            return 1000000 + PIECE_VALUE[victim] * 10 - PIECE_VALUE[attacker] / 10 + (promo == QUEEN ? 9000 : 0);
        }
        if (promo == QUEEN) return 950000;
        if (promo) return -100000;  // underpromotions last
        if (m == killers[ply][0]) return 900000;
        if (m == killers[ply][1]) return 890000;
        return g_history[b.sq[moveFrom(m)]][moveTo(m)];
    }

    static void pickMove(MoveList& ml, int* scores, int i) {
        int best = i;
        for (int j = i + 1; j < ml.count; j++)
            if (scores[j] > scores[best]) best = j;
        std::swap(ml.moves[i], ml.moves[best]);
        std::swap(scores[i], scores[best]);
    }

    int qsearch(int alpha, int beta, int ply) {
        pvLen[ply] = ply;
        if ((++nodes & 2047) == 0) checkTime();
        if (stopped) return 0;
        if (b.pieceCount <= 4) {
            int v;
            if (tb::probe(b, v)) return tbScore(v, ply);
        }
        if (ply >= MAX_PLY - 1) return evaluate(b);

        bool inCheck = b.inCheck();
        int best = -INF;
        if (!inCheck) {
            best = evaluate(b);
            if (best >= beta) return best;
            if (best > alpha) alpha = best;
        }
        MoveList ml;
        b.genMoves(ml, !inCheck);
        int scores[256];
        for (int i = 0; i < ml.count; i++) scores[i] = scoreMove(ml.moves[i], NO_MOVE, ply);
        int legal = 0;
        for (int i = 0; i < ml.count; i++) {
            pickMove(ml, scores, i);
            Move m = ml.moves[i];
            if (!b.makeMove(m)) continue;
            legal++;
            int s = -qsearch(-beta, -alpha, ply + 1);
            b.unmakeMove();
            if (stopped) return 0;
            if (s > best) {
                best = s;
                if (s > alpha) {
                    alpha = s;
                    if (s >= beta) break;
                }
            }
        }
        if (inCheck && !legal) return -MATE + ply;
        return best;
    }

    int negamax(int depth, int alpha, int beta, int ply, bool nullOk) {
        pvLen[ply] = ply;
        if ((++nodes & 2047) == 0) checkTime();
        if (stopped) return 0;
        const bool root = ply == 0;
        const bool pvNode = beta - alpha > 1;

        if (!root) {
            if (b.halfmove >= 100 || b.isRepetition() || b.insufficientMaterial()) return 0;
            if (b.pieceCount <= 4) {
                int v;
                if (tb::probe(b, v)) return tbScore(v, ply);
            }
            alpha = std::max(alpha, -MATE + ply);
            beta = std::min(beta, MATE - ply - 1);
            if (alpha >= beta) return alpha;
        }
        if (ply >= MAX_PLY - 1) return evaluate(b);

        bool inCheck = b.inCheck();
        if (inCheck) depth++;
        if (depth <= 0) return qsearch(alpha, beta, ply);

        TTEntry& e = g_tt[b.hash & (TT_SIZE - 1)];
        Move ttMove = NO_MOVE;
        if (e.key == b.hash) {
            ttMove = e.move;
            if (!pvNode && e.depth >= depth) {
                int s = scoreFromTT(e.score, ply);
                if (e.flag == TT_EXACT || (e.flag == TT_LOWER && s >= beta) || (e.flag == TT_UPPER && s <= alpha))
                    return s;
            }
        }

        int staticEval = inCheck ? -INF : evaluate(b);
        if (!pvNode && !inCheck) {
            // Reverse futility pruning.
            if (depth <= 3 && std::abs(beta) < MATE_BOUND && staticEval - 120 * depth >= beta) return staticEval;
            // Null-move pruning.
            if (nullOk && depth >= 3 && staticEval >= beta && b.hasNonPawnMaterial(b.stm)) {
                int R = 2 + depth / 4;
                b.makeNull();
                int s = -negamax(depth - 1 - R, -beta, -beta + 1, ply + 1, false);
                b.unmakeNull();
                if (stopped) return 0;
                if (s >= beta) return s >= MATE_BOUND ? beta : s;
            }
        }

        MoveList ml;
        b.genMoves(ml);
        int scores[256];
        for (int i = 0; i < ml.count; i++) scores[i] = scoreMove(ml.moves[i], ttMove, ply);

        int legal = 0, best = -INF;
        Move bestMove = NO_MOVE;
        const int origAlpha = alpha;
        for (int i = 0; i < ml.count; i++) {
            pickMove(ml, scores, i);
            Move m = ml.moves[i];
            bool quiet = !isCapture(m) && !movePromo(m);
            if (!b.makeMove(m)) continue;
            legal++;
            bool givesCheck = b.inCheck();
            int s;
            if (legal == 1) {
                s = -negamax(depth - 1, -beta, -alpha, ply + 1, true);
            } else {
                int r = 0;
                if (depth >= 3 && legal > 3 && quiet && !inCheck && !givesCheck) {
                    r = 1 + (legal > 8) + (depth > 7);
                    if (pvNode) r--;
                    r = std::min(r, depth - 2);
                }
                s = -negamax(depth - 1 - r, -alpha - 1, -alpha, ply + 1, true);
                if (s > alpha && r > 0) s = -negamax(depth - 1, -alpha - 1, -alpha, ply + 1, true);
                if (s > alpha && s < beta) s = -negamax(depth - 1, -beta, -alpha, ply + 1, true);
            }
            b.unmakeMove();
            if (stopped) return 0;

            if (s > best) {
                best = s;
                bestMove = m;
                if (s > alpha) {
                    alpha = s;
                    pv[ply][ply] = m;
                    for (int k = ply + 1; k < pvLen[ply + 1]; k++) pv[ply][k] = pv[ply + 1][k];
                    pvLen[ply] = std::max(pvLen[ply + 1], ply + 1);
                    if (root) {
                        rootBest = m;
                        rootScore = s;
                    }
                    if (s >= beta) {
                        if (quiet) {
                            if (killers[ply][0] != m) {
                                killers[ply][1] = killers[ply][0];
                                killers[ply][0] = m;
                            }
                            int& h = g_history[b.sq[moveFrom(m)]][moveTo(m)];
                            h = std::min(h + depth * depth, 800000);
                        }
                        break;
                    }
                }
            }
        }
        if (!legal) return inCheck ? -MATE + ply : 0;

        uint8_t flag = best >= beta ? TT_LOWER : (best > origAlpha ? TT_EXACT : TT_UPPER);
        if (e.key != b.hash || depth >= e.depth || flag == TT_EXACT) {
            e.key = b.hash;
            e.move = bestMove;
            e.score = int16_t(scoreToTT(best, ply));
            e.depth = uint8_t(std::min(depth, 255));
            e.flag = flag;
        }
        return best;
    }

    std::string pvToSAN(const Move* moves, int n) {
        Board copy = b;
        std::string out;
        for (int i = 0; i < n; i++) {
            Move m = copy.parseUCI(copy.moveToUCI(moves[i]));
            if (!m) break;
            if (!out.empty()) out += ' ';
            out += copy.moveToSAN(m);
            copy.makeMove(m);
        }
        return out;
    }

    // With 4 or fewer pieces, pick the move with the best distance to mate straight from the tables.
    bool tablebaseRoot(Move& best, const InfoFn& info) {
        if (b.pieceCount > 4 || b.ep >= 0 || b.castle) return false;
        MoveList ml;
        b.genLegal(ml);
        if (!ml.count) return false;
        int bestScore = -INF;
        for (int i = 0; i < ml.count; i++) {
            b.makeMove(ml.moves[i]);
            int v;
            bool ok = tb::probe(b, v);
            int s = ok ? -tbScore(v, 1) : 0;
            // Prefer not to repeat in drawn positions so the game keeps making progress.
            if (ok && v == 0 && b.isRepetition()) s -= 1;
            b.unmakeMove();
            if (!ok) return false;
            if (s > bestScore) bestScore = s, best = ml.moves[i];
        }
        if (info) {
            SearchInfo si;
            si.depth = 0;
            si.score = bestScore;
            si.ms = elapsed();
            si.pv = pvToSAN(&best, 1);
            int plies = MATE - std::abs(bestScore);
            if (bestScore > MATE_BOUND)
                si.note = "Tablebase: win, mate in " + std::to_string((plies + 1) / 2);
            else if (bestScore < -MATE_BOUND)
                si.note = "Tablebase: loss, mated in " + std::to_string((plies + 1) / 2);
            else
                si.note = "Tablebase: draw";
            info(si);
        }
        return true;
    }

    Move think(const InfoFn& info) {
        start = Clock::now();
        std::memset(killers, 0, sizeof(killers));
        for (auto& row : g_history)
            for (auto& h : row) h /= 4;

        Move best = NO_MOVE;
        if (tablebaseRoot(best, info)) return best;

        MoveList legal;
        b.genLegal(legal);
        if (!legal.count) return NO_MOVE;
        best = legal.moves[0];
        if (legal.count == 1) limitMs = std::min(limitMs, 50);

        for (int depth = 1; depth < MAX_PLY - 8; depth++) {
            iterDepth = depth;
            rootBest = NO_MOVE;
            int s = negamax(depth, -INF, INF, 0, false);
            if (rootBest) best = rootBest;  // a move fully searched this iteration is trustworthy
            if (stopped) break;
            if (info) {
                SearchInfo si;
                si.depth = depth;
                si.score = s;
                si.nodes = nodes;
                si.ms = elapsed();
                si.pv = pvToSAN(pv[0], pvLen[0]);
                info(si);
            }
            if (elapsed() > limitMs / 2) break;
            if (std::abs(s) > MATE_BOUND && depth > MATE - std::abs(s) + 2) break;
        }
        return best;
    }
};

}  // namespace

Move findBestMove(const Board& root, int timeMs, std::atomic<bool>* stop, const InfoFn& info) {
    auto s = std::make_unique<Searcher>();
    s->b = root;
    s->limitMs = timeMs;
    s->stopFlag = stop;
    return s->think(info);
}

void clearSearchMemory() {
    std::fill(g_tt.begin(), g_tt.end(), TTEntry{});
    std::memset(g_history, 0, sizeof(g_history));
}

std::string scoreToString(int score) {
    if (score > MATE_BOUND) return "M" + std::to_string((MATE - score + 1) / 2);
    if (score < -MATE_BOUND) return "-M" + std::to_string((MATE + score + 1) / 2);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%+.2f", score / 100.0);
    return buf;
}
