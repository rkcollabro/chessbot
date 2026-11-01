// Offline tools: PGN reading, opening book building, tuning-position extraction and Texel tuning.
#include "datagen.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <unordered_map>

#include "book.h"
#include "eval.h"
#include "search.h"
#include "tablebase.h"

namespace {

struct Game {
    double result = -1;  // 1, 0.5, 0 from White's view; -1 unknown
    int whiteElo = 0, blackElo = 0;
    int baseSeconds = -1;  // from the TimeControl tag, e.g. "180+2" -> 180
    std::vector<std::string> moves;
};

// Reads the next game from a PGN stream. Returns false at end of file.
bool readGame(std::istream& in, Game& g) {
    g = Game();
    std::string line, text;
    bool inMoves = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) {
            if (inMoves) break;
            continue;
        }
        if (line[0] == '[') {
            if (inMoves) break;  // malformed: headers after moves
            auto val = [&](const char* tag) -> std::string {
                if (line.compare(1, std::strlen(tag), tag) || line[1 + std::strlen(tag)] != ' ') return "";
                size_t a = line.find('"'), b = line.rfind('"');
                return a != std::string::npos && b > a ? line.substr(a + 1, b - a - 1) : "";
            };
            std::string r = val("Result");
            if (r == "1-0") g.result = 1;
            else if (r == "0-1") g.result = 0;
            else if (r == "1/2-1/2") g.result = 0.5;
            std::string tc = val("TimeControl");
            if (!tc.empty() && std::isdigit((unsigned char)tc[0])) g.baseSeconds = std::atoi(tc.c_str());
            std::string we = val("WhiteElo"), be = val("BlackElo");
            if (!we.empty()) g.whiteElo = std::atoi(we.c_str());
            if (!be.empty()) g.blackElo = std::atoi(be.c_str());
            continue;
        }
        inMoves = true;
        text += line + ' ';
    }
    if (!inMoves && in.eof()) return false;

    // Tokenize movetext, skipping comments, variations, NAGs, move numbers and the result.
    int braces = 0, parens = 0;
    std::string tok;
    auto flush = [&] {
        if (tok.empty()) return;
        size_t dot = tok.find_last_of('.');
        if (dot != std::string::npos) tok = tok.substr(dot + 1);
        if (!tok.empty() && tok[0] != '$' && tok != "1-0" && tok != "0-1" && tok != "1/2-1/2" && tok != "*" &&
            !std::isdigit((unsigned char)tok[0]))
            g.moves.push_back(tok);
        tok.clear();
    };
    for (char c : text) {
        if (c == '{') { flush(); braces++; continue; }
        if (c == '}') { braces--; continue; }
        if (braces) continue;
        if (c == '(') { flush(); parens++; continue; }
        if (c == ')') { parens--; continue; }
        if (parens) continue;
        if (std::isspace((unsigned char)c)) flush();
        else tok += c;
    }
    flush();
    return true;
}

template <class F>
void forEachGame(const std::vector<std::string>& files, F&& fn) {
    long long n = 0;
    for (const auto& f : files) {
        std::ifstream in(f);
        if (!in) {
            std::fprintf(stderr, "cannot open %s\n", f.c_str());
            continue;
        }
        Game g;
        while (readGame(in, g)) {
            if (!fn(g)) return;
            if (++n % 100000 == 0) std::fprintf(stderr, "  %lld games\n", n);
        }
    }
}

}  // namespace

// ---------------------------------------------------------------- opening book

// Pass 1 counts (position, move) pairs from qualifying games; pass 2 records a FEN for each kept
// position. The result is a text file "fen;uci;weight;games;score" so the moves can be vetted
// (tools/vet_book.py) before `booktxt` turns it into book.bin.
int buildBook(const std::string& outTxt, int maxPly, int minElo, int minBaseSeconds,
              const std::vector<std::string>& pgns) {
    struct Stat {
        uint64_t hash = 0;
        uint16_t move = 0;
        uint32_t games = 0;
        float score = 0;  // for the side making the move
    };
    auto qualifies = [&](const Game& g) {
        return g.result >= 0 && std::min(g.whiteElo, g.blackElo) >= minElo && g.baseSeconds >= minBaseSeconds;
    };
    std::unordered_map<uint64_t, Stat> stats;  // keyed by a mix of position hash and move
    stats.reserve(1 << 22);
    long long used = 0;
    forEachGame(pgns, [&](const Game& g) {
        if (!qualifies(g)) return true;
        used++;
        Board b;
        b.setStartPos();
        for (int ply = 0; ply < maxPly && ply < int(g.moves.size()); ply++) {
            Move m = b.parseSAN(g.moves[ply]);
            if (!m) break;
            uint16_t mv = uint16_t(m & 0x7FFF);
            Stat& s = stats[b.hash * 0x9E3779B97F4A7C15ULL ^ mv];
            s.hash = b.hash;
            s.move = mv;
            s.games++;
            s.score += float(b.stm == WHITE ? g.result : 1 - g.result);
            b.makeMove(m);
        }
        return true;
    });

    std::vector<Stat> kept;
    std::unordered_map<uint64_t, std::string> fens;  // positions that have at least one kept move
    for (auto& kv : stats) {
        const Stat& s = kv.second;
        if (s.games < 20) continue;  // too rare to trust
        if (s.score / s.games < 0.40) continue;  // moves that score badly for the mover
        kept.push_back(s);
        fens.emplace(s.hash, "");
    }
    stats.clear();

    forEachGame(pgns, [&](const Game& g) {
        if (!qualifies(g)) return true;
        Board b;
        b.setStartPos();
        for (int ply = 0; ply < maxPly && ply < int(g.moves.size()); ply++) {
            auto it = fens.find(b.hash);
            if (it == fens.end()) break;  // left the book tree
            if (it->second.empty()) it->second = b.getFEN();
            Move m = b.parseSAN(g.moves[ply]);
            if (!m) break;
            b.makeMove(m);
        }
        return true;
    });

    std::ofstream f(outTxt);
    for (const Stat& s : kept) {
        const std::string& fen = fens[s.hash];
        if (fen.empty()) continue;
        Board b;
        b.setFEN(fen);
        double pct = s.score / s.games;
        int weight = int(std::min(65535.0, std::sqrt(s.games * pct * pct) * 10));
        f << fen << ';' << b.moveToUCI(Move(s.move)) << ';' << weight << ';' << s.games << ';' << pct << '\n';
    }
    std::printf("book: %lld qualifying games, %zu moves in %zu positions written to %s\n", used, kept.size(),
                fens.size(), outTxt.c_str());
    return 0;
}

// Converts vetted "fen;uci;weight;..." lines into the binary book format.
int bookFromText(const std::string& txt, const std::string& outBin) {
    std::ifstream in(txt);
    if (!in) return std::printf("cannot open %s\n", txt.c_str()), 1;
    std::vector<book::Entry> entries;
    std::string line;
    while (std::getline(in, line)) {
        std::vector<std::string> parts;
        std::stringstream ss(line);
        for (std::string p; std::getline(ss, p, ';');) parts.push_back(p);
        if (parts.size() < 3) continue;
        Board b;
        if (!b.setFEN(parts[0])) continue;
        Move m = b.parseUCI(parts[1]);
        if (!m) continue;
        entries.push_back({b.hash, uint16_t(m & 0x7FFF), uint16_t(std::atoi(parts[2].c_str()))});
    }
    std::sort(entries.begin(), entries.end(), [](const book::Entry& a, const book::Entry& b) {
        return a.key != b.key ? a.key < b.key : a.weight > b.weight;
    });
    std::ofstream f(outBin, std::ios::binary);
    f.write(reinterpret_cast<const char*>(entries.data()), entries.size() * sizeof(book::Entry));
    std::printf("book: %zu entries written to %s\n", entries.size(), outBin.c_str());
    return 0;
}

// ---------------------------------------------------------------- tuning positions

int extractPositions(const std::string& out, int perGame, int maxPositions, const std::vector<std::string>& pgns) {
    std::ofstream f(out);
    std::mt19937 rng(12345);
    long long written = 0;
    forEachGame(pgns, [&](const Game& g) {
        if (g.result < 0 || g.moves.size() < 30) return true;
        Board b;
        b.setStartPos();
        std::vector<std::string> fens;
        for (size_t ply = 0; ply < g.moves.size(); ply++) {
            Move m = b.parseSAN(g.moves[ply]);
            if (!m) break;
            b.makeMove(m);
            if (ply >= 16 && !b.inCheck() && b.pieceCount > 4) fens.push_back(b.getFEN());
        }
        std::shuffle(fens.begin(), fens.end(), rng);
        for (int i = 0; i < perGame && i < int(fens.size()); i++) {
            f << fens[i] << ';' << g.result << '\n';
            written++;
        }
        return written < maxPositions;
    });
    std::printf("wrote %lld positions to %s\n", written, out.c_str());
    return 0;
}

// ---------------------------------------------------------------- NNUE training data

namespace {

// One training position, 68 bytes: up to 32 (square, piece) pairs (0xFF-padded), the search score and the
// game result, both from the side to move's point of view.
#pragma pack(push, 1)
struct PackedPos {
    uint8_t squares[32];
    uint8_t pieces[32];  // makePiece(color, type)
    int16_t score;       // centipawns
    uint8_t result;      // 2 win, 1 draw, 0 loss
    uint8_t stm;
};
#pragma pack(pop)
static_assert(sizeof(PackedPos) == 68, "PackedPos layout");

}  // namespace

// Self-play games at a fixed node count per move, starting from random 8-9 ply openings. Writes quiet
// positions (not in check, best move not a capture or promotion, no mate score) with the engine's score and
// the final result. Games end by the rules, by tablebase, by agreement on a +-20 pawn score, or at 400 plies.
int selfplay(const std::string& out, int games, int nodes, unsigned seed) {
    tb::loadOrGenerateAll(nullptr);
    std::FILE* f = std::fopen(out.c_str(), "ab");
    if (!f) return std::printf("cannot open %s\n", out.c_str()), 1;
    std::mt19937 rng(seed);
    long long positions = 0;
    for (int g = 0; g < games; g++) {
        Board b;
        b.setStartPos();
        clearSearchMemory();
        // Random opening, discarded if it already loses material or the game.
        int randomPlies = 8 + int(rng() % 2);
        bool ok = true;
        for (int i = 0; i < randomPlies && ok; i++) {
            MoveList ml;
            b.genLegal(ml);
            if (!ml.count) ok = false;
            else b.makeMove(ml.moves[rng() % ml.count]);
        }
        if (!ok) continue;
        int openingScore = 0;
        findBestMove(b, 1 << 30, nullptr, [&](const SearchInfo& si) { openingScore = si.score; }, false, 1, 2000);
        if (std::abs(openingScore) > 200) continue;

        std::vector<PackedPos> recorded;
        double whiteResult = -1;  // 1, 0.5, 0
        int decisive = 0;
        while (whiteResult < 0) {
            MoveList ml;
            b.genLegal(ml);
            if (!ml.count) {
                whiteResult = b.inCheck() ? (b.stm == WHITE ? 0 : 1) : 0.5;
                break;
            }
            if (b.halfmove >= 100 || b.repetitionCount() >= 3 || b.insufficientMaterial() || b.fullmove > 200) {
                whiteResult = 0.5;
                break;
            }
            int v;
            if (b.pieceCount <= 4 && tb::probe(b, v)) {
                whiteResult = v == 0 ? 0.5 : ((v > 0) == (b.stm == WHITE) ? 1 : 0);
                break;
            }
            int score = 0;
            Move m = findBestMove(b, 1 << 30, nullptr, [&](const SearchInfo& si) { score = si.score; }, false, 1,
                                  nodes);
            if (!m) break;
            decisive = std::abs(score) >= 2000 ? decisive + 1 : 0;
            if (decisive >= 4) {
                whiteResult = (score > 0) == (b.stm == WHITE) ? 1 : 0;
                break;
            }
            if (!b.inCheck() && !isCapture(m) && !movePromo(m) && std::abs(score) < 2000) {
                PackedPos p;
                std::memset(&p, 0xFF, sizeof(p));
                int k = 0;
                for (int s = 0; s < 64; s++)
                    if (b.sq[s] != EMPTY && k < 32) p.squares[k] = uint8_t(s), p.pieces[k++] = uint8_t(b.sq[s]);
                p.score = int16_t(score);
                p.stm = uint8_t(b.stm);
                recorded.push_back(p);
            }
            b.makeMove(m);
        }
        if (whiteResult < 0) continue;
        for (PackedPos& p : recorded) {
            double r = p.stm == WHITE ? whiteResult : 1 - whiteResult;
            p.result = uint8_t(r > 0.75 ? 2 : r < 0.25 ? 0 : 1);
        }
        std::fwrite(recorded.data(), sizeof(PackedPos), recorded.size(), f);
        std::fflush(f);
        positions += recorded.size();
        if ((g + 1) % 100 == 0) std::printf("games %d positions %lld\n", g + 1, positions), std::fflush(stdout);
    }
    std::fclose(f);
    std::printf("done: %lld positions\n", positions);
    return 0;
}

// ---------------------------------------------------------------- Texel tuning

namespace {

struct TuneEntry {
    float result;
    int phase;
    uint32_t start;
    uint16_t count;
};

const char* paramGroup(int i, int& base) {
    struct G { int off; const char* name; };
    static const G groups[] = {
        {OFF_MATERIAL, "material"}, {OFF_PST, "pst"}, {OFF_MOB_N, "mobility knight"},
        {OFF_MOB_B, "mobility bishop"}, {OFF_MOB_R, "mobility rook"}, {OFF_MOB_Q, "mobility queen"},
        {OFF_PASSED, "passed pawn"}, {OFF_ISOLATED, "isolated pawn"}, {OFF_DOUBLED, "doubled pawn"},
        {OFF_BISHOP_PAIR, "bishop pair"}, {OFF_ROOK_OPEN, "rook open file"},
        {OFF_ROOK_SEMI_OPEN, "rook semi-open file"}, {OFF_KING_ATTACK, "king attack"},
        {OFF_SAFE_CHECK, "safe check"}, {OFF_SHIELD, "king shield"}, {OFF_PAWN_THREAT, "pawn threat"},
        {OFF_TEMPO, "tempo"}};
    const G* best = &groups[0];
    for (const auto& g : groups)
        if (g.off <= i) best = &g;
    base = best->off;
    return best->name;
}

}  // namespace

int tune(const std::string& dataFile, const std::string& outInc, int epochs) {
    std::ifstream in(dataFile);
    if (!in) return std::printf("cannot open %s\n", dataFile.c_str()), 1;

    std::vector<TuneEntry> entries;
    std::vector<int16_t> fidx, fcnt;
    std::string line;
    long long read = 0, skipped = 0;
    EvalTrace tr;
    Board b;
    while (std::getline(in, line)) {
        size_t semi = line.find(';');
        if (semi == std::string::npos || !b.setFEN(line.substr(0, semi))) continue;
        read++;
        // Keep only quiet positions: the static evaluation must equal the quiescence result.
        int stat = evaluate(b);
        if (b.inCheck() || quiescenceScore(b) != stat) {
            skipped++;
            continue;
        }
        traceEval(b, tr);
        if (!tr.linear) {
            skipped++;
            continue;
        }
        TuneEntry e;
        e.result = float(std::atof(line.c_str() + semi + 1));
        e.phase = tr.phase;
        e.start = uint32_t(fidx.size());
        e.count = uint16_t(tr.features.size());
        for (auto& f : tr.features) fidx.push_back(f.first), fcnt.push_back(f.second);
        entries.push_back(e);
    }
    std::printf("tuning on %zu quiet positions (%lld read, %lld skipped)\n", entries.size(), read, skipped);
    if (entries.empty()) return 1;

    const int N = NUM_PARAMS;
    std::vector<double> mg(N), eg(N);
    for (int i = 0; i < N; i++) mg[i] = PARAMS[i].mg, eg[i] = PARAMS[i].eg;

    auto evalEntry = [&](const TuneEntry& e) {
        double m = 0, g = 0;
        for (uint32_t k = e.start; k < e.start + e.count; k++) {
            m += mg[fidx[k]] * fcnt[k];
            g += eg[fidx[k]] * fcnt[k];
        }
        return (m * e.phase + g * (MAX_PHASE - e.phase)) / MAX_PHASE;
    };
    auto sigmoid = [](double K, double x) { return 1.0 / (1.0 + std::pow(10.0, -K * x / 400.0)); };
    auto loss = [&](double K) {
        double sum = 0;
        for (const auto& e : entries) {
            double d = e.result - sigmoid(K, evalEntry(e));
            sum += d * d;
        }
        return sum / entries.size();
    };

    // Fit the scaling constant K for the starting weights.
    double K = 1.0, step = 0.5, bestLoss = loss(K);
    while (step > 0.001) {
        bool improved = false;
        for (double cand : {K - step, K + step}) {
            double l = loss(cand);
            if (cand > 0 && l < bestLoss) bestLoss = l, K = cand, improved = true;
        }
        if (!improved) step /= 2;
    }
    std::printf("K = %.3f, initial loss %.6f\n", K, bestLoss);

    // Adam on the mean squared error.
    std::vector<double> gm(N), ge(N), m1m(N), m1e(N), m2m(N), m2e(N);
    const double lr = 1.0, b1 = 0.9, b2 = 0.999, eps = 1e-8;
    for (int epoch = 1; epoch <= epochs; epoch++) {
        std::fill(gm.begin(), gm.end(), 0);
        std::fill(ge.begin(), ge.end(), 0);
        double total = 0;
        for (const auto& e : entries) {
            double s = sigmoid(K, evalEntry(e));
            double err = e.result - s;
            total += err * err;
            double gcoef = -2 * err * s * (1 - s) * K * std::log(10.0) / 400.0;
            double wm = gcoef * e.phase / MAX_PHASE, we = gcoef * (MAX_PHASE - e.phase) / MAX_PHASE;
            for (uint32_t k = e.start; k < e.start + e.count; k++) {
                gm[fidx[k]] += wm * fcnt[k];
                ge[fidx[k]] += we * fcnt[k];
            }
        }
        double n = double(entries.size());
        for (int i = 0; i < N; i++) {
            double g1 = gm[i] / n, g2 = ge[i] / n;
            m1m[i] = b1 * m1m[i] + (1 - b1) * g1;
            m2m[i] = b2 * m2m[i] + (1 - b2) * g1 * g1;
            m1e[i] = b1 * m1e[i] + (1 - b1) * g2;
            m2e[i] = b2 * m2e[i] + (1 - b2) * g2 * g2;
            double c1 = 1 - std::pow(b1, epoch), c2 = 1 - std::pow(b2, epoch);
            mg[i] -= lr * (m1m[i] / c1) / (std::sqrt(m2m[i] / c2) + eps);
            eg[i] -= lr * (m1e[i] / c1) / (std::sqrt(m2e[i] / c2) + eps);
        }
        if (epoch % 25 == 0 || epoch == 1) {
            std::printf("epoch %4d  loss %.6f\n", epoch, total / n);
            std::fflush(stdout);
        }
    }

    std::ofstream out(outInc);
    out << "// Generated by `chesscli tune` (Texel tuning on " << entries.size() << " positions). Do not edit.\n";
    int lastBase = -1;
    for (int i = 0; i < N; i++) {
        int base;
        const char* name = paramGroup(i, base);
        if (base != lastBase) {
            out << (lastBase >= 0 ? "\n" : "") << "// " << name << "\n";
            lastBase = base;
        }
        out << "{" << std::lround(mg[i]) << ", " << std::lround(eg[i]) << "},";
        out << ((i - base) % 8 == 7 ? "\n" : " ");
    }
    std::printf("wrote %s\n", outInc.c_str());
    return 0;
}
