#include "tablebase.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>

namespace tb {
namespace {

struct Table {
    std::string key;
    int n = 0;           // number of pieces
    int piece[4] = {};   // piece codes: white king, white pieces, black king, black pieces
    bool pawns = false;
    uint32_t size = 0;   // positions per side to move
    std::vector<int8_t> data;
    std::atomic<bool> ready{false};
};

std::map<std::string, std::unique_ptr<Table>> g_tables;
std::string g_dir;
std::atomic<int> g_ready{0};

const int TRI[10] = {0, 1, 2, 3, 9, 10, 11, 18, 19, 27};  // a1-d1-d4 triangle
int g_triIndex[64];

int orderRank(int t) {
    switch (t) {
        case QUEEN: return 0;
        case ROOK: return 1;
        case BISHOP: return 2;
        case KNIGHT: return 3;
        default: return 4;
    }
}
int roughValue(int t) {
    switch (t) {
        case QUEEN: return 9;
        case ROOK: return 5;
        case BISHOP:
        case KNIGHT: return 3;
        default: return 1;
    }
}
char typeChar(int t) { return " PNBRQK"[t]; }

void sortSide(std::vector<int>& v) {
    std::sort(v.begin(), v.end(), [](int a, int b) { return orderRank(a) < orderRank(b); });
}

// True if side a should be the "white" (stronger) side of the canonical key.
bool stronger(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() != b.size()) return a.size() > b.size();
    int va = 0, vb = 0;
    for (int t : a) va += roughValue(t);
    for (int t : b) vb += roughValue(t);
    if (va != vb) return va > vb;
    for (size_t i = 0; i < a.size(); i++)
        if (orderRank(a[i]) != orderRank(b[i])) return orderRank(a[i]) < orderRank(b[i]);
    return true;
}

std::string sideString(const std::vector<int>& v) {
    std::string s = "K";
    for (int t : v) s += typeChar(t);
    return s;
}

std::string makeKey(std::vector<int> w, std::vector<int> b, bool& flip) {
    sortSide(w);
    sortSide(b);
    flip = !stronger(w, b);
    return flip ? sideString(b) + "v" + sideString(w) : sideString(w) + "v" + sideString(b);
}

int charType(char c) {
    switch (c) {
        case 'P': return PAWN;
        case 'N': return KNIGHT;
        case 'B': return BISHOP;
        case 'R': return ROOK;
        case 'Q': return QUEEN;
        default: return KING;
    }
}

void parseKey(const std::string& key, std::vector<int>& w, std::vector<int>& b) {
    size_t v = key.find('v');
    for (size_t i = 1; i < v; i++) w.push_back(charType(key[i]));
    for (size_t i = v + 2; i < key.size(); i++) b.push_back(charType(key[i]));
}

inline int xform(int s, int t) {
    int f = s & 7, r = s >> 3;
    if (t & 1) f = 7 - f;
    if (t & 2) r = 7 - r;
    if (t & 4) std::swap(f, r);
    return r * 8 + f;
}

// Canonical index. Pawnless tables use the 8-fold board symmetry with the white king in the
// a1-d1-d4 triangle; pawn tables only use the left/right mirror.
uint32_t indexOf(const Table& T, const int* s, int stm) {
    if (T.pawns) {
        int m = (s[0] & 7) >= 4 ? 7 : 0;
        int k = s[0] ^ m;
        uint32_t idx = (k >> 3) * 4 + (k & 7);
        for (int i = 1; i < T.n; i++) idx = idx * 64 + (s[i] ^ m);
        return stm * T.size + idx;
    }
    uint32_t best = UINT32_MAX;
    for (int t = 0; t < 8; t++) {
        int k = xform(s[0], t);
        if (g_triIndex[k] < 0) continue;
        uint32_t idx = g_triIndex[k];
        for (int i = 1; i < T.n; i++) idx = idx * 64 + xform(s[i], t);
        best = std::min(best, idx);
    }
    return stm * T.size + best;
}

void decode(const Table& T, uint32_t idx, int* s, int& stm) {
    stm = idx / T.size;
    uint32_t r = idx % T.size;
    for (int i = T.n - 1; i >= 1; i--) {
        s[i] = r & 63;
        r >>= 6;
    }
    s[0] = T.pawns ? int((r / 4) * 8 + r % 4) : TRI[r];
}

void setup(Board& b, const Table& T, const int* s, int stm) {
    b.clear();
    for (int i = 0; i < T.n; i++) b.put(s[i], T.piece[i]);
    b.stm = Color(stm);
    b.checkCache = -1;
}

// Index of the board's position in table T. If flip, colors are swapped and ranks mirrored.
uint32_t boardIndex(const Table& T, const Board& b, bool flip) {
    int s[4];
    bool used[4] = {false, false, false, false};
    for (int q = 0; q < 64; q++) {
        int p = b.sq[q];
        if (!p) continue;
        int pq = flip ? (q ^ 56) : q;
        int pp = flip ? (p ^ 8) : p;
        for (int i = 0; i < T.n; i++)
            if (!used[i] && T.piece[i] == pp) {
                used[i] = true;
                s[i] = pq;
                break;
            }
    }
    return indexOf(T, s, flip ? (b.stm ^ 1) : b.stm);
}

Table* findTable(const Board& b, bool& flip) {
    std::vector<int> w, bl;
    for (int q = 0; q < 64; q++) {
        int p = b.sq[q];
        if (p && typeOf(p) != KING) (colorOf(p) == WHITE ? w : bl).push_back(typeOf(p));
    }
    auto it = g_tables.find(makeKey(w, bl, flip));
    return it == g_tables.end() ? nullptr : it->second.get();
}

std::string tablePath(const Table& T) { return g_dir + "/" + T.key + ".tb"; }

bool loadTable(Table& T) {
    std::ifstream in(tablePath(T), std::ios::binary);
    if (!in) return false;
    std::vector<int8_t> data(size_t(T.size) * 2);
    in.read(reinterpret_cast<char*>(data.data()), data.size());
    if (size_t(in.gcount()) != data.size() || in.peek() != EOF) return false;
    T.data = std::move(data);
    return true;
}

void saveTable(const Table& T) {
    std::string path = tablePath(T), tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        out.write(reinterpret_cast<const char*>(T.data.data()), T.data.size());
        if (!out) return;
    }
    std::remove(path.c_str());
    std::rename(tmp.c_str(), path.c_str());
}

std::vector<std::string> dependencies(const Table& T) {
    std::vector<int> side[2];
    parseKey(T.key, side[0], side[1]);
    std::set<std::string> deps;
    auto add = [&](std::vector<int> a, std::vector<int> b) {
        if (a.empty() && b.empty()) return;  // bare kings
        bool flip;
        deps.insert(makeKey(a, b, flip));
    };
    for (int c = 0; c < 2; c++) {
        const auto& us = side[c];
        const auto& them = side[c ^ 1];
        auto withSides = [&](const std::vector<int>& u, const std::vector<int>& t) {
            if (c == 0) add(u, t); else add(t, u);
        };
        // Captures of an enemy piece.
        for (size_t j = 0; j < them.size(); j++) {
            auto t2 = them;
            t2.erase(t2.begin() + j);
            withSides(us, t2);
        }
        // Promotions, with or without a capture.
        for (size_t i = 0; i < us.size(); i++) {
            if (us[i] != PAWN) continue;
            for (int pr : {QUEEN, ROOK, BISHOP, KNIGHT}) {
                auto u2 = us;
                u2[i] = pr;
                withSides(u2, them);
                for (size_t j = 0; j < them.size(); j++) {
                    auto t2 = them;
                    t2.erase(t2.begin() + j);
                    withSides(u2, t2);
                }
            }
        }
    }
    deps.erase(T.key);
    return {deps.begin(), deps.end()};
}

enum : uint8_t { F_VALID = 1, F_DONE = 2, F_NEVERLOSS = 4 };

bool isExit(Move m) { return isCapture(m) || movePromo(m); }

// Retrograde analysis. Positions are resolved in order of increasing distance to mate:
// a position is a win at ply q+1 if some move reaches a loss at ply q, and a loss once every
// move reaches a resolved win. A per-position cursor remembers how many moves are already
// known to be wins for the opponent, so each move is examined only a few times.
bool generate(Table& T) {
    const uint32_t N = T.size * 2;
    const uint32_t SEED = 1u << 31;
    const int MAXPLY = 400;
    std::vector<uint8_t> flags(N, 0), cursor(N, 0), maxd(N, 0);
    std::vector<int16_t> val(N, 0);
    std::vector<std::vector<uint32_t>> buckets(MAXPLY + 2);
    Board b;
    int s[4], stm;
    int bk = 1;
    while (typeOf(T.piece[bk]) != KING) bk++;

    for (uint32_t idx = 0; idx < N; idx++) {
        decode(T, idx, s, stm);
        bool ok = true;
        for (int i = 0; i < T.n && ok; i++) {
            for (int j = 0; j < i; j++)
                if (s[i] == s[j]) ok = false;
            if (typeOf(T.piece[i]) == PAWN && (rankOf(s[i]) == 0 || rankOf(s[i]) == 7)) ok = false;
        }
        if (!ok) continue;
        if (std::abs(fileOf(s[0]) - fileOf(s[bk])) <= 1 && std::abs(rankOf(s[0]) - rankOf(s[bk])) <= 1) continue;
        if (indexOf(T, s, stm) != idx) continue;  // non-canonical duplicate
        setup(b, T, s, stm);
        if (b.attacked(b.kingSq[stm ^ 1], stm)) continue;
        flags[idx] = F_VALID;

        MoveList ml;
        b.genMoves(ml);
        int legal = 0, bestWin = INT_MAX, maxLoss = 0;
        bool anyQuiet = false, neverLoss = false;
        for (int i = 0; i < ml.count; i++) {
            Move m = ml.moves[i];
            if (!b.makeMove(m)) continue;
            legal++;
            if (isExit(m)) {
                int cv;
                if (!probe(b, cv)) {
                    std::fprintf(stderr, "tb: missing dependency while generating %s\n", T.key.c_str());
                    return false;
                }
                if (cv < 0)
                    bestWin = std::min(bestWin, -cv);  // opponent mated in -cv-1 plies
                else if (cv == 0)
                    neverLoss = true;
                else
                    maxLoss = std::max(maxLoss, cv);
            } else {
                anyQuiet = true;
            }
            b.unmakeMove();
        }
        if (!legal) {
            flags[idx] |= F_DONE;
            if (b.inCheck()) {
                val[idx] = -1;  // checkmated
                buckets[0].push_back(idx);
            }
            continue;
        }
        if (bestWin != INT_MAX) {
            buckets[bestWin].push_back(idx | SEED);
            neverLoss = true;
        }
        if (neverLoss) flags[idx] |= F_NEVERLOSS;
        maxd[idx] = uint8_t(maxLoss);
        if (!anyQuiet && !neverLoss) {
            int L = maxLoss + 1;
            val[idx] = int16_t(-(L + 1));
            flags[idx] |= F_DONE;
            buckets[L].push_back(idx);
        }
    }

    // Returns true if every move from pidx now leads to a resolved win for the opponent.
    auto allMovesLose = [&](uint32_t pidx) {
        int ps[4], pstm;
        decode(T, pidx, ps, pstm);
        setup(b, T, ps, pstm);
        MoveList ml;
        b.genMoves(ml);
        int k = cursor[pidx], md = maxd[pidx];
        for (; k < ml.count; k++) {
            Move m = ml.moves[k];
            if (isExit(m) || !b.makeMove(m)) continue;
            uint32_t cidx = boardIndex(T, b, false);
            b.unmakeMove();
            if (!(flags[cidx] & F_DONE) || val[cidx] <= 0) {
                cursor[pidx] = uint8_t(k);
                maxd[pidx] = uint8_t(md);
                return false;
            }
            md = std::max<int>(md, val[cidx]);
        }
        cursor[pidx] = uint8_t(k);
        maxd[pidx] = uint8_t(md);
        return true;
    };

    for (int q = 0; q <= MAXPLY; q++) {
        auto& B = buckets[q];
        // Finalize capture/promotion wins seeded at this ply before propagating anything.
        for (uint32_t& e : B) {
            if (!(e & SEED)) continue;
            e &= ~SEED;
            if (flags[e] & F_DONE)
                e = UINT32_MAX;
            else
                flags[e] |= F_DONE, val[e] = int16_t(q);
        }
        for (size_t j = 0; j < B.size(); j++) {
            uint32_t idx = B[j];
            if (idx == UINT32_MAX) continue;
            bool win = val[idx] > 0;
            decode(T, idx, s, stm);
            int mover = stm ^ 1;  // side that made the last move
            // Pieces move reversibly, so a piece's origin squares are its own quiet move targets.
            MoveList back;
            setup(b, T, s, mover);
            b.genMoves(back);
            for (int i = 0; i < T.n; i++) {
                int p = T.piece[i];
                if (colorOf(p) != mover) continue;
                int t = typeOf(p), cur = s[i];
                int from[32], nf = 0;
                auto occupied = [&](int q2) {
                    for (int j2 = 0; j2 < T.n; j2++)
                        if (s[j2] == q2) return true;
                    return false;
                };
                if (t == PAWN) {
                    int dir = mover == WHITE ? 8 : -8;
                    int f1 = cur - dir;
                    int r1 = rankOf(f1);
                    if (r1 >= 1 && r1 <= 6 && !occupied(f1)) {
                        from[nf++] = f1;
                        int f2 = f1 - dir;
                        if ((mover == WHITE ? rankOf(cur) == 3 : rankOf(cur) == 4) && !occupied(f2)) from[nf++] = f2;
                    }
                } else {
                    for (int k = 0; k < back.count; k++) {
                        Move m = back.moves[k];
                        if (moveFrom(m) == cur && !isCapture(m)) from[nf++] = moveTo(m);
                    }
                }
                for (int k = 0; k < nf; k++) {
                    int s2[4] = {s[0], s[1], s[2], s[3]};
                    s2[i] = from[k];
                    uint32_t pidx = indexOf(T, s2, mover);
                    uint8_t fl = flags[pidx];
                    if (!(fl & F_VALID) || (fl & F_DONE)) continue;
                    if (!win) {
                        flags[pidx] |= F_DONE;
                        val[pidx] = int16_t(q + 1);
                        buckets[q + 1].push_back(pidx);
                    } else if (!(fl & F_NEVERLOSS) && allMovesLose(pidx)) {
                        int L = maxd[pidx] + 1;
                        flags[pidx] |= F_DONE;
                        val[pidx] = int16_t(-(L + 1));
                        buckets[L].push_back(pidx);
                    }
                }
            }
        }
        std::vector<uint32_t>().swap(B);
    }

    T.data.assign(N, 0);
    for (uint32_t i = 0; i < N; i++) {
        if (!(flags[i] & F_DONE)) continue;
        if (val[i] > 127 || val[i] < -128) {
            std::fprintf(stderr, "tb: %s distance overflow\n", T.key.c_str());
            return false;
        }
        T.data[i] = int8_t(val[i]);
    }
    return true;
}

bool ensure(Table& T, const StatusFn& status) {
    if (T.ready.load(std::memory_order_acquire)) return true;
    for (const auto& d : dependencies(T)) {
        auto it = g_tables.find(d);
        if (it == g_tables.end() || !ensure(*it->second, status)) return false;
    }
    if (!loadTable(T)) {
        if (status)
            status("Generating " + T.key + " (" + std::to_string(g_ready.load()) + "/" +
                   std::to_string(totalCount()) + " ready)");
        if (!generate(T)) return false;
        saveTable(T);
    }
    T.ready.store(true, std::memory_order_release);
    g_ready++;
    return true;
}

}  // namespace

void init(const std::string& directory) {
    initTables();
    g_dir = directory;
    std::error_code ec;
    std::filesystem::create_directories(g_dir, ec);
    for (int i = 0; i < 64; i++) g_triIndex[i] = -1;
    for (int i = 0; i < 10; i++) g_triIndex[TRI[i]] = i;
    if (!g_tables.empty()) return;

    const int types[5] = {QUEEN, ROOK, BISHOP, KNIGHT, PAWN};
    std::set<std::string> keys;
    bool flip;
    for (int a : types) {
        keys.insert(makeKey({a}, {}, flip));
        for (int c : types) {
            keys.insert(makeKey({a, c}, {}, flip));
            keys.insert(makeKey({a}, {c}, flip));
        }
    }
    for (const auto& key : keys) {
        auto T = std::make_unique<Table>();
        T->key = key;
        std::vector<int> w, b;
        parseKey(key, w, b);
        int n = 0;
        T->piece[n++] = makePiece(WHITE, KING);
        for (int t : w) T->piece[n++] = makePiece(WHITE, t);
        T->piece[n++] = makePiece(BLACK, KING);
        for (int t : b) T->piece[n++] = makePiece(BLACK, t);
        T->n = n;
        T->pawns = key.find('P') != std::string::npos;
        T->size = T->pawns ? 32 : 10;
        for (int i = 1; i < n; i++) T->size *= 64;
        g_tables[key] = std::move(T);
    }
}

bool loadOrGenerateAll(const StatusFn& status) {
    std::vector<Table*> order;
    for (auto& kv : g_tables) order.push_back(kv.second.get());
    std::stable_sort(order.begin(), order.end(), [](Table* a, Table* b) {
        if (a->n != b->n) return a->n < b->n;
        return a->pawns < b->pawns;
    });
    bool ok = true;
    for (Table* T : order) ok = ensure(*T, status) && ok;
    if (status) status(ok ? "All " + std::to_string(totalCount()) + " tables ready (up to 4 pieces)"
                          : "Some tables failed to generate");
    return ok;
}

bool probe(const Board& b, int& value) {
    if (b.pieceCount > 4) return false;
    if (b.pieceCount == 2) {
        value = 0;
        return true;
    }
    if (b.ep >= 0 || b.castle) return false;
    bool flip;
    Table* T = findTable(b, flip);
    if (!T || !T->ready.load(std::memory_order_acquire)) return false;
    value = T->data[boardIndex(*T, b, flip)];
    return true;
}

int readyCount() { return g_ready.load(); }
int totalCount() { return int(g_tables.size()); }

std::vector<TableStats> stats() {
    std::vector<TableStats> out;
    for (auto& kv : g_tables) {
        Table& T = *kv.second;
        if (!T.ready) continue;
        TableStats st;
        st.key = T.key;
        for (uint32_t i = 0; i < T.size; i++) {  // white to move
            int v = T.data[i];
            if (v > 0) st.wins++, st.longestMate = std::max(st.longestMate, (v + 1) / 2);
            if (v < 0) st.losses++;
        }
        out.push_back(st);
    }
    return out;
}

}  // namespace tb
