#include "board.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace {

Bitboard PAWN_ATT[2][64], KNIGHT_ATT[64], KING_ATT[64];
Bitboard RAY[8][64];  // N, E, NE, NW (increasing), S, W, SE, SW (decreasing)
Bitboard LINES[64];   // all eight rays from a square on an empty board
// Drawn after the original keys so standard-chess hashes (and the opening book) are unchanged.
uint64_t Z_piece[16][64], Z_side, Z_castle[16], Z_ep[8];
uint64_t Z_checks[2][8], Z_pocket[2][7][32];

// Squares from a to b inclusive, both on the same rank.
Bitboard rankSpan(int a, int b) {
    if (a > b) std::swap(a, b);
    Bitboard r = 0;
    for (int s = a; s <= b; s++) r |= bit(s);
    return r;
}

uint64_t splitmix(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

}  // namespace

namespace attacks {
Bitboard pawn(int color, int s) { return PAWN_ATT[color][s]; }
Bitboard knight(int s) { return KNIGHT_ATT[s]; }
Bitboard king(int s) { return KING_ATT[s]; }

// Magic bitboards: the blockers that matter (the slider's rays minus their edge squares) are
// multiplied by a per-square magic number whose top bits index a table of precomputed attacks.
// Tables are filled from the classical ray scan below, so the two always agree.
struct Magic {
    Bitboard mask;
    uint64_t magic;
    const Bitboard* table;
    int shift;
};
Magic ROOK_MAGIC[64], BISHOP_MAGIC[64];
Bitboard ROOK_TABLE[102400], BISHOP_TABLE[5248];

inline Bitboard magicLookup(const Magic& m, Bitboard occ) {
    return m.table[((occ & m.mask) * m.magic) >> m.shift];
}
Bitboard bishop(int s, Bitboard occ) { return magicLookup(BISHOP_MAGIC[s], occ); }
Bitboard rook(int s, Bitboard occ) { return magicLookup(ROOK_MAGIC[s], occ); }

// Classical ray attacks: the nearest blocker along each ray cuts off the rest of the ray.
Bitboard bishopSlow(int s, Bitboard occ) {
    Bitboard a = 0;
    for (int d : {2, 3, 6, 7}) {
        Bitboard r = RAY[d][s], blockers = r & occ;
        if (blockers) r ^= RAY[d][d < 4 ? lsb(blockers) : msb(blockers)];
        a |= r;
    }
    return a;
}
Bitboard rookSlow(int s, Bitboard occ) {
    Bitboard a = 0;
    for (int d : {0, 1, 4, 5}) {
        Bitboard r = RAY[d][s], blockers = r & occ;
        if (blockers) r ^= RAY[d][d < 4 ? lsb(blockers) : msb(blockers)];
        a |= r;
    }
    return a;
}

// Finds a magic for each square by trial: a candidate works if no two blocker sets that give
// different attacks land on the same table slot. Fixed seed, so the tables are the same every run.
void initMagics(Magic* magics, Bitboard* table, const int* dirs, Bitboard (*slow)(int, Bitboard)) {
    uint64_t seed = 0x5EED0F0A61C5ULL;
    static Bitboard occs[4096], refs[4096];
    static int epoch[4096], tries = 0;
    Bitboard* next = table;
    for (int s = 0; s < 64; s++) {
        Magic& m = magics[s];
        m.mask = 0;
        for (int i = 0; i < 4; i++) {
            Bitboard r = RAY[dirs[i]][s];
            if (r) m.mask |= r & ~bit(dirs[i] < 4 ? msb(r) : lsb(r));  // the edge square never blocks anything
        }
        const int bits = popcount(m.mask), n = 1 << bits;
        m.shift = 64 - bits;
        m.table = next;
        Bitboard sub = 0;
        for (int i = 0; i < n; i++) {  // every subset of the mask (carry-rippler)
            occs[i] = sub;
            refs[i] = slow(s, sub);
            sub = (sub - m.mask) & m.mask;
        }
        for (;;) {
            m.magic = splitmix(seed) & splitmix(seed) & splitmix(seed);  // sparse candidates work best
            if (popcount((m.mask * m.magic) >> 56) < 6) continue;
            tries++;
            bool ok = true;
            for (int i = 0; i < n && ok; i++) {
                size_t idx = (occs[i] * m.magic) >> m.shift;
                if (epoch[idx] != tries) {
                    epoch[idx] = tries;
                    next[idx] = refs[i];
                } else if (next[idx] != refs[i]) {
                    ok = false;
                }
            }
            if (ok) break;
        }
        next += n;
    }
}
}  // namespace attacks

bool checkMagics() {
    // Every relevant blocker subset on every square, with random pieces outside the mask too.
    uint64_t seed = 42;
    long bad = 0;
    for (int bishopPass = 0; bishopPass < 2; bishopPass++)
        for (int s = 0; s < 64; s++) {
            const attacks::Magic& m = bishopPass ? attacks::BISHOP_MAGIC[s] : attacks::ROOK_MAGIC[s];
            Bitboard sub = 0;
            do {
                for (Bitboard occ : {sub, sub | (splitmix(seed) & ~m.mask)}) {
                    occ &= ~bit(s);
                    Bitboard fast = bishopPass ? attacks::bishop(s, occ) : attacks::rook(s, occ);
                    Bitboard slow = bishopPass ? attacks::bishopSlow(s, occ) : attacks::rookSlow(s, occ);
                    bad += fast != slow;
                }
                sub = (sub - m.mask) & m.mask;
            } while (sub);
        }
    return bad == 0;
}

void initTables() {
    static bool done = false;
    if (done) return;
    done = true;

    const int KN[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    const int KG[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    // Ray directions (df, dr) in the order used by RAY.
    const int DIR[8][2] = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}, {0, -1}, {-1, 0}, {1, -1}, {-1, -1}};
    auto on = [](int f, int r) { return f >= 0 && f < 8 && r >= 0 && r < 8; };
    for (int s = 0; s < 64; s++) {
        int f = fileOf(s), r = rankOf(s);
        KNIGHT_ATT[s] = KING_ATT[s] = 0;
        for (auto& k : KN)
            if (on(f + k[0], r + k[1])) KNIGHT_ATT[s] |= bit((r + k[1]) * 8 + f + k[0]);
        for (auto& k : KG)
            if (on(f + k[0], r + k[1])) KING_ATT[s] |= bit((r + k[1]) * 8 + f + k[0]);
        PAWN_ATT[0][s] = PAWN_ATT[1][s] = 0;
        LINES[s] = 0;
        for (int df : {-1, 1}) {
            if (on(f + df, r + 1)) PAWN_ATT[WHITE][s] |= bit((r + 1) * 8 + f + df);
            if (on(f + df, r - 1)) PAWN_ATT[BLACK][s] |= bit((r - 1) * 8 + f + df);
        }
        for (int d = 0; d < 8; d++) {
            RAY[d][s] = 0;
            for (int nf = f + DIR[d][0], nr = r + DIR[d][1]; on(nf, nr); nf += DIR[d][0], nr += DIR[d][1])
                RAY[d][s] |= bit(nr * 8 + nf);
            LINES[s] |= RAY[d][s];
        }
    }

    uint64_t seed = 0xC0FFEE1234ULL;
    for (auto& row : Z_piece)
        for (auto& z : row) z = splitmix(seed);
    Z_side = splitmix(seed);
    for (auto& z : Z_castle) z = splitmix(seed);
    for (auto& z : Z_ep) z = splitmix(seed);
    for (auto& row : Z_checks)
        for (auto& z : row) z = splitmix(seed);
    for (auto& side : Z_pocket)
        for (auto& row : side)
            for (auto& z : row) z = splitmix(seed);

    const int ROOK_DIRS[4] = {0, 1, 4, 5}, BISHOP_DIRS[4] = {2, 3, 6, 7};
    attacks::initMagics(attacks::ROOK_MAGIC, attacks::ROOK_TABLE, ROOK_DIRS, attacks::rookSlow);
    attacks::initMagics(attacks::BISHOP_MAGIC, attacks::BISHOP_TABLE, BISHOP_DIRS, attacks::bishopSlow);
}

const char* variantName(int v) {
    static const char* N[] = {"Standard", "Chess960", "Three-check", "Crazyhouse", "King of the Hill", "Horde"};
    return v >= 0 && v < NUM_VARIANTS ? N[v] : "?";
}

const char* variantUciName(int v) {
    static const char* N[] = {"chess", "chess960", "3check", "crazyhouse", "kingofthehill", "horde"};
    return v >= 0 && v < NUM_VARIANTS ? N[v] : "?";
}

int variantFromUciName(const std::string& s) {
    std::string l;
    for (char c : s) l += char(std::tolower((unsigned char)c));
    for (int v = 0; v < NUM_VARIANTS; v++)
        if (l == variantUciName(v)) return v;
    if (l == "standard" || l == "normal") return STANDARD;
    if (l == "threecheck" || l == "three-check") return THREE_CHECK;
    if (l == "koth" || l == "king of the hill") return KING_OF_THE_HILL;
    return -1;
}

std::string variantStartFEN(int v) {
    switch (v) {
        case THREE_CHECK: return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 3+3 0 1";
        case CRAZYHOUSE: return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1";
        case HORDE: return "rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP w kq - 0 1";
        default: return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    }
}

std::string chess960StartFEN(int n) {
    n = ((n % 960) + 960) % 960;
    char rank[8] = {0};
    auto placeOnFree = [&](int index, char piece) {
        for (int f = 0; f < 8; f++)
            if (!rank[f] && index-- == 0) {
                rank[f] = piece;
                return;
            }
    };
    rank[(n % 4) * 2 + 1] = 'b';  // light-squared bishop
    n /= 4;
    rank[(n % 4) * 2] = 'b';      // dark-squared bishop
    n /= 4;
    placeOnFree(n % 6, 'q');
    n /= 6;
    static const int KN[10][2] = {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4}};
    placeOnFree(KN[n][1], 'n');  // higher index first so the lower one still counts the same free squares
    placeOnFree(KN[n][0], 'n');
    placeOnFree(0, 'r');
    placeOnFree(0, 'k');
    placeOnFree(0, 'r');
    std::string black(rank, 8), white = black;
    for (char& c : white) c = char(std::toupper((unsigned char)c));
    return black + "/pppppppp/8/8/8/8/PPPPPPPP/" + white + " w KQkq - 0 1";
}

std::string squareName(int s) {
    std::string r;
    r += char('a' + fileOf(s));
    r += char('1' + rankOf(s));
    return r;
}

Board::Board() {
    initTables();
    clear();
}

void Board::clear() {
    std::memset(sq, 0, sizeof(sq));
    std::memset(pieces, 0, sizeof(pieces));
    occ[0] = occ[1] = occAll = 0;
    stm = WHITE;
    castle = 0;
    ep = -1;
    halfmove = 0;
    fullmove = 1;
    hash = 0;
    kingSq[0] = kingSq[1] = -1;
    pieceCount = 0;
    checkCache = -1;
    checks[0] = checks[1] = 0;
    std::memset(pocket, 0, sizeof(pocket));
    promoted = 0;
    castleRook[0] = 7, castleRook[1] = 0, castleRook[2] = 63, castleRook[3] = 56;
    std::memset(castleMask, 15, sizeof(castleMask));
    undoStack.clear();
    hashHistory.clear();
    nnueStack.resize(1);
    nnueStack[0].computed = false;
    nnueStack[0].refresh = true;
}

NnueAccum& Board::pushAccum() {
    size_t i = undoStack.size();
    if (nnueStack.size() <= i) nnueStack.resize(i + 1);
    NnueAccum& a = nnueStack[i];
    a.computed = false;
    a.refresh = false;
    a.ndirty = 0;
    return a;
}

inline void Board::addPiece(int s, int p) {
    sq[s] = p;
    Bitboard b = bit(s);
    pieces[colorOf(p)][typeOf(p)] |= b;
    occ[colorOf(p)] |= b;
    occAll |= b;
}

inline void Board::removePiece(int s) {
    int p = sq[s];
    Bitboard b = ~bit(s);
    pieces[colorOf(p)][typeOf(p)] &= b;
    occ[colorOf(p)] &= b;
    occAll &= b;
    sq[s] = EMPTY;
}

void Board::put(int s, int piece) {
    checkCache = -1;
    nnueStack[undoStack.size()].computed = false;
    nnueStack[undoStack.size()].refresh = true;
    if (sq[s]) {
        removePiece(s);
        pieceCount--;
    }
    if (piece) {
        addPiece(s, piece);
        pieceCount++;
        if (typeOf(piece) == KING) kingSq[colorOf(piece)] = s;
    }
}

uint64_t Board::computeHash() const {
    uint64_t h = 0;
    for (int s = 0; s < 64; s++)
        if (sq[s]) h ^= Z_piece[sq[s]][s];
    if (stm == BLACK) h ^= Z_side;
    h ^= Z_castle[castle];
    if (ep >= 0) h ^= Z_ep[fileOf(ep)];
    if (variant == THREE_CHECK) h ^= Z_checks[0][checks[0]] ^ Z_checks[1][checks[1]];
    if (variant == CRAZYHOUSE)
        for (int c = 0; c < 2; c++)
            for (int t = PAWN; t <= QUEEN; t++) h ^= Z_pocket[c][t][pocket[c][t]];
    return h;
}

void Board::changePocket(int c, int t, int delta) {
    hash ^= Z_pocket[c][t][pocket[c][t]];
    pocket[c][t] += delta;
    hash ^= Z_pocket[c][t][pocket[c][t]];
}

void Board::setStartPos() { setFEN(variantStartFEN(variant)); }

bool Board::setFEN(const std::string& fenIn) {
    clear();
    std::istringstream ss(fenIn);
    std::string pos, side, cas = "-", eps = "-";
    if (!(ss >> pos >> side)) return false;
    ss >> cas >> eps;
    std::vector<std::string> rest;
    for (std::string t; ss >> t;) rest.push_back(t);
    // Three-check counters: "3+3" before the clocks (checks remaining) or "+1+0" after them (checks given).
    for (size_t i = 0; i < rest.size(); i++) {
        const std::string& t = rest[i];
        int a, b;
        if (t.size() >= 4 && t[0] == '+' && std::sscanf(t.c_str(), "+%d+%d", &a, &b) == 2) {
            checks[WHITE] = a, checks[BLACK] = b;
        } else if (t.find('+') != std::string::npos && std::sscanf(t.c_str(), "%d+%d", &a, &b) == 2) {
            checks[WHITE] = 3 - a, checks[BLACK] = 3 - b;
        } else {
            continue;
        }
        rest.erase(rest.begin() + i--);
    }
    for (int& c : checks) c = std::max(0, std::min(c, 3));
    int hm = rest.size() > 0 ? std::atoi(rest[0].c_str()) : 0;
    int fm = rest.size() > 1 ? std::atoi(rest[1].c_str()) : 1;

    // Crazyhouse hands: "board[Qn]" or a ninth "rank" after the eighth slash.
    std::string hand;
    size_t br = pos.find('[');
    if (br != std::string::npos) {
        size_t end = pos.find(']', br);
        hand = pos.substr(br + 1, end == std::string::npos ? std::string::npos : end - br - 1);
        pos = pos.substr(0, br);
    } else if (std::count(pos.begin(), pos.end(), '/') == 8) {
        size_t last = pos.rfind('/');
        hand = pos.substr(last + 1);
        pos = pos.substr(0, last);
    }

    auto typeFor = [](char c) {
        switch (std::tolower((unsigned char)c)) {
            case 'p': return int(PAWN);
            case 'n': return int(KNIGHT);
            case 'b': return int(BISHOP);
            case 'r': return int(ROOK);
            case 'q': return int(QUEEN);
            case 'k': return int(KING);
            default: return 0;
        }
    };
    int r = 7, f = 0;
    int kings[2] = {0, 0};
    for (char c : pos) {
        if (c == '/') {
            r--;
            f = 0;
            continue;
        }
        if (std::isdigit((unsigned char)c)) {
            f += c - '0';
            continue;
        }
        if (c == '~') {  // the piece just placed is promoted
            if (f > 0 && r >= 0) promoted |= bit(r * 8 + f - 1);
            continue;
        }
        int t = typeFor(c);
        if (!t || r < 0 || f > 7) return false;
        int c2 = std::isupper((unsigned char)c) ? WHITE : BLACK;
        if (t == KING) kings[c2]++;
        put(r * 8 + f, makePiece(c2, t));
        f++;
    }
    for (char c : hand) {
        int t = typeFor(c);
        if (t && t != KING) pocket[std::isupper((unsigned char)c) ? WHITE : BLACK][t]++;
    }
    if (variant != CRAZYHOUSE) std::memset(pocket, 0, sizeof(pocket)), promoted = 0;
    if (variant != THREE_CHECK) checks[0] = checks[1] = 0;
    if (kings[BLACK] != 1 || kings[WHITE] != (variant == HORDE ? 0 : 1)) return false;
    if (side != "w" && side != "b") return false;
    stm = side == "b" ? BLACK : WHITE;
    checkCache = -1;

    // Castling: KQkq picks the outermost rook on that side of the king; A-H / a-h name the rook's file.
    for (char c : cas) {
        int color = std::isupper((unsigned char)c) ? WHITE : BLACK;
        int k = kingSq[color], base = color == WHITE ? 0 : 56;
        if (k < 0 || rankOf(k) != rankOf(base)) continue;
        int rook = makePiece(color, ROOK), rookSq = -1;
        char l = char(std::tolower((unsigned char)c));
        if (l == 'k') {
            for (int s = base + 7; s > k && rookSq < 0; s--)
                if (sq[s] == rook) rookSq = s;
        } else if (l == 'q') {
            for (int s = base; s < k && rookSq < 0; s++)
                if (sq[s] == rook) rookSq = s;
        } else if (l >= 'a' && l <= 'h' && sq[base + (l - 'a')] == rook) {
            rookSq = base + (l - 'a');
        }
        if (rookSq < 0) continue;
        int right = color * 2 + (rookSq > k ? 0 : 1);
        castle |= 1 << right;
        castleRook[right] = rookSq;
    }
    for (int i = 0; i < 4; i++) {
        if (!(castle & (1 << i))) continue;
        castleMask[castleRook[i]] &= uint8_t(~(1 << i));
        castleMask[kingSq[i / 2]] &= uint8_t(~(3 << (i / 2 * 2)));
    }
    if (eps.size() == 2 && eps[0] >= 'a' && eps[0] <= 'h' && eps[1] >= '1' && eps[1] <= '8')
        ep = (eps[1] - '1') * 8 + (eps[0] - 'a');
    halfmove = std::max(0, hm);
    fullmove = fm < 1 ? 1 : fm;
    // The side not to move must not be in check.
    if (kingSq[stm ^ 1] >= 0 && attacked(kingSq[stm ^ 1], stm)) return false;
    hash = computeHash();
    return true;
}

std::string Board::getFEN() const {
    const char* PC = " PNBRQK  pnbrqk";
    std::string s;
    for (int r = 7; r >= 0; r--) {
        int empty = 0;
        for (int f = 0; f < 8; f++) {
            int p = sq[r * 8 + f];
            if (!p) {
                empty++;
                continue;
            }
            if (empty) s += char('0' + empty), empty = 0;
            s += PC[p];
            if (promoted & bit(r * 8 + f)) s += '~';
        }
        if (empty) s += char('0' + empty);
        if (r) s += '/';
    }
    if (variant == CRAZYHOUSE) {
        s += '[';
        for (int c = 0; c < 2; c++)
            for (int t = QUEEN; t >= PAWN; t--) s += std::string(pocket[c][t], PC[makePiece(c, t)]);
        s += ']';
    }
    s += stm == WHITE ? " w " : " b ";
    if (!castle) s += '-';
    for (int i = 0; i < 4; i++) {
        if (!(castle & (1 << i))) continue;
        int color = i / 2, rookSq = castleRook[i];
        // Use KQkq when the castling rook is the outermost one (X-FEN), else the rook's file.
        bool outermost = true;
        for (int t = rookSq & 56; t < (rookSq & 56) + 8; t++)
            if (sq[t] == makePiece(color, ROOK) && (i % 2 == 0 ? t > rookSq : t < rookSq)) outermost = false;
        char c = outermost ? "KQkq"[i] : char('a' + fileOf(rookSq));
        s += color == WHITE ? char(std::toupper((unsigned char)c)) : c;
    }
    s += ' ';
    s += ep >= 0 ? squareName(ep) : "-";
    if (variant == THREE_CHECK) s += ' ' + std::to_string(3 - checks[WHITE]) + '+' + std::to_string(3 - checks[BLACK]);
    s += ' ' + std::to_string(halfmove) + ' ' + std::to_string(fullmove);
    return s;
}

Bitboard Board::attackersTo(int s, Bitboard o) const {
    Bitboard bq = pieces[0][BISHOP] | pieces[1][BISHOP] | pieces[0][QUEEN] | pieces[1][QUEEN];
    Bitboard rq = pieces[0][ROOK] | pieces[1][ROOK] | pieces[0][QUEEN] | pieces[1][QUEEN];
    return (PAWN_ATT[BLACK][s] & pieces[WHITE][PAWN]) | (PAWN_ATT[WHITE][s] & pieces[BLACK][PAWN]) |
           (KNIGHT_ATT[s] & (pieces[0][KNIGHT] | pieces[1][KNIGHT])) |
           (KING_ATT[s] & (pieces[0][KING] | pieces[1][KING])) | (attacks::bishop(s, o) & bq) |
           (attacks::rook(s, o) & rq);
}

bool Board::attacked(int s, int by) const {
    if (s < 0) return false;
    if (PAWN_ATT[by ^ 1][s] & pieces[by][PAWN]) return true;
    if (KNIGHT_ATT[s] & pieces[by][KNIGHT]) return true;
    if (KING_ATT[s] & pieces[by][KING]) return true;
    Bitboard bq = pieces[by][BISHOP] | pieces[by][QUEEN];
    if (bq && (attacks::bishop(s, occAll) & bq)) return true;
    Bitboard rq = pieces[by][ROOK] | pieces[by][QUEEN];
    if (rq && (attacks::rook(s, occAll) & rq)) return true;
    return false;
}

void Board::genMoves(MoveList& ml, bool capturesOnly) const {
    ml.count = 0;
    const int us = stm, them = stm ^ 1;
    const Bitboard enemies = occ[them], empty = ~occAll;

    // Pawns
    const int up = us == WHITE ? 8 : -8;
    const Bitboard promoRank = rankMask(us == WHITE ? 7 : 0);
    const Bitboard thirdRank = rankMask(us == WHITE ? 2 : 5);
    Bitboard pawns = pieces[us][PAWN];
    auto shift = [&](Bitboard b) { return us == WHITE ? b << 8 : b >> 8; };
    Bitboard push1 = shift(pawns) & empty;
    Bitboard push2 = shift(push1 & thirdRank) & empty;
    auto addPromos = [&](int from, int to, uint32_t flags) {
        for (int pr : {QUEEN, KNIGHT, ROOK, BISHOP}) ml.add(encodeMove(from, to, pr, flags));
    };
    for (Bitboard b = push1 & promoRank; b;) {
        int to = popLsb(b);
        addPromos(to - up, to, 0);
    }
    if (!capturesOnly) {
        for (Bitboard b = push1 & ~promoRank; b;) {
            int to = popLsb(b);
            ml.add(encodeMove(to - up, to));
        }
        for (Bitboard b = push2; b;) {
            int to = popLsb(b);
            ml.add(encodeMove(to - 2 * up, to, 0, MF_DOUBLE));
        }
        // Horde: White's pawns on the first rank may also advance two squares, but (as on lichess)
        // that step cannot be taken en passant, so it is not flagged as a double push.
        if (variant == HORDE && us == WHITE)
            for (Bitboard b = shift(push1 & rankMask(1)) & empty; b;) {
                int to = popLsb(b);
                ml.add(encodeMove(to - 16, to));
            }
    }
    for (Bitboard b = pawns; b;) {
        int from = popLsb(b);
        Bitboard att = PAWN_ATT[us][from];
        for (Bitboard c = att & enemies; c;) {
            int to = popLsb(c);
            if (bit(to) & promoRank)
                addPromos(from, to, MF_CAPTURE);
            else
                ml.add(encodeMove(from, to, 0, MF_CAPTURE));
        }
        if (ep >= 0 && (att & bit(ep))) ml.add(encodeMove(from, ep, 0, MF_CAPTURE | MF_EP));
    }

    // Pieces
    const Bitboard targets = capturesOnly ? enemies : ~occ[us];
    for (int t = KNIGHT; t <= KING; t++) {
        for (Bitboard b = pieces[us][t]; b;) {
            int from = popLsb(b);
            Bitboard a;
            switch (t) {
                case KNIGHT: a = KNIGHT_ATT[from]; break;
                case BISHOP: a = attacks::bishop(from, occAll); break;
                case ROOK: a = attacks::rook(from, occAll); break;
                case QUEEN: a = attacks::queen(from, occAll); break;
                default: a = KING_ATT[from]; break;
            }
            for (a &= targets; a;) {
                int to = popLsb(a);
                ml.add(encodeMove(from, to, 0, (bit(to) & enemies) ? MF_CAPTURE : 0));
            }
        }
    }

    // Castling (also Chess960): the king goes to the g- or c-file and the rook next to it. Every square
    // either piece crosses must be empty apart from those two, and the king may not pass through check.
    if (!capturesOnly && (castle & (us == WHITE ? 3 : 12))) {
        const int k = kingSq[us], base = us == WHITE ? 0 : 56;
        for (int right = us * 2; right < us * 2 + 2; right++) {
            if (!(castle & (1 << right))) continue;
            const bool kingSide = right % 2 == 0;
            const int rf = castleRook[right], kt = base + (kingSide ? 6 : 2), rt = base + (kingSide ? 5 : 3);
            if ((rankSpan(k, kt) | rankSpan(rf, rt)) & occAll & ~bit(k) & ~bit(rf)) continue;
            bool safe = true;
            for (Bitboard path = rankSpan(k, kt); path && safe;)
                if (attacked(popLsb(path), them)) safe = false;
            if (safe) ml.add(encodeMove(k, rf, 0, MF_CASTLE));
        }
    }

    // Crazyhouse drops: any piece in hand onto an empty square, pawns not on the first or last rank.
    if (!capturesOnly && variant == CRAZYHOUSE) {
        for (int t = PAWN; t <= QUEEN; t++) {
            if (!pocket[us][t]) continue;
            Bitboard to = empty & (t == PAWN ? ~(rankMask(0) | rankMask(7)) : ~Bitboard(0));
            while (to) ml.add(encodeDrop(popLsb(to), t));
        }
    }
}

void Board::genLegal(MoveList& ml) {
    ml.count = 0;
    if (variantOutcome()) return;  // the game is already over
    MoveList pseudo;
    genMoves(pseudo);
    for (int i = 0; i < pseudo.count; i++)
        if (makeMove(pseudo.moves[i])) {
            unmakeMove();
            ml.add(pseudo.moves[i]);
        }
}

void Board::castleSquares(Move m, int& kingTo, int& rookFrom, int& rookTo) const {
    int from = moveFrom(m), base = from & 56;
    rookFrom = moveTo(m);
    bool kingSide = rookFrom > from;
    kingTo = base + (kingSide ? 6 : 2);
    rookTo = base + (kingSide ? 5 : 3);
}

bool Board::makeMove(Move m) {
    const int from = moveFrom(m), to = moveTo(m);
    const uint32_t flags = moveFlags(m);
    const bool drop = isDrop(m), castling = (flags & MF_CASTLE) != 0;
    const int p = drop ? makePiece(stm, dropType(m)) : sq[from];
    const int capSq = (flags & MF_EP) ? to + (stm == WHITE ? -8 : 8) : to;
    const int cap = drop || castling ? EMPTY : sq[capSq];
    // A move can only expose our king if we are in check, the king moves, it is en passant,
    // or the moving piece sits on a line through our king (a potential pin). A drop only adds a blocker.
    const int ks = kingSq[stm];
    const bool mustTest = ks >= 0 && (inCheck() || typeOf(p) == KING || (flags & MF_EP) ||
                                      (!drop && (LINES[ks] & bit(from))));

    undoStack.push_back({m, cap, castle, ep, halfmove, hash, checkCache, promoted,
                         {uint8_t(checks[0]), uint8_t(checks[1])}});
    hashHistory.push_back(hash);
    NnueAccum& acc = pushAccum();

    if (ep >= 0) hash ^= Z_ep[fileOf(ep)];
    if (drop) {
        dirty(acc, p, to, 1);
        addPiece(to, p);
        hash ^= Z_piece[p][to];
        pieceCount++;
        changePocket(stm, typeOf(p), -1);
    } else if (castling) {
        int kt, rf, rt;
        castleSquares(m, kt, rf, rt);
        int rook = sq[rf];
        dirty(acc, p, from, -1);
        dirty(acc, rook, rf, -1);
        dirty(acc, p, kt, 1);
        dirty(acc, rook, rt, 1);
        removePiece(from);
        removePiece(rf);
        addPiece(kt, p);
        addPiece(rt, rook);
        hash ^= Z_piece[p][from] ^ Z_piece[p][kt] ^ Z_piece[rook][rf] ^ Z_piece[rook][rt];
        kingSq[stm] = kt;
    } else {
        if (cap) {
            hash ^= Z_piece[cap][capSq];
            dirty(acc, cap, capSq, -1);
            removePiece(capSq);
            pieceCount--;
            if (variant == CRAZYHOUSE) {
                changePocket(stm, (promoted & bit(capSq)) ? PAWN : typeOf(cap), 1);
                promoted &= ~bit(capSq);
            }
        }
        hash ^= Z_piece[p][from];
        removePiece(from);
        int np = movePromo(m) ? makePiece(stm, movePromo(m)) : p;
        dirty(acc, p, from, -1);
        dirty(acc, np, to, 1);
        addPiece(to, np);
        hash ^= Z_piece[np][to];
        if (typeOf(p) == KING) kingSq[stm] = to;
        if (variant == CRAZYHOUSE) {
            if (promoted & bit(from))
                promoted ^= bit(from) | bit(to);
            else if (movePromo(m))
                promoted |= bit(to);
        }
    }
    hash ^= Z_castle[castle];
    castle &= castleMask[from] & castleMask[to];
    hash ^= Z_castle[castle];
    ep = -1;
    if (flags & MF_DOUBLE) {
        ep = (from + to) / 2;
        hash ^= Z_ep[fileOf(ep)];
    }
    halfmove = (typeOf(p) == PAWN || cap) ? 0 : halfmove + 1;
    if (stm == BLACK) fullmove++;
    stm = Color(stm ^ 1);
    hash ^= Z_side;
    checkCache = -1;

    if (mustTest && attacked(kingSq[stm ^ 1], stm)) {
        unmakeMove();
        return false;
    }
    if (variant == THREE_CHECK && inCheck()) {
        int mover = stm ^ 1;
        hash ^= Z_checks[mover][checks[mover]];
        checks[mover] = std::min(checks[mover] + 1, 7);
        hash ^= Z_checks[mover][checks[mover]];
    }
    return true;
}

void Board::unmakeMove() {
    Undo u = undoStack.back();
    undoStack.pop_back();
    hashHistory.pop_back();
    stm = Color(stm ^ 1);
    if (stm == BLACK) fullmove--;
    Move m = u.move;
    int from = moveFrom(m), to = moveTo(m);
    uint32_t flags = moveFlags(m);
    if (isDrop(m)) {
        removePiece(to);
        pieceCount--;
        pocket[stm][dropType(m)]++;
    } else if (flags & MF_CASTLE) {
        int kt, rf, rt;
        castleSquares(m, kt, rf, rt);
        int king = sq[kt], rook = sq[rt];
        removePiece(kt);
        removePiece(rt);
        addPiece(from, king);
        addPiece(rf, rook);
        kingSq[stm] = from;
    } else {
        int p = movePromo(m) ? makePiece(stm, PAWN) : sq[to];
        removePiece(to);
        addPiece(from, p);
        if (typeOf(p) == KING) kingSq[stm] = from;
        if (u.captured) {
            int capSq = (flags & MF_EP) ? to + (stm == WHITE ? -8 : 8) : to;
            addPiece(capSq, u.captured);
            pieceCount++;
            if (variant == CRAZYHOUSE) pocket[stm][(u.promoted & bit(capSq)) ? PAWN : typeOf(u.captured)]--;
        }
    }
    castle = u.castle;
    ep = u.ep;
    halfmove = u.halfmove;
    hash = u.hash;
    checkCache = u.checkCache;
    promoted = u.promoted;
    checks[0] = u.checks[0];
    checks[1] = u.checks[1];
}

void Board::makeNull() {
    undoStack.push_back({NO_MOVE, 0, castle, ep, halfmove, hash, checkCache, promoted,
                         {uint8_t(checks[0]), uint8_t(checks[1])}});
    checkCache = -1;
    hashHistory.push_back(hash);
    pushAccum();
    if (ep >= 0) hash ^= Z_ep[fileOf(ep)];
    ep = -1;
    halfmove++;
    stm = Color(stm ^ 1);
    hash ^= Z_side;
}

void Board::unmakeNull() {
    Undo u = undoStack.back();
    undoStack.pop_back();
    hashHistory.pop_back();
    stm = Color(stm ^ 1);
    ep = u.ep;
    halfmove = u.halfmove;
    hash = u.hash;
    checkCache = u.checkCache;
}

int Board::variantOutcome() const {
    const int them = stm ^ 1;
    switch (variant) {
        case THREE_CHECK: return checks[them] >= 3 ? -1 : 0;
        case KING_OF_THE_HILL: return kingSq[them] >= 0 && (bit(kingSq[them]) & CENTER) ? -1 : 0;
        case HORDE: return occ[stm] ? 0 : -1;
        default: return 0;
    }
}

std::string Board::variantOutcomeText() const {
    if (!variantOutcome()) return {};
    std::string winner = stm == WHITE ? "Black" : "White";
    switch (variant) {
        case THREE_CHECK: return winner + " wins by three checks";
        case KING_OF_THE_HILL: return winner + " wins - king reached the center";
        default: return "Black wins - the horde is destroyed";
    }
}

bool Board::isRepetition() const {
    int n = (int)hashHistory.size();
    int limit = halfmove < n ? halfmove : n;
    for (int k = 4; k <= limit; k += 2)
        if (hashHistory[n - k] == hash) return true;
    return false;
}

int Board::repetitionCount() const {
    int n = (int)hashHistory.size();
    int limit = halfmove < n ? halfmove : n;
    int count = 1;
    for (int k = 4; k <= limit; k += 2)
        if (hashHistory[n - k] == hash) count++;
    return count;
}

bool Board::insufficientMaterial() const {
    // Other variants: pieces come back (crazyhouse), checks or the center win, or the horde must be wiped out.
    if (variant != STANDARD && variant != CHESS960) return false;
    for (int c = 0; c < 2; c++)
        if (pieces[c][PAWN] | pieces[c][ROOK] | pieces[c][QUEEN]) return false;
    Bitboard knights = pieces[0][KNIGHT] | pieces[1][KNIGHT];
    Bitboard bishops = pieces[0][BISHOP] | pieces[1][BISHOP];
    if (popcount(knights | bishops) <= 1) return true;
    const Bitboard DARK = 0xAA55AA55AA55AA55ULL;
    return !knights && (!(bishops & DARK) || !(bishops & ~DARK));  // only same-colored bishops
}

bool Board::hasNonPawnMaterial(int c) const {
    return (pieces[c][KNIGHT] | pieces[c][BISHOP] | pieces[c][ROOK] | pieces[c][QUEEN]) != 0;
}

std::string Board::moveToUCI(Move m) const {
    if (isDrop(m)) return std::string(1, " PNBRQK"[dropType(m)]) + '@' + squareName(moveTo(m));
    int to = moveTo(m);
    if ((moveFlags(m) & MF_CASTLE) && variant != CHESS960) {
        int rf, rt;
        castleSquares(m, to, rf, rt);  // standard notation names the king's destination (e1g1)
    }
    std::string s = squareName(moveFrom(m)) + squareName(to);
    if (movePromo(m)) s += " pnbrqk"[movePromo(m)];
    return s;
}

Move Board::parseUCI(const std::string& str) {
    MoveList ml;
    genLegal(ml);
    for (int i = 0; i < ml.count; i++) {
        Move m = ml.moves[i];
        if (moveToUCI(m) == str) return m;
        // Accept both castling notations: king to its destination and king onto the rook.
        if ((moveFlags(m) & MF_CASTLE) && squareName(moveFrom(m)) + squareName(moveTo(m)) == str) return m;
        if (moveFlags(m) & MF_CASTLE) {
            int kt, rf, rt;
            castleSquares(m, kt, rf, rt);
            if (kt != moveFrom(m) && squareName(moveFrom(m)) + squareName(kt) == str) return m;
        }
    }
    return NO_MOVE;
}

// Parses standard algebraic notation ("Nbd7", "exd5", "e8=Q+", "O-O") against the legal moves.
Move Board::parseSAN(const std::string& in) {
    std::string s;
    for (char c : in)
        if (c != '+' && c != '#' && c != '!' && c != '?' && c != 'x' && c != '=') s += c;
    MoveList ml;
    genLegal(ml);
    if (s == "O-O" || s == "0-0" || s == "O-O-O" || s == "0-0-0") {
        bool longSide = s.size() == 5;
        for (int i = 0; i < ml.count; i++) {
            Move m = ml.moves[i];
            if ((moveFlags(m) & MF_CASTLE) && (moveTo(m) < moveFrom(m)) == longSide) return m;
        }
        return NO_MOVE;
    }
    size_t at = s.find('@');
    if (at != std::string::npos && s.size() == at + 3) {  // drop: "N@f3", "P@e4" or "@e4"
        int type = at == 0 ? int(PAWN) : int(std::string(" PNBRQK").find(s[0]));
        int to = (s[at + 2] - '1') * 8 + (s[at + 1] - 'a');
        for (int i = 0; i < ml.count; i++)
            if (isDrop(ml.moves[i]) && moveTo(ml.moves[i]) == to && dropType(ml.moves[i]) == type) return ml.moves[i];
        return NO_MOVE;
    }
    if (s.size() < 2) return NO_MOVE;
    int type = PAWN;
    size_t i = 0;
    if (std::strchr("NBRQK", s[0])) type = int(std::string(" PNBRQK").find(s[0])), i = 1;
    int promo = 0;
    if (std::strchr("NBRQ", s.back())) {
        promo = int(std::string(" PNBRQK").find(s.back()));
        s.pop_back();
    }
    if (s.size() < i + 2) return NO_MOVE;
    int to = (s[s.size() - 1] - '1') * 8 + (s[s.size() - 2] - 'a');
    std::string dis = s.substr(i, s.size() - 2 - i);
    int disFile = -1, disRank = -1;
    for (char c : dis) {
        if (c >= 'a' && c <= 'h') disFile = c - 'a';
        if (c >= '1' && c <= '8') disRank = c - '1';
    }
    for (int k = 0; k < ml.count; k++) {
        Move m = ml.moves[k];
        int from = moveFrom(m);
        if (moveTo(m) != to || typeOf(sq[from]) != type || movePromo(m) != promo) continue;
        if ((moveFlags(m) & MF_CASTLE) || isDrop(m)) continue;
        if (disFile >= 0 && fileOf(from) != disFile) continue;
        if (disRank >= 0 && rankOf(from) != disRank) continue;
        return m;
    }
    return NO_MOVE;
}

std::string Board::moveToSAN(Move m) {
    const char* LET = "  NBRQK";
    int from = moveFrom(m), to = moveTo(m);
    int t = isDrop(m) ? dropType(m) : typeOf(sq[from]);
    std::string s;
    if (isDrop(m)) {
        s = std::string(1, " PNBRQK"[t]) + '@' + squareName(to);
    } else if (moveFlags(m) & MF_CASTLE) {
        s = to > from ? "O-O" : "O-O-O";
    } else if (t == PAWN) {
        if (isCapture(m)) {
            s += char('a' + fileOf(from));
            s += 'x';
        }
        s += squareName(to);
        if (movePromo(m)) {
            s += '=';
            s += LET[movePromo(m)];
        }
    } else {
        s += LET[t];
        MoveList ml;
        genLegal(ml);
        bool ambiguous = false, sameFile = false, sameRank = false;
        for (int i = 0; i < ml.count; i++) {
            Move o = ml.moves[i];
            int of = moveFrom(o);
            if (o == m || moveTo(o) != to || of == from || isDrop(o) || (moveFlags(o) & MF_CASTLE) ||
                typeOf(sq[of]) != t)
                continue;
            ambiguous = true;
            if (fileOf(of) == fileOf(from)) sameFile = true;
            if (rankOf(of) == rankOf(from)) sameRank = true;
        }
        if (ambiguous) {
            if (!sameFile)
                s += char('a' + fileOf(from));
            else if (!sameRank)
                s += char('1' + rankOf(from));
            else
                s += squareName(from);
        }
        if (isCapture(m)) s += 'x';
        s += squareName(to);
    }
    if (makeMove(m)) {
        if (inCheck()) {
            MoveList replies;
            genLegal(replies);
            s += replies.count || variantOutcome() ? '+' : '#';
        }
        unmakeMove();
    }
    return s;
}
