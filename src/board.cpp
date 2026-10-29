#include "board.h"

#include <cctype>
#include <cstring>
#include <sstream>

namespace {

Bitboard PAWN_ATT[2][64], KNIGHT_ATT[64], KING_ATT[64];
Bitboard RAY[8][64];  // N, E, NE, NW (increasing), S, W, SE, SW (decreasing)
Bitboard LINES[64];   // all eight rays from a square on an empty board
int castleMask[64];

uint64_t Z_piece[16][64], Z_side, Z_castle[16], Z_ep[8];

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
// Classical ray attacks: the nearest blocker along each ray cuts off the rest of the ray.
Bitboard bishop(int s, Bitboard occ) {
    Bitboard a = 0;
    for (int d : {2, 3, 6, 7}) {
        Bitboard r = RAY[d][s], blockers = r & occ;
        if (blockers) r ^= RAY[d][d < 4 ? lsb(blockers) : msb(blockers)];
        a |= r;
    }
    return a;
}
Bitboard rook(int s, Bitboard occ) {
    Bitboard a = 0;
    for (int d : {0, 1, 4, 5}) {
        Bitboard r = RAY[d][s], blockers = r & occ;
        if (blockers) r ^= RAY[d][d < 4 ? lsb(blockers) : msb(blockers)];
        a |= r;
    }
    return a;
}
}  // namespace attacks

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
        castleMask[s] = 15;
    }
    castleMask[0] = 15 & ~2;
    castleMask[7] = 15 & ~1;
    castleMask[4] = 15 & ~3;
    castleMask[56] = 15 & ~8;
    castleMask[63] = 15 & ~4;
    castleMask[60] = 15 & ~12;

    uint64_t seed = 0xC0FFEE1234ULL;
    for (auto& row : Z_piece)
        for (auto& z : row) z = splitmix(seed);
    Z_side = splitmix(seed);
    for (auto& z : Z_castle) z = splitmix(seed);
    for (auto& z : Z_ep) z = splitmix(seed);
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
    undoStack.clear();
    hashHistory.clear();
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
    return h;
}

void Board::setStartPos() { setFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"); }

bool Board::setFEN(const std::string& fen) {
    clear();
    std::istringstream ss(fen);
    std::string pos, side, cas = "-", eps = "-";
    if (!(ss >> pos >> side)) return false;
    ss >> cas >> eps;
    int hm = 0, fm = 1;
    if (!(ss >> hm)) hm = 0;
    if (!(ss >> fm)) fm = 1;

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
        int t;
        switch (std::tolower((unsigned char)c)) {
            case 'p': t = PAWN; break;
            case 'n': t = KNIGHT; break;
            case 'b': t = BISHOP; break;
            case 'r': t = ROOK; break;
            case 'q': t = QUEEN; break;
            case 'k': t = KING; break;
            default: return false;
        }
        if (r < 0 || f > 7) return false;
        int c2 = std::isupper((unsigned char)c) ? WHITE : BLACK;
        if (t == KING) kings[c2]++;
        put(r * 8 + f, makePiece(c2, t));
        f++;
    }
    if (kings[0] != 1 || kings[1] != 1) return false;
    if (side != "w" && side != "b") return false;
    stm = side == "b" ? BLACK : WHITE;
    checkCache = -1;

    for (char c : cas) {
        if (c == 'K' && sq[4] == makePiece(WHITE, KING) && sq[7] == makePiece(WHITE, ROOK)) castle |= 1;
        if (c == 'Q' && sq[4] == makePiece(WHITE, KING) && sq[0] == makePiece(WHITE, ROOK)) castle |= 2;
        if (c == 'k' && sq[60] == makePiece(BLACK, KING) && sq[63] == makePiece(BLACK, ROOK)) castle |= 4;
        if (c == 'q' && sq[60] == makePiece(BLACK, KING) && sq[56] == makePiece(BLACK, ROOK)) castle |= 8;
    }
    if (eps.size() == 2 && eps[0] >= 'a' && eps[0] <= 'h' && eps[1] >= '1' && eps[1] <= '8')
        ep = (eps[1] - '1') * 8 + (eps[0] - 'a');
    halfmove = hm;
    fullmove = fm < 1 ? 1 : fm;
    // The side not to move must not be in check.
    if (attacked(kingSq[stm ^ 1], stm)) return false;
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
        }
        if (empty) s += char('0' + empty);
        if (r) s += '/';
    }
    s += stm == WHITE ? " w " : " b ";
    if (!castle) s += '-';
    if (castle & 1) s += 'K';
    if (castle & 2) s += 'Q';
    if (castle & 4) s += 'k';
    if (castle & 8) s += 'q';
    s += ' ';
    s += ep >= 0 ? squareName(ep) : "-";
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

    // Castling
    if (!capturesOnly && castle) {
        if (us == WHITE && kingSq[us] == 4) {
            if ((castle & 1) && !(occAll & 0x60ULL) && !attacked(4, BLACK) && !attacked(5, BLACK) &&
                !attacked(6, BLACK))
                ml.add(encodeMove(4, 6, 0, MF_CASTLE));
            if ((castle & 2) && !(occAll & 0x0EULL) && !attacked(4, BLACK) && !attacked(3, BLACK) &&
                !attacked(2, BLACK))
                ml.add(encodeMove(4, 2, 0, MF_CASTLE));
        } else if (us == BLACK && kingSq[us] == 60) {
            if ((castle & 4) && !(occAll & (0x60ULL << 56)) && !attacked(60, WHITE) && !attacked(61, WHITE) &&
                !attacked(62, WHITE))
                ml.add(encodeMove(60, 62, 0, MF_CASTLE));
            if ((castle & 8) && !(occAll & (0x0EULL << 56)) && !attacked(60, WHITE) && !attacked(59, WHITE) &&
                !attacked(58, WHITE))
                ml.add(encodeMove(60, 58, 0, MF_CASTLE));
        }
    }
}

void Board::genLegal(MoveList& ml) {
    MoveList pseudo;
    genMoves(pseudo);
    ml.count = 0;
    for (int i = 0; i < pseudo.count; i++)
        if (makeMove(pseudo.moves[i])) {
            unmakeMove();
            ml.add(pseudo.moves[i]);
        }
}

static void rookCastleSquares(int kingTo, int& rf, int& rt) {
    switch (kingTo) {
        case 6: rf = 7, rt = 5; break;
        case 2: rf = 0, rt = 3; break;
        case 62: rf = 63, rt = 61; break;
        default: rf = 56, rt = 59; break;
    }
}

bool Board::makeMove(Move m) {
    int from = moveFrom(m), to = moveTo(m);
    uint32_t flags = moveFlags(m);
    int p = sq[from];
    int capSq = (flags & MF_EP) ? to + (stm == WHITE ? -8 : 8) : to;
    int cap = sq[capSq];
    // A move can only expose our king if we are in check, the king moves, it is en passant,
    // or the moving piece sits on a line through our king (a potential pin).
    bool mustTest = inCheck() || typeOf(p) == KING || (flags & MF_EP) || (LINES[kingSq[stm]] & bit(from));

    undoStack.push_back({m, cap, castle, ep, halfmove, hash, checkCache});
    hashHistory.push_back(hash);

    if (ep >= 0) hash ^= Z_ep[fileOf(ep)];
    if (cap) {
        hash ^= Z_piece[cap][capSq];
        removePiece(capSq);
        pieceCount--;
    }
    hash ^= Z_piece[p][from];
    removePiece(from);
    int np = movePromo(m) ? makePiece(stm, movePromo(m)) : p;
    addPiece(to, np);
    hash ^= Z_piece[np][to];
    if (typeOf(p) == KING) kingSq[stm] = to;
    if (flags & MF_CASTLE) {
        int rf, rt;
        rookCastleSquares(to, rf, rt);
        int rook = sq[rf];
        removePiece(rf);
        addPiece(rt, rook);
        hash ^= Z_piece[rook][rf] ^ Z_piece[rook][rt];
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
    int p = movePromo(m) ? makePiece(stm, PAWN) : sq[to];
    removePiece(to);
    addPiece(from, p);
    if (typeOf(p) == KING) kingSq[stm] = from;
    if (flags & MF_CASTLE) {
        int rf, rt;
        rookCastleSquares(to, rf, rt);
        int rook = sq[rt];
        removePiece(rt);
        addPiece(rf, rook);
    }
    if (u.captured) {
        int capSq = (flags & MF_EP) ? to + (stm == WHITE ? -8 : 8) : to;
        addPiece(capSq, u.captured);
        pieceCount++;
    }
    castle = u.castle;
    ep = u.ep;
    halfmove = u.halfmove;
    hash = u.hash;
    checkCache = u.checkCache;
}

void Board::makeNull() {
    undoStack.push_back({NO_MOVE, 0, castle, ep, halfmove, hash, checkCache});
    checkCache = -1;
    hashHistory.push_back(hash);
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
    std::string s = squareName(moveFrom(m)) + squareName(moveTo(m));
    if (movePromo(m)) s += " pnbrqk"[movePromo(m)];
    return s;
}

Move Board::parseUCI(const std::string& str) {
    MoveList ml;
    genLegal(ml);
    for (int i = 0; i < ml.count; i++)
        if (moveToUCI(ml.moves[i]) == str) return ml.moves[i];
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
            if ((moveFlags(m) & MF_CASTLE) && (fileOf(moveTo(m)) == 2) == longSide) return m;
        }
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
        if (moveFlags(m) & MF_CASTLE) continue;
        if (disFile >= 0 && fileOf(from) != disFile) continue;
        if (disRank >= 0 && rankOf(from) != disRank) continue;
        return m;
    }
    return NO_MOVE;
}

std::string Board::moveToSAN(Move m) {
    const char* LET = "  NBRQK";
    int from = moveFrom(m), to = moveTo(m);
    int t = typeOf(sq[from]);
    std::string s;
    if (moveFlags(m) & MF_CASTLE) {
        s = fileOf(to) == 6 ? "O-O" : "O-O-O";
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
            if (o == m || moveTo(o) != to || of == from || typeOf(sq[of]) != t) continue;
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
            s += replies.count ? '+' : '#';
        }
        unmakeMove();
    }
    return s;
}
