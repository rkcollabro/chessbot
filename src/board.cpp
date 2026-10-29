#include "board.h"

#include <cctype>
#include <cstring>
#include <sstream>

namespace {

int knightTo[64][8], knightN[64];
int kingTo[64][8], kingN[64];
int rays[64][8][7], rayN[64][8];  // dirs 0-3 orthogonal, 4-7 diagonal
int castleMask[64];

uint64_t Z_piece[16][64], Z_side, Z_castle[16], Z_ep[8];

uint64_t splitmix(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

const int WP = 1, BP = 9;

}  // namespace

void initTables() {
    static bool done = false;
    if (done) return;
    done = true;

    const int KN[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    const int DF[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    const int DR[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    for (int s = 0; s < 64; s++) {
        int f = fileOf(s), r = rankOf(s);
        knightN[s] = kingN[s] = 0;
        for (auto& k : KN) {
            int nf = f + k[0], nr = r + k[1];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) knightTo[s][knightN[s]++] = nr * 8 + nf;
        }
        for (int d = 0; d < 8; d++) {
            int nf = f + DF[d], nr = r + DR[d];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) kingTo[s][kingN[s]++] = nr * 8 + nf;
            rayN[s][d] = 0;
            while (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
                rays[s][d][rayN[s][d]++] = nr * 8 + nf;
                nf += DF[d];
                nr += DR[d];
            }
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
    stm = WHITE;
    castle = 0;
    ep = -1;
    halfmove = 0;
    fullmove = 1;
    hash = 0;
    kingSq[0] = kingSq[1] = -1;
    pieceCount = 0;
    undoStack.clear();
    hashHistory.clear();
}

void Board::put(int s, int piece) {
    if (sq[s]) pieceCount--;
    sq[s] = piece;
    if (piece) {
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

bool Board::attacked(int s, int by) const {
    int f = fileOf(s), r = rankOf(s);
    if (by == WHITE) {
        if (r > 0) {
            if (f > 0 && sq[s - 9] == WP) return true;
            if (f < 7 && sq[s - 7] == WP) return true;
        }
    } else {
        if (r < 7) {
            if (f > 0 && sq[s + 7] == BP) return true;
            if (f < 7 && sq[s + 9] == BP) return true;
        }
    }
    int kn = makePiece(by, KNIGHT);
    for (int i = 0; i < knightN[s]; i++)
        if (sq[knightTo[s][i]] == kn) return true;
    int kg = makePiece(by, KING);
    for (int i = 0; i < kingN[s]; i++)
        if (sq[kingTo[s][i]] == kg) return true;
    int R = makePiece(by, ROOK), B = makePiece(by, BISHOP), Q = makePiece(by, QUEEN);
    for (int d = 0; d < 4; d++)
        for (int i = 0; i < rayN[s][d]; i++) {
            int p = sq[rays[s][d][i]];
            if (p) {
                if (p == R || p == Q) return true;
                break;
            }
        }
    for (int d = 4; d < 8; d++)
        for (int i = 0; i < rayN[s][d]; i++) {
            int p = sq[rays[s][d][i]];
            if (p) {
                if (p == B || p == Q) return true;
                break;
            }
        }
    return false;
}

void Board::genMoves(MoveList& ml, bool capturesOnly) const {
    ml.count = 0;
    const int us = stm, them = stm ^ 1;
    for (int s = 0; s < 64; s++) {
        int p = sq[s];
        if (!p || colorOf(p) != us) continue;
        int t = typeOf(p);
        if (t == PAWN) {
            int dir = us == WHITE ? 8 : -8;
            int startRank = us == WHITE ? 1 : 6;
            int promoRank = us == WHITE ? 7 : 0;
            int to = s + dir;
            auto addPawn = [&](int from, int dest, uint32_t flags) {
                if (rankOf(dest) == promoRank) {
                    for (int pr : {QUEEN, KNIGHT, ROOK, BISHOP}) ml.add(encodeMove(from, dest, pr, flags));
                } else {
                    ml.add(encodeMove(from, dest, 0, flags));
                }
            };
            if (sq[to] == EMPTY) {
                if (!capturesOnly || rankOf(to) == promoRank) addPawn(s, to, 0);
                if (!capturesOnly && rankOf(s) == startRank && sq[to + dir] == EMPTY)
                    ml.add(encodeMove(s, to + dir, 0, MF_DOUBLE));
            }
            for (int df : {-1, 1}) {
                int nf = fileOf(s) + df;
                if (nf < 0 || nf > 7) continue;
                int c = to + df;
                if (sq[c] && colorOf(sq[c]) == them)
                    addPawn(s, c, MF_CAPTURE);
                else if (c == ep)
                    ml.add(encodeMove(s, c, 0, MF_CAPTURE | MF_EP));
            }
        } else if (t == KNIGHT || t == KING) {
            const int* tgt = t == KNIGHT ? knightTo[s] : kingTo[s];
            int n = t == KNIGHT ? knightN[s] : kingN[s];
            for (int i = 0; i < n; i++) {
                int to = tgt[i];
                if (!sq[to]) {
                    if (!capturesOnly) ml.add(encodeMove(s, to));
                } else if (colorOf(sq[to]) == them) {
                    ml.add(encodeMove(s, to, 0, MF_CAPTURE));
                }
            }
            if (t == KING && !capturesOnly && castle) {
                if (us == WHITE && s == 4) {
                    if ((castle & 1) && !sq[5] && !sq[6] && !attacked(4, BLACK) && !attacked(5, BLACK) &&
                        !attacked(6, BLACK))
                        ml.add(encodeMove(4, 6, 0, MF_CASTLE));
                    if ((castle & 2) && !sq[3] && !sq[2] && !sq[1] && !attacked(4, BLACK) &&
                        !attacked(3, BLACK) && !attacked(2, BLACK))
                        ml.add(encodeMove(4, 2, 0, MF_CASTLE));
                } else if (us == BLACK && s == 60) {
                    if ((castle & 4) && !sq[61] && !sq[62] && !attacked(60, WHITE) && !attacked(61, WHITE) &&
                        !attacked(62, WHITE))
                        ml.add(encodeMove(60, 62, 0, MF_CASTLE));
                    if ((castle & 8) && !sq[59] && !sq[58] && !sq[57] && !attacked(60, WHITE) &&
                        !attacked(59, WHITE) && !attacked(58, WHITE))
                        ml.add(encodeMove(60, 58, 0, MF_CASTLE));
                }
            }
        } else {
            int d0 = t == BISHOP ? 4 : 0, d1 = t == ROOK ? 4 : 8;
            for (int d = d0; d < d1; d++)
                for (int i = 0; i < rayN[s][d]; i++) {
                    int to = rays[s][d][i];
                    if (!sq[to]) {
                        if (!capturesOnly) ml.add(encodeMove(s, to));
                        continue;
                    }
                    if (colorOf(sq[to]) == them) ml.add(encodeMove(s, to, 0, MF_CAPTURE));
                    break;
                }
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

    undoStack.push_back({m, cap, castle, ep, halfmove, hash});
    hashHistory.push_back(hash);

    if (ep >= 0) hash ^= Z_ep[fileOf(ep)];
    if (cap) {
        hash ^= Z_piece[cap][capSq];
        sq[capSq] = EMPTY;
        pieceCount--;
    }
    hash ^= Z_piece[p][from];
    sq[from] = EMPTY;
    int np = movePromo(m) ? makePiece(stm, movePromo(m)) : p;
    sq[to] = np;
    hash ^= Z_piece[np][to];
    if (typeOf(p) == KING) kingSq[stm] = to;
    if (flags & MF_CASTLE) {
        int rf, rt;
        rookCastleSquares(to, rf, rt);
        int rook = sq[rf];
        sq[rf] = EMPTY;
        sq[rt] = rook;
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

    if (attacked(kingSq[stm ^ 1], stm)) {
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
    sq[from] = p;
    sq[to] = EMPTY;
    if (typeOf(p) == KING) kingSq[stm] = from;
    if (flags & MF_CASTLE) {
        int rf, rt;
        rookCastleSquares(to, rf, rt);
        sq[rf] = sq[rt];
        sq[rt] = EMPTY;
    }
    if (u.captured) {
        int capSq = (flags & MF_EP) ? to + (stm == WHITE ? -8 : 8) : to;
        sq[capSq] = u.captured;
        pieceCount++;
    }
    castle = u.castle;
    ep = u.ep;
    halfmove = u.halfmove;
    hash = u.hash;
}

void Board::makeNull() {
    undoStack.push_back({NO_MOVE, 0, castle, ep, halfmove, hash});
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
    int minors = 0, bishopColors = 0, knights = 0;
    for (int s = 0; s < 64; s++) {
        int t = typeOf(sq[s]);
        if (t == PAWN || t == ROOK || t == QUEEN) return false;
        if (t == KNIGHT) minors++, knights++;
        if (t == BISHOP) minors++, bishopColors |= 1 << ((fileOf(s) + rankOf(s)) & 1);
    }
    if (minors <= 1) return true;
    return knights == 0 && bishopColors != 3;  // only same-colored bishops
}

bool Board::hasNonPawnMaterial(int c) const {
    for (int s = 0; s < 64; s++) {
        int p = sq[s];
        if (p && colorOf(p) == c && typeOf(p) != PAWN && typeOf(p) != KING) return true;
    }
    return false;
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
