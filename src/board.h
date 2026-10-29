#pragma once
#include <cstdint>
#include <string>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif

enum Color : int { WHITE = 0, BLACK = 1 };
enum PieceType : int { NO_TYPE = 0, PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6 };

// Piece code = type | color << 3. White pieces are 1..6, black 9..14, 0 is empty.
constexpr int EMPTY = 0;
inline int makePiece(int c, int t) { return t | (c << 3); }
inline int typeOf(int p) { return p & 7; }
inline int colorOf(int p) { return p >> 3; }
inline int fileOf(int s) { return s & 7; }
inline int rankOf(int s) { return s >> 3; }

// ---------------------------------------------------------------- bitboards
using Bitboard = uint64_t;
inline Bitboard bit(int s) { return Bitboard(1) << s; }
inline int lsb(Bitboard b) {
#ifdef _MSC_VER
    unsigned long i;
    _BitScanForward64(&i, b);
    return int(i);
#else
    return __builtin_ctzll(b);
#endif
}
inline int msb(Bitboard b) {
#ifdef _MSC_VER
    unsigned long i;
    _BitScanReverse64(&i, b);
    return int(i);
#else
    return 63 - __builtin_clzll(b);
#endif
}
inline int popcount(Bitboard b) {
#ifdef _MSC_VER
    return int(__popcnt64(b));
#else
    return __builtin_popcountll(b);
#endif
}
inline int popLsb(Bitboard& b) {
    int s = lsb(b);
    b &= b - 1;
    return s;
}

constexpr Bitboard FILE_A = 0x0101010101010101ULL;
constexpr Bitboard RANK_1 = 0xFFULL;
inline Bitboard fileMask(int f) { return FILE_A << f; }
inline Bitboard rankMask(int r) { return RANK_1 << (8 * r); }

namespace attacks {
Bitboard pawn(int color, int s);  // squares a pawn of `color` on s attacks
Bitboard knight(int s);
Bitboard king(int s);
Bitboard bishop(int s, Bitboard occ);
Bitboard rook(int s, Bitboard occ);
inline Bitboard queen(int s, Bitboard occ) { return bishop(s, occ) | rook(s, occ); }
}  // namespace attacks

// ---------------------------------------------------------------- moves
// Squares: a1 = 0, h1 = 7, a8 = 56, h8 = 63.
// Move packing: from(6) | to(6) << 6 | promo type(3) << 12 | flags << 16
using Move = uint32_t;
enum MoveFlag : uint32_t { MF_CAPTURE = 1, MF_EP = 2, MF_CASTLE = 4, MF_DOUBLE = 8 };
constexpr Move NO_MOVE = 0;
inline Move encodeMove(int from, int to, int promo = 0, uint32_t flags = 0) {
    return Move(from | (to << 6) | (promo << 12) | (flags << 16));
}
inline int moveFrom(Move m) { return m & 63; }
inline int moveTo(Move m) { return (m >> 6) & 63; }
inline int movePromo(Move m) { return (m >> 12) & 7; }
inline uint32_t moveFlags(Move m) { return m >> 16; }
inline bool isCapture(Move m) { return (moveFlags(m) & MF_CAPTURE) != 0; }

struct MoveList {
    Move moves[256];
    int count = 0;
    void add(Move m) { moves[count++] = m; }
};

struct Undo {
    Move move;
    int captured;
    int castle;
    int ep;
    int halfmove;
    uint64_t hash;
    int checkCache;
};

class Board {
public:
    int sq[64];
    Bitboard pieces[2][7];  // [color][type]; index 0 unused
    Bitboard occ[2];
    Bitboard occAll = 0;
    Color stm = WHITE;
    int castle = 0;      // 1 = K, 2 = Q, 4 = k, 8 = q
    int ep = -1;         // en passant target square or -1
    int halfmove = 0;
    int fullmove = 1;
    uint64_t hash = 0;
    int kingSq[2] = {-1, -1};
    int pieceCount = 0;  // all pieces including kings
    mutable int checkCache = -1;  // -1 unknown, else whether the side to move is in check
    std::vector<Undo> undoStack;
    std::vector<uint64_t> hashHistory;  // hash before each move made

    Board();
    void clear();
    void setStartPos();
    bool setFEN(const std::string& fen);
    std::string getFEN() const;
    void put(int s, int piece);
    uint64_t computeHash() const;

    Bitboard bb(int c, int t) const { return pieces[c][t]; }
    Bitboard attackersTo(int s, Bitboard occupancy) const;  // both colors
    bool attacked(int s, int byColor) const;
    bool inCheck() const {
        if (checkCache < 0) checkCache = attacked(kingSq[stm], stm ^ 1) ? 1 : 0;
        return checkCache != 0;
    }

    // Pseudo-legal moves. capturesOnly also includes promotions.
    void genMoves(MoveList& ml, bool capturesOnly = false) const;
    void genLegal(MoveList& ml);

    // Returns false (and leaves the board unchanged) if the move leaves the king in check.
    bool makeMove(Move m);
    void unmakeMove();
    void makeNull();
    void unmakeNull();

    bool isRepetition() const;
    int repetitionCount() const;
    bool insufficientMaterial() const;
    bool hasNonPawnMaterial(int c) const;

    std::string moveToUCI(Move m) const;
    std::string moveToSAN(Move m);
    Move parseUCI(const std::string& s);
    Move parseSAN(const std::string& s);

private:
    void addPiece(int s, int p);
    void removePiece(int s);
};

void initTables();
std::string squareName(int s);
