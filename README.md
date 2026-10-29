# ChessBot

A C++17 chess engine with a native Win32 GUI and self-generated endgame tablebases. No external dependencies.

## Build & run

```
build.bat        # finds MSVC via vswhere, builds chessbot.exe (GUI) and chesscli.exe (tests)
chessbot.exe
```

On first launch the endgame tables are generated in the background (~5 min, ~260 MB in `tb/`); after that they load from disk in a second or two. The engine plays normally while they build and uses each table as soon as it is ready.

## Controls

| Action | How |
|---|---|
| Move | Click a piece, then a highlighted square (promotion pops up a menu) |
| New game as White / Black | Ctrl+N / Ctrl+B |
| Undo (takes back your move and the engine's reply) | Ctrl+Z |
| Flip board | F |
| Switch sides with the engine | Ctrl+E |
| Force the engine to move now | Space |
| Copy / paste FEN | Ctrl+C / Ctrl+V (pasting gives you the side to move) |
| Think time | Engine menu: 1, 3, 5, 10, 30 seconds |

## Engine

- **Board** (`board.cpp`): mailbox board with precomputed attack tables, incremental Zobrist hashing, and full rules (castling, en passant, promotion, 50-move rule, repetition).
- **Search** (`search.cpp`): iterative-deepening negamax with alpha-beta pruning and a principal variation search (PVS) window. It also uses:
  - a transposition table;
  - null-move pruning, reverse futility pruning and late move reductions;
  - check extensions;
  - move ordering by MVV-LVA (captures), killer moves and the history heuristic;
  - quiescence search with check evasions.
- **Evaluation** (`eval.cpp`):
  - material plus piece-square tables, with king tables blended between middlegame and endgame;
  - pawn structure: passed, isolated and doubled pawns;
  - bishop pair and rooks on open files;
  - mop-up terms for converting a lone-king endgame.
- **Tablebases** (`tablebase.cpp`): retrograde analysis over every material combination with ≤4 pieces (35 tables). The tables store distance to mate in plies.
  - Pawnless tables use 8-fold board symmetry; pawn tables use the left/right mirror.
  - Captures and promotions resolve through the smaller tables, which are generated first.
  - The search probes the tables at every node with ≤4 pieces. At the root, it picks the move that mates fastest (or loses slowest) straight from the table.

## Tests

```
chesscli test                  # perft suite (6 standard positions) + hash consistency
chesscli tbgen                 # build/load tables and print longest mate per table
chesscli tbprobe <fen>
chesscli search <ms> [fen]
```

The tablebase longest-mate values match the published ones (KQvK 10, KRvK 16, KBNvK 33, KQvKR 35, KRvKN 40, KPvK 28, …).
