# ChessBot

A C++17 chess engine with a native Win32 GUI, a tuned evaluation, an opening book and self-generated endgame tablebases. It has no external dependencies.

## Build & run

```
build.bat        # finds MSVC via vswhere, builds chessbot.exe (GUI) and chesscli.exe (tools/UCI)
chessbot.exe
```

On first launch the endgame tables are generated in the background (~5 min, ~260 MB in `tb/`); after that they load from disk in a second or two. `book.bin` must sit next to the exe for the opening book.

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

- **Board** (`board.cpp`): bitboards with classical ray attacks, incremental Zobrist hashing, full rules. The legality check is skipped for moves that cannot expose the king.
- **Search** (`search.cpp`): iterative-deepening negamax with alpha-beta pruning, a principal variation search (PVS) window and aspiration windows.
  - **Pruning:**
    - null-move pruning and reverse futility pruning, both adjusted by an "improving" signal;
    - futility pruning and late-move pruning;
    - SEE pruning of losing moves at low depth;
    - log-scaled late move reductions.
  - **Extensions:** check extensions and singular extensions (with multi-cut), plus internal iterative reduction when there is no remembered best move.
  - **Move ordering:** MVV-LVA with SEE, killer moves, counter-moves, and a history table that also penalizes moves that failed.
  - **Quiescence search:** drops SEE-losing captures, uses delta pruning and probes the transposition table.
- **Multi-threading:** Lazy SMP. All threads search the same position and share a lockless transposition table. The GUI defaults to 4 threads (Engine menu: 1/2/4/8); UCI has a `Threads` option.
- **Evaluation** (`eval.cpp`): a tapered (middlegame/endgame) linear evaluation with about 560 weights. Its terms are:
  - material and piece-square tables;
  - mobility, measured as safe squares per piece;
  - king safety: attacks on the king zone, safe checks and the pawn shield;
  - passed, isolated and doubled pawns;
  - bishop pair, rooks on open files and pawn threats.
- **Tuning** (`datagen.cpp`): Texel tuning, which fits the weights to game results with Adam on about 1.06M quiet positions from Lichess elite games. The weights live in `src/eval_weights.inc`.
- **Opening book** (`book.bin`): 8,764 moves in about 6,000 positions, built from 2023 Lichess elite games between 2500+ players at 3 minutes or longer, covering the first 16 plies. Every move was vetted offline with Stockfish 19 at depth 16, and any move more than 0.4 pawns worse than the best was dropped. Moves are chosen at random, weighted by popularity and score.
- **Tablebases** (`tablebase.cpp`): retrograde analysis over every material combination with 4 or fewer pieces (35 tables). The tables store distance to mate in plies. They are probed at every search node with 4 or fewer pieces, and at the root the engine plays the table's best move directly.

### Strength history (self-play, 8s + 0.08s per game)

| Change | Result | Elo |
|---|---|---|
| v2: bitboards + mobility/king safety + SEE pruning, vs v1 | +55 -16 =29 | about +143 |
| v3: Texel-tuned weights, vs v2 | +57 -48 =15 | about +26 |
| v4: search improvements (single thread), vs v3 | +158 -2 =40 | about +363 |
| v4 with 4 threads, vs v4 with 1 thread | +43 -5 =32 | about +179 |
| Experiment: pawn storms + quadratic king danger, vs v4 (not merged; branch `experiment/king-danger`) | +54 -73 =73 | about -33 |

## Tools

```
chesscli test                                    # perft suite + hash consistency
chesscli tbgen | tbprobe <fen> | search <ms> [fen]
chesscli uci                                     # UCI engine (Arena, Cute Chess, lichess-bot, ...)
chesscli mkbook data/book_raw.txt 16 2500 180 games.pgn...   # plies, min Elo, min base seconds
python tools/vet_book.py data/book_raw.txt data/book_vetted.txt  # needs Stockfish in tools/stockfish
chesscli booktxt data/book_vetted.txt book.bin
chesscli extract data/positions.txt 3 1500000 games.pgn...
chesscli tune data/positions.txt src/eval_weights.inc 400
python tools/match.py "engines/new.exe uci" "engines/old.exe uci" --games 100 --tc 8+0.08 [--threads-a 4]
```

## Credits

The piece images are the "cburnett" set by Colin M.L. Burnett, which lichess uses by default. It is licensed GPLv2+ / BSD / GFDL. The images are in `assets/pieces` and are embedded into the exe.

The opening book and tuning data come from the Lichess Elite Database (database.nikonoel.fr), which is built from lichess.org games (CC0).
