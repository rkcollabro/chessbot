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
- **Search** (`search.cpp`): iterative-deepening negamax with alpha-beta pruning and a principal variation search (PVS) window. It also uses:
  - a transposition table;
  - null-move pruning, reverse futility pruning and late move reductions;
  - check extensions;
  - move ordering by MVV-LVA with static exchange evaluation (SEE), killer moves and the history heuristic;
  - quiescence search that drops SEE-losing captures and uses delta pruning.
- **Evaluation** (`eval.cpp`): a tapered (middlegame/endgame) linear evaluation with about 560 weights. Its terms are:
  - material and piece-square tables;
  - mobility, measured as safe squares per piece;
  - king safety: attacks on the king zone, safe checks and the pawn shield;
  - passed, isolated and doubled pawns;
  - bishop pair, rooks on open files and pawn threats.
- **Tuning** (`datagen.cpp`): Texel tuning, which fits the weights to game results with Adam on about 1.06M quiet positions from Lichess elite games. The weights live in `src/eval_weights.inc`.
- **Opening book** (`book.bin`): about 29k positions from 2023 Lichess elite games where both players are rated 2300+, covering the first 20 plies. Moves are chosen at random, weighted by popularity and score.
- **Tablebases** (`tablebase.cpp`): retrograde analysis over every material combination with 4 or fewer pieces (35 tables). The tables store distance to mate in plies. They are probed at every search node with 4 or fewer pieces, and at the root the engine plays the table's best move directly.

### Strength history (self-play, 8s + 0.08s per game)

| Change | Result | Elo |
|---|---|---|
| v2: bitboards + mobility/king safety + SEE pruning, vs v1 | +55 -16 =29 | about +143 |
| v3: Texel-tuned weights, vs v2 | +57 -48 =15 | about +26 |

## Tools

```
chesscli test                                    # perft suite + hash consistency
chesscli tbgen | tbprobe <fen> | search <ms> [fen]
chesscli uci                                     # UCI engine (Arena, Cute Chess, lichess-bot, ...)
chesscli mkbook book.bin 20 games.pgn...
chesscli extract data/positions.txt 3 1500000 games.pgn...
chesscli tune data/positions.txt src/eval_weights.inc 400
python tools/match.py "engines/new.exe uci" "engines/old.exe uci" --games 100 --tc 8+0.08
```

## Credits

The piece images are the "cburnett" set by Colin M.L. Burnett, which lichess uses by default. It is licensed GPLv2+ / BSD / GFDL. The images are in `assets/pieces` and are embedded into the exe.

The opening book and tuning data come from the Lichess Elite Database (database.nikonoel.fr), which is built from lichess.org games (CC0).
