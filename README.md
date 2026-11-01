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
| Strength | Level menu: Level 1 to Level 6, or MAX (see Levels) |
| Variant | Variant menu (starts a new game as White) |

## Variants

Besides standard chess the engine plays five lichess variants, picked from the Variant menu or with the UCI `UCI_Variant` option:

| Variant | Rules | In the GUI |
|---|---|---|
| Chess960 | Random back rank (one of 960); castling puts the king on g/c and the rook on f/d as usual | Each new game gets a random start. Castle by clicking the king, then its rook |
| Three-check | Giving a third check also wins | The panel counts checks for both sides |
| Crazyhouse | Captured pieces join your hand and can be dropped on any empty square instead of moving (pawns not on the first or last rank). Promoted pieces go back to the hand as pawns | Both hands show in the panel: click a piece in your hand, then an empty square |
| King of the Hill | Getting your king to d4, e4, d5 or e5 also wins | The four center squares are tinted |
| Horde | White has 36 pawns and no king, and wins by checkmate; Black wins by capturing every white piece. White pawns on the first rank may also move two squares | |

FENs follow lichess / python-chess: crazyhouse hands in brackets with `~` after promoted pieces (`...RNBQKBNR[Qp] w ...`), three-check counters as checks remaining before the clocks (`... - 3+3 0 1`, the lichess `+1+0` suffix is also read), and Chess960 castling as `KQkq` or the rooks' files (`HFhf`). In UCI, drops are written `N@f3`, and Chess960 castling as king-takes-rook (`e1h1`).

The opening book and the endgame tables only apply to standard chess (the tables also to Chess960 once castling is gone). The evaluation adds a few hand-set terms per variant on top of the tuned weights: pieces in hand and holes around the king for crazyhouse, a bonus per check given, the king's distance to the hill, and extra worth for the horde's pawns.

## Levels

| Level | How it plays | Elo vs Level 6 |
|---|---|---|
| Level 1 | 1,000 nodes a move, heavy evaluation noise, no book | about −2500 |
| Level 2 | 4,000 nodes, strong noise, no book | about −2000 |
| Level 3 | 20,000 nodes, light noise | about −1500 |
| Level 4 | 120,000 nodes | about −1000 |
| Level 5 | 800,000 nodes | about −500 |
| Level 6 | full engine, 3 seconds a move, all threads (default) | 0 |
| MAX | full engine, 30 seconds a move, all threads | about +500 |

Levels 1–5 cap the nodes searched on one thread, so they play the same strength on any PC and usually move within a second or two. The ratings are spaced evenly and are only a rough guide: self-play matches (`tools/rate_levels.py`) put levels 1–5 about 450–630 Elo apart, and the gaps up to Level 6 and MAX were not measured. MAX's real gain from ten times the thinking time is probably smaller than 500. In UCI, the `Level` option (1–6) picks the level, and the time still comes from the `go` command.

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
- **Time management:** with a fixed time per move (GUI), it stops starting new iterations after half the time. On a clock (UCI `wtime`/`btime`), it aims for 1/25 of the remaining time plus 3/4 of the increment. It stretches that up to 5× (never more than a third of the clock) while the best move keeps changing, the score is dropping, or the root move fails low, and cuts it to 0.7× once the same move has held for six iterations. It supports pondering (`go ponder` / `ponderhit`), reporting the expected reply with `bestmove ... ponder ...`. After a ponderhit it moves at once if the pondering already took longer than the move's target time.
- **Multi-threading:** Lazy SMP. All threads search the same position and share a lockless transposition table. The GUI always uses every core, up to 16; UCI has a `Threads` option.
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
chesscli test                                    # perft suite (standard + variants) + hash consistency
chesscli perft <d> --variant crazyhouse [fen]    # variants: chess960, 3check, crazyhouse, kingofthehill, horde
python tools/variant_perft.py [games] [seed]     # random variant positions, perft vs python-chess
chesscli tbgen | tbprobe <fen> | search <ms> [fen]
chesscli uci                                     # UCI engine (Arena, Cute Chess, lichess-bot, ...); UCI_Variant, UCI_Chess960
chesscli mkbook data/book_raw.txt 16 2500 180 games.pgn...   # plies, min Elo, min base seconds
python tools/vet_book.py data/book_raw.txt data/book_vetted.txt  # needs Stockfish in tools/stockfish
chesscli booktxt data/book_vetted.txt book.bin
chesscli extract data/positions.txt 3 1500000 games.pgn...
chesscli tune data/positions.txt src/eval_weights.inc 400
python tools/match.py "engines/new.exe uci" "engines/old.exe uci" --games 100 --tc 8+0.08 [--threads-a 4]
chesscli selfplay data/sp.bin 100000 5000 1             # NNUE training data: games, nodes per move, seed
python tools/nnue_train.py nnue.bin data/sp*.bin --epochs 10 [--hidden 512]
chesscli nnuecheck 100                                   # incremental NNUE updates vs a full rebuild
python tools/sprt.py <candidate> <base> --rule           # 60 games: pass with >30 wins and <=10 losses
```

GitHub Actions (run from the Actions tab or `gh workflow run`):

- **Match**: candidate ref vs base ref, games split across up to 20 runners, results combined in the run summary.
  `gh workflow run match.yml -f candidate=my-branch -f base=master -f games=400`
- **Generate data**: self-play on up to 20 runners for up to 5.5 hours each; positions are uploaded as artifacts.
  `gh workflow run datagen.yml -f ref=master -f jobs=20 -f minutes=300`

## Credits

The piece images are the "cburnett" set by Colin M.L. Burnett, which lichess uses by default. It is licensed GPLv2+ / BSD / GFDL. The images are in `assets/pieces` and are embedded into the exe.

The opening book and tuning data come from the Lichess Elite Database (database.nikonoel.fr), which is built from lichess.org games (CC0).
