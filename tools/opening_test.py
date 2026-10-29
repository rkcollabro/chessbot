"""Opening-quality test: the engine with its book vs the same engine without it.

Plays each opening from the start position for --plies plies (colors alternate), then has Stockfish score
the final position. A positive average means the book side came out of the opening better.

usage: python tools/opening_test.py "engines/v4_book.exe uci" [--openings 60] [--movetime 3] [--threads 4]
"""
import argparse
import concurrent.futures as cf
import math
import os
import shlex
import statistics
import time

import chess
import chess.engine

SF = os.path.abspath("tools/stockfish/stockfish/stockfish-windows-x86-64-universal.exe")


def command(cmd):
    parts = shlex.split(cmd, posix=False)
    parts[0] = os.path.abspath(parts[0])
    return parts


def play_opening(cmd, book_white, plies, movetime, threads):
    engines = {}
    try:
        for side in (chess.WHITE, chess.BLACK):
            e = chess.engine.SimpleEngine.popen_uci(command(cmd))
            e.configure({"OwnBook": side == (chess.WHITE if book_white else chess.BLACK), "Threads": threads})
            engines[side] = e
        book_side = chess.WHITE if book_white else chess.BLACK
        board = chess.Board()
        exit_board = None  # position when the book side first had to think for itself
        while board.ply() < plies and not board.is_game_over():
            t0 = time.monotonic()
            move = engines[board.turn].play(board, chess.engine.Limit(time=movetime)).move
            if board.turn == book_side and exit_board is None and time.monotonic() - t0 > 0.1:
                exit_board = board.copy()
            board.push(move)
        return board, exit_board or board.copy()
    finally:
        for e in engines.values():
            e.quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--openings", type=int, default=60)
    ap.add_argument("--plies", type=int, default=20)
    ap.add_argument("--movetime", type=float, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--depth", type=int, default=20)
    args = ap.parse_args()

    jobs = [i % 2 == 0 for i in range(args.openings)]  # True: book side plays White
    results = []
    with cf.ThreadPoolExecutor(args.concurrency) as pool:
        futures = {pool.submit(play_opening, args.engine, bw, args.plies, args.movetime, args.threads): bw
                   for bw in jobs}
        for n, f in enumerate(cf.as_completed(futures), 1):
            results.append((futures[f], f.result()))
            print(f"played {n}/{args.openings}", flush=True)

    sf = chess.engine.SimpleEngine.popen_uci(SF)
    sf.configure({"Threads": 8, "Hash": 256})
    scored = []
    exits = []
    for book_white, (board, exit_board) in results:
        pov = chess.WHITE if book_white else chess.BLACK
        info = sf.analyse(board, chess.engine.Limit(depth=args.depth))
        scored.append((info["score"].pov(pov).score(mate_score=10000), book_white, board))
        info = sf.analyse(exit_board, chess.engine.Limit(depth=args.depth))
        exits.append((info["score"].pov(pov).score(mate_score=10000), book_white, exit_board))
    sf.quit()

    def summary(name, vals):
        if not vals:
            return
        mean = statistics.mean(vals)
        se = statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0
        print(f"{name:28s} n={len(vals):3d}  mean {mean:+6.0f} cp  (+-{1.96 * se:.0f} at 95%)")

    print("\nStockfish score for the book side when its book ran out (avg ply %.1f):"
          % statistics.mean(b.ply() for _, _, b in exits))
    summary("all openings", [s for s, _, _ in exits])
    summary("book side as White", [s for s, bw, _ in exits if bw])
    summary("book side as Black", [s for s, bw, _ in exits if not bw])
    print("\nStockfish score of the book side's position after", args.plies // 2, "moves:")
    summary("all openings", [s for s, _, _ in scored])
    summary("book side as White", [s for s, bw, _ in scored if bw])
    summary("book side as Black", [s for s, bw, _ in scored if not bw])
    print("\nworst 5 for the book side:")
    for cp, bw, board in sorted(scored, key=lambda x: x[0])[:5]:
        print(f"  {cp:+5d}  book={'White' if bw else 'Black'}  {chess.Board().variation_san(board.move_stack)}")
    print("best 5 for the book side:")
    for cp, bw, board in sorted(scored, key=lambda x: -x[0])[:5]:
        print(f"  {cp:+5d}  book={'White' if bw else 'Black'}  {chess.Board().variation_san(board.move_stack)}")


if __name__ == "__main__":
    main()
