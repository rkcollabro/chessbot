"""Engine-vs-engine match over UCI, alternating colors from a set of balanced openings.

usage: python tools/match.py <engineA> <engineB> [--games N] [--tc base+inc] [--concurrency K]
Engines are commands, e.g. "chesscli.exe uci". Prints the score and an Elo estimate for A.
"""
import argparse
import concurrent.futures as cf
import math
import os
import random
import shlex
import time

import chess
import chess.engine

# Short, mostly balanced opening lines so games don't repeat.
OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6 f1b5", "e2e4 e7e5 g1f3 b8c6 f1c4", "e2e4 c7c5 g1f3 d7d6", "e2e4 c7c5 g1f3 b8c6",
    "e2e4 c7c5 b1c3", "e2e4 e7e6 d2d4 d7d5", "e2e4 c7c6 d2d4 d7d5", "e2e4 d7d6 d2d4 g8f6",
    "e2e4 g7g6 d2d4 f8g7", "e2e4 d7d5 e4d5 d8d5", "d2d4 d7d5 c2c4 e7e6", "d2d4 d7d5 c2c4 c7c6",
    "d2d4 g8f6 c2c4 e7e6 g1f3", "d2d4 g8f6 c2c4 g7g6 b1c3", "d2d4 g8f6 c2c4 c7c5 d4d5",
    "d2d4 f7f5 g2g3", "c2c4 e7e5 b1c3", "c2c4 g8f6 b1c3 e7e6", "g1f3 d7d5 g2g3", "g1f3 g8f6 c2c4 c7c5",
    "e2e4 e7e5 f2f4", "d2d4 d7d5 c1f4", "e2e4 c7c5 c2c3", "d2d4 g8f6 c1g5", "b2b3 e7e5",
    "e2e4 e7e5 b1c3 g8f6", "e2e4 b8c6", "d2d4 e7e6 c2c4 f8b4", "c2c4 c7c5 g1f3 b8c6", "e2e4 g8f6 e4e5 f6d5",
]


THREADS = {}  # engine command -> thread count


def command(cmd):
    parts = shlex.split(cmd, posix=False)
    parts[0] = os.path.abspath(parts[0])
    return parts


def play(cmd_white, cmd_black, opening, tc):
    base, inc = tc
    board = chess.Board()
    for uci in opening.split():
        board.push_uci(uci)
    engines = {}
    try:
        engines[chess.WHITE] = chess.engine.SimpleEngine.popen_uci(command(cmd_white))
        engines[chess.BLACK] = chess.engine.SimpleEngine.popen_uci(command(cmd_black))
        for e in engines.values():
            if "OwnBook" in e.options:
                e.configure({"OwnBook": False})  # the match supplies its own openings
        for side, cmd in ((chess.WHITE, cmd_white), (chess.BLACK, cmd_black)):
            t = THREADS.get(cmd)
            if t and "Threads" in engines[side].options:
                engines[side].configure({"Threads": t})
        clock = {chess.WHITE: base, chess.BLACK: base}
        while not board.is_game_over(claim_draw=True):
            side = board.turn
            limit = chess.engine.Limit(white_clock=clock[chess.WHITE], black_clock=clock[chess.BLACK],
                                       white_inc=inc, black_inc=inc)
            t0 = time.monotonic()
            result = engines[side].play(board, limit)
            clock[side] -= time.monotonic() - t0
            if clock[side] < -0.5 or result.move is None or result.move not in board.legal_moves:
                return "0-1" if side == chess.WHITE else "1-0"  # flag or illegal move loses
            clock[side] += inc
            board.push(result.move)
            if board.ply() > 400:
                return "1/2-1/2"
        return board.result(claim_draw=True)
    finally:
        for e in engines.values():
            e.quit()


def elo(score, n):
    p = min(max(score / n, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / p - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--tc", default="10+0.1", help="seconds per game + increment")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--threads-a", type=int, default=0)
    ap.add_argument("--threads-b", type=int, default=0)
    args = ap.parse_args()
    base, inc = (float(x) for x in args.tc.split("+"))
    if args.a == args.b and (args.threads_a or args.threads_b):
        args.b += " "  # distinguish identical commands so each side gets its own thread count
    THREADS[args.a] = args.threads_a
    THREADS[args.b] = args.threads_b

    openings = OPENINGS[:]
    random.Random(1).shuffle(openings)
    jobs = []
    for i in range(args.games):
        opening = openings[(i // 2) % len(openings)]
        a_white = i % 2 == 0
        jobs.append((a_white, opening))

    score = wins = losses = draws = 0
    with cf.ThreadPoolExecutor(args.concurrency) as pool:
        futures = {
            pool.submit(play, args.a if aw else args.b, args.b if aw else args.a, op, (base, inc)): aw
            for aw, op in jobs
        }
        for n, f in enumerate(cf.as_completed(futures), 1):
            aw = futures[f]
            res = f.result()
            if res == "1/2-1/2":
                draws += 1
                score += 0.5
            elif (res == "1-0") == aw:
                wins += 1
                score += 1
            else:
                losses += 1
            print(f"[{n}/{args.games}] A {wins}W {losses}L {draws}D  score {score / n:.3f}  elo {elo(score, n):+.0f}",
                  flush=True)


if __name__ == "__main__":
    main()
