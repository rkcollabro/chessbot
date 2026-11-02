"""Sequential probability ratio test (SPRT) between two engine builds: plays game pairs (each opening once
with each color) until the result is statistically clear, then stops.

H0: candidate is elo0 stronger than base; H1: elo1 stronger (logistic Elo). With the default bounds a pass
means the candidate is very likely at least as strong, and probably about elo1 stronger or more.

    python tools/sprt.py <candidate> <base> [--tc 5+0.05] [--concurrency 4] [--elo0 0] [--elo1 10]

--fixed N plays exactly N pairs starting at opening --offset and writes the tallies to --json; the Match workflow
runs one such shard per job and tools/combine.py adds them up.
"""
import json
import argparse
import concurrent.futures as cf
import math
import os
import threading
import time

import chess
import chess.engine

ALPHA = BETA = 0.05


def logistic(elo):
    return 1 / (1 + 10 ** (-elo / 400))


def to_elo(score):
    score = min(max(score, 1e-4), 1 - 1e-4)
    return -400 * math.log10(1 / score - 1)


def play(engines, opening, tc):
    """One game; engines = {chess.WHITE: engine, chess.BLACK: engine}. Returns "1-0", "0-1" or "1/2-1/2"."""
    base, inc = tc
    board = chess.Board()
    for uci in opening.split():
        board.push_uci(uci)
    game = object()  # a new game key makes python-chess send "ucinewgame"
    clock = {chess.WHITE: base, chess.BLACK: base}
    lopsided = 0
    while not board.is_game_over(claim_draw=True):
        side = board.turn
        limit = chess.engine.Limit(white_clock=clock[chess.WHITE], black_clock=clock[chess.BLACK],
                                   white_inc=inc, black_inc=inc)
        t0 = time.monotonic()
        result = engines[side].play(board, limit, info=chess.engine.INFO_SCORE, game=game)
        clock[side] -= time.monotonic() - t0
        if clock[side] < -0.5 or result.move is None or result.move not in board.legal_moves:
            return "0-1" if side == chess.WHITE else "1-0"  # flag or illegal move loses
        if "score" in result.info:  # adjudicate once both engines see a +-10 pawn score for 3 moves each
            cp = result.info["score"].pov(side).score(mate_score=100000)
            lopsided = lopsided + 1 if abs(cp) >= 1000 else 0
            if lopsided >= 6:
                return "1-0" if (cp > 0) == (side == chess.WHITE) else "0-1"
        clock[side] += inc
        board.push(result.move)
        if board.ply() > 400:
            return "1/2-1/2"
    return board.result(claim_draw=True)


def open_engine(path, book=False):
    e = chess.engine.SimpleEngine.popen_uci([os.path.abspath(path), "uci"])
    e.configure({"OwnBook": book, "Threads": 1})
    return e


def llr(pairs, elo0, elo1):
    """Pentanomial GSPRT log-likelihood ratio from per-pair scores in {0, .25, .5, .75, 1}."""
    n = len(pairs)
    if n < 2:
        return 0.0
    mean = sum(pairs) / n
    var = sum((p - mean) ** 2 for p in pairs) / n
    if var <= 0:
        return 0.0
    s0, s1 = logistic(elo0), logistic(elo1)
    return n * (s1 - s0) * (2 * mean - s0 - s1) / (2 * var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("base")
    ap.add_argument("--tc", default="5+0.05")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--elo0", type=float, default=0)
    ap.add_argument("--elo1", type=float, default=10)
    ap.add_argument("--max-pairs", type=int, default=3000)
    ap.add_argument("--rule", action="store_true",
                    help="fixed-games acceptance rule instead of SPRT: --games games, pass with more than --win-over "
                         "wins and at most --max-losses losses")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--win-over", type=int, default=30)
    ap.add_argument("--max-losses", type=int, default=10)
    ap.add_argument("--fixed", type=int, default=0, help="play exactly this many pairs (no early stop)")
    ap.add_argument("--offset", type=int, default=0, help="index of the first opening")
    ap.add_argument("--json", help="write W/L/D and per-pair scores here")
    ap.add_argument("--openings", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "openings.txt"))
    ap.add_argument("--book", choices=["none", "cand", "base", "both"], default="none",
                    help="which side plays from its own opening book (book.bin next to the exe); with a book in "
                         "play every game starts from the initial position instead of the opening suite")
    args = ap.parse_args()
    tc = tuple(float(x) for x in args.tc.split("+"))
    if args.book == "none":
        openings = [l.strip() for l in open(args.openings) if l.strip()]
    else:
        openings = [""]  # start position: the book (or the search) chooses the opening
    lower, upper = math.log(BETA / (1 - ALPHA)), math.log((1 - BETA) / ALPHA)

    pairs, wdl = [], [0, 0, 0]
    lock = threading.Lock()
    done = threading.Event()
    start = time.time()

    local = threading.local()  # each worker keeps its two engines running between games

    def run_pair(i):
        if done.is_set():
            return
        if not hasattr(local, "cand"):
            local.cand = open_engine(args.candidate, args.book in ("cand", "both"))
            local.base = open_engine(args.base, args.book in ("base", "both"))
        opening = openings[(args.offset + i) % len(openings)]
        points = 0.0
        for cand_white in (True, False):
            w, b = (local.cand, local.base) if cand_white else (local.base, local.cand)
            res = play({chess.WHITE: w, chess.BLACK: b}, opening, tc)
            pts = 0.5 if res == "1/2-1/2" else float((res == "1-0") == cand_white)
            points += pts
            with lock:
                wdl[0 if pts == 1 else 1 if pts == 0 else 2] += 1
        with lock:
            pairs.append(points / 2)
            n = len(pairs)
            mean = sum(pairs) / n
            sd = math.sqrt(sum((p - mean) ** 2 for p in pairs) / n / n) if n > 1 else 0
            ratio = llr(pairs, args.elo0, args.elo1)
            print(f"pairs {n:4d}  W {wdl[0]} L {wdl[1]} D {wdl[2]}  elo {to_elo(mean):+6.1f} "
                  f"[{to_elo(mean - 1.96 * sd):+.0f},{to_elo(mean + 1.96 * sd):+.0f}]  "
                  f"LLR {ratio:+.2f} ({lower:.2f},{upper:.2f})  {(time.time() - start) / 60:.0f} min", flush=True)
            if args.fixed:
                pass
            elif args.rule:
                games = sum(wdl)
                # Stop early once the rule can no longer be met.
                if wdl[1] > args.max_losses or wdl[0] + (args.games - games) <= args.win_over or games >= args.games:
                    done.set()
            elif ratio >= upper or ratio <= lower or n >= args.max_pairs:
                done.set()

    total_pairs = args.fixed or (args.games // 2 if args.rule else args.max_pairs)
    with cf.ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(run_pair, i) for i in range(total_pairs)]
        for f in futures:
            f.result()
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"wdl": wdl, "pairs": pairs}, f)
    ratio = llr(pairs, args.elo0, args.elo1)
    mean = sum(pairs) / len(pairs)
    if args.rule:
        verdict = "PASSED" if wdl[0] > args.win_over and wdl[1] <= args.max_losses else "FAILED"
    else:
        verdict = "PASSED" if ratio >= upper else "FAILED" if ratio <= lower else "INCONCLUSIVE"
    print(f"RESULT {verdict}  elo {to_elo(mean):+.1f}  pairs {len(pairs)}  games {sum(wdl)}  "
          f"W {wdl[0]} L {wdl[1]} D {wdl[2]}  LLR {ratio:+.2f}", flush=True)
    os._exit(0)  # the engines' I/O threads would otherwise keep the process alive


if __name__ == "__main__":
    main()
