"""Book audit: walks the opening book the way the engine plays it (moves with at least 15% of the top move's
weight, chosen in proportion to weight), collects every position where a line leaves the book, with the chance of
reaching it, and scores each one with our engine and with Stockfish.

usage: python tools/book_audit.py data/book_vetted.txt out.json [--ours chesscli.exe] [--sf stockfish.exe]
           [--ours-nodes 2000000] [--sf-depth 22] [--min-prob 0.0005] [--workers 6]

Scores are centipawns from White's point of view. Output: JSON list of
{fen, line (SAN), prob, prob_white_choices, prob_black_choices, ours, sf}.
"""
import argparse
import collections
import concurrent.futures as cf
import json
import os
import threading

import chess
import chess.engine


def key(board):
    return " ".join(board.fen(en_passant="fen").split()[:4])


def load_book(path):
    book = collections.defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) >= 3:
                book[" ".join(parts[0].split()[:4])].append((parts[1], int(parts[2])))
    return book


def walk(book, min_prob):
    """Leaves of the book tree: (board, san line, prob, prob from White's choices, prob from Black's choices)."""
    leaves = []

    def rec(board, line, p, pw, pb):
        opts = book.get(key(board), [])
        legal = {m.uci() for m in board.legal_moves}
        # Castling: the book stores king-to-g1/c1, which is also python-chess's uci, so no conversion needed.
        opts = [(u, w) for u, w in opts if u in legal]
        best = max((w for _, w in opts), default=0)
        opts = [(u, w) for u, w in opts if w >= best * 15 // 100 and w > 0]
        # A book exit, or a line already too rare to split further (its position stands for everything below it,
        # so the leaves always cover all of book play). Ply cap against transposition cycles.
        total = sum(w for _, w in opts)
        if not opts or len(line) >= 40:
            leaves.append((board.copy(), line, p, pw, pb))
            return
        for u, w in opts:
            q = w / total
            m = chess.Move.from_uci(u)
            san = board.san(m)
            board.push(m)
            white_moved = not board.turn  # the side that just moved
            args = (line + [san], p * q, pw * (q if white_moved else 1), pb * (1 if white_moved else q))
            if p * q < min_prob:
                leaves.append((board.copy(),) + args)
            else:
                rec(board, *args)
            board.pop()

    rec(chess.Board(), [], 1.0, 1.0, 1.0)
    return leaves


_local = threading.local()


def engines(args):
    if not hasattr(_local, "ours"):
        _local.ours = chess.engine.SimpleEngine.popen_uci([os.path.abspath(args.ours), "uci"])
        _local.ours.configure({"Threads": 1, "Hash": 64, "OwnBook": False})
        _local.sf = chess.engine.SimpleEngine.popen_uci(os.path.abspath(args.sf))
        _local.sf.configure({"Threads": 1, "Hash": 64})
    return _local.ours, _local.sf


def score(info, board):
    return info["score"].white().score(mate_score=10000)


def evaluate(leaf, args):
    board, line, p, pw, pb = leaf
    ours, sf = engines(args)
    o = score(ours.analyse(board, chess.engine.Limit(nodes=args.ours_nodes), game=object()), board)
    s = score(sf.analyse(board, chess.engine.Limit(depth=args.sf_depth), game=object()), board)
    return {"fen": board.fen(), "line": " ".join(line), "prob": p, "prob_white_choices": pw,
            "prob_black_choices": pb, "ours": o, "sf": s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("book")
    ap.add_argument("out")
    ap.add_argument("--ours", default="chesscli.exe")
    ap.add_argument("--sf", default="tools/stockfish/stockfish/stockfish-windows-x86-64-universal.exe")
    ap.add_argument("--ours-nodes", type=int, default=2_000_000)
    ap.add_argument("--sf-depth", type=int, default=22)
    ap.add_argument("--min-prob", type=float, default=0.0005)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    leaves = walk(load_book(args.book), args.min_prob)
    covered = sum(l[2] for l in leaves)
    print(f"{len(leaves)} book exits covering {covered:.1%} of book play", flush=True)
    results = []
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for n, r in enumerate(pool.map(lambda l: evaluate(l, args), leaves), 1):
            results.append(r)
            if n % 50 == 0:
                print(f"{n}/{len(leaves)}", flush=True)
                json.dump(results, open(args.out, "w"))
    json.dump(results, open(args.out, "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
