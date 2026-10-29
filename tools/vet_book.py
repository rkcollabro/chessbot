"""Vets opening-book moves with Stockfish: drops any move that is more than --margin centipawns
worse than the best move in its position.

usage: python tools/vet_book.py data/book_raw.txt data/book_vetted.txt [--depth 16] [--margin 40]
Input/output lines: fen;uci;weight;games;score
"""
import argparse
import collections
import concurrent.futures as cf
import os
import threading

import chess
import chess.engine

SF = os.path.abspath("tools/stockfish/stockfish/stockfish-windows-x86-64-universal.exe")
_local = threading.local()


def engine():
    if not hasattr(_local, "sf"):
        _local.sf = chess.engine.SimpleEngine.popen_uci(SF)
        _local.sf.configure({"Threads": 1, "Hash": 64})
    return _local.sf


def vet(fen, lines, depth, margin):
    sf = engine()
    board = chess.Board(fen)
    limit = chess.engine.Limit(depth=depth)
    infos = sf.analyse(board, limit, multipv=min(5, board.legal_moves.count()))
    scores = {i["pv"][0]: i["score"].pov(board.turn).score(mate_score=10000) for i in infos}
    best = max(scores.values())
    kept, dropped = [], []
    for line in lines:
        move = chess.Move.from_uci(line.split(";")[1])
        if move not in scores:
            info = sf.analyse(board, limit, root_moves=[move])
            scores[move] = info["score"].pov(board.turn).score(mate_score=10000)
        (kept if best - scores[move] <= margin else dropped).append((line, best - scores[move]))
    return kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--margin", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    by_fen = collections.defaultdict(list)
    with open(args.inp) as f:
        for line in f:
            line = line.strip()
            if line:
                by_fen[line.split(";")[0]].append(line)

    kept_all, dropped_all = [], []
    with cf.ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(vet, fen, lines, args.depth, args.margin) for fen, lines in by_fen.items()]
        for n, fut in enumerate(cf.as_completed(futures), 1):
            kept, dropped = fut.result()
            kept_all += kept
            dropped_all += dropped
            if n % 500 == 0:
                print(f"{n}/{len(futures)} positions, kept {len(kept_all)}, dropped {len(dropped_all)}", flush=True)

    with open(args.out, "w") as f:
        for line, _ in kept_all:
            f.write(line + "\n")
    print(f"kept {len(kept_all)} moves, dropped {len(dropped_all)}")
    worst = sorted(dropped_all, key=lambda x: -x[1])[:10]
    print("worst dropped moves (cp worse than best):")
    for line, loss in worst:
        fen, uci = line.split(";")[:2]
        b = chess.Board(fen)
        print(f"  {loss:5d}  {b.san(chess.Move.from_uci(uci)):7s} in {fen}")


if __name__ == "__main__":
    main()
