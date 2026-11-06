"""Rebuilds the opening book from a book_audit.py run: at every book position, the side to move keeps only its
best move, or its two best (played 50/50) when the second is within --margin centipawns of the best.

A move's value is backed up from the scored book exits (the average of our engine and Stockfish, clipped to
+-1000 cp): where the side we are playing is to move, it plays this policy; where the opponent is to move, its
replies are averaged by how often the old book (built from master games) has them played. Positions the audit
did not reach keep no book move, so the engine thinks for itself there.

usage: python tools/book_rebuild.py data/book_vetted.txt book_audit.json data/book_best.txt [--margin 30]
"""
import argparse
import json
import sys

import chess

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import book_audit as audit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("book")
    ap.add_argument("audit")
    ap.add_argument("out")
    ap.add_argument("--margin", type=int, default=30)
    ap.add_argument("--min-prob", type=float, default=0.0005, help="must match the audit run")
    args = ap.parse_args()

    book = audit.load_book(args.book)
    scored = {}
    for r in json.load(open(args.audit)):
        b = chess.Board(r["fen"])
        scored[audit.key(b)] = (max(-1000, min(1000, r["ours"])) + max(-1000, min(1000, r["sf"]))) / 2

    # Rebuild the audited tree: node key -> list of (uci, weight, child key); leaves have a score instead.
    children, seen = {}, set()

    def rec(board, line, p):
        k = audit.key(board)
        legal = {m.uci() for m in board.legal_moves}
        opts = [(u, w) for u, w in book.get(k, []) if u in legal]
        best = max((w for _, w in opts), default=0)
        opts = [(u, w) for u, w in opts if w >= best * 15 // 100 and w > 0]
        if not opts or len(line) >= 40 or k in seen:
            return
        seen.add(k)
        total = sum(w for _, w in opts)
        kids = []
        for u, w in opts:
            board.push(chess.Move.from_uci(u))
            ck = audit.key(board)
            if p * w / total >= args.min_prob:
                rec(board, line + [u], p * w / total)
            kids.append((u, w, ck))
            board.pop()
        children[k] = kids

    rec(chess.Board(), [], 1.0)

    memo = {}

    def value(k, white_to_move, us_white):
        """Expected score (White's view) at node k when we play White (us_white) or Black."""
        mk = (k, us_white)
        if mk in memo:
            return memo[mk]
        memo[mk] = 0  # cycle guard
        if k not in children:
            v = scored.get(k)
        else:
            vals = [(u, w, value(ck, not white_to_move, us_white)) for u, w, ck in children[k]]
            vals = [x for x in vals if x[2] is not None]
            if not vals:
                v = scored.get(k)
            elif white_to_move == us_white:  # our move: the policy
                sign = 1 if us_white else -1
                ranked = sorted(vals, key=lambda x: -sign * x[2])
                top = [ranked[0]] + [x for x in ranked[1:2] if sign * (ranked[0][2] - x[2]) <= args.margin]
                v = sum(x[2] for x in top) / len(top)
            else:  # opponent: averaged by the book's master-game frequencies
                v = sum(w * x for _, w, x in vals) / sum(w for _, w, _ in vals)
        memo[mk] = v
        return v

    fen_of = {}
    for line in open(args.book):
        parts = line.strip().split(";")
        if len(parts) >= 3:
            fen_of.setdefault(" ".join(parts[0].split()[:4]), parts[0])

    out = []
    for k, kids in children.items():
        white_to_move = k.split()[1] == "w"
        sign = 1 if white_to_move else -1
        vals = [(u, value(ck, not white_to_move, white_to_move)) for u, _, ck in kids]
        vals = [x for x in vals if x[1] is not None]
        if not vals:
            continue
        ranked = sorted(vals, key=lambda x: -sign * x[1])
        top = [ranked[0]] + [x for x in ranked[1:2] if sign * (ranked[0][1] - x[1]) <= args.margin]
        for u, v in top:
            out.append(f"{fen_of[k]};{u};100;0;{v:.0f}")
    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"{len(children)} book positions, {len(out)} moves written to {args.out}")
    start = audit.key(chess.Board())
    old = [(r["prob"], (max(-1000, min(1000, r["ours"])) + max(-1000, min(1000, r["sf"]))) / 2)
           for r in json.load(open(args.audit))]
    old_v = sum(p * v for p, v in old) / sum(p for p, _ in old)
    print(f"expected score at book exit (avg of both engines, White's view): old book {old_v:+.0f} cp; "
          f"new book as White {value(start, True, True):+.0f} cp, as Black {value(start, True, False):+.0f} cp")


if __name__ == "__main__":
    main()
