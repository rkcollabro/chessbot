"""Summarises tools/book_audit.py output: how well our engine's and Stockfish's scores of the book exits agree,
and which first moves / lines leave each side worse off.   python tools/book_audit_report.py book_audit.json"""
import collections
import json
import statistics
import sys

rows = json.load(open(sys.argv[1]))
clip = lambda v: max(-1000, min(1000, v))
o = [clip(r["ours"]) for r in rows]
s = [clip(r["sf"]) for r in rows]
w = [r["prob"] for r in rows]


def wmean(xs, ws):
    return sum(x * y for x, y in zip(xs, ws)) / sum(ws)


def wcorr(a, b, ws):
    ma, mb = wmean(a, ws), wmean(b, ws)
    cov = wmean([(x - ma) * (y - mb) for x, y in zip(a, b)], ws)
    va = wmean([(x - ma) ** 2 for x in a], ws)
    vb = wmean([(y - mb) ** 2 for y in b], ws)
    return cov / (va * vb) ** 0.5


n = len(rows)
print(f"{n} book exits (weighted by how often each is reached)")
print(f"correlation ours vs Stockfish: {wcorr(o, s, w):.2f} (weighted), {statistics.correlation(o, s):.2f} (unweighted)")
print(f"mean score, White's view: ours {wmean(o, w):+.0f} cp, Stockfish {wmean(s, w):+.0f} cp")
# Scale: least-squares slope of ours on Stockfish (our engine may simply run hotter or colder).
ms, mo = wmean(s, w), wmean(o, w)
slope = wmean([(x - ms) * (y - mo) for x, y in zip(s, o)], w) / wmean([(x - ms) ** 2 for x in s], w)
print(f"slope ours ~ {slope:.2f} x Stockfish")
sign = sum(r["prob"] for r in rows if (r["ours"] > 25 and r["sf"] < -25) or (r["ours"] < -25 and r["sf"] > 25))
print(f"share of play where they disagree on who is better (both beyond 25 cp, opposite signs): {sign:.1%}")
big = sorted(rows, key=lambda r: -abs(clip(r["ours"]) - clip(r["sf"])))[:12]
print("\nlargest disagreements:")
for r in big:
    print(f"  ours {r['ours']:+5d}  sf {r['sf']:+5d}  p {r['prob']:.4f}  {r['line']}")

# By our colour: lines where the book leaves that side worse (Stockfish, from that side's view).
for side, pk in (("White", "prob_white_choices"), ("Black", "prob_black_choices")):
    sg = 1 if side == "White" else -1
    print(f"\nAs {side}: score at book exit, averaged over the opponent's replies (Stockfish / ours):")
    by_first = collections.defaultdict(list)
    for r in rows:
        mv = r["line"].split()
        # Our first choice: White's move 1, or Black's reply to White's move 1.
        k = mv[0] if side == "White" else " ".join(mv[:2])
        if k:
            by_first[k].append(r)
    out = []
    for k, rs in by_first.items():
        p = sum(r["prob"] for r in rs)
        if p < 0.01:
            continue
        out.append((wmean([sg * clip(r["sf"]) for r in rs], [r["prob"] for r in rs]),
                    wmean([sg * clip(r["ours"]) for r in rs], [r["prob"] for r in rs]), p, k))
    for sfv, ov, p, k in sorted(out):
        print(f"  {k:12s} share {p:5.1%}  sf {sfv:+5.0f}  ours {ov:+5.0f}")
    bad = sorted((r for r in rows if sg * r["sf"] < -40), key=lambda r: -r["prob"])[:15]
    print(f"  most common exits where {side} is worse than -40 cp (Stockfish):")
    for r in bad:
        print(f"    sf {sg * r['sf']:+5d}  ours {sg * r['ours']:+5d}  p {r['prob']:.4f}  {r['line']}")
