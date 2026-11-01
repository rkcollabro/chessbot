"""Adds up the shard results of a Match workflow run (tools/sprt.py --json files) and prints a Markdown summary.

The acceptance rule is the 60-game one scaled to the number of games played: more than half wins, and losses
at most a sixth of the games.

    python tools/combine.py <result.json>... [--title "nnue3 vs nnue2"]
"""
import argparse
import json
import math


def to_elo(score):
    score = min(max(score, 1e-4), 1 - 1e-4)
    return -400 * math.log10(1 / score - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--title", default="Match")
    args = ap.parse_args()
    wdl, pairs = [0, 0, 0], []
    for path in args.files:
        with open(path) as f:
            r = json.load(f)
        wdl = [a + b for a, b in zip(wdl, r["wdl"])]
        pairs += r["pairs"]
    games = sum(wdl)
    n = len(pairs)
    mean = sum(pairs) / n
    sd = math.sqrt(sum((p - mean) ** 2 for p in pairs) / n / n) if n > 1 else 0
    passed = wdl[0] > games / 2 and wdl[1] <= games / 6
    print(f"## {args.title}\n")
    print("| Games | Wins | Losses | Draws | Elo (95% range) | Rule |")
    print("|---|---|---|---|---|---|")
    print(f"| {games} | {wdl[0]} | {wdl[1]} | {wdl[2]} | {to_elo(mean):+.0f} "
          f"({to_elo(mean - 1.96 * sd):+.0f} to {to_elo(mean + 1.96 * sd):+.0f}) | {'PASSED' if passed else 'FAILED'} |")
    print(f"\nRESULT {'PASSED' if passed else 'FAILED'}  elo {to_elo(mean):+.1f}  games {games}  "
          f"W {wdl[0]} L {wdl[1]} D {wdl[2]}  shards {len(args.files)}")


if __name__ == "__main__":
    main()
