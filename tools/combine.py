"""Adds up the shard results of a Match or SPRT workflow run (tools/sprt.py --json files) and prints a Markdown
summary.

Without --sprt the verdict is the 60-game acceptance rule scaled to the number of games played: more than half
wins, and losses at most a sixth of the games. With --sprt ELO0 ELO1 it is the same pentanomial SPRT that
tools/sprt.py runs locally (alpha = beta = 0.05): PASSED, FAILED, or INCONCLUSIVE when more games are needed.

    python tools/combine.py <result.json>... [--title "nnue3 vs nnue2"] [--sprt 0 10] [--out total.json]
"""
import argparse
import json
import math

ALPHA = BETA = 0.05


def to_elo(score):
    score = min(max(score, 1e-4), 1 - 1e-4)
    return -400 * math.log10(1 / score - 1)


def logistic(elo):
    return 1 / (1 + 10 ** (-elo / 400))


def llr(pairs, elo0, elo1):
    """Pentanomial GSPRT log-likelihood ratio from per-pair scores (same formula as tools/sprt.py)."""
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
    ap.add_argument("files", nargs="+")
    ap.add_argument("--title", default="Match")
    ap.add_argument("--sprt", nargs=2, type=float, metavar=("ELO0", "ELO1"))
    ap.add_argument("--out", help="write the combined W/L/D and pair scores here")
    args = ap.parse_args()
    wdl, pairs = [0, 0, 0], []
    for path in args.files:
        with open(path) as f:
            r = json.load(f)
        wdl = [a + b for a, b in zip(wdl, r["wdl"])]
        pairs += r["pairs"]
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"wdl": wdl, "pairs": pairs}, f)
    games = sum(wdl)
    n = len(pairs)
    mean = sum(pairs) / n
    sd = math.sqrt(sum((p - mean) ** 2 for p in pairs) / n / n) if n > 1 else 0
    elo = f"{to_elo(mean):+.0f} ({to_elo(mean - 1.96 * sd):+.0f} to {to_elo(mean + 1.96 * sd):+.0f})"
    print(f"## {args.title}\n")
    if args.sprt:
        elo0, elo1 = args.sprt
        ratio = llr(pairs, elo0, elo1)
        lower, upper = math.log(BETA / (1 - ALPHA)), math.log((1 - BETA) / ALPHA)
        verdict = "PASSED" if ratio >= upper else "FAILED" if ratio <= lower else "INCONCLUSIVE"
        print(f"| Games | Wins | Losses | Draws | Elo (95% range) | LLR (bounds) | SPRT [{elo0:g}, {elo1:g}] |")
        print("|---|---|---|---|---|---|---|")
        print(f"| {games} | {wdl[0]} | {wdl[1]} | {wdl[2]} | {elo} | {ratio:+.2f} ({lower:.2f}, {upper:.2f}) | {verdict} |")
        print(f"\nRESULT {verdict}  elo {to_elo(mean):+.1f}  games {games}  W {wdl[0]} L {wdl[1]} D {wdl[2]}  "
              f"LLR {ratio:+.2f}")
    else:
        passed = wdl[0] > games / 2 and wdl[1] <= games / 6
        print("| Games | Wins | Losses | Draws | Elo (95% range) | Rule |")
        print("|---|---|---|---|---|---|")
        print(f"| {games} | {wdl[0]} | {wdl[1]} | {wdl[2]} | {elo} | {'PASSED' if passed else 'FAILED'} |")
        print(f"\nRESULT {'PASSED' if passed else 'FAILED'}  elo {to_elo(mean):+.1f}  games {games}  "
              f"W {wdl[0]} L {wdl[1]} D {wdl[2]}  shards {len(args.files)}")


if __name__ == "__main__":
    main()
