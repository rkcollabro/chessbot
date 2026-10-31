"""Measures the engine levels' strength relative to Level 6 by self-play.

Neighbouring levels are too far apart (often 95%+ scores) to measure directly, so each gap is split by an
intermediate setting and measured as two closer matches. The top end plays Level 6 itself (3 s, all threads);
MAX (30 s) is estimated from a 10x time-odds match one step down (0.3 s vs 3 s), since playing at 30 s a move
would take many hours.

    python tools/rate_levels.py [engine] [games]
"""
import re
import subprocess
import sys

ENGINE = sys.argv[1] if len(sys.argv) > 1 else "engines/levels.exe"
GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 60
THREADS = 8

# (nodes, noise) for levels 1-5, matching LEVELS in src/search.cpp; intermediates split each gap.
LADDER = [
    ("Level 1", 1000, 150), ("mid", 2000, 100),
    ("Level 2", 4000, 60), ("mid", 9000, 40),
    ("Level 3", 20000, 25), ("mid", 50000, 10),
    ("Level 4", 120000, 0), ("mid", 300000, 0),
    ("Level 5", 800000, 0), ("mid", 3000000, 0),
]


def match(opts_a, opts_b, games, movetime, concurrency, extra=()):
    cmd = [sys.executable, "tools/match.py", f"{ENGINE} uci", f"{ENGINE} uci", "--games", str(games),
           "--movetime", str(movetime), "--concurrency", str(concurrency), "--adjudicate", *extra]
    for k, v in opts_a.items():
        cmd += ["--option-a", f"{k}={v}"]
    for k, v in opts_b.items():
        cmd += ["--option-b", f"{k}={v}"]
    last = subprocess.run(cmd, capture_output=True, text=True).stdout.strip().splitlines()[-1]
    print("   ", last, flush=True)
    return float(re.search(r"elo ([+-]\d+)", last).group(1))


def limited(nodes, noise):
    return {"Level": 1, "LevelNodes": nodes, "LevelNoise": noise}


def main():
    # gaps[i] = Elo of LADDER[i+1] over LADDER[i]; the last entry is Level 6 over the top intermediate.
    gaps = []
    for (name_a, na, za), (name_b, nb, zb) in zip(LADDER, LADDER[1:]):
        print(f"{name_a} {na}/{za} vs {name_b} {nb}/{zb}", flush=True)
        gaps.append(-match(limited(na, za), limited(nb, zb), GAMES, 10, 8))
    print("top intermediate vs Level 6 (3 s, all threads)", flush=True)
    gaps.append(-match(limited(*LADDER[-1][1:]), {"Level": 6}, max(20, GAMES // 3), 3, 1,
                       ("--threads-b", str(THREADS))))
    print("Level 6 at 0.3 s vs 3 s (MAX estimate: the same 10x time ratio)", flush=True)
    max_gain = -match({"Level": 6}, {"Level": 6}, max(20, GAMES // 3), 0, 1,
                      ("--threads-a", str(THREADS), "--threads-b", str(THREADS), "--movetime-a", "0.3",
                       "--movetime-b", "3"))

    # Elo of each level relative to Level 6: walk down from the top.
    rel = 0.0
    ratings = {"Level 6": 0}
    for i in range(len(LADDER) - 1, -1, -1):
        rel -= gaps[i]
        if LADDER[i][0] != "mid":
            ratings[LADDER[i][0]] = round(rel / 10) * 10
    ratings["MAX"] = round(max_gain / 10) * 10
    print("\nRelative to Level 6:")
    for name in ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6", "MAX"]:
        print(f"  {name:8} {ratings[name]:+5d}")


if __name__ == "__main__":
    main()
