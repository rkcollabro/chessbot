"""Builds a test-opening suite from the first moves of Lichess elite games (both players 2500+), from games of
3 minutes or longer, de-duplicated and shuffled. Strong players' first moves are sound and roughly balanced.

    python tools/make_openings.py data/elite_2023-01.zip tools/openings.txt [count] [plies]
"""
import io
import random
import re
import sys
import zipfile

import chess


def main():
    src, out = sys.argv[1], sys.argv[2]
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    plies = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    z = zipfile.ZipFile(src)
    pgn = io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8", errors="replace")
    lines = set()
    base = 0
    for row in pgn:
        if row.startswith("[TimeControl"):
            m = re.match(r'\[TimeControl "(\d+)', row)
            base = int(m.group(1)) if m else 0
        elif row.startswith("1.") and base >= 180:
            sans = [t for t in re.sub(r"\{[^}]*\}", "", row).split() if not re.match(r"^\d+\.", t)][:plies]
            if len(sans) < plies:
                continue
            b = chess.Board()
            try:
                lines.add(" ".join(b.push_san(s).uci() for s in sans))
            except ValueError:
                continue
            if len(lines) >= count * 2:
                break
    lines = sorted(lines)
    random.Random(1).shuffle(lines)
    with open(out, "w") as f:
        f.write("\n".join(lines[:count]) + "\n")
    print(f"{min(count, len(lines))} openings of {plies} plies")


if __name__ == "__main__":
    main()
