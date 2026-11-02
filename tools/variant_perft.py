"""Cross-checks chesscli's variant move generation against python-chess.

Plays random games in each variant, and at sampled positions compares perft counts from
`chesscli perft <depth> --variant <name> <fen>` with python-chess. On a mismatch it prints the
per-move breakdown (divide) so the offending move is easy to find.

    python tools/variant_perft.py [games_per_variant] [seed] [variant,...]
"""
import random
import subprocess
import sys
from pathlib import Path

import chess
import chess.variant

ROOT = Path(__file__).resolve().parent.parent
CLI = str(ROOT / "chesscli.exe")

VARIANTS = {
    # name: (python-chess board factory, perft depth)
    "chess960": (lambda fen: chess.Board(fen, chess960=True), 3),
    "3check": (chess.variant.ThreeCheckBoard, 3),
    "crazyhouse": (chess.variant.CrazyhouseBoard, 2),
    "kingofthehill": (chess.variant.KingOfTheHillBoard, 3),
    "horde": (chess.variant.HordeBoard, 3),
    "racingkings": (chess.variant.RacingKingsBoard, 3),
    "antichess": (chess.variant.AntichessBoard, 3),
    "atomic": (chess.variant.AtomicBoard, 3),
}


def py_perft(board, depth):
    if depth == 0:
        return 1
    if board.is_variant_end():
        return 0
    n = 0
    for m in board.legal_moves:
        board.push(m)
        n += py_perft(board, depth - 1)
        board.pop()
    return n


def cli_perft(variant, fen, depth):
    out = subprocess.run([CLI, "perft", str(depth), "--variant", variant, *fen.split()],
                         capture_output=True, text=True).stdout
    try:
        return int(out.split()[0])
    except (ValueError, IndexError):
        return f"error: {out.strip()}"


def divide(variant, factory, fen, depth):
    board = factory(fen)
    print(f"    divide at depth {depth}:")
    for m in sorted(board.legal_moves, key=lambda m: m.uci()):
        board.push(m)
        expect = py_perft(board, depth - 1)
        got = cli_perft(variant, board.fen(), depth - 1)
        board.pop()
        if expect != got:
            print(f"      {m.uci():8} python {expect:>8}  chesscli {got}")


def start_fen(name, rng):
    if name == "chess960":
        return chess.Board.from_chess960_pos(rng.randrange(960)).fen()
    return VARIANTS[name][0]().fen()


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else set(VARIANTS)
    failures = 0
    for name, (factory, depth) in VARIANTS.items():
        if name not in only:
            continue
        checked = 0
        for _ in range(games):
            board = factory(start_fen(name, rng))
            for ply in range(80):
                if ply % 7 == 0 or board.is_check():
                    fen = board.fen()
                    expect = py_perft(factory(fen), depth)
                    got = cli_perft(name, fen, depth)
                    checked += 1
                    if expect != got:
                        failures += 1
                        print(f"FAIL [{name}] perft({depth}) python {expect} chesscli {got}\n    {fen}")
                        divide(name, factory, fen, depth)
                moves = list(board.legal_moves)
                if not moves or board.is_variant_end():
                    break
                board.push(rng.choice(moves))
        print(f"{name:14} {checked} positions checked")
    print("ALL MATCH" if not failures else f"{failures} MISMATCHES")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
