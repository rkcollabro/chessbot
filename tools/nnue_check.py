"""Checks the engine's quantized NNUE against the trainer's float network on positions from a data file.

    python3 tools/nnue_check.py <engine> <net.pt> <data.bin> [count]
"""
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from nnue_train import RECORD, SCALE, Net, features  # noqa: E402

PIECES = "?PNBRQK"


def fen(rec):
    board = [""] * 64
    for s, p in zip(rec["sq"], rec["pc"]):
        if p != 0xFF:
            ch = PIECES[p & 7]
            board[s] = ch if p >> 3 == 0 else ch.lower()
    rows = []
    for r in range(7, -1, -1):
        row, empty = "", 0
        for f in range(8):
            ch = board[r * 8 + f]
            if not ch:
                empty += 1
                continue
            row += (str(empty) if empty else "") + ch
            empty = 0
        rows.append(row + (str(empty) if empty else ""))
    return "/".join(rows) + (" w" if rec["stm"] == 0 else " b") + " - - 0 1"


def main():
    engine, netfile, datafile = sys.argv[1:4]
    count = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    data = np.fromfile(datafile, dtype=RECORD)[:count]
    net = Net()
    net.load_state_dict(torch.load(netfile))
    stm = data["stm"].astype(np.int64)
    with torch.no_grad():
        floats = net(torch.from_numpy(features(data, stm)), torch.from_numpy(features(data, 1 - stm))) * SCALE
    worst = 0
    for rec, f in zip(data, floats.tolist()):
        e = int(subprocess.run([engine, "eval", fen(rec)], capture_output=True, text=True).stdout.strip())
        worst = max(worst, abs(e - f))
        print(f"{fen(rec):70s} float {f:8.1f}  engine {e:6d}  search {rec['score']:6d}")
    print(f"max |engine - float| = {worst:.1f} cp")


if __name__ == "__main__":
    main()
