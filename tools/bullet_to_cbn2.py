"""Bullet quantised.bin -> chessbot CBN2 network file.

Bullet's saved layout (docs/4-saved-networks.md, little endian, padded with zeros to a multiple of 64 bytes):
  l0w i16 [kbuckets * 768][hidden]   feature-major, factoriser already merged
  l0b i16 [hidden]
  l1w i16 [8][2 * hidden]            transposed: per output bucket, side-to-move half first
  l1b i16 [8]                        scaled by QA * QB
CBN2 (src/nnue.cpp): "CBN2" hidden buckets qb kbuckets mirror kingBucket[64] activation, then the same four
blocks, with the output biases widened to int32.

usage: python tools/bullet_to_cbn2.py quantised.bin out.bin [--hidden 1024] [--kbuckets 10] [--layout example]
"""
import argparse
import struct

import numpy as np

# Bullet's 3_input_buckets example layout, indexed rank * 4 + file for files a-d of the king's own view.
LAYOUTS = {
    "example": [0, 1, 2, 3,
                4, 4, 5, 5,
                6, 6, 6, 6,
                7, 7, 7, 7,
                8, 8, 8, 8,
                8, 8, 8, 8,
                9, 9, 9, 9,
                9, 9, 9, 9],
}


def expand(layout32):
    # Same expansion as Bullet's ChessBucketsMirrored::new: files e-h mirror a-d.
    return [layout32[(sq // 8) * 4 + [0, 1, 2, 3, 3, 2, 1, 0][sq % 8]] for sq in range(64)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--buckets", type=int, default=8)
    ap.add_argument("--layout", default="example")
    ap.add_argument("--qb", type=int, default=64)
    a = ap.parse_args()
    layout = expand(LAYOUTS[a.layout])
    kb = max(layout) + 1
    h, nb = a.hidden, a.buckets
    raw = np.fromfile(a.src, dtype="<i2")
    sizes = [kb * 768 * h, h, nb * 2 * h, nb]
    need = sum(sizes)
    if len(raw) < need or len(raw) - need >= 32:  # at most 63 bytes of padding
        raise SystemExit(f"{a.src}: {len(raw)} int16 values, expected {need} (+padding) for hidden {h}, {kb} king buckets")
    l0w, l0b, l1w, l1b = np.split(raw[:need], np.cumsum(sizes)[:-1])
    if np.any(np.abs(l1w.astype(np.int32)) > 127):
        print("warning: output weights beyond +-127; the engine's 16-bit SCReLU product could overflow")
    with open(a.out, "wb") as f:
        f.write(b"CBN2")
        f.write(struct.pack("<iiiii", h, nb, a.qb, kb, 1))
        f.write(bytes(layout))
        f.write(struct.pack("<i", 1))  # SCReLU
        f.write(l0w.astype("<i2").tobytes())
        f.write(l0b.astype("<i2").tobytes())
        f.write(l1w.astype("<i2").tobytes())
        f.write(l1b.astype("<i4").tobytes())
    print(f"wrote {a.out}: hidden {h}, {kb} king buckets (mirrored), {nb} output buckets, SCReLU")


if __name__ == "__main__":
    main()
