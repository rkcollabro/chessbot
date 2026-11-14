"""Bullet quantised.bin from tools/bignet/main.rs with ARCH=ml -> chessbot CBN3 network file.

Bullet's saved layout (little endian, padded to a multiple of 64 bytes):
  l0w i16 [kbuckets * 768][hidden]   factoriser merged, QA 255
  l0b i16 [hidden]
  l1w i16 [buckets * l2][hidden]     QB 256, row = bucket * l2 + neuron
  l1b f32 [buckets * l2]
  l2w f32 [buckets * l3][l2]
  l2b f32 [buckets * l3]
  l3w f32 [buckets][l3]
  l3b f32 [buckets]
CBN3 (src/nnue.cpp): "CBN3" hidden buckets qb kbuckets mirror kingBucket[64] activation(2) l2 l3, then the
same blocks in the same order. Only a header is added, so the bytes are copied unchanged.

usage: python tools/bullet_to_cbn3.py quantised.bin out.bin [--hidden 1024] [--l2 16] [--l3 32]
"""
import argparse
import struct

# Bullet's 3_input_buckets example layout, indexed rank * 4 + file for files a-d of the king's own view.
LAYOUT32 = [0, 1, 2, 3,
            4, 4, 5, 5,
            6, 6, 6, 6,
            7, 7, 7, 7,
            8, 8, 8, 8,
            8, 8, 8, 8,
            9, 9, 9, 9,
            9, 9, 9, 9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--buckets", type=int, default=8)
    ap.add_argument("--l2", type=int, default=16)
    ap.add_argument("--l3", type=int, default=32)
    ap.add_argument("--qb", type=int, default=256)
    a = ap.parse_args()
    # Same expansion as Bullet's ChessBucketsMirrored::new: files e-h mirror a-d.
    layout = [LAYOUT32[(sq // 8) * 4 + [0, 1, 2, 3, 3, 2, 1, 0][sq % 8]] for sq in range(64)]
    kb = max(layout) + 1
    h, nb, l2, l3 = a.hidden, a.buckets, a.l2, a.l3
    raw = open(a.src, "rb").read()
    blocks = [2 * kb * 768 * h, 2 * h, 2 * nb * l2 * h,4 * nb * l2, 4 * nb * l3 * l2, 4 * nb * l3, 4 * nb * l3, 4 * nb]
    need = sum(blocks)
    if len(raw) < need or len(raw) - need >= 64:
        raise SystemExit(f"{a.src}: {len(raw)} bytes, expected {need} (+ under 64 padding) for hidden {h}, "
                         f"l2 {l2}, l3 {l3}, {kb} king buckets, {nb} output buckets")
    with open(a.out, "wb") as f:
        f.write(b"CBN3")
        f.write(struct.pack("<iiiii", h, nb, a.qb, kb, 1))
        f.write(bytes(layout))
        f.write(struct.pack("<iii", 2, l2, l3))  # pairwise activation, layer sizes
        f.write(raw[:need])
    print(f"wrote {a.out}: hidden {h} pairwise -> {l2} -> {l3} -> 1, {kb} king buckets (mirrored), {nb} output buckets")


if __name__ == "__main__":
    main()
