# Bullet quantised.bin (main.rs save format) -> Bullet weights file, to start a new run from a trained net.
#   quantised: l0w i16 [buckets*768][HL] (QA 255, factoriser already merged), l0b i16 [HL] (QA), l1w i16 [8][2*HL]
#              (QB 64, saved transposed), l1b i16 [8] (QA*QB), zero-padded to 64 bytes.
#   ml (ARCH=ml): l0w, l0b as above, l1w i16 [8*16][HL] (QB 256, saved transposed), then f32: l1b [8*16],
#              l2w [8*32][16] (saved transposed), l2b [8*32], l3w [8][32] (saved transposed), l3b [8].
#   weights file: per weight "id\n", u64 element count, f32 values in Bullet's column-major order.
#   A saved (transposed) matrix [out][in] is raw[in_col * out + out_row], i.e. saved.reshape(out, in).T.
# usage: python3 dequant.py quantised.bin out_weights.bin [hl 1024] [buckets 10] [arch: cbn2|ml]
import struct, sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
hl = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
nb = int(sys.argv[4]) if len(sys.argv) > 4 else 10
arch = sys.argv[5] if len(sys.argv) > 5 else "cbn2"
raw = open(src, "rb").read()
n0 = nb * 768 * hl


def i16(off, n):
    return np.frombuffer(raw, dtype="<i2", count=n, offset=off).astype(np.float32), off + 2 * n


def f32(off, n):
    return np.frombuffer(raw, dtype="<f4", count=n, offset=off).astype(np.float32), off + 4 * n


def untranspose(saved, rows, cols):
    return saved.reshape(rows, cols).T.reshape(-1)


off = 0
l0w, off = i16(off, n0)
l0b, off = i16(off, hl)
out = {"l0w": l0w / 255, "l0b": l0b / 255}  # raw order == saved order
if arch == "ml":
    L2, L3, NOB = 16, 32, 8
    l1w, off = i16(off, NOB * L2 * hl)
    l1b, off = f32(off, NOB * L2)
    l2w, off = f32(off, NOB * L3 * L2)
    l2b, off = f32(off, NOB * L3)
    l3w, off = f32(off, NOB * L3)
    l3b, off = f32(off, NOB)
    out.update({
        "l1w": untranspose(l1w, NOB * L2, hl) / 256,
        "l1b": l1b,
        "l2w": untranspose(l2w, NOB * L3, L2),
        "l2b": l2b,
        "l3w": untranspose(l3w, NOB, L3),
        "l3b": l3b,
    })
else:
    l1w, off = i16(off, 8 * 2 * hl)
    l1b, off = i16(off, 8)
    out.update({
        "l1w": untranspose(l1w, 8, 2 * hl) / 64,  # raw[col*8 + row] = saved[row][col]
        "l1b": l1b / (255 * 64),
    })
assert off <= len(raw) < off + 64, f"size mismatch: parsed {off} bytes of {len(raw)} ({arch})"
with open(dst, "wb") as f:
    for k, v in out.items():
        f.write(k.encode() + b"\n" + struct.pack("<Q", v.size) + v.astype("<f4").tobytes())
print(f"wrote {dst}: " + ", ".join(f"{k} {v.size}" for k, v in out.items()))
