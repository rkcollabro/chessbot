# Bullet quantised.bin (main.rs save format) -> Bullet weights file, to start a new run from a trained net.
#   quantised: l0w i16 [buckets*768][HL] (QA 255, factoriser already merged), l0b i16 [HL] (QA), l1w i16 [8][2*HL]
#              (QB 64, saved transposed), l1b i16 [8] (QA*QB), zero-padded to 64 bytes.
#   weights file: per weight "id\n", u64 element count, f32 values in Bullet's column-major order.
# usage: python3 dequant.py quantised.bin out_weights.bin [hl 1024] [buckets 10]
import struct, sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
hl = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
nb = int(sys.argv[4]) if len(sys.argv) > 4 else 10
q = np.fromfile(src, dtype="<i2")
n0, n1 = nb * 768 * hl, 8 * 2 * hl
l0w, q = q[:n0], q[n0:]
l0b, q = q[:hl], q[hl:]
l1w, q = q[:n1], q[n1:]
l1b = q[:8]
assert len(l1b) == 8
out = {
    "l0w": l0w.astype(np.float32) / 255,                                   # raw order == saved order
    "l0b": l0b.astype(np.float32) / 255,
    "l1w": (l1w.reshape(8, 2 * hl).T.reshape(-1)).astype(np.float32) / 64,  # raw[col*8 + row] = saved[row][col]
    "l1b": l1b.astype(np.float32) / (255 * 64),
}
with open(dst, "wb") as f:
    for k, v in out.items():
        f.write(k.encode() + b"\n" + struct.pack("<Q", v.size) + v.astype("<f4").tobytes())
print(f"wrote {dst}: " + ", ".join(f"{k} {v.size}" for k, v in out.items()))
