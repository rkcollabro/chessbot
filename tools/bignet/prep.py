# Converts chessbot self-play records (68 bytes: squares[32], pieces[32] 0xFF-padded, score i16 and result u8
# from the side to move's view, stm u8) into Bullet's 32-byte ChessBoard records, shuffled.
# ChessBoard (bulletformat src/chess.rs) is side-to-move relative: when Black is to move the board is flipped
# vertically (sq ^ 56) and the colours swapped. occ u64; pcs[16] 4-bit pieces in increasing square order,
# type 0-5 = P N B R Q K, bit 3 set for the opponent's pieces; score i16; result u8 (0 loss, 1 draw, 2 win);
# ksq (own king); opp_ksq (opponent king, ^ 56); 3 zero bytes.
#
# usage: python prep.py <out.bin> <max positions (0 = all)> <in.bin.zst or in.bin>...
import sys, os, io, glob
import numpy as np
import zstandard

REC = np.dtype([("sq", "u1", 32), ("pc", "u1", 32), ("score", "<i2"), ("result", "u1"), ("stm", "u1")])
assert REC.itemsize == 68
NBUCKET = 64


def records(paths):
    for p in paths:
        f = open(p, "rb")
        r = zstandard.ZstdDecompressor().stream_reader(f) if p.endswith(".zst") else f
        rest = b""
        while True:
            buf = r.read(68 * 4_000_000)
            if not buf:
                break
            buf = rest + buf
            n = len(buf) // 68
            rest = buf[n * 68:]
            yield np.frombuffer(buf[: n * 68], dtype=REC)
        f.close()


def convert(a):
    n = len(a)
    sq = a["sq"].astype(np.int16)
    pc = a["pc"].astype(np.int16)
    valid = sq != 255
    black = (a["stm"] == 1)[:, None]
    sq = np.where(valid, np.where(black, sq ^ 56, sq), 64)
    color = (pc >> 3) & 1
    color = np.where(black, 1 - color, color)  # 1 = the side not to move
    ptype = (pc & 7) - 1
    nib = np.where(valid, (color << 3) | ptype, 0).astype(np.uint8)
    order = np.argsort(sq, axis=1, kind="stable")
    sq_s = np.take_along_axis(sq, order, axis=1)
    nib = np.take_along_axis(nib, order, axis=1)
    out = np.zeros((n, 32), np.uint8)
    bits = np.where(sq_s < 64, np.left_shift(np.uint64(1), np.minimum(sq_s, 63).astype(np.uint64)), np.uint64(0))
    occ = np.bitwise_or.reduce(bits, axis=1).astype("<u8")
    out[:, 0:8] = occ.view(np.uint8).reshape(n, 8)
    out[:, 8:24] = nib[:, 0::2] | (nib[:, 1::2] << 4)
    out[:, 24:26] = a["score"].astype("<i2").view(np.uint8).reshape(n, 2)
    out[:, 26] = a["result"]
    is_k = valid & (ptype == 5)
    own_k = np.where(is_k & (color == 0), sq, 0).max(axis=1)
    opp_k = np.where(is_k & (color == 1), sq, 0).max(axis=1)
    out[:, 27] = own_k
    out[:, 28] = opp_k ^ 56
    keep = (is_k & (color == 0)).any(axis=1) & (is_k & (color == 1)).any(axis=1) & (a["result"] <= 2)
    return out[keep]


def main():
    out, limit, paths = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    rng = np.random.default_rng(12345)
    tmp = out + ".parts"
    os.makedirs(tmp, exist_ok=True)
    parts = [open(f"{tmp}/{i}.bin", "wb") for i in range(NBUCKET)]
    total = 0
    for a in records(paths):
        b = convert(a)
        if limit and total + len(b) > limit:
            b = b[: limit - total]
        ids = rng.integers(0, NBUCKET, len(b))
        for i in range(NBUCKET):
            parts[i].write(b[ids == i].tobytes())
        total += len(b)
        if limit and total >= limit:
            break
    for p in parts:
        p.close()
    with open(out, "wb") as o:
        for i in range(NBUCKET):
            x = np.fromfile(f"{tmp}/{i}.bin", dtype=np.uint8).reshape(-1, 32)
            rng.shuffle(x, axis=0)
            o.write(x.tobytes())
            os.remove(f"{tmp}/{i}.bin")
    os.rmdir(tmp)
    print(f"prep: {total} positions -> {out}", flush=True)


if __name__ == "__main__":
    main()
