"""Trains the NNUE evaluation (src/nnue.cpp) on self-play data from `chesscli selfplay`.

Network: 768 inputs per side (own/enemy x piece type x square, mirrored for Black) -> 256 shared hidden units,
clipped ReLU, the side to move's and the opponent's halves concatenated -> 1 output, 1.0 = 400 centipawns.
With --buckets N the output layer has N sets of weights, picked by the number of pieces on the board
(bucket = min(N - 1, (pieces - 2) * N // 32)), so openings, middlegames and endgames get their own output.
The target mixes the engine's search score and the game result:
    target = LAMBDA * sigmoid(score / 400) + (1 - LAMBDA) * result
Endgame positions (at most --endgame-pieces pieces) can use their own --lambda-endgame. Self-play used to shuffle
won endgames into 50-move draws, so their result labels are less trustworthy than their search scores.

The data is streamed from disk: each epoch reads the files in shuffled 1M-position chunks, mixes several chunks
in memory and shuffles them, so the data set can be far larger than RAM. Features are built on the training
device (CUDA when available).

    python tools/nnue_train.py <out.bin> <data.bin>... [--epochs 10] [--lambda 0.7] [--lambda-endgame 0.9]
                               [--init old.bin.pt] [--hidden 256] [--buckets 8] [--device cuda]
"""
import argparse
import math
import struct
import time

import numpy as np
import torch

HIDDEN = 256
BUCKETS = 1
QB_OUT = 64  # output-layer quantization written to the file (QB is the original format's fixed 64)
INPUTS = 768
PAD = INPUTS  # feature index for empty slots
QA, QB, SCALE = 255, 64, 400
RECORD = np.dtype([("sq", "u1", 32), ("pc", "u1", 32), ("score", "<i2"), ("result", "u1"), ("stm", "u1")])
CHUNK = 1 << 20      # positions read from one file at a time
MIX = 8              # chunks shuffled together
VAL_TOTAL = 100_000  # validation positions, taken from the tail of every file


class Batchify:
    """Turns raw records (numpy) into feature indices and targets on the training device."""

    def __init__(self, device, lam, lam_eg, eg_pieces):
        self.device, self.lam, self.lam_eg, self.eg_pieces = device, lam, lam_eg, eg_pieces

    def __call__(self, rec):
        rec = rec[np.abs(rec["score"]) < 2000]
        d = self.device
        sq = torch.from_numpy(np.ascontiguousarray(rec["sq"])).to(d, non_blocking=True).long()
        pc = torch.from_numpy(np.ascontiguousarray(rec["pc"])).to(d, non_blocking=True).long()
        stm = torch.from_numpy(np.ascontiguousarray(rec["stm"])).to(d, non_blocking=True).long()
        score = torch.from_numpy(np.ascontiguousarray(rec["score"])).to(d, non_blocking=True).float()
        result = torch.from_numpy(np.ascontiguousarray(rec["result"])).to(d, non_blocking=True).float() / 2

        empty = pc == 0xFF
        color, ptype = pc >> 3, pc & 7

        def feats(persp):
            p = persp[:, None]
            square = torch.where(p == 0, sq, sq ^ 56)
            idx = (color != p).long() * 384 + (ptype - 1) * 64 + square
            return idx.masked_fill(empty, PAD)

        us, them = feats(stm), feats(1 - stm)
        pieces = (~empty).sum(dim=1)
        bucket = torch.clamp((pieces - 2) * BUCKETS // 32, 0, BUCKETS - 1)
        lam = torch.full_like(score, self.lam)
        if self.lam_eg is not None:
            lam = torch.where(pieces <= self.eg_pieces, torch.full_like(score, self.lam_eg), lam)
        target = lam * torch.sigmoid(score / SCALE) + (1 - lam) * result
        return us, them, bucket, target


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ft = torch.nn.EmbeddingBag(INPUTS + 1, HIDDEN, mode="sum", padding_idx=PAD)
        self.ft_bias = torch.nn.Parameter(torch.zeros(HIDDEN))
        self.out = torch.nn.Linear(2 * HIDDEN, BUCKETS)
        torch.nn.init.normal_(self.ft.weight, std=0.05)
        with torch.no_grad():
            self.ft.weight[PAD].zero_()

    def forward(self, us, them, bucket):
        a = torch.clamp(self.ft(us) + self.ft_bias, 0, 1)
        b = torch.clamp(self.ft(them) + self.ft_bias, 0, 1)
        out = self.out(torch.cat([a, b], dim=1))  # in units of SCALE centipawns, one column per bucket
        return out.gather(1, bucket[:, None]).squeeze(1)


def export(net, path):
    w = net.ft.weight.detach().cpu().numpy()[:INPUTS]
    ft_w = np.clip(np.round(w * QA), -32768, 32767).astype("<i2")
    ft_b = np.clip(np.round(net.ft_bias.detach().cpu().numpy() * QA), -32768, 32767).astype("<i2")
    out_w = np.clip(np.round(net.out.weight.detach().cpu().numpy() * QB_OUT), -32768, 32767).astype("<i2")  # [B][2H]
    out_b = np.round(net.out.bias.detach().cpu().numpy().astype(np.float64) * QA * QB_OUT).astype("<i4")
    with open(path, "wb") as f:
        if BUCKETS == 1 and QB_OUT == QB:  # original format, readable by older engines
            f.write(b"CBNN" + struct.pack("<i", HIDDEN))
        else:
            f.write(b"CBNB" + struct.pack("<iii", HIDDEN, BUCKETS, QB_OUT))
        f.write(ft_w.tobytes() + ft_b.tobytes() + out_w.tobytes() + out_b.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("data", nargs="+")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.7)
    ap.add_argument("--lambda-endgame", dest="lam_eg", type=float, default=None,
                    help="lambda for positions with at most --endgame-pieces pieces (default: same as --lambda)")
    ap.add_argument("--endgame-pieces", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--buckets", type=int, default=1, help="output buckets by piece count (1 = single output)")
    ap.add_argument("--qb", type=int, default=QB, help="output-layer quantization in the exported file (64 = original)")
    ap.add_argument("--init", help="start from this .pt state dict (same --hidden) instead of random weights; "
                                   "a single-output network's output is copied into every bucket")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    global HIDDEN, BUCKETS
    global QB_OUT
    HIDDEN, BUCKETS, QB_OUT = args.hidden, args.buckets, args.qb
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    # Memory-map every file; the last few positions of each are held out for validation.
    files = [np.memmap(p, dtype=RECORD, mode="r", shape=(np.memmap(p, dtype="u1", mode="r").size // RECORD.itemsize,))
             for p in args.data]
    total = sum(len(f) for f in files)
    per_file_val = math.ceil(VAL_TOTAL / len(files))
    train_end = [max(0, len(f) - min(per_file_val, len(f) // 20)) for f in files]
    chunks = [(i, s, min(s + CHUNK, end)) for i, end in enumerate(train_end) for s in range(0, end, CHUNK)]
    batchify = Batchify(device, args.lam, args.lam_eg, args.endgame_pieces)
    val = batchify(np.concatenate([np.asarray(f[e:]) for f, e in zip(files, train_end)]))
    print(f"{total} positions in {len(files)} files, {len(val[3])} for validation, device {device}", flush=True)

    net = Net().to(device)
    if args.init:
        state = torch.load(args.init, map_location=device)
        for k in ("out.weight", "out.bias"):
            if state[k].shape[0] == 1 and BUCKETS > 1:
                state[k] = state[k].repeat(BUCKETS, *([1] * (state[k].dim() - 1)))
        net.load_state_dict(state)
        print(f"initialised from {args.init}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    for epoch in range(args.epochs):
        t0 = time.time()
        order = rng.permutation(len(chunks))
        loss_sum, seen = 0.0, 0
        for g in range(0, len(order), MIX):
            rec = np.concatenate([np.asarray(files[i][s:e]) for i, s, e in (chunks[k] for k in order[g:g + MIX])])
            rec = rec[rng.permutation(len(rec))]
            for b in range(0, len(rec), args.batch):
                us, them, bucket, target = batchify(rec[b:b + args.batch])
                if not len(target):
                    continue
                loss = torch.mean((torch.sigmoid(net(us, them, bucket)) - target) ** 2)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    net.ft.weight[PAD].zero_()
                loss_sum += loss.item() * len(target)
                seen += len(target)
        sched.step()
        with torch.no_grad():
            n = len(val[3])
            vl = [torch.sum((torch.sigmoid(net(*(v[i:i + 65536] for v in val[:3]))) - val[3][i:i + 65536]) ** 2)
                  for i in range(0, n, 65536)]
            vloss = (sum(vl) / n).item()
        print(f"epoch {epoch + 1}/{args.epochs}  train {loss_sum / max(seen, 1):.5f}  val {vloss:.5f}  "
              f"{time.time() - t0:.0f}s", flush=True)
        export(net, args.out)  # keep the latest network on disk
        torch.save(net.state_dict(), args.out + ".pt")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
