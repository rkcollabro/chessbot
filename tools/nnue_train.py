"""Trains the NNUE evaluation (src/nnue.cpp) on self-play data from `chesscli selfplay`.

Network: 768 inputs per side (own/enemy x piece type x square, mirrored for Black) -> 256 shared hidden units,
clipped ReLU, the side to move's and the opponent's halves concatenated -> 1 output, 1.0 = 400 centipawns.
The target mixes the engine's search score and the game result:
    target = LAMBDA * sigmoid(score / 400) + (1 - LAMBDA) * result

    python3 tools/nnue_train.py <out.bin> <data.bin>... [--epochs 20] [--lambda 0.7]
"""
import argparse
import struct
import time

import numpy as np
import torch

HIDDEN = 256
INPUTS = 768
PAD = INPUTS  # feature index for empty slots
QA, QB, SCALE = 255, 64, 400
RECORD = np.dtype([("sq", "u1", 32), ("pc", "u1", 32), ("score", "<i2"), ("result", "u1"), ("stm", "u1")])


def all_features(data, perspective, chunk=1 << 20):
    """features() for the whole dataset, built in chunks and stored as int16 to keep memory down."""
    out = np.empty((len(data), 32), dtype=np.int16)
    for i in range(0, len(data), chunk):
        out[i : i + chunk] = features(data[i : i + chunk], perspective[i : i + chunk])
    return torch.from_numpy(out)


def features(data, perspective):
    """[N, 32] feature indices from `perspective` (array of 0 = White / 1 = Black per position)."""
    sq = data["sq"].astype(np.int64)
    pc = data["pc"].astype(np.int64)
    empty = pc == 0xFF
    color, ptype = pc >> 3, pc & 7
    persp = perspective[:, None].astype(np.int64)
    square = np.where(persp == 0, sq, sq ^ 56)
    idx = np.where(color == persp, 0, 384) + (ptype - 1) * 64 + square
    idx[empty] = PAD
    return idx


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ft = torch.nn.EmbeddingBag(INPUTS + 1, HIDDEN, mode="sum", padding_idx=PAD)
        self.ft_bias = torch.nn.Parameter(torch.zeros(HIDDEN))
        self.out = torch.nn.Linear(2 * HIDDEN, 1)
        torch.nn.init.normal_(self.ft.weight, std=0.05)
        with torch.no_grad():
            self.ft.weight[PAD].zero_()

    def forward(self, us, them):
        a = torch.clamp(self.ft(us) + self.ft_bias, 0, 1)
        b = torch.clamp(self.ft(them) + self.ft_bias, 0, 1)
        return self.out(torch.cat([a, b], dim=1)).squeeze(1)  # in units of SCALE centipawns


def export(net, path):
    w = net.ft.weight.detach().numpy()[:INPUTS]
    ft_w = np.clip(np.round(w * QA), -32768, 32767).astype("<i2")
    ft_b = np.clip(np.round(net.ft_bias.detach().numpy() * QA), -32768, 32767).astype("<i2")
    out_w = np.clip(np.round(net.out.weight.detach().numpy()[0] * QB), -32768, 32767).astype("<i2")
    out_b = int(round(float(net.out.bias.detach()[0]) * QA * QB))
    with open(path, "wb") as f:
        f.write(b"CBNN" + struct.pack("<i", HIDDEN))
        f.write(ft_w.tobytes() + ft_b.tobytes() + out_w.tobytes() + struct.pack("<i", out_b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("data", nargs="+")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.7)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    args = ap.parse_args()
    global HIDDEN
    HIDDEN = args.hidden

    data = np.concatenate([np.fromfile(p, dtype=RECORD) for p in args.data])
    data = data[np.abs(data["score"]) < 2000]
    print(f"{len(data)} positions", flush=True)
    stm = data["stm"].astype(np.int64)
    us = all_features(data, stm)
    them = all_features(data, 1 - stm)
    score_t = torch.sigmoid(torch.from_numpy(data["score"].astype(np.float32)) / SCALE)
    result_t = torch.from_numpy(data["result"].astype(np.float32)) / 2
    target = args.lam * score_t + (1 - args.lam) * result_t

    n = len(data)
    val = torch.randperm(n)[: min(50000, n // 20)]
    train_mask = torch.ones(n, dtype=torch.bool)
    train_mask[val] = False
    train = torch.nonzero(train_mask).squeeze(1)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for epoch in range(args.epochs):
        t0 = time.time()
        perm = train[torch.randperm(len(train))]
        total = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i : i + args.batch]
            pred = torch.sigmoid(net(us[idx].long(), them[idx].long()))
            loss = torch.mean((pred - target[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                net.ft.weight[PAD].zero_()
            total += loss.item() * len(idx)
        sched.step()
        with torch.no_grad():
            vloss = torch.mean((torch.sigmoid(net(us[val].long(), them[val].long())) - target[val]) ** 2).item()
        print(f"epoch {epoch + 1}/{args.epochs}  train {total / len(perm):.5f}  val {vloss:.5f}  "
              f"{time.time() - t0:.0f}s", flush=True)
        export(net, args.out)  # keep the latest network on disk
        torch.save(net.state_dict(), args.out + ".pt")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
