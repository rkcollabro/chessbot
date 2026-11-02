"""SPSA tuning of the search parameters that a -DTUNE build exposes as UCI options.

Each iteration perturbs every tuned parameter by +-c at random, plays a game pair (one opening, both colors)
between theta+c*delta and theta-c*delta, and records result*delta, where result is the plus engine's wins minus
losses in the pair (-2..2). Summed over many pairs that points uphill. The SPSA workflow runs `play` on many
runners from the same theta, then `update` adds up their sums and moves theta:

    theta_i += R * c_i * sum(result * delta_i)        (R = learning rate per pair result, as in fishtest)

    python tools/spsa.py play <engine> --spec tools/spsa_params.txt [--theta theta.json] --pairs 50 --out g.json
    python tools/spsa.py update --spec tools/spsa_params.txt --theta theta.json g*.json --out theta.json

The spec file lists one tuned parameter per line: NAME C (perturbation size). Parameters start from the engine's
own defaults; their bounds come from the engine's UCI options.
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import threading
import time

import chess
import chess.engine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sprt import play as play_game  # noqa: E402


def read_spec(path):
    spec = []
    for line in open(path):
        line = line.split("#")[0].split()
        if line:
            spec.append((line[0], float(line[1])))
    return spec


def engine_bounds(path, names):
    e = chess.engine.SimpleEngine.popen_uci([os.path.abspath(path), "uci"])
    out = {}
    for n in names:
        o = e.options[n]
        out[n] = (float(o.default), float(o.min), float(o.max))
    e.quit()
    return out


def stochastic_round(x, rng):
    lo = int(x // 1)
    return lo + (rng.random() < x - lo)


def cmd_play(args):
    spec = read_spec(args.spec)
    names = [n for n, _ in spec]
    bounds = engine_bounds(args.engine, names)
    theta = {n: bounds[n][0] for n in names}
    if args.theta and os.path.exists(args.theta):
        theta.update(json.load(open(args.theta))["theta"])
    openings = [l.strip() for l in open(args.openings) if l.strip()]
    tc = tuple(float(x) for x in args.tc.split("+"))
    grad = {n: 0.0 for n in names}
    wdl = [0, 0, 0]
    lock = threading.Lock()
    local = threading.local()
    start = time.time()
    done = [0]

    def config(delta_sign, delta, rng):
        cfg = {}
        for n, c in spec:
            _, lo, hi = bounds[n]
            v = min(max(theta[n] + delta_sign * c * delta[n], lo), hi)
            cfg[n] = stochastic_round(v, rng)
        return cfg

    def run(i):
        rng = random.Random(args.seed * 1000003 + i)
        if not hasattr(local, "plus"):
            local.plus = chess.engine.SimpleEngine.popen_uci([os.path.abspath(args.engine), "uci"])
            local.minus = chess.engine.SimpleEngine.popen_uci([os.path.abspath(args.engine), "uci"])
            for e in (local.plus, local.minus):
                e.configure({"OwnBook": False, "Threads": 1})
        delta = {n: rng.choice((-1, 1)) for n in names}
        local.plus.configure(config(+1, delta, rng))
        local.minus.configure(config(-1, delta, rng))
        opening = openings[(args.offset + i) % len(openings)]
        result = 0
        for plus_white in (True, False):
            w, b = (local.plus, local.minus) if plus_white else (local.minus, local.plus)
            res = play_game({chess.WHITE: w, chess.BLACK: b}, opening, tc)
            if res != "1/2-1/2":
                result += 1 if (res == "1-0") == plus_white else -1
            with lock:
                wdl[0 if res == "1-0" else 1 if res == "0-1" else 2] += 1
        with lock:
            for n in names:
                grad[n] += result * delta[n]
            done[0] += 1
            if done[0] % 10 == 0:
                print(f"pairs {done[0]}  white W {wdl[0]} L {wdl[1]} D {wdl[2]}  {(time.time() - start) / 60:.0f} min",
                      flush=True)

    with cf.ThreadPoolExecutor(args.concurrency) as pool:
        for f in [pool.submit(run, i) for i in range(args.pairs)]:
            f.result()
    json.dump({"grad": grad, "pairs": args.pairs, "wdl": wdl, "theta": theta,
               "bounds": {n: [bounds[n][1], bounds[n][2]] for n in names}}, open(args.out, "w"))
    print("done", flush=True)
    os._exit(0)


def cmd_update(args):
    spec = read_spec(args.spec)
    shards = [json.load(open(f)) for f in args.grads]
    if not shards:
        sys.exit("no gradient files")
    theta = dict(shards[0]["theta"])  # every shard played from the same theta
    history = []
    if args.theta and os.path.exists(args.theta):
        prev = json.load(open(args.theta))
        history = prev.get("history", [])
    pairs = sum(s["pairs"] for s in shards)
    wdl = [sum(s["wdl"][k] for s in shards) for k in range(3)]
    lo_hi = shards[0].get("bounds", {})
    rows = []
    for n, c in spec:
        g = sum(s["grad"][n] for s in shards)
        step = args.r * c * g
        new = theta[n] + step
        if n in lo_hi:
            new = min(max(new, lo_hi[n][0]), lo_hi[n][1])
        rows.append((n, theta[n], new, g))
        theta[n] = new
    history.append({"pairs": pairs, "theta": dict(theta)})
    json.dump({"theta": theta, "history": history}, open(args.out, "w"), indent=1)
    total_pairs = sum(h["pairs"] for h in history)
    print(f"### SPSA round {len(history)}: {pairs} pairs this round, {total_pairs} in total "
          f"(white W {wdl[0]} L {wdl[1]} D {wdl[2]})\n")
    print("| parameter | before | after | change | gradient sum |")
    print("|---|---:|---:|---:|---:|")
    for n, before, after, g in rows:
        print(f"| {n} | {before:.1f} | {after:.1f} | {after - before:+.2f} | {g:+.0f} |")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play")
    p.add_argument("engine")
    p.add_argument("--spec", required=True)
    p.add_argument("--theta")
    p.add_argument("--pairs", type=int, default=50)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--tc", default="8+0.08")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--openings", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "openings.txt"))
    p.add_argument("--out", required=True)
    u = sub.add_parser("update")
    u.add_argument("grads", nargs="+")
    u.add_argument("--spec", required=True)
    u.add_argument("--theta", help="previous theta.json (for its history)")
    u.add_argument("--r", type=float, default=0.002)
    u.add_argument("--out", required=True)
    args = ap.parse_args()
    cmd_play(args) if args.cmd == "play" else cmd_update(args)


if __name__ == "__main__":
    main()
