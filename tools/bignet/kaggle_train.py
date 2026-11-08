# chessbot big-net training on a Kaggle 2x T4 session: one Bullet run per GPU.
#   GPU 0: our gen6 self-play data (Kaggle dataset raykay98/gen6-data, converted by prep.py)
#   GPU 1: Leela-derived data (linrock/test80-2024 binpacks from Hugging Face)
# Same net and recipe for both (tools/bignet/main.rs on branch bignet), so the comparison is data only.
# Checkpoints' quantised.bin files are copied to /kaggle/working as they appear; the trainers are stopped
# before the session limit so the outputs are kept.
import glob, os, shutil, signal, subprocess, threading, time, urllib.request

T0 = time.time()
SB = int("__SB__")                 # superbatches (x 100M positions) per net
WDL = "__WDL__"
HOURS = float("__HOURS__")         # stop the trainers after this many hours of session time
OURS_LIMIT = int("__OURS_LIMIT__") # positions of our data to use (0 = all)
LEELA = "__LEELA__".split(",")
HL = "1024"
REV = "5d70177"
RAW = "https://raw.githubusercontent.com/rkcollabro/chessbot/bignet/tools/bignet/"
W = "/kaggle/working"
os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + os.environ["PATH"]
os.environ["CUDA_PATH"] = "/usr/local/cuda"
logf = open(f"{W}/run.log", "a")


def log(m):
    s = f"[{(time.time() - T0) / 60:6.1f} min] {m}"
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


def sh(c):
    log("+ " + c)
    subprocess.run(["bash", "-o", "pipefail", "-c", c], check=True)


def fetch(url, dst):
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 24)


# ---- sources and build
os.makedirs("/tmp/gtrain/src", exist_ok=True)
fetch(RAW + "main.rs", "/tmp/gtrain/src/main.rs")
fetch(RAW + "prep.py", "/tmp/gtrain/prep.py")
open("/tmp/gtrain/Cargo.toml", "w").write(
    '[package]\nname = "gtrain"\nversion = "0.1.0"\nedition = "2024"\n\n'
    '[dependencies]\nbullet_lib = { path = "/tmp/bullet/crates/bullet_lib", features = ["cuda"] }\n\n'
    '[profile.release]\nopt-level = 3\n')


def build():
    sh("curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal > /dev/null")
    sh(f"git clone -q https://github.com/jw1912/bullet /tmp/bullet && cd /tmp/bullet && git checkout -q {REV}")
    # The CUDA driver library (libcuda.so) lives with the NVIDIA driver, outside the toolkit's lib64.
    os.environ["RUSTFLAGS"] = "-L /usr/local/nvidia/lib64"
    sh("cd /tmp/gtrain && cargo build -q --release 2>&1 | tail -20")
    log("build done")


def leela():
    import zstandard
    os.makedirs("/tmp/leela", exist_ok=True)
    for name in LEELA:
        url = "https://huggingface.co/datasets/linrock/test80-2024/resolve/main/" + name
        out = "/tmp/leela/" + name.removesuffix(".zst")
        with urllib.request.urlopen(url) as r, open(out, "wb") as f:
            zstandard.ZstdDecompressor().copy_stream(r, f, read_size=1 << 24, write_size=1 << 24)
        log(f"leela {name}: {os.path.getsize(out) / 1e9:.1f} GB")


MIN_PER_SB = 2.15  # measured: ~2.0-2.1 min per superbatch per T4 with both GPUs busy
STOP_AT = HOURS * 60 - 8  # minutes; every run must reach its last superbatch before this
procs = {}
built = threading.Event()


def start(name, gpu, kind, files):
    # Size the run to the time left so the cosine schedule completes (an early stop would keep an
    # un-annealed checkpoint).
    left = STOP_AT - (time.time() - T0) / 60
    sb = min(SB, int(left / MIN_PER_SB))
    d = f"/tmp/run_{name}"
    os.makedirs(d, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    args = [kind, f"big-{name}", str(sb), WDL, HL] + files
    procs[name] = subprocess.Popen(["/tmp/gtrain/target/release/gtrain"] + args, cwd=d, env=env,
                                   stdout=open(f"{W}/{name}.log", "w"), stderr=subprocess.STDOUT)
    log(f"started {name} on GPU {gpu}: {sb} superbatches")


def leela_pipeline():
    leela()
    built.wait()
    start("leela", "1", "binpack", sorted(glob.glob("/tmp/leela/*.binpack")))


def ours_pipeline():
    files = sorted(glob.glob("/kaggle/input/**/*.bin.zst", recursive=True))
    log(f"our data: {len(files)} files")
    # Four converters over interleaved slices of the files (one per CPU core), each writing its own shuffled
    # part; the loader reads the parts in turn.
    per = OURS_LIMIT // 4 if OURS_LIMIT else 0
    ps = [subprocess.Popen(["python3", "/tmp/gtrain/prep.py", f"/tmp/ours_{k}.bin", str(per)] + files[k::4],
                           stdout=logf, stderr=subprocess.STDOUT) for k in range(4)]
    if any(p.wait() for p in ps):
        log("prep failed")
        return
    parts = [f"/tmp/ours_{k}.bin" for k in range(4)]
    log(f"ours: {sum(os.path.getsize(f) for f in parts) // 32 / 1e6:.0f}M positions")
    built.wait()
    start("ours", "0", "bullet", parts)


def build_thread():
    build()
    built.set()


threads = [threading.Thread(target=f) for f in (build_thread, leela_pipeline, ours_pipeline)]
for t in threads:
    t.start()


def harvest():
    for q in glob.glob("/tmp/run_*/checkpoints/*/quantised.bin"):
        tag = os.path.basename(os.path.dirname(q))
        dst = f"{W}/nets/{tag}.bin"
        if not os.path.exists(dst):
            os.makedirs(f"{W}/nets", exist_ok=True)
            shutil.copy(q, dst)
            log(f"saved {tag}")


while any(t.is_alive() for t in threads) or any(p.poll() is None for p in procs.values()):
    time.sleep(60)
    harvest()
    if time.time() - T0 > HOURS * 3600:
        log("time limit: stopping trainers")
        for p in procs.values():
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        time.sleep(30)
        for p in procs.values():
            if p.poll() is None:
                p.kill()
        break
harvest()
for name, p in procs.items():
    log(f"{name} exit {p.poll()}")
log("done")
