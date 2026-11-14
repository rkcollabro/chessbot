// gen6 "big net" trainer for chessbot, on Bullet (https://github.com/jw1912/bullet, rev 5d70177).
// Net: 768 x 10 king buckets (mirrored, Bullet's example layout) -> HL SCReLU (x2 perspectives) -> 8 material
// output buckets. Saved quantised (QA 255, QB 64): l0w [buckets*768][HL], l0b [HL], l1w [8][2*HL] (side to
// move half first), l1b [8] - exactly what tools/bullet_to_cbn2.py turns into the engine's CBN2 format.
//
// usage: gtrain <bullet|binpack|mix<f>> <net_id> <superbatches> <wdl> <hl> <data file>...
//   bullet: our data converted by prep.py; binpack: Stockfish binpacks (Leela data);
//   mix<f>, e.g. mix0.25: both at once, files ending in .binpack read as binpacks and the rest as bullet files,
//   with a fraction f of every batch taken from the bullet files.
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc,
};

use bullet_lib::{
    game::{
        formats::bulletformat::ChessBoard,
        inputs::{ChessBucketsMirrored, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};
use bullet_trainer::reader::DataReader;

// Two readers blended into one stream: each runs on its own thread, and every output chunk takes a fixed share
// from each and is shuffled, so every batch holds both kinds of data in the set proportion.
#[derive(Clone)]
struct Mix<A, B> {
    a: A,
    b: B,
    frac_b: f64,
}

fn feed<R: DataReader<ChessBoard>>(r: R, stop: Arc<AtomicBool>) -> mpsc::Receiver<Vec<ChessBoard>> {
    let (tx, rx) = mpsc::sync_channel(2);
    std::thread::spawn(move || r.read_chunks(0, |c| stop.load(Ordering::Relaxed) || tx.send(c.to_vec()).is_err()));
    rx
}

fn take(rx: &mpsc::Receiver<Vec<ChessBoard>>, buf: &mut Vec<ChessBoard>, at: &mut usize, mut n: usize,
        out: &mut Vec<ChessBoard>) -> bool {
    while n > 0 {
        if *at == buf.len() {
            match rx.recv() {
                Ok(v) => (*buf, *at) = (v, 0),
                Err(_) => return false,
            }
            continue;
        }
        let k = n.min(buf.len() - *at);
        out.extend_from_slice(&buf[*at..*at + k]);
        *at += k;
        n -= k;
    }
    true
}

impl<A: DataReader<ChessBoard>, B: DataReader<ChessBoard>> DataReader<ChessBoard> for Mix<A, B> {
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, _skip: usize, mut f: F) {
        const OUT: usize = 1 << 20;
        let nb = (OUT as f64 * self.frac_b).round() as usize;
        let stop = Arc::new(AtomicBool::new(false));
        let (ra, rb) = (feed(self.a.clone(), stop.clone()), feed(self.b.clone(), stop.clone()));
        let (mut ba, mut bb, mut ia, mut ib) = (Vec::new(), Vec::new(), 0, 0);
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        loop {
            let mut out = Vec::with_capacity(OUT);
            if !take(&ra, &mut ba, &mut ia, OUT - nb, &mut out) || !take(&rb, &mut bb, &mut ib, nb, &mut out) {
                break;
            }
            for i in (1..out.len()).rev() {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                out.swap(i, (seed % (i as u64 + 1)) as usize);
            }
            if f(&out) {
                break;
            }
        }
        stop.store(true, Ordering::Relaxed);
    }
}

#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
    0, 1, 2, 3,
    4, 4, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    8, 8, 8, 8,
    9, 9, 9, 9,
    9, 9, 9, 9,
];
const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);
const NUM_OUTPUT_BUCKETS: usize = 8;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let kind = args[1].clone();
    let net_id = args[2].clone();
    let superbatches: usize = args[3].parse().unwrap();
    let wdl_proportion: f32 = args[4].parse().unwrap();
    let hl_size: usize = args[5].parse().unwrap();
    let data: Vec<String> = args[6..].to_vec();
    // INIT_LR: a lower starting rate when continuing from INIT_WEIGHTS (a second cosine cycle).
    let initial_lr: f32 = std::env::var("INIT_LR").map(|s| s.parse().unwrap()).unwrap_or(0.001);
    let final_lr = 0.001 * 0.3f32.powi(5);

    // ARCH=ml: multi-layer net (Bullet's 4_multi_layer example): l0 HL CReLU, each perspective's two halves
    // multiplied pairwise (HL/2 each) -> l1 16 SCReLU -> l2 32 SCReLU -> l3 1, l1-l3 per output bucket.
    // Saved: l0w/l0b i16 (QA 255) as above, l1w i16 [8*16][HL] (QB 256, row = bucket * 16 + neuron; i8 with
    // QB 64 was too coarse for the small l1 weights and moved evaluations by up to 20%), then f32:
    // l1b [8*16], l2w [8*32][16], l2b [8*32], l3w [8][32], l3b [8] - tools/bullet_to_cbn3.py makes CBN3 from it.
    let ml = std::env::var("ARCH").map(|s| s == "ml").unwrap_or(false);
    const L2: usize = 16;
    const L3: usize = 32;
    let l0w_fmt = SavedFormat::id("l0w")
        .transform(|store, weights| {
            let factoriser = store.get("l0f").values.f32().repeat(NUM_INPUT_BUCKETS);
            weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
        })
        .round()
        .quantise::<i16>(255);
    let formats = if ml {
        vec![
            l0w_fmt,
            SavedFormat::id("l0b").round().quantise::<i16>(255),
            SavedFormat::id("l1w").transpose().round().quantise::<i16>(256),
            SavedFormat::id("l1b"),
            SavedFormat::id("l2w").transpose(),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w").transpose(),
            SavedFormat::id("l3b"),
        ]
    } else {
        vec![
            l0w_fmt,
            SavedFormat::id("l0b").round().quantise::<i16>(255),
            SavedFormat::id("l1w").round().quantise::<i16>(64).transpose(),
            SavedFormat::id("l1b").round().quantise::<i16>(255 * 64),
        ]
    };
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&formats)
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0f = builder.new_weights("l0f", Shape::new(hl_size, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, hl_size);
            if ml {
                l0.init_with_effective_input_size(32);
            }
            l0.weights = l0.weights + expanded_factoriser;
            if ml {
                let l1 = builder.new_affine("l1", hl_size, NUM_OUTPUT_BUCKETS * L2);
                let l2 = builder.new_affine("l2", L2, NUM_OUTPUT_BUCKETS * L3);
                let l3 = builder.new_affine("l3", L3, NUM_OUTPUT_BUCKETS);
                let ft = |input, start, end| l0.slice(start, end).forward(input).crelu();
                let stm_hidden = ft(stm_inputs, 0, hl_size / 2) * ft(stm_inputs, hl_size / 2, hl_size);
                let ntm_hidden = ft(ntm_inputs, 0, hl_size / 2) * ft(ntm_inputs, hl_size / 2, hl_size);
                let hl1 = stm_hidden.concat(ntm_hidden);
                let hl2 = l1.forward(hl1).select(output_buckets).screlu();
                let hl3 = l2.forward(hl2).select(output_buckets).screlu();
                l3.forward(hl3).select(output_buckets)
            } else {
                let l1 = builder.new_affine("l1", 2 * hl_size, NUM_OUTPUT_BUCKETS);
                let stm_hidden = l0.forward(stm_inputs).screlu();
                let ntm_hidden = l0.forward(ntm_inputs).screlu();
                l1.forward(stm_hidden.concat(ntm_hidden)).select(output_buckets)
            }
        });

    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
    trainer.optimiser.set_params_for_weight("l0f", stricter_clipping);

    // INIT_WEIGHTS: start from a previous net (Bullet weights file made by tools/bignet/dequant.py from a
    // quantised.bin; l0f stays zero since the saved l0w already includes it). init_check.bin is that net
    // quantised again, for the caller to compare with the original.
    if let Ok(p) = std::env::var("INIT_WEIGHTS") {
        trainer.optimiser.load_weights_from_file(&p).map_err(|e| format!("{e:?}")).unwrap();
        trainer.save_quantised("init_check.bin").unwrap();
        println!("loaded initial weights from {p}");
        // EVAL_ONLY: just print that net's evaluations of EVAL_FENS.
        if std::env::var("EVAL_ONLY").is_ok() {
            for fen in EVAL_FENS {
                println!("EVAL {:.2} {fen}", 400.0 * trainer.eval(&format!("{fen} | 0 | 0.5")));
            }
            return;
        }
    }

    let schedule = TrainingSchedule {
        net_id: net_id.clone(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_proportion },
        lr_scheduler: lr::CosineDecayLR { initial_lr, final_lr, final_superbatch: superbatches },
        save_rate: 20,
    };
    let settings = LocalSettings { threads: 2, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };
    let paths: Vec<&str> = data.iter().map(|s| s.as_str()).collect();

    use loader::sfbinpack::{MoveType, PieceType, SfBinpackLoader, TrainingDataEntry};
    // Same quiet-position filter our self-play data already applies when it is written.
    fn filter(entry: &TrainingDataEntry) -> bool {
        entry.ply >= 16
            && !entry.pos.is_checked(entry.pos.side_to_move())
            && entry.score.unsigned_abs() <= 10000
            && entry.mv.mtype() == MoveType::Normal
            && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
    }
    if kind == "binpack" {
        let dl = SfBinpackLoader::new_concat_multiple(&paths, 1024, 2, filter);
        trainer.run(&schedule, &settings, &dl);
    } else if let Some(frac) = kind.strip_prefix("mix") {
        let (bp, ours): (Vec<&str>, Vec<&str>) = paths.iter().copied().partition(|p| p.ends_with(".binpack"));
        let dl = Mix {
            a: SfBinpackLoader::new_concat_multiple(&bp, 1024, 2, filter),
            b: loader::DirectSequentialDataLoader::new(&ours),
            frac_b: frac.parse().unwrap(),
        };
        trainer.run(&schedule, &settings, &dl);
    } else {
        let dl = loader::DirectSequentialDataLoader::new(&paths);
        trainer.run(&schedule, &settings, &dl);
    }

    // The final net's own evaluations (centipawns, side to move), to check the engine's inference against.
    for fen in EVAL_FENS {
        println!("EVAL {:.2} {fen}", 400.0 * trainer.eval(&format!("{fen} | 0 | 0.5")));
    }
}

// The last three are colour-flipped copies of three Black-to-move positions above: same evaluation expected.
const EVAL_FENS: [&str; 11] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 b - - 0 1",
    "2kr3r/ppp2ppp/2n5/2b1q3/4P3/2N2N2/PPP2PPP/R2QKB1R b KQ - 3 10",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "3r2k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
    "r2qkb1r/ppp2ppp/2n2n2/4p3/2B1Q3/2N5/PPP2PPP/2KR3R w kq - 3 10",
];
