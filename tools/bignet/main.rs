// gen6 "big net" trainer for chessbot, on Bullet (https://github.com/jw1912/bullet, rev 5d70177).
// Net: 768 x 10 king buckets (mirrored, Bullet's example layout) -> HL SCReLU (x2 perspectives) -> 8 material
// output buckets. Saved quantised (QA 255, QB 64): l0w [buckets*768][HL], l0b [HL], l1w [8][2*HL] (side to
// move half first), l1b [8] - exactly what tools/bullet_to_cbn2.py turns into the engine's CBN2 format.
//
// usage: gtrain <bullet|binpack> <net_id> <superbatches> <wdl> <hl> <data file>...
use bullet_lib::{
    game::{
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
    let initial_lr = 0.001;
    let final_lr = 0.001 * 0.3f32.powi(5);

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w")
                .transform(|store, weights| {
                    let factoriser = store.get("l0f").values.f32().repeat(NUM_INPUT_BUCKETS);
                    weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                })
                .round()
                .quantise::<i16>(255),
            SavedFormat::id("l0b").round().quantise::<i16>(255),
            SavedFormat::id("l1w").round().quantise::<i16>(64).transpose(),
            SavedFormat::id("l1b").round().quantise::<i16>(255 * 64),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0f = builder.new_weights("l0f", Shape::new(hl_size, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, hl_size);
            l0.weights = l0.weights + expanded_factoriser;
            let l1 = builder.new_affine("l1", 2 * hl_size, NUM_OUTPUT_BUCKETS);
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            l1.forward(stm_hidden.concat(ntm_hidden)).select(output_buckets)
        });

    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
    trainer.optimiser.set_params_for_weight("l0f", stricter_clipping);

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

    if kind == "binpack" {
        use loader::sfbinpack::{MoveType, PieceType, SfBinpackLoader, TrainingDataEntry};
        // Same quiet-position filter our self-play data already applies when it is written.
        fn filter(entry: &TrainingDataEntry) -> bool {
            entry.ply >= 16
                && !entry.pos.is_checked(entry.pos.side_to_move())
                && entry.score.unsigned_abs() <= 10000
                && entry.mv.mtype() == MoveType::Normal
                && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
        }
        let dl = SfBinpackLoader::new_concat_multiple(&paths, 1024, 2, filter);
        trainer.run(&schedule, &settings, &dl);
    } else {
        let dl = loader::DirectSequentialDataLoader::new(&paths);
        trainer.run(&schedule, &settings, &dl);
    }
}
