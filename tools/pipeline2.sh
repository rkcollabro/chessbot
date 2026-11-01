#!/bin/sh
# Network 1b: stop the generators, train on everything collected so far, rule-test it, resume generating.
cd ~/chessbot
EPOCHS=${1:-10}
log() { echo "$(date +'%m-%d %H:%M') $*" >> logs/pipeline.log; }

pkill -f "engines/datagen selfplay"
sleep 5
mkdir -p nets/1b_data
mv data/sp_*.bin nets/1b_data/
log "training 1b on $(( $(cat nets/1b_data/*.bin | wc -c) / 68 )) positions, $EPOCHS epochs"
rm -f nets/gen1b.bin
nice -n 5 python3 tools/nnue_train.py nets/gen1b.bin nets/1b_data/*.bin --epochs $EPOCHS > logs/train_gen1b.log 2>&1
if [ ! -s nets/gen1b.bin ]; then
  log "training FAILED: $(tail -1 logs/train_gen1b.log)"
  exit 1
fi
cp nets/gen1b.bin engines/nnue.bin

# Record the candidate's evaluation so the log shows the network is in use (base predates NNUE support).
FEN="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
log "eval check: nnue1 $(./engines/nnue1 eval "$FEN" 2>&1 | tail -1)"

log "rule test nnue1b vs base"
python3 tools/sprt.py engines/nnue1 engines/base --rule > logs/nnue1b.log 2>&1
log "$(tail -1 logs/nnue1b.log)"

log "restarting generators"
for i in 1 2 3 4; do
  nohup nice -n 5 ./engines/datagen selfplay data/sp_$i.bin 10000000 5000 $((3000 + i)) > logs/datagen3_$i.out 2>&1 < /dev/null &
done
