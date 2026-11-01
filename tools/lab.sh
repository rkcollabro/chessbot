#!/bin/sh
# Helper for the self-play lab on the test machine (native, run from ~/chessbot).
#   lab.sh build <name> [srcdir]     build engines/<name> from srcdir (default src)
#   lab.sh test <name>               run the perft/hash test suite
#   lab.sh sprt <cand> <base> [args] start an SPRT in the background, logging to logs/<cand>.log
#   lab.sh rule <cand> <base>        start a 60-game rule test (more than 30 wins, at most 10 losses)
#   lab.sh stop                      stop any running test and its engines
#   lab.sh status                    last line of every SPRT log, and whether one is running
cd "$HOME/chessbot" || exit 1
cmd=$1; shift
case $cmd in
  build) sh tools/build_linux.sh "engines/$1" "${2:-src}" ;;
  test)  "./engines/$1" test 2>&1 | tail -1 ;;
  sprt)  c=$1; b=$2; shift 2
         nohup nice -n 5 python3 tools/sprt.py "engines/$c" "engines/$b" "$@" > "logs/$c.log" 2>&1 &
         echo "started $c vs $b" ;;
  rule)  c=$1; b=$2; shift 2
         nohup nice -n 5 python3 tools/sprt.py "engines/$c" "engines/$b" --rule "$@" > "logs/$c.log" 2>&1 &
         echo "started rule test $c vs $b" ;;
  stop)  pkill -f "tools/sprt.py"; sleep 1; pkill -f "$HOME/chessbot/engines/"; echo stopped ;;
  status) for f in logs/*.log; do [ -f "$f" ] && echo "$(basename "$f" .log): $(tail -1 "$f")"; done
          pgrep -fa "tools/sprt.py" || echo "no SPRT running" ;;
esac
