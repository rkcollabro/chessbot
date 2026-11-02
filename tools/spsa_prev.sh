#!/bin/sh
# SPSA workflow helper: puts the round's starting theta in prev/theta.json. From the previous round's run
# ($PREV) when set; its artifact can only be fetched once that run has completed, which happens just after it
# dispatches this round, so retry for a while. Otherwise from $THETA, or nothing (the engine's own values).
mkdir -p prev
if [ -n "$PREV" ]; then
  for i in $(seq 1 30); do
    gh run download "$PREV" -R "$GITHUB_REPOSITORY" -n spsa-theta -D prev 2>/dev/null && break
    sleep 20
  done
  [ -f prev/theta.json ] || { echo "no spsa-theta artifact from run $PREV"; exit 1; }
elif [ -n "$THETA" ]; then
  # JSON as is, or base64 (survives shells that mangle quotes when passing it to gh)
  case $THETA in
    \{*) printf '%s' "$THETA" > prev/theta.json ;;
    *) printf '%s' "$THETA" | base64 -d > prev/theta.json ;;
  esac
  python3 -c "import json; json.load(open('prev/theta.json'))" || { echo "theta input is not valid JSON"; exit 1; }
fi
[ -f prev/theta.json ] && cat prev/theta.json || echo "starting from the engine's values"
