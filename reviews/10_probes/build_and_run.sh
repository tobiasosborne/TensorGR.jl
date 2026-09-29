#!/bin/bash
# Build the blind cross-check against both prototype branches and run it.
# usage: bash reviews/10_probes/build_and_run.sh   (from repo root)
set -euo pipefail
D=$(mktemp -d); trap 'rm -rf "$D"' EXIT
for b in proto/canon-ir-baseline proto/canon-ir-baseline-opus; do
  n=$([ "$b" = proto/canon-ir-baseline ] && echo s || echo o)
  mkdir -p "$D/$n/include" "$D/$n/src"
  git show "$b:proto/canonir/include/canonir.h" > "$D/$n/include/canonir.h"
  git show "$b:proto/canonir/src/canonir.c"     > "$D/$n/src/canonir.c"
  gcc -O2 -march=native -std=c11 -c -o "$D/$n.o" "$D/$n/src/canonir.c" -I"$D/$n/include"
done
gcc -O2 -march=native -std=c11 -Ireviews/10_probes -o "$D/xcheck" reviews/10_probes/xcheck.c "$D/s.o" "$D/o.o" -lm
for cfg in "100000 12 11" "100000 20 12" "30000 32 13" "5000 64 14" "1000 128 15"; do "$D/xcheck" check $cfg; done
CORRUPT=500 "$D/xcheck" check 3000 16 7 || echo "(expected: planted corruption detected)"
taskset -c 0 "$D/xcheck" bench 42
