#!/bin/bash
# tools/dsv41/ab_decode.sh - emitted decode tok/s of two builds of strata-dsv41, interleaved on the same prompt.
#
#   tools/dsv41/ab_decode.sh <exe A> <exe B> <pack> <ids file> [<pairs>]
#
# Each pair runs A and B once, and the order alternates from pair to pair, so a drift in the machine hits both arms alike.
# The first pair is a warm-up and is left out of the summary (WARM=0 keeps it).  The number is the engine's own line
#   decode: N tokens in T ms (X tok/s)
# for the N tokens after the prompt.  DS4_NO_EOS=1 keeps N fixed, so the two arms make the same amount of work.
#
#   env: N (128 tokens), DSPARK (on), CTX (8192), WARM (1), GPU (0: CUDA_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES),
#        OUT (folder for the logs, default /tmp/ab_decode.<pid>), EXTRA (more engine args, the same for both arms).
#   The other environment variables (DS4_TIER_*, DS4_AHEAD, ...) are inherited: set them the same for both arms.
#   Use the same pack, the same tier args (EXTRA) and the same GPU in both arms, on the same day.
set -u
A=$1
B=$2
pack=$3
ids=$(cat "$4")
pairs=${5:-5}
N=${N:-128}
DSPARK=${DSPARK:-on}
CTX=${CTX:-8192}
WARM=${WARM:-1}
GPU=${GPU:-0}
OUT=${OUT:-/tmp/ab_decode.$$}
mkdir -p "$OUT"
: > "$OUT/results.txt"
extra=()
[ -n "${EXTRA:-}" ] && read -r -a extra <<< "$EXTRA"

run_one() {   # <arm> <exe> <pair>
    local arm=$1 exe=$2 p=$3 rc tps
    DS4_NO_EOS=1 CUDA_VISIBLE_DEVICES=$GPU HIP_VISIBLE_DEVICES=$GPU "$exe" --pack "$pack" --max-context "$CTX" --vision off \
        --dspark "$DSPARK" --n "$N" --ids "$ids" ${extra[@]+"${extra[@]}"} > "$OUT/${arm}_$p.out" 2> "$OUT/${arm}_$p.err"
    rc=$?
    tps=$(sed -n 's/^decode: [0-9][0-9]* tokens in [0-9.][0-9.]* ms (\([0-9.][0-9.]*\) tok\/s).*/\1/p' "$OUT/${arm}_$p.err" | tail -1)
    echo "$arm $p rc=$rc tok_s=${tps:-NA}"
    printf '%s %s %s\n' "$arm" "$p" "${tps:-NA}" >> "$OUT/results.txt"
}

for p in $(seq 1 "$pairs"); do
    if [ $((p % 2)) -eq 1 ]; then
        run_one A "$A" "$p"
        run_one B "$B" "$p"
    else
        run_one B "$B" "$p"
        run_one A "$A" "$p"
    fi
done

echo "== summary (pairs: $pairs, warm-up pair dropped: $([ "$WARM" = 1 ] && echo yes || echo no))"
awk -v warm="$WARM" '
    function median(arr, m,    i, j, x, t) {   # insertion sort of arr[1..m], then the middle
        for (i = 2; i <= m; i++) { x = arr[i]; j = i - 1; while (j >= 1 && arr[j] > x) { arr[j + 1] = arr[j]; j-- } arr[j + 1] = x }
        return (m % 2) ? arr[(m + 1) / 2] : (arr[m / 2] + arr[m / 2 + 1]) / 2
    }
    $3 == "NA" || (warm == 1 && $2 == 1) { next }
    $1 == "A" { na++; va[na] = $3 }
    $1 == "B" { nb++; vb[nb] = $3 }
    END {
        if (na == 0 || nb == 0) { print "no runs with a decode line for both arms"; exit }
        ma = median(va, na); mb = median(vb, nb)
        printf "A: median %.2f tok/s over %d runs (min %.2f, max %.2f)\n", ma, na, min(va, na), max(va, na)
        printf "B: median %.2f tok/s over %d runs (min %.2f, max %.2f)\n", mb, nb, min(vb, nb), max(vb, nb)
        printf "B / A (medians): %.4f (%+.2f%%)\n", mb / ma, 100 * (mb / ma - 1)
    }
    function min(arr, m,    i, lo) { lo = arr[1]; for (i = 2; i <= m; i++) if (arr[i] < lo) lo = arr[i]; return lo }
    function max(arr, m,    i, hi) { hi = arr[1]; for (i = 2; i <= m; i++) if (arr[i] > hi) hi = arr[i]; return hi }' "$OUT/results.txt"
echo "logs and results: $OUT"
