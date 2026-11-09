#!/bin/bash
# Every valid tile config at every shape -> results.csv + autotune_summary.md
#   bash sweep.sh
#
# Rank on `ratio` (kernel ms / cuBLAS ms, both timed in the same process). Raw
# ms is not comparable across a multi-hour run.
#
# BK=8 and TM/TN of 2/2, 2/4, 4/2 are deliberately absent from the lists below:
# BK=8 leaves the A loader only 2 float4 chunks per row of A, and those TM/TN
# give 1.00-1.33 FMAs per shared-memory element against 4.00 for 8/8, the wrong
# direction for a kernel bound by shared-memory throughput.

set -u
cd "$(dirname "$0")" || exit 1

ARCH=sm_120
MIN_ITERS=10        # floor only; the driver calibrates each timed loop to ~100 ms
BUILD=/tmp/gemm_sweep
OUT=results.csv
SUMMARY=autotune_summary.md
SKIPPED=skipped.txt
DEFAULT=BM128_BN64_BK32_TM4_TN4

BMS="48 64 96 128 192 256"
BNS="48 64 96 128 192"
BKS="16 32 64"
TMS="2 4 6 8"
TNS="2 4 6 8"

SHAPES=(
  "32 32 32"        "64 64 64"        "128 128 128"
  "256 256 256"     "512 512 512"     "1024 1024 1024"
  "2048 2048 2048"  "4096 4096 4096"  "8192 8192 8192"
  "8192 128 2048"   "128 8192 2048"   "4096 64 4096"
  "64 4096 4096"    "256 256 8192"    "4096 4096 64"
  "2048 512 1024"   "512 2048 1024"   "1024 4096 256"
  "8192 8192 128"
)

# BK%4, BN%4     global loaders index in float4 chunks
# BM%TM, BN%TN   thread -> tile mapping
# TM%4 -> BM%4   float4 smem read needs 16B-aligned rows (TM%2 -> BM%2)
# threads        32..1024, multiple of 32
# smem           BK*(BM+BN)*4 <= 48 KB, the static shared memory cap
# intensity      TM*TN/(TM+TN) >= 1.5
valid() {
  local BM=$1 BN=$2 BK=$3 TM=$4 TN=$5 t
  (( BK % 4 )) && return 1
  (( BN % 4 )) && return 1
  (( BM % TM )) && return 1
  (( BN % TN )) && return 1
  if   (( TM % 4 == 0 )); then (( BM % 4 )) && return 1
  elif (( TM % 2 == 0 )); then (( BM % 2 )) && return 1; fi
  if   (( TN % 4 == 0 )); then (( BN % 4 )) && return 1
  elif (( TN % 2 == 0 )); then (( BN % 2 )) && return 1; fi
  t=$(( (BM/TM) * (BN/TN) ))
  (( t > 1024 || t < 32 || t % 32 )) && return 1
  (( BK * (BM+BN) * 4 > 49152 )) && return 1
  (( 2 * TM * TN < 3 * (TM + TN) )) && return 1
  return 0
}

CONFIGS=()
for BM in $BMS; do for BN in $BNS; do for BK in $BKS; do
for TM in $TMS; do for TN in $TNS; do
  valid "$BM" "$BN" "$BK" "$TM" "$TN" && CONFIGS+=("$BM,$BN,$BK,$TM,$TN")
done; done; done; done; done

rm -rf "$BUILD"; mkdir -p "$BUILD"; rm -f "$SKIPPED"
trap 'rm -rf "$BUILD"' EXIT        # also covers early exit and Ctrl-C
nvcc -arch=$ARCH -O3 -c cublas_ref.cu -o "$BUILD/cublas_ref.o" || exit 1

TESTS=$(( ${#CONFIGS[@]} * ${#SHAPES[@]} ))
echo "configs=${#CONFIGS[@]}  shapes=${#SHAPES[@]}  tests=$TESTS"
echo "config,M,N,K,ms,cublas_ms,ratio,tflops,max_err,k_iters,c_iters" > $OUT

START=$(date +%s)
n=0
# Shuffled so any slow drift over the run shows up as noise, not as a bias
# toward whichever configs happened to run first.
for cfg in $(printf '%s\n' "${CONFIGS[@]}" | shuf); do
  IFS=, read -r BM BN BK TM TN <<< "$cfg"
  name="BM${BM}_BN${BN}_BK${BK}_TM${TM}_TN${TN}"
  n=$(( n + 1 ))

  # Printed before the work so a hang or crash names the config in flight.
  EL=$(( $(date +%s) - START ))
  printf "  %d / %d  %-28s  %dm elapsed  ~%dm left\n" "$n" "${#CONFIGS[@]}" "$name" \
    $(( EL / 60 )) $(( n > 1 ? EL * (${#CONFIGS[@]} - n + 1) / (n - 1) / 60 : 0 ))

  if ! nvcc -arch=$ARCH -O3 \
            -DCFG_BM=$BM -DCFG_BN=$BN -DCFG_BK=$BK -DCFG_TM=$TM -DCFG_TN=$TN \
            sweep_driver.cu "$BUILD/cublas_ref.o" -lcublas -o "$BUILD/k" 2>/dev/null; then
    echo "compile failed: $name" >> "$SKIPPED"
    continue
  fi

  for shape in "${SHAPES[@]}"; do
    "$BUILD/k" $shape $MIN_ITERS >> $OUT \
      || echo "run failed: $name at $shape" >> "$SKIPPED"
  done
done
ELAPSED=$(( $(date +%s) - START ))

best_for() {
  awk -F, -v m="$1" -v n="$2" -v k="$3" '$2==m && $3==n && $4==k' $OUT \
    | sort -t, -k7,7g | head -1
}

{
  echo "# Autotune Summary"
  echo
  echo "Generated $(date -u '+%Y-%m-%d %H:%M UTC'), $(( ELAPSED / 60 )) min."
  echo
  echo "**$TESTS tests** — ${#CONFIGS[@]} configs x ${#SHAPES[@]} shapes."
  echo "\`ratio\` = kernel ms / cuBLAS ms, both timed in the same process. Below 1.00 beats cuBLAS."
  echo
  echo "## Best config per shape"
  echo
  echo "| M | N | K | best config | ratio | TFLOP/s | max abs err |"
  echo "|---:|---:|---:|---|---:|---:|---:|"
  for shape in "${SHAPES[@]}"; do
    read -r M N K <<< "$shape"
    line=$(best_for "$M" "$N" "$K")
    [ -n "$line" ] && echo "$line" | awk -F, \
      '{printf "| %s | %s | %s | `%s` | %s | %s | %s |\n", $2,$3,$4,$1,$7,$8,$9}'
  done
  echo
  echo "## Best single config overall"
  echo
  echo "| rank | config | mean ratio | shapes |"
  echo "|---:|---|---:|---:|"
  tail -n +2 $OUT | awk -F, '{s[$1]+=$7; c[$1]++} END {for (k in s) printf "%.4f %s %d\n", s[k]/c[k], k, c[k]}' \
    | sort -n | head -10 | awk '{printf "| %d | `%s` | %.4f | %s |\n", NR, $2, $1, $3}'
  echo
  echo "## Current default \`$DEFAULT\`"
  echo
  echo "Baseline the sweep has to beat."
  echo
  echo "| M | N | K | ratio | TFLOP/s |"
  echo "|---:|---:|---:|---:|---:|"
  awk -F, -v d="$DEFAULT" '$1==d {printf "| %s | %s | %s | %s | %s |\n", $2,$3,$4,$7,$8}' $OUT
  echo
  echo "## Grid"
  echo
  echo "| param | values |"
  echo "|---|---|"
  echo "| BM | $BMS |"
  echo "| BN | $BNS |"
  echo "| BK | $BKS |"
  echo "| TM | $TMS |"
  echo "| TN | $TNS |"
  echo "| pruned | \`BK=8\`; \`TM/TN\` 2/2, 2/4, 4/2 |"
  echo
  echo "Per-run data in \`$OUT\`."
} > $SUMMARY

echo
sed -n '/^| M | N | K | best/,/^$/p' $SUMMARY
echo "wrote $SUMMARY and $OUT in $(( ELAPSED / 60 ))m"
if [ -s "$SKIPPED" ]; then
  echo "skipped: $(wc -l < "$SKIPPED") (see $SKIPPED)"
fi
exit 0

# 1. Build list of valid tile configs
# 2. Compile one driver binary per config
# 3. Run that binary on every shape
# 4. Append each result to results.csv
# 5. Pick lowest-ratio config per shape
# 6. Write tables to autotune_summary.md
