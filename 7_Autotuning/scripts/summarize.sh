#!/bin/bash
# Rebuild autotune_summary.md from results.csv. Safe to rerun any time.
#
# Ranks on TFLOP/s (column 8), not ratio (column 7). Ratio = kernel_ms/cublas_ms
# is biased: cuBLAS loses more throughput than this kernel when GPU clocks dip,
# so a config measured during a slow patch gets a flattering ratio. At 4096^3
# cuBLAS readings spanned 14.0-26.8 ms across the run, and the ratio-ranked
# winner there was 371st of 548 by actual speed.

set -u
cd "$(dirname "$0")/.." || exit 1    # stage dir, where results.csv lives
OUT=results.csv
SUMMARY=autotune_summary.md
DEFAULT=BM128_BN64_BK32_TM4_TN4
[ -f $OUT ] || { echo "no $OUT"; exit 1; }

# Shapes in the order they appear for one config, i.e. the sweep's shape order.
mapfile -t SHAPES < <(awk -F, -v d="$DEFAULT" 'NR>1 && $1==d {print $2, $3, $4}' $OUT)
[ "${#SHAPES[@]}" -gt 0 ] || mapfile -t SHAPES < <(awk -F, 'NR>1 {print $2, $3, $4}' $OUT | awk '!s[$0]++')

CFG=$(tail -n +2 $OUT | cut -d, -f1 | sort -u | wc -l)
ROWS=$(( $(wc -l < $OUT) - 1 ))

best_for() {   # M N K -> row with highest TFLOP/s
  awk -F, -v m="$1" -v n="$2" -v k="$3" '$2==m && $3==n && $4==k' $OUT \
    | sort -t, -k8,8gr | head -1
}
default_for() {
  awk -F, -v d="$DEFAULT" -v m="$1" -v n="$2" -v k="$3" \
    '$1==d && $2==m && $3==n && $4==k' $OUT
}

{
  echo "# Autotune Summary"
  echo
  echo "$ROWS measured runs across $CFG configs and ${#SHAPES[@]} shapes."
  echo "Ranked on **TFLOP/s**. \`gain\` is versus \`$DEFAULT\`."
  echo
  echo "## Best config per shape"
  echo
  echo "| M | N | K | best config | ms | TFLOP/s | default TFLOP/s | gain | max abs err |"
  echo "|---:|---:|---:|---|---:|---:|---:|---:|---:|"
  for shape in "${SHAPES[@]}"; do
    read -r M N K <<< "$shape"
    b=$(best_for "$M" "$N" "$K"); d=$(default_for "$M" "$N" "$K")
    [ -n "$b" ] || continue
    bt=$(echo "$b" | cut -d, -f8); dt=$(echo "$d" | cut -d, -f8)
    g="-"
    [ -n "$dt" ] && g="$(echo "scale=1; 100*($bt-$dt)/$dt" | bc)%"
    echo "$b" | awk -F, -v dt="${dt:--}" -v g="$g" \
      '{printf "| %s | %s | %s | `%s` | %s | %s | %s | +%s | %s |\n", $2,$3,$4,$1,$5,$8,dt,g,$9}'
  done
  echo
  echo "## Configs by shapes won"
  echo
  echo "| config | shapes won |"
  echo "|---|---:|"
  for shape in "${SHAPES[@]}"; do
    read -r M N K <<< "$shape"
    best_for "$M" "$N" "$K" | cut -d, -f1
  done | sort | uniq -c | sort -rn | awk '{printf "| `%s` | %s |\n", $2, $1}'
  echo
  echo "## Current default \`$DEFAULT\`"
  echo
  echo "| M | N | K | ms | TFLOP/s |"
  echo "|---:|---:|---:|---:|---:|"
  awk -F, -v d="$DEFAULT" '$1==d {printf "| %s | %s | %s | %s | %s |\n", $2,$3,$4,$5,$8}' $OUT
  echo
  echo "## Grid"
  echo
  echo "| param | values |"
  echo "|---|---|"
  for p in BM BN BK TM TN; do
    v=$(tail -n +2 $OUT | cut -d, -f1 | sed -n "s/.*${p}\([0-9]\+\).*/\1/p" | sort -n -u | tr '\n' ' ')
    echo "| $p | $v|"
  done
  echo
  echo "Per-run data in \`$OUT\`. Ratio column retained there but not used for ranking."
} > $SUMMARY

echo "wrote $SUMMARY"
sed -n '/^## Configs by shapes won/,/^## Current/p' $SUMMARY
