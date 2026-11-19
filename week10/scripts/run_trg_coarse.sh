#!/usr/bin/env bash
set -euo pipefail

L="${1:?Usage: ./scripts/run_trg_coarse.sh 64|256|1024}"

if [[ "$L" != "64" && "$L" != "256" && "$L" != "1024" ]]; then
    echo "L must be 64, 256 or 1024"
    exit 1
fi

ROOT="runs/trg_scan"
CSV="$ROOT/coarse_L${L}_chi16.csv"

mkdir -p "$ROOT"

TMP="${CSV}.tmp"
echo "L,K,chi,g,log_Z,solver_seconds" > "$TMP"

for i in $(seq 0 40); do
    K=$(awk -v i="$i" 'BEGIN {printf "%.4f", 0.40+0.0025*i}')
    TAG="${K/./p}"
    OUT="$ROOT/L${L}_K${TAG}_chi16"

    echo "======================================"
    echo "TRG coarse scan: L=$L K=$K chi=16"
    echo "======================================"

    if [[ ! -f "$OUT/summary.txt" ]]; then
        cargo run --release --bin thermo -- \
            --method trg \
            --l "$L" \
            --k "$K" \
            --chi 16 \
            --out "$OUT"
    else
        echo "Existing result found, skipping: $OUT"
    fi

    if [[ ! -f "$OUT/summary.txt" ]]; then
        echo "ERROR: summary.txt missing in $OUT"
        exit 1
    fi

    G=$(awk -F': *' '$1=="g"{print $2; exit}' "$OUT/summary.txt")
    LOGZ=$(awk -F': *' '$1=="log_Z"{print $2; exit}' "$OUT/summary.txt")
    SEC=$(awk -F': *' '$1=="solver_seconds"{print $2; exit}' "$OUT/summary.txt")

    if [[ -z "$G" || -z "$LOGZ" || -z "$SEC" ]]; then
        echo "ERROR: Cannot parse summary for K=$K"
        exit 1
    fi

    echo "$L,$K,16,$G,$LOGZ,$SEC" >> "$TMP"
done

mv "$TMP" "$CSV"

echo "======================================"
echo "SCAN COMPLETED"
echo "Saved: $CSV"
echo "======================================"

wc -l "$CSV"
head -5 "$CSV"
tail -5 "$CSV"
