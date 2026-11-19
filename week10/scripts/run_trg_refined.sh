#!/usr/bin/env bash
set -euo pipefail

L="${1:?Usage: ./scripts/run_trg_refined.sh 64|256|1024}"

if [[ "$L" != "64" && "$L" != "256" && "$L" != "1024" ]]; then
    echo "L must be 64, 256 or 1024"
    exit 1
fi

CENTER=$(awk -F',' -v L="$L" \
    'NR>1 && $1==L {print $2; exit}' \
    runs/trg_scan/coarse_peak_estimates.csv)

if [[ -z "$CENTER" ]]; then
    echo "ERROR: Cannot find coarse peak for L=$L"
    exit 1
fi

mkdir -p runs/trg_scan

echo "===================================="
echo "Refined TRG scan"
echo "L = $L"
echo "Center K = $CENTER"
echo "Step = 0.00125"
echo "===================================="

cargo build --release --bin thermo

for CHI in 16 32; do

    CSV="runs/trg_scan/refined_L${L}_chi${CHI}.csv"
    TMP="${CSV}.tmp"

    echo "L,K,chi,g,log_Z,solver_seconds" > "$TMP"

    for j in -4 -3 -2 -1 0 1 2 3 4; do

        K=$(awk -v c="$CENTER" -v j="$j" \
            'BEGIN {printf "%.5f", c+j*0.00125}')

        TAG="${K/./p}"
        OUT="runs/trg_scan/L${L}_K${TAG}_chi${CHI}"

        echo "------------------------------------"
        echo "L=$L K=$K chi=$CHI"

        if [[ ! -f "$OUT/summary.txt" ]]; then
            ./target/release/thermo \
                --method trg \
                --l "$L" \
                --k "$K" \
                --chi "$CHI" \
                --out "$OUT"
        else
            echo "Existing result found, skipping."
        fi

        if [[ ! -f "$OUT/summary.txt" ]]; then
            echo "ERROR: Missing summary.txt"
            exit 1
        fi

        G=$(awk -F': *' '$1=="g"{print $2; exit}' "$OUT/summary.txt")
        LOGZ=$(awk -F': *' '$1=="log_Z"{print $2; exit}' "$OUT/summary.txt")
        SEC=$(awk -F': *' '$1=="solver_seconds"{print $2; exit}' "$OUT/summary.txt")

        if [[ -z "$G" || -z "$LOGZ" || -z "$SEC" ]]; then
            echo "ERROR: Cannot parse summary"
            exit 1
        fi

        echo "$L,$K,$CHI,$G,$LOGZ,$SEC" >> "$TMP"

    done

    mv "$TMP" "$CSV"

    echo "Completed: $CSV"
    wc -l "$CSV"

done

echo "===================================="
echo "REFINED SCAN COMPLETED: L=$L"
echo "===================================="
