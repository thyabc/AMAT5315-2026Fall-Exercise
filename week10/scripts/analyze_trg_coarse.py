import csv
import os

sizes = [64, 256, 1024]
results = []

for L in sizes:
    path = f"runs/trg_scan/coarse_L{L}_chi16.csv"

    if not os.path.exists(path):
        print(f"ERROR: Missing {path}")
        raise SystemExit(1)

    with open(path) as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 41:
        print(f"ERROR: L={L}, expected 41 points, got {len(rows)}")
        raise SystemExit(1)

    K = [float(r["K"]) for r in rows]
    g = [float(r["g"]) for r in rows]

    h = K[1] - K[0]
    cv = []

    for i in range(1, len(K)-1):
        value = K[i]**2 * (
            g[i+1] - 2*g[i] + g[i-1]
        ) / h**2

        cv.append((K[i], value))

    peak_K, peak_cv = max(cv, key=lambda x: x[1])
    peak_T = 1.0 / peak_K

    results.append({
        "L": L,
        "K_peak": peak_K,
        "T_peak": peak_T,
        "cV_peak": peak_cv
    })

    print(f"\nL = {L}")
    print(f"Coarse peak K = {peak_K:.7f}")
    print(f"Coarse peak T = {peak_T:.7f}")
    print(f"Coarse peak cV = {peak_cv:.7f}")

os.makedirs("runs/trg_scan", exist_ok=True)

with open("runs/trg_scan/coarse_peak_estimates.csv", "w") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["L", "K_peak", "T_peak", "cV_peak"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nSaved: runs/trg_scan/coarse_peak_estimates.csv")
