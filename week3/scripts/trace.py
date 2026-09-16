import json
import matplotlib.pyplot as plt
import os


def read_series(path, target_T):

    M = []

    with open(path, "r") as f:
        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            # 浮点数容差
            if abs(row["T"] - target_T) < 1e-6:
                M.append(row["M"])

    return M



file = "artifacts/window-l64/series.jsonl"


print("Reading data...")


# below Tc
M_low = read_series(
    file,
    2.25
)


# above Tc
M_high = read_series(
    file,
    2.50
)


print("T=2.25 points:",len(M_low))
print("T=2.50 points:",len(M_high))


plt.figure(figsize=(10,4))


plt.plot(
    M_low,
    label="T=2.25 (<Tc)"
)


plt.plot(
    M_high,
    label="T=2.50 (>Tc)"
)


plt.xlabel("measurement sweep")

plt.ylabel("|M|")

plt.title(
    "Magnetization time trace"
)


plt.legend()

plt.grid(True)


# 创建保存目录
os.makedirs(
    "evidence",
    exist_ok=True
)


plt.savefig(
    "evidence/trace.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()