import json
import os
import numpy as np
import matplotlib.pyplot as plt


def read_series(folder):

    filename = os.path.join(folder, "series.jsonl")

    data = {}

    with open(filename, "r") as f:
        for line in f:
            row = json.loads(line)

            T = row["T"]
            M = row["M"]

            if T not in data:
                data[T] = []

            data[T].append(M)

    temps = []
    mean_absM = []
    chi = []

    L = None

    for T in sorted(data.keys()):

        M = np.array(data[T])

        # 从第一行恢复L
        if L is None:
            L = row["L"]

        m_abs = np.mean(np.abs(M))

        susceptibility = (
            L**2 *
            (
                np.mean(M**2)
                -
                np.mean(np.abs(M))**2
            )
            /
            T
        )

        temps.append(T)
        mean_absM.append(m_abs)
        chi.append(susceptibility)

    return (
        np.array(temps),
        np.array(mean_absM),
        np.array(chi)
    )


# ======================
# 读取四组数据
# ======================

runs = {
    "L=32": "artifacts/window-l32",
    "L=64": "artifacts/window-l64"
}


results = {}

for name, folder in runs.items():

    print("Reading", folder)

    results[name] = read_series(folder)



# ======================
# 图1 magnetization
# ======================

plt.figure(figsize=(7,5))


for name, value in results.items():

    T, M, chi = value

    plt.plot(
        T,
        M,
        marker="o",
        label=name
    )


plt.xlabel("Temperature T")
plt.ylabel("<|M|>")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "evidence/magnetization.png",
    dpi=300
)

plt.close()



# ======================
# 图2 susceptibility
# ======================


plt.figure(figsize=(7,5))


for name, value in results.items():

    T, M, chi = value

    plt.plot(
        T,
        chi,
        marker="o",
        label=name
    )


plt.xlabel("Temperature T")
plt.ylabel("Susceptibility")


plt.legend()

plt.grid(True)

plt.tight_layout()


plt.savefig(
    "evidence/susceptibility.png",
    dpi=300
)


plt.close()


print("Done!")