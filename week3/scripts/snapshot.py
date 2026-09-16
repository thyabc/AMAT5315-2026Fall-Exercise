import json
import numpy as np
import matplotlib.pyplot as plt
import os


def read_spin_configuration(path, target_T):

    last_spin = None
    L = None
    count = 0

    with open(path, "r") as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            # 浮点数容差
            if abs(row["T"] - target_T) < 1e-6:

                last_spin = row["spins"]
                L = row["L"]

                count += 1


    if last_spin is None:
        print("Cannot find T =", target_T)
        return None


    print(
        "T =", target_T,
        "samples =", count,
        "L =", L
    )


    # 一维数组转二维
    spins = np.array(last_spin)

    spins = spins.reshape((L,L))


    return spins



# 数据文件

file = "artifacts/window-l64/spins.jsonl"



temperatures = [
    2.0,
    2.3,
    2.5
]


fig, axes = plt.subplots(
    1,
    3,
    figsize=(12,4)
)



for ax,T in zip(axes,temperatures):

    spins = read_spin_configuration(
        file,
        T
    )


    if spins is None:
        continue


    im=ax.imshow(
        spins,
        cmap="gray",
        vmin=-1,
        vmax=1
    )


    ax.set_title(
        f"T={T}"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")



plt.suptitle(
    "Spin configuration snapshots (L=64)"
)


plt.tight_layout()



# 创建输出文件夹

os.makedirs(
    "evidence",
    exist_ok=True
)



plt.savefig(
    "evidence/spin_snapshots.png",
    dpi=300,
    bbox_inches="tight"
)



plt.show()