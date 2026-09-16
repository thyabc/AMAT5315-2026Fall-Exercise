import json
import numpy as np
import matplotlib.pyplot as plt


def read_energy(path):
    E=[]
    with open(path) as f:
        for line in f:
            row=json.loads(line)
            E.append(row["E"])
    return np.array(E)


E30=read_energy(
"runs/T3.0/series.jsonl"
)

E31=read_energy(
"runs/T3.1/series.jsonl"
)


# 注意：
# series里面E是energy per site
# 转换total energy

E30=E30*64*64
E31=E31*64*64


bins=np.arange(
min(E30.min(),E31.min()),
max(E30.max(),E31.max()),
40
)


h30,_=np.histogram(E30,bins=bins)
h31,_=np.histogram(E31,bins=bins)


centers=(bins[:-1]+bins[1:])/2


mask=(h30>5)&(h31>5)


ratio=np.log(
h31[mask]/h30[mask]
)


plt.figure(figsize=(6,4))

plt.plot(
centers[mask],
ratio,
'o',
label="simulation"
)


# 理论斜率
slope=1/3.0-1/3.1

x=centers[mask]

plt.plot(
x,
slope*x,
label="theory"
)


plt.xlabel("Energy E")
plt.ylabel(
"ln(P3.1/P3.0)"
)

plt.legend()

plt.savefig(
"evidence/boltzmann.png",
dpi=300
)

plt.show()