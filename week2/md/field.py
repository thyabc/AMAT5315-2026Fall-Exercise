import numpy as np
import matplotlib.pyplot as plt

r = np.linspace(0.8, 3.0, 500)

# Lennard-Jones potential
V = 4*((1/r)**12 - (1/r)**6)

plt.figure(figsize=(6,4))
plt.plot(r, V)

plt.xlabel("r")
plt.ylabel("V(r)")
plt.title("Lennard-Jones potential")

plt.grid(True)
plt.savefig("field.png", dpi=300)
plt.close()
