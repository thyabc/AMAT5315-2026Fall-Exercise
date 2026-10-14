from pathlib import Path
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from part1_datasets import load_mnist

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence"
OUT.mkdir(exist_ok=True)

def main():
    images, labels = load_mnist("train")

    # Convert grayscale images to binary spins
    sample = jnp.asarray(images[:16])
    binary = (sample >= 128).astype(jnp.int8)
    spins = np.asarray(2 * binary - 1)

    assert spins.shape == (16, 28, 28)
    assert np.all(np.isin(spins, [-1, 1]))

    print("Original images:", images.shape)
    print("Binary spin array:", spins.shape)
    print("Visible units:", 28 * 28)
    print("Hidden units:", 128)
    print("Spin values:", np.unique(spins))

    # Figure 1: Binarized MNIST and enlarged spin image
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(
        4, 6,
        width_ratios=[1, 1, 1, 1, 1.5, 1.5],
        wspace=0.2,
        hspace=0.4
    )

    for i in range(16):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ax.imshow(
            spins[i],
            cmap="gray",
            vmin=-1,
            vmax=1,
            interpolation="nearest"
        )
        ax.set_title(f"Digit {int(labels[i])}", fontsize=9)
        ax.axis("off")

    ax_big = fig.add_subplot(gs[:, 4:])
    im = ax_big.imshow(
        spins[0],
        cmap="gray",
        vmin=-1,
        vmax=1,
        interpolation="nearest"
    )
    ax_big.set_title("First image as spins", fontsize=12)
    ax_big.axis("off")

    cbar = fig.colorbar(
        im, ax=ax_big,
        ticks=[-1, 1],
        fraction=0.045,
        pad=0.03
    )
    cbar.set_label("Spin value")

    fig.suptitle(
        "Binarized MNIST: First 16 Training Images",
        fontsize=15
    )

    fig.savefig(
        OUT / "part2_spin_gallery.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    # Figure 2: RBM architecture schematic
    fig, ax = plt.subplots(figsize=(12, 5))

    x_visible = np.linspace(0.8, 9.2, 9)
    x_hidden = np.linspace(1.4, 8.6, 6)

    y_visible = 0.22
    y_hidden = 0.75

    # Draw visible-hidden connections
    for xv in x_visible:
        for xh in x_hidden:
            ax.plot(
                [xv, xh],
                [y_visible, y_hidden],
                color="gray",
                alpha=0.18,
                linewidth=0.8,
                zorder=1
            )

    ax.scatter(
        x_visible,
        np.full(len(x_visible), y_visible),
        s=480,
        color="#2563eb",
        edgecolors="black",
        zorder=2
    )

    ax.scatter(
        x_hidden,
        np.full(len(x_hidden), y_hidden),
        s=480,
        color="#f59e0b",
        edgecolors="black",
        zorder=2
    )

    for i, x in enumerate(x_visible):
        ax.text(
            x, y_visible, f"v{i+1}",
            ha="center", va="center",
            fontsize=8, color="white",
            zorder=3
        )

    for j, x in enumerate(x_hidden):
        ax.text(
            x, y_hidden, f"h{j+1}",
            ha="center", va="center",
            fontsize=8, zorder=3
        )

    ax.text(
        5, 0.93,
        "Hidden layer: 128 binary units",
        ha="center", fontsize=13
    )

    ax.text(
        5, 0.05,
        "Visible layer: 784 pixel bits (28 x 28)",
        ha="center", fontsize=13
    )

    ax.set_title(
        "Restricted Boltzmann Machine (schematic)",
        fontsize=15
    )
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.05)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(
        OUT / "part2_rbm_structure.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("\nPart 2 initial verification PASSED")
    print("Saved: evidence/part2_spin_gallery.png")
    print("Saved: evidence/part2_rbm_structure.png")
    print("Note: network diagram shows only a subset of nodes.")

if __name__ == "__main__":
    main()
