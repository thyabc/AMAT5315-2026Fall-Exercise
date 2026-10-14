from pathlib import Path
import gzip
import json
import struct
import tarfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "inputs"
EVIDENCE = ROOT / "evidence"
ARTIFACTS = ROOT / "artifacts"

EVIDENCE.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)

CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_mnist(split):
    prefix = "train" if split == "train" else "t10k"

    image_path = INPUT / "mnist" / f"{prefix}-images-idx3-ubyte.gz"
    label_path = INPUT / "mnist" / f"{prefix}-labels-idx1-ubyte.gz"

    with gzip.open(image_path, "rb") as f:
        raw = f.read()

    magic, n, height, width = struct.unpack(">IIII", raw[:16])
    assert magic == 2051
    assert (height, width) == (28, 28)

    images = np.frombuffer(raw, dtype=np.uint8, offset=16)
    assert images.size == n * height * width
    images = images.reshape(n, height, width)

    with gzip.open(label_path, "rb") as f:
        raw_labels = f.read()

    label_magic, label_count = struct.unpack(">II", raw_labels[:8])
    assert label_magic == 2049
    assert label_count == n

    labels = np.frombuffer(
        raw_labels, dtype=np.uint8, offset=8
    )
    assert labels.size == n

    return images, labels


def load_cifar(split):
    archive = INPUT / "cifar10" / "cifar-10-binary.tar.gz"

    if split == "train":
        filenames = [f"data_batch_{i}.bin" for i in range(1, 6)]
    else:
        filenames = ["test_batch.bin"]

    all_images = []
    all_labels = []

    with tarfile.open(archive, "r:gz") as tar:
        for filename in filenames:
            member = tar.extractfile(
                f"cifar-10-batches-bin/{filename}"
            )
            if member is None:
                raise FileNotFoundError(filename)

            raw = member.read()
            data = np.frombuffer(raw, dtype=np.uint8)

            assert data.size % 3073 == 0
            data = data.reshape(-1, 3073)

            labels = data[:, 0].copy()

            images = data[:, 1:].reshape(
                -1, 3, 32, 32
            ).transpose(0, 2, 3, 1)

            all_images.append(images)
            all_labels.append(labels)

    return (
        np.concatenate(all_images, axis=0),
        np.concatenate(all_labels, axis=0)
    )


def verify_dataset(name, train, test):
    x_train, y_train = train
    x_test, y_test = test

    if name == "MNIST":
        assert x_train.shape == (60000, 28, 28)
        assert x_test.shape == (10000, 28, 28)
    else:
        assert x_train.shape == (50000, 32, 32, 3)
        assert x_test.shape == (10000, 32, 32, 3)

    assert x_train.dtype == np.uint8
    assert x_test.dtype == np.uint8
    assert y_train.min() >= 0 and y_train.max() <= 9
    assert y_test.min() >= 0 and y_test.max() <= 9

    train_counts = np.bincount(y_train, minlength=10)
    test_counts = np.bincount(y_test, minlength=10)

    assert train_counts.sum() == len(y_train)
    assert test_counts.sum() == len(y_test)

    if name == "MNIST":
        assert np.argmax(train_counts) == 1
        assert np.argmin(train_counts) == 5
        assert np.argmax(test_counts) == 1
        assert np.argmin(test_counts) == 5
    else:
        assert np.all(train_counts == 5000)
        assert np.all(test_counts == 1000)

    print(f"\n===== {name} verification =====")
    print("Training images:", len(y_train))
    print("Test images:", len(y_test))
    print("Training image shape:", x_train.shape[1:])
    print("Pixel dtype:", x_train.dtype)
    print("Class | Train count | Test count")

    for i in range(10):
        print(
            f"{i:5d} | {train_counts[i]:11d} | "
            f"{test_counts[i]:10d}"
        )

    print(f"{name}: PASSED")

    return {
        "train_total": len(y_train),
        "test_total": len(y_test),
        "train_counts": train_counts.tolist(),
        "test_counts": test_counts.tolist()
    }


def plot_gallery(name, images, labels, class_names):
    fig, axes = plt.subplots(2, 10, figsize=(16, 4))

    for c in range(10):
        indices = np.flatnonzero(labels == c)[:2]
        assert len(indices) == 2

        for row in range(2):
            ax = axes[row, c]
            image = images[indices[row]]

            if name == "MNIST":
                ax.imshow(
                    image, cmap="gray",
                    vmin=0, vmax=255,
                    interpolation="nearest"
                )
            else:
                ax.imshow(
                    image, interpolation="nearest"
                )

            ax.axis("off")

            if row == 0:
                ax.set_title(class_names[c], fontsize=9)

    fig.suptitle(
        f"{name}: Training Image Gallery",
        fontsize=15
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    output = EVIDENCE / f"{name.lower()}_gallery.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output)


def plot_frequencies(name, train_labels, test_labels, class_names):
    train_counts = np.bincount(train_labels, minlength=10)
    test_counts = np.bincount(test_labels, minlength=10)

    train_freq = train_counts / len(train_labels)
    test_freq = test_counts / len(test_labels)

    x = np.arange(10)
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar(
        x - 0.2, train_freq, width=0.4,
        label="Training"
    )
    ax.bar(
        x + 0.2, test_freq, width=0.4,
        label="Test"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_ylabel("Relative Class Frequency")
    ax.set_xlabel("Class")
    ax.set_title(f"{name}: Training vs Test Class Frequencies")
    ax.set_ylim(0, max(train_freq.max(), test_freq.max()) * 1.2)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    output = EVIDENCE / f"{name.lower()}_frequency.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output)


def main():
    print("Loading MNIST...")
    mnist_train = load_mnist("train")
    mnist_test = load_mnist("test")

    mnist_info = verify_dataset(
        "MNIST", mnist_train, mnist_test
    )

    digit_names = [str(i) for i in range(10)]

    plot_gallery(
        "MNIST", mnist_train[0],
        mnist_train[1], digit_names
    )
    plot_frequencies(
        "MNIST", mnist_train[1],
        mnist_test[1], digit_names
    )

    print("\nLoading CIFAR-10...")
    cifar_train = load_cifar("train")
    cifar_test = load_cifar("test")

    cifar_info = verify_dataset(
        "CIFAR-10", cifar_train, cifar_test
    )

    plot_gallery(
        "CIFAR-10", cifar_train[0],
        cifar_train[1], CIFAR_CLASSES
    )
    plot_frequencies(
        "CIFAR-10", cifar_train[1],
        cifar_test[1], CIFAR_CLASSES
    )

    summary = {
        "MNIST": mnist_info,
        "CIFAR-10": cifar_info
    }

    with open(
        ARTIFACTS / "part1_summary.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print("\nAll Part 1 checks PASSED.")
    print("Four figures saved to evidence/")


if __name__ == "__main__":
    main()
