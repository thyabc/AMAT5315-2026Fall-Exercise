
<!-- WEEK6_PART1 -->
# Week 6: Energy-based Models

## Environment
- Python 3.12 (uv virtual environment)
- NumPy, Matplotlib, JAX
- CPU execution

## Part 1: Understand the Datasets

### Dataset sources

**MNIST**
- Source: https://storage.googleapis.com/cvdf-datasets/mnist
- Training: 60,000 images
- Test: 10,000 images
- Resolution: 28 x 28 grayscale
- Pixel values: 0 to 255
- Original train/test splits are preserved.

**CIFAR-10**
- Source: https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz
- Training: 50,000 images
- Test: 10,000 images
- Resolution: 32 x 32 RGB
- Pixel values: 0 to 255
- Original train/test splits are preserved.

The datasets are stored in inputs/ and excluded from Git.

### Reproduce the experiment

From the week6 directory, run:

```bash
uv run --no-project --python .venv/bin/python src/part1_datasets.py
```

This command generates all four figures below.

### Evidence

**1. MNIST image gallery**

![MNIST gallery](evidence/mnist_gallery.png)

The gallery contains two handwritten examples from each digit
class. Different handwriting styles can be observed within a class.
Images are displayed without smoothing interpolation.

**2. MNIST class frequencies**

![MNIST frequencies](evidence/mnist_frequency.png)

MNIST is not perfectly class-balanced. Digit 1 is the most
frequent and digit 5 is the least frequent in both splits.
The training and test distributions are similar but not identical.

**3. CIFAR-10 image gallery**

![CIFAR-10 gallery](evidence/cifar-10_gallery.png)

The gallery displays two RGB images from each object class.
Some fine details are difficult to recognize because the
original resolution is only 32 x 32 pixels.

**4. CIFAR-10 class frequencies**

![CIFAR-10 frequencies](evidence/cifar-10_frequency.png)

All ten classes are exactly balanced.
Each class contains 5,000 training examples and 1,000 test
examples, corresponding to a relative frequency of 0.1.

### Conclusion

MNIST and CIFAR-10 have different image representations
and class distributions. MNIST contains grayscale handwritten
digits with unequal class frequencies, while CIFAR-10 contains
low-resolution color photographs with balanced classes.

The galleries show differences in image appearance, whereas
class-frequency plots describe the empirical label distributions.

These datasets will be reused in later experiments.
