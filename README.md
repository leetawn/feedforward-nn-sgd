# Feedforward Neural Network trained via Stochastic Gradient Descent in C

A lightweight, zero-dependency feedforward neural network to classify handwritten digits from the MNIST dataset.

All technical details, math, and architecture explanations can be found in the `docs` folder.

---

## Quick Start

### 1. Prerequisites

Make sure you have a C compiler (like `gcc`) and `make` installed on your system.

### 2. Run the Project

Run the following commands in your terminal:

```bash
git clone https://github.com/leetawn/feedforward-nn-sgd
cd feedforward-nn-sgd
make
make run
```

---

## Note

- **MNIST Alignment:** The network is trained on the standard MNIST dataset. Because of this, accuracy depends heavily on how closely your hand-drawn digits match the style, thickness, and centering of the original training images.
- **Brush Radius:** The brush size is currently locked to a specific radius that yields the most optimal results. I might add a slider for the brush radius but this will do for now.
