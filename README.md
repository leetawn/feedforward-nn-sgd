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

## Configuration & Hyperparameters

To change the hyperparameters, open `include/config.h` and edit the following values directly:

```c
#define LEARNING_RATE 0.01
#define EPOCHS 1
#define VALIDATION_SET_SIZE 5000
```

The epoch hyperparameter is currently set to 1 for faster debug purposes (although it already suffices for basic functionality). Higher epoch values would generally result into higher accuracy, but only up to a point, experimentation is highly encouraged.

## Testing Single Images

Once training completes, the program will prompt you in the terminal to test the network on a specific digit. 

If you select **Y**:
1. Enter the target digit (0–9) you want to look up.
2. The application will search the test dataset for a matching sample.
3. It will run a forward pass to predict the digit and output its confidence percentages.
4. It will automatically export the tested sample as a grayscale image to `out/test.png` so you can visually verify the prediction.
