# Neural Networks: From Scratch (NumPy) vs PyTorch

**Course:** Machine Learning and Applications  
**Institution:** Harokopio University of Athens — Department of Informatics and Telematics  
**Author:** Konstantinos Kotsaras

---

## Overview

This project implements a fully connected neural network for binary classification from scratch using NumPy, and compares it against an equivalent PyTorch implementation. The primary objective is to develop a deep understanding of neural network internals — including forward propagation, backpropagation, and gradient descent — while also evaluating the practical trade-offs between a manual implementation and a modern deep learning framework.

Experiments are conducted on both synthetic and real-world datasets, covering a range of problem complexities to assess model performance, training stability, and generalization capability.

---

## Project Structure

```
.
├── fully_connected_nn.py   # NumPy neural network implementation
├── experiment1.py          # Synthetic datasets: Blobs and Flower
├── experiment2.py          # Breast Cancer dataset (NumPy model, 20 runs)
├── experiment3.py          # PyTorch model comparison and architecture experiments
├── generate_datasets.py    # Synthetic dataset generators
├── helpers.py              # Plotting utilities
└── README.md
```

---

## Datasets

| Dataset | Type | Task |
|---|---|---|
| Binary Blobs | Synthetic | Linearly separable binary classification |
| Flower | Synthetic | Non-linearly separable binary classification |
| Breast Cancer Wisconsin | Real-world (scikit-learn) | Binary classification (569 samples, 30 features) |

---

## Experiments

### Experiment 1 — Synthetic Datasets (NumPy)

Trains and evaluates the NumPy neural network on multiple configurations of the Blobs and Flower datasets. Examines the effect of:

- Class separation distance (blob center spacing)
- Number of hidden neurons
- Number of training iterations
- Learning rate
- Mini-batch vs full-batch gradient descent

Decision boundaries are visualized for each configuration.

### Experiment 2 — Breast Cancer Dataset (NumPy)

Applies the NumPy model to the Breast Cancer Wisconsin dataset over 20 independent runs. Each run uses a random 70/30 train/test split with min-max normalization fitted on the training set. Reports mean accuracy, standard deviation, and total training time.

**Configuration:** hidden=16, lr=0.01, batch\_size=64, iterations=5000

### Experiment 3 — PyTorch Comparison and Architecture Study

**Part 1:** Reimplements the same model as Experiment 2 using PyTorch (`nn.Module`, `optim.SGD`, `nn.BCELoss`) under identical conditions (hidden=16, lr=0.01, batch\_size=64, 800 epochs, 20 runs). Enables a direct comparison of accuracy, stability, and training time between the two implementations.

**Part 2:** Evaluates three distinct architectures across all three datasets (Blobs, Flower, Breast Cancer):

| Architecture | Description |
|---|---|
| WideNN | 1 hidden layer, 64 neurons, ReLU |
| DeepNN | 2 hidden layers (32 → 16), ReLU |
| TanhNN | 1 hidden layer, 16 neurons, Tanh |

---

## Key Concepts

- Forward propagation and backpropagation
- Binary cross-entropy (BCE) loss
- Stochastic gradient descent (SGD) with mini-batches
- Min-max normalization (fit on training set only)
- Repeated evaluation for statistical reliability (mean / std over 20 runs)
- Architecture comparison: width, depth, and activation functions

---

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- scikit-learn
- PyTorch

Install dependencies:

```bash
pip install numpy matplotlib scikit-learn torch
```

---

## How to Run

```bash
python experiment1.py   # Synthetic datasets
python experiment2.py   # Breast Cancer (NumPy)
python experiment3.py   # PyTorch comparison and architecture experiments
```

---

## Notes

- All normalization uses training set statistics exclusively to prevent data leakage.
- Epoch count in Experiment 3 (800) is calibrated to match the total number of gradient updates in Experiment 2 (5000 iterations), ensuring a fair comparison.
- The NumPy model is significantly faster than the PyTorch equivalent for this network size due to lower per-operation overhead.
