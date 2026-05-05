import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_breast_cancer
from generate_datasets import generate_binary_problem, generate_flower_problem


# Train/Test Split
def train_test_split(X, y, test_ratio=0.3):
    N = X.shape[0]
    indices = np.random.permutation(N)
    split = int(N * (1 - test_ratio))
    return X[indices[:split]], X[indices[split:]], y[indices[:split]], y[indices[split:]]

# Accuracy
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Normalization
def normalize(X_train, X_test):
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    denom = X_max - X_min
    denom[denom == 0] = 1
    return (X_train - X_min) / denom, (X_test - X_min) / denom


# Training
def train_model(model, X_train, y_train, epochs=800, lr=0.01, batch_size=64, show_step=None):
    model.train()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    N = X_train.shape[0]

    for epoch in range(epochs):
        indices = np.random.permutation(N)
        for start in range(0, N, batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch = torch.tensor(X_train[batch_idx], dtype=torch.float32)
            y_batch = torch.tensor(y_train[batch_idx], dtype=torch.float32)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        if show_step is not None and epoch % show_step == 0:
            model.eval()
            with torch.no_grad():
                X_t = torch.tensor(X_train, dtype=torch.float32)
                y_t = torch.tensor(y_train, dtype=torch.float32)
                current_loss = criterion(model(X_t), y_t).item()
            print(f"  Epoch {epoch}, Loss: {current_loss:.5f}")
            model.train()


# Evaluation
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        probs = model(X_t).numpy()
    preds = (probs >= 0.5).astype(int)
    return accuracy(y, preds)


# Experiment function to run multiple runs and report mean/std
def run_experiment(model_class, X, y, label, runs=20, epochs=800, lr=0.01, batch_size=64):
    print(f"\n  [{label}]")
    accuracies = []
    for _ in range(runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        X_train_n, X_test_n = normalize(X_train, X_test)

        model = model_class(X.shape[1])
        train_model(model, X_train_n, y_train, epochs=epochs, lr=lr, batch_size=batch_size)

        acc = evaluate(model, X_test_n, y_test)
        accuracies.append(acc)

    print(f"  Mean: {np.mean(accuracies):.4f} | Std: {np.std(accuracies):.4f}")
    return np.mean(accuracies), np.std(accuracies)


# ==================================================
# PART 1: Same as experiment2 but with PyTorch
# (SGD, lr=0.01, batch_size=64, 800 epochs, 20 runs)
# ==================================================

print("\n------ PyTorch - Breast Cancer (compare with experiment2) ------")

data = load_breast_cancer()
X_bc = data.data
y_bc = data.target.reshape(-1, 1)


class TorchNN(nn.Module):
    def __init__(self, input_size, hidden_size=16):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


accuracies_part1 = []
start_time = time.time()

for i in range(20):
    print(f"\n--- Run {i+1} ---")

    X_train, X_test, y_train, y_test = train_test_split(X_bc, y_bc)
    X_train, X_test = normalize(X_train, X_test)

    model = TorchNN(input_size=X_bc.shape[1], hidden_size=16)
    train_model(model, X_train, y_train, epochs=800, lr=0.01, batch_size=64, show_step=100)

    acc = evaluate(model, X_test, y_test)
    accuracies_part1.append(acc)
    print(f"Test Accuracy: {acc}")

end_time = time.time()

print("\n---- RESULTS (PyTorch - compare with experiment2) ----")
print("Mean Accuracy:", np.mean(accuracies_part1))
print("Std Accuracy:", np.std(accuracies_part1))
print("Total Time (seconds):", end_time - start_time)


# ==================================================
# PART 2: Architecture experiments
# Tested on: Blobs, Flower, Breast Cancer
# ==================================================

# --- Architecture definitions ---

class WideNN(nn.Module):
    """1 hidden layer, 64 neurons, ReLU"""
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class DeepNN(nn.Module):
    """2 hidden layers: input → 32 → 16 → output, ReLU"""
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class TanhNN(nn.Module):
    """1 hidden layer, 16 neurons, Tanh activation"""
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


architectures = [
    (WideNN,  "WideNN  — 1 hidden layer, 64 neurons, ReLU"),
    (DeepNN,  "DeepNN  — 2 hidden layers (32→16), ReLU"),
    (TanhNN,  "TanhNN  — 1 hidden layer, 16 neurons, Tanh"),
]

# --- Dataset preparation ---

# Blobs
centers = np.array([[0, 8], [0, 8]])
X_blobs, y_blobs = generate_binary_problem(centers, N=500)
y_blobs = y_blobs.reshape(-1, 1)

# Flower
X_flower, y_flower = generate_flower_problem()
X_flower = X_flower.T
y_flower = y_flower.reshape(-1, 1)


# --- Run architecture experiments ---

print("\n------ A: Architecture Experiments — Blobs ------")
for model_class, label in architectures:
    run_experiment(model_class, X_blobs, y_blobs, label, runs=20, epochs=300)

print("\n------ B: Architecture Experiments — Flower ------")
for model_class, label in architectures:
    run_experiment(model_class, X_flower, y_flower, label, runs=20, epochs=1000)

print("\n------ C: Architecture Experiments — Breast Cancer ------")
for model_class, label in architectures:
    run_experiment(model_class, X_bc, y_bc, label, runs=20, epochs=800)
