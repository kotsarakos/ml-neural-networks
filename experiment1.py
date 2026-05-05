import numpy as np
from generate_datasets import generate_binary_problem, generate_flower_problem
from helpers import plot_decision_boundary
from fully_connected_nn import FullyConnectedNN


# Train/test split
def train_test_split(X, y, test_ratio=0.3):
    N = X.shape[0]
    indices = np.random.permutation(N)

    split = int(N * (1 - test_ratio))

    train_idx = indices[:split]
    test_idx = indices[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# Accuracy 
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# 1. BINARY BLOBS
print("\n----- BINARY BLOBS -----")

# Centers
centers = np.array([[0, 8],
                    [0, 8]])

# Generate dataset (N = 500 for each class, so 1000 total)
X, y = generate_binary_problem(centers, N=500)

# Ensure y shape is (N,1)
y = y.reshape(-1, 1)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Model
model = FullyConnectedNN(
    input_size=2,
    hidden_size=10,
    output_size=1,
    lr=0.01
)

# Train
model.fit(X_train, y_train, iterations=5000, batch_size=None, show_step=500)

# Evaluate
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print("Train Accuracy:", accuracy(y_train, train_preds))
print("Test Accuracy:", accuracy(y_test, test_preds))

# Plot decision boundary
plot_decision_boundary(model.predict, X, y.ravel())


# 2. BINARY BLOBS (New centers)
print("\n----- BINARY BLOBS -----")

# New Centers
centers = np.array([[0, 2],
                    [0, 2]])

# Generate dataset (N = 500 for each class, so 1000 total)
X, y = generate_binary_problem(centers, N=500)
y = y.reshape(-1, 1)

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Model
model = FullyConnectedNN(
    input_size=2,
    hidden_size=10,
    output_size=1,
    lr=0.01
)

# Train
model.fit(X_train, y_train, iterations=5000, batch_size=None, show_step=500)

# Evaluate
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print("Train Accuracy:", accuracy(y_train, train_preds))
print("Test Accuracy:", accuracy(y_test, test_preds))

plot_decision_boundary(model.predict, X, y.ravel())


# 3. BINARY BLOBS (Asymmetric centers)
print("\n----- BINARY BLOBS (ASYMMETRIC) -----")

# Centers: class 0 at (0, 0), class 1 at (6, 2) — large gap on x, small on y
centers = np.array([[0, 6],
                    [0, 2]])

# Generate dataset (N = 500 for each class, so 1000 total)
X, y = generate_binary_problem(centers, N=500)
y = y.reshape(-1, 1)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Model
model = FullyConnectedNN(
    input_size=2,
    hidden_size=10,
    output_size=1,
    lr=0.01
)

# Train
model.fit(X_train, y_train, iterations=5000, batch_size=None, show_step=500)

# Evaluate
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print("Train Accuracy:", accuracy(y_train, train_preds))
print("Test Accuracy:", accuracy(y_test, test_preds))

plot_decision_boundary(model.predict, X, y.ravel())


# 4. FLOWER DATASET
print("\n----- FLOWER DATASET -----")

# Generate dataset
X, y = generate_flower_problem()

# Shape: (features, N) --> (N, features)
X = X.T
y = y.reshape(-1, 1)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Model
model = FullyConnectedNN(
    input_size=2,
    hidden_size=10,
    output_size=1,
    lr=0.01
)

# Train
model.fit(X_train, y_train, iterations=10000, batch_size=None, show_step=1000)

# Evaluate
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print("Train Accuracy:", accuracy(y_train, train_preds))
print("Test Accuracy:", accuracy(y_test, test_preds))

plot_decision_boundary(model.predict, X, y.ravel())


# 5. FLOWER DATASET (More neurons: hidden_size=50)
print("\n----- FLOWER DATASET (MORE NEURONS) -----")

X, y = generate_flower_problem()
X = X.T
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = FullyConnectedNN(
    input_size=2,
    hidden_size=50,
    output_size=1,
    lr=0.01
)

model.fit(X_train, y_train, iterations=10000, batch_size=None, show_step=2000)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print("Train Accuracy:", accuracy(y_train, train_preds))
print("Test Accuracy:", accuracy(y_test, test_preds))

plot_decision_boundary(model.predict, X, y.ravel())


# 6. FLOWER DATASET (More iterations: 50000)
print("\n----- FLOWER DATASET (MORE ITERATIONS) -----")

X, y = generate_flower_problem()
X = X.T
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = FullyConnectedNN(
    input_size=2,
    hidden_size=10,
    output_size=1,
    lr=0.01
)

model.fit(X_train, y_train, iterations=50000, batch_size=None, show_step=10000)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print("Train Accuracy:", accuracy(y_train, train_preds))
print("Test Accuracy:", accuracy(y_test, test_preds))

plot_decision_boundary(model.predict, X, y.ravel())


# 7. FLOWER DATASET (More iterations + smaller learning rate)
print("\n----- FLOWER DATASET (MORE ITERATIONS + SMALLER LR) -----")

X, y = generate_flower_problem()
X = X.T
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = FullyConnectedNN(
    input_size=2,
    hidden_size=10,
    output_size=1,
    lr=0.001
)

model.fit(X_train, y_train, iterations=50000, batch_size=None, show_step=10000)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print("Train Accuracy:", accuracy(y_train, train_preds))
print("Test Accuracy:", accuracy(y_test, test_preds))

plot_decision_boundary(model.predict, X, y.ravel())


# 8. FLOWER DATASET (Mini-batch: batch_size=32)
print("\n----- FLOWER DATASET (MINI-BATCH) -----")

X, y = generate_flower_problem()
X = X.T
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = FullyConnectedNN(
    input_size=2,
    hidden_size=10,
    output_size=1,
    lr=0.01
)

model.fit(X_train, y_train, iterations=10000, batch_size=32, show_step=2000)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print("Train Accuracy:", accuracy(y_train, train_preds))
print("Test Accuracy:", accuracy(y_test, test_preds))

plot_decision_boundary(model.predict, X, y.ravel())
