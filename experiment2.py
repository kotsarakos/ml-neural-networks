import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from fully_connected_nn import FullyConnectedNN


# Train/Test Split
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


# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target.reshape(-1, 1)  # (N,) --> (N,1)

accuracies = []

start_time = time.time()

for i in range(20):
    print(f"\n--- Run {i+1} ---")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Normalize 
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)

    denom = X_max - X_min

    # Avoid division by zero
    denom[denom == 0] = 1

    # Normalize train and test using the same parameters, scaling based on the training set
    # This ensures consistent scaling and prevents data leakage
    X_train = (X_train - X_min) / denom
    X_test  = (X_test  - X_min) / denom

    # Model
    model = FullyConnectedNN(
        input_size=X.shape[1], # We take the number of features from the dataset
        hidden_size=16,
        output_size=1,
        lr=0.01
    )

    # Train
    model.fit(X_train, y_train, iterations=5000, batch_size=64, show_step=1000)
    
    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy(y_test, preds)

    print("Test Accuracy:", acc)
    accuracies.append(acc)

end_time = time.time()

accuracies = np.array(accuracies)

mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
total_time = end_time - start_time

print("\n---- FINAL RESULTS ----")
print("Mean Accuracy:", mean_acc)
print("Std Accuracy:", std_acc)
print("Total Time (seconds):", total_time)