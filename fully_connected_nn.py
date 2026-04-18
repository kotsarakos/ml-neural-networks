import numpy as np


class FullyConnectedNN:
    """
    Fully Connected Neural Network
    """

    def __init__(self, input_size, hidden_size, output_size, lr=1e-2):
        """
        Initialization of model structure and hyperparameters.

        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of hidden neurons
            output_size (int): Number of output neurons
            lr (float): Learning rate
        """

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        # Parameters
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        # Forward cache
        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None

        # Gradients dictionary
        self.gradients = {}


    def init_parameters(self):
        """
        Initialization of weights and biases using Gaussian distribution:
        - mean = 0 
        - std = 0.1
        """

        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.b1 = np.random.randn(1, self.hidden_size) * 0.1

        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.b2 = np.random.randn(1, self.output_size) * 0.1


    def forward(self, X):
        """
        Forward Pass
        Args:
            X (np.array): Shape (N, p)

        Returns:
            np.array: Predictions (A2)
        """

        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2
    

    def predict(self, X):
        """
        Return binary predictions using threshold 0.5.
        """

        probs = self.forward(X)
        return (probs >= 0.5).astype(int)
    

    def loss(self, X, y):
        """
        Compute Binary Cross Entropy (BCE).

        Args:
            X (np.array)
            y (np.array)

        Returns:
            float
        """

        y_hat = self.forward(X)

        eps = 1e-8
        loss = -np.mean(
            y * np.log(y_hat + eps) +
            (1 - y) * np.log(1 - y_hat + eps)
        )

        return loss
    


    def backward(self, X, y):
        """
        Compute gradients and store them.
        """

        N = X.shape[0]

        # Output layer gradients
        dZ2 = self.A2 - y
        dW2 = (self.A1.T @ dZ2) / N
        db2 = np.sum(dZ2, axis=0, keepdims=True) / N

        # Hidden layer gradients
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = (X.T @ dZ1) / N
        db1 = np.sum(dZ1, axis=0, keepdims=True) / N

        # Store in dictionary
        self.gradients = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2
        }


    def step(self):
        """
        Perform one gradient descent update step.
        """

        self.W1 -= self.lr * self.gradients["dW1"]
        self.b1 -= self.lr * self.gradients["db1"]

        self.W2 -= self.lr * self.gradients["dW2"]
        self.b2 -= self.lr * self.gradients["db2"]


    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))


    def fit(self, X, y, iterations=10000, batch_size=None, show_step=1000):
        """
        Train the model

        Args:
            X (np.array): Shape (N, p)
            y (np.array): Shape (N, 1)
        """

        # Validations
        # Check if X and y are numpy arrays
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays!")

        # Check if X and y have compatible dimensions
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have compatible dimensions!")

        N = X.shape[0]

        # Initialize parameters 
        self.init_parameters()

        # Shuffle data
        indices = np.random.permutation(N)
        X = X[indices]
        y = y[indices]

        # Training loop
        for it in range(iterations):

            # Select batch
            if batch_size is None:
                X_batch = X
                y_batch = y
            else:
                start = (it * batch_size) % N
                end = start + batch_size

                if end <= N:
                    X_batch = X[start:end]
                    y_batch = y[start:end]
                else:
                    # Wrap around
                    X_batch = np.vstack((X[start:], X[:end - N]))
                    y_batch = np.vstack((y[start:], y[:end - N]))

            
            self.forward(X_batch)
            self.backward(X_batch, y_batch)
            self.step()

            # Logging
            if show_step is not None and it % show_step == 0:
                current_loss = self.loss(X, y)
                print(f"Iteration {it}, Loss: {current_loss:.5f}")