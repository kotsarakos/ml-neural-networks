# Neural Networks: From Scratch (NumPy) vs PyTorch

## Overview

This project implements a fully connected neural network from scratch
using NumPy and compares it with an equivalent implementation using
PyTorch. The goal is to understand the internal mechanics of neural
networks while also leveraging modern deep learning frameworks for
efficiency and scalability.

The project includes multiple experiments on both synthetic and
real-world datasets, focusing on model performance, training behavior,
and generalization.

------------------------------------------------------------------------

## Objectives

-   Implement a neural network from first principles (NumPy)
-   Understand forward and backward propagation
-   Apply binary classification using sigmoid activation and binary
    cross-entropy loss
-   Compare manual implementation with PyTorch
-   Evaluate model performance across different datasets
-   Analyze stability using repeated experiments

------------------------------------------------------------------------

## Project Structure

.
├── fully_connected_nn.py
├── experiment1.py
├── experiment2.py
├── experiment3.py
├── helpers.py
├── generate_datasets.py
└── README.md

------------------------------------------------------------------------

## Datasets

### Synthetic Datasets

-   Binary blobs (linearly separable)
-   Flower dataset (non-linearly separable)

### Real Dataset

-   Breast Cancer Wisconsin Dataset (from scikit-learn)

------------------------------------------------------------------------

## Experiments

### Experiment 1

-   Train and evaluate the NumPy neural network on synthetic datasets
-   Visualize decision boundaries
-   Analyze model behavior on simple vs complex patterns

### Experiment 2

-   Apply the NumPy model to a real dataset
-   Perform normalization using training statistics
-   Run multiple iterations (20 runs)
-   Compute mean accuracy and standard deviation

### Experiment 3

-   Reimplement the model using PyTorch
-   Use mini-batch gradient descent
-   Compare performance with NumPy implementation
-   Experiment with different architectures and hyperparameters

------------------------------------------------------------------------

## Key Concepts

-   Forward propagation
-   Backpropagation
-   Binary cross-entropy loss
-   Gradient descent
-   Data normalization and data leakage
-   Model evaluation

------------------------------------------------------------------------

## Requirements

-   Python 3.x
-   NumPy
-   Matplotlib
-   scikit-learn
-   PyTorch

Install dependencies:

pip install numpy matplotlib scikit-learn torch

------------------------------------------------------------------------

## How to Run

python experiment1.py
python experiment2.py
python experiment3.py

------------------------------------------------------------------------

## Notes

-   Input size is inferred from dataset features
-   Normalization uses training set statistics to avoid data leakage
-   Data is shuffled during training for better generalization

------------------------------------------------------------------------

## Conclusion

This project demonstrates both the theoretical and practical aspects of
neural networks, comparing a manual NumPy implementation with a
PyTorch-based approach.

------------------------------------------------------------------------

## Author

Konstantinos Kotsaras
