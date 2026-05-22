# Architecture — Neural Network from Scratch

## Goal
Implement a neural network from scratch using only NumPy.
No PyTorch, no TensorFlow. Every mathematical component
implemented and explained manually.

## Project Structure

### notebooks/
Educational and progressive notebooks. Each one is a
self-contained chapter with math and code together.

| Notebook | Content |
- 01_perceptron: Single neuron, forward pass, activation function 
- 02_activations: ReLU, Sigmoid, Softmax — implementation and visualization 
- 03_loss_functions: MSE, Cross-Entropy — intuition and math 
- 04_backpropagation: Gradients, chain rule — the core of the project 
- 05_optimizers: SGD, Momentum, Adam 
- 06_full_network: Everything integrated, training on MNIST 
- 07_regularization: Dropout, L2 

### neuralnet/
Own library resulting from the notebooks. Clean and modular code.

| File | Content |
- layers.py: Dense layers 
- activations.py: Activation functions 
- losses.py: Loss functions 
- optimizers.py: Optimizers

## Stack
- Python 3.12
- NumPy — matrix operations and gradients
- Matplotlib — visualizations
- Jupyter — educational notebooks

## Development Flow
1. Each concept is first developed in a notebook
2. Once validated, clean code is migrated to neuralnet/
3. At the end, neuralnet/ is a functional mini-library
4. Validated against PyTorch in 06_full_network

## Dataset
MNIST — 70,000 handwritten digit images.
Chosen for being simple, well-known, and sufficient
to demonstrate all concepts.